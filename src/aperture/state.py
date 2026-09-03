"""Persisted desk state.

Alpaca knows what positions exist. It does not know which strategy opened them,
what the structure's maximum loss was at entry, or what the desk paid. Without
that, the risk engine cannot enforce a per-strategy budget and the allocator
cannot attribute P&L — so the desk keeps its own ledger alongside the broker's.

The file is the source of truth for *intent*; the broker is the source of truth
for *reality*. Where they disagree, reality wins and the discrepancy is logged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def audit_path_for(state_path: "Path | str") -> Path:
    """The audit trail belonging to one ledger.

    Derived from the ledger's own name, never a fixed filename. A shared
    audit.jsonl silently merges accounts, and the allocator reads that file for
    its veto-rate signal -- so a strategy refused repeatedly while testing on a
    throwaway account would be fired on the scored account before it had placed
    a single trade there. The evidence must belong to the book it describes.
    """
    path = Path(state_path)
    return path.with_name(f"{path.stem}.audit.jsonl")


@dataclass
class OpenTrade:
    """One submitted structure and its broker-confirmed lifecycle.

    An accepted limit order is only a reservation (``pending_entry``).  It does
    not become ``open`` until Alpaca reports a fill, and a close does not leave
    the ledger until its own mleg order fills.  Keeping those states explicit is
    what prevents an unfilled day order from being mistaken for a vanished
    position and submitted again every cycle.
    """

    client_order_id: str
    strategy_id: str
    underlying: str
    legs: list[str]
    qty: int
    net_price: float  # Alpaca sign convention: + debit, - credit
    max_loss: float
    opened_at: str
    # Side per leg, kept explicitly: the structure's closing price cannot be
    # reconstructed from the net price alone for anything past two legs.
    leg_sides: dict[str, str] = field(default_factory=dict)
    leg_ratios: dict[str, int] = field(default_factory=dict)
    rationale: str = ""
    order_id: str | None = None
    status: str = "pending_entry"
    filled_at: str | None = None
    close_order_id: str | None = None
    close_client_order_id: str | None = None
    close_submitted_at: str | None = None
    close_reason: str | None = None
    close_limit_price: float | None = None
    close_attempts: int = 0
    partial_close_qty: int = 0
    partial_close_pnl: float = 0.0
    partial_close_risk: float = 0.0
    exit_policy: dict[str, Any] = field(default_factory=dict)

    @property
    def leg_set(self) -> set[str]:
        return set(self.legs)


@dataclass
class DeskState:
    path: Path = Path("state/desk.json")
    high_water_mark: float = 0.0
    day_start_equity: float = 0.0
    day_stamp: str = ""
    start_equity: float = 0.0
    # Hash of the account this ledger was created against. Only the hash: the
    # account number is a submission-form value, not a file value.
    account_fingerprint: str = ""
    # How far through the half-spread the desk currently reaches for a
    # fill. Learned from observed outcomes; persisted so a restart does
    # not throw away what a whole session taught it.
    aggression: float = 0.60
    open_trades: dict[str, OpenTrade] = field(default_factory=dict)
    closed: list[dict[str, Any]] = field(default_factory=list)
    allocations: dict[str, float] = field(default_factory=dict)
    hired_strategies: list[dict[str, Any]] = field(default_factory=list)
    research_history: list[dict[str, Any]] = field(default_factory=list)
    research_trials: int = 0
    last_research_date: str = ""
    latest_letter: dict[str, Any] = field(default_factory=dict)
    last_letter_date: str = ""

    # -- persistence ------------------------------------------------------ #

    @classmethod
    def load(cls, path: Path | str = "state/desk.json") -> "DeskState":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt ledger must not silently become an empty one: an empty
            # ledger reads as "no open risk", which would let the desk double up.
            raise RuntimeError(f"desk state at {path} is unreadable: {exc}") from exc

        trades = {k: OpenTrade(**v) for k, v in (raw.get("open_trades") or {}).items()}
        return cls(
            path=path,
            high_water_mark=raw.get("high_water_mark", 0.0),
            day_start_equity=raw.get("day_start_equity", 0.0),
            day_stamp=raw.get("day_stamp", ""),
            start_equity=raw.get("start_equity", 0.0),
            account_fingerprint=raw.get("account_fingerprint", ""),
            aggression=float(raw.get("aggression", 0.60)),
            open_trades=trades,
            closed=raw.get("closed") or [],
            allocations=raw.get("allocations") or {},
            hired_strategies=raw.get("hired_strategies") or [],
            research_history=raw.get("research_history") or [],
            research_trials=int(raw.get("research_trials") or 0),
            last_research_date=raw.get("last_research_date", ""),
            latest_letter=raw.get("latest_letter") or {},
            last_letter_date=raw.get("last_letter_date", ""),
        )

    def save(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "high_water_mark": self.high_water_mark,
            "day_start_equity": self.day_start_equity,
            "day_stamp": self.day_stamp,
            "start_equity": self.start_equity,
            "account_fingerprint": self.account_fingerprint,
            "aggression": self.aggression,
            "open_trades": {k: asdict(v) for k, v in self.open_trades.items()},
            "closed": self.closed,
            "allocations": self.allocations,
            "hired_strategies": self.hired_strategies,
            "research_history": self.research_history[-30:],
            "research_trials": self.research_trials,
            "last_research_date": self.last_research_date,
            "latest_letter": self.latest_letter,
            "last_letter_date": self.last_letter_date,
        }
        # Write to a sibling then replace, so a crash mid-write cannot leave a
        # truncated ledger that the loader would refuse on the next cycle.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    # -- bookkeeping ------------------------------------------------------ #

    def observe_equity(
        self,
        equity: float,
        today: date,
        *,
        prior_close_equity: float | None = None,
    ) -> None:
        """Record equity without mistaking the first intraday mark for the open.

        Alpaca's ``last_equity`` is the prior session's official close.  That is
        the baseline a daily-loss breaker needs: using the first runner sample of
        a new date silently erases any overnight or opening-auction loss.

        ``prior_close_equity`` remains optional for simulations and old callers,
        but a valid broker value always wins over the locally sampled fallback.
        """
        stamp = today.isoformat()
        if self.start_equity <= 0:
            self.start_equity = equity
        if self.day_stamp != stamp:
            self.day_stamp = stamp
            self.day_start_equity = (
                prior_close_equity
                if prior_close_equity is not None and prior_close_equity > 0
                else equity
            )
        elif prior_close_equity is not None and prior_close_equity > 0:
            # Repair a process that already sampled this session using the old
            # first-observation behaviour.  ``last_equity`` is stable intraday.
            self.day_start_equity = prior_close_equity
        self.high_water_mark = max(
            self.high_water_mark,
            equity,
            prior_close_equity or 0.0,
        )

    def record_open(self, trade: OpenTrade) -> None:
        self.open_trades[trade.client_order_id] = trade

    def confirm_entry(
        self,
        client_order_id: str,
        *,
        qty: int,
        net_price: float,
        max_loss: float,
        filled_at: str | None,
    ) -> OpenTrade | None:
        """Promote a reserved entry only after the broker reports a fill."""
        trade = self.open_trades.get(client_order_id)
        if trade is None or qty <= 0:
            return None
        trade.qty = qty
        trade.net_price = net_price
        trade.max_loss = max_loss
        trade.status = "open"
        trade.filled_at = filled_at
        if filled_at:
            trade.opened_at = filled_at
        return trade

    def discard_pending(self, client_order_id: str) -> OpenTrade | None:
        """Remove an entry that reached a terminal state without any fill.

        It deliberately does not enter ``closed``: an order that never traded
        is execution evidence, not a losing trade for the allocator.
        """
        trade = self.open_trades.get(client_order_id)
        if trade is None or trade.status not in {"submitting_entry", "pending_entry"}:
            return None
        return self.open_trades.pop(client_order_id)

    def mark_close_pending(
        self,
        client_order_id: str,
        *,
        order_id: str | None,
        close_client_order_id: str,
        reason: str,
        submitted_at: str,
        limit_price: float,
    ) -> OpenTrade | None:
        trade = self.open_trades.get(client_order_id)
        if trade is None:
            return None
        trade.status = "pending_close"
        trade.close_order_id = order_id
        trade.close_client_order_id = close_client_order_id
        trade.close_submitted_at = submitted_at
        trade.close_reason = reason
        trade.close_limit_price = limit_price
        trade.close_attempts += 1
        trade.status = "submitting_close"
        return trade

    def confirm_close_submission(
        self,
        client_order_id: str,
        *,
        order_id: str | None,
        limit_price: float,
    ) -> OpenTrade | None:
        trade = self.open_trades.get(client_order_id)
        if trade is None or trade.status not in {"submitting_close", "pending_close"}:
            return None
        trade.close_order_id = order_id
        trade.close_limit_price = limit_price
        trade.status = "pending_close"
        return trade

    def reopen_after_unfilled_close(self, client_order_id: str) -> OpenTrade | None:
        trade = self.open_trades.get(client_order_id)
        if trade is None or trade.status not in {"submitting_close", "pending_close"}:
            return None
        trade.status = "open"
        trade.close_order_id = None
        trade.close_client_order_id = None
        trade.close_submitted_at = None
        trade.close_reason = None
        trade.close_limit_price = None
        return trade

    def record_filled_close(
        self,
        client_order_id: str,
        *,
        qty: int,
        reason: str,
        pnl: float,
        close_price: float,
        closed_at: str | None,
    ) -> dict[str, Any] | None:
        """Record a confirmed full or partial close and retain any remainder."""
        trade = self.open_trades.get(client_order_id)
        if trade is None or qty <= 0:
            return None

        closed_qty = min(qty, trade.qty)
        risk_per_unit = trade.max_loss / trade.qty if trade.qty > 0 else 0.0
        closed_risk = risk_per_unit * closed_qty
        if closed_qty >= trade.qty:
            row = {
                **asdict(trade),
                "qty": trade.partial_close_qty + closed_qty,
                "max_loss": trade.partial_close_risk + closed_risk,
                "status": "closed",
                "closed_at": closed_at or datetime.now(timezone.utc).isoformat(),
                "close_reason": reason,
                "close_price": close_price,
                "pnl": round(trade.partial_close_pnl + pnl, 2),
            }
            self.closed.append(row)
            self.open_trades.pop(client_order_id, None)
        else:
            trade.partial_close_qty += closed_qty
            trade.partial_close_pnl += pnl
            trade.partial_close_risk += closed_risk
            trade.qty -= closed_qty
            trade.max_loss = risk_per_unit * trade.qty
            trade.status = "open"
            trade.close_order_id = None
            trade.close_client_order_id = None
            trade.close_submitted_at = None
            trade.close_reason = None
            trade.close_limit_price = None
            return None
        return row

    def record_close(self, client_order_id: str, reason: str, pnl: float | None = None) -> None:
        trade = self.open_trades.pop(client_order_id, None)
        if trade is None:
            return
        self.closed.append(
            {
                **asdict(trade),
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "close_reason": reason,
                "pnl": pnl,
            }
        )

    # -- views the risk engine needs -------------------------------------- #

    def open_risk_by_strategy(self) -> dict[str, float]:
        table: dict[str, float] = {}
        for trade in self.open_trades.values():
            table[trade.strategy_id] = table.get(trade.strategy_id, 0.0) + trade.max_loss
        return table

    def open_risk_by_underlying(self) -> dict[str, float]:
        table: dict[str, float] = {}
        for trade in self.open_trades.values():
            table[trade.underlying] = table.get(trade.underlying, 0.0) + trade.max_loss
        return table

    def open_expiries_by_strategy_underlying(self) -> dict[tuple[str, str], set]:
        """Which expiries each sleeve already holds in each name.

        A second structure on the same name but a *different* expiry is a
        different position, not averaging down. The convex sleeve needs that
        distinction to add a short-dated layer beside a longer-dated one.
        """
        from .contracts import parse_occ

        table: dict[tuple[str, str], set] = {}
        for trade in self.open_trades.values():
            key = (trade.strategy_id, trade.underlying)
            for leg in trade.legs:
                try:
                    table.setdefault(key, set()).add(parse_occ(leg).expiry)
                except Exception:  # noqa: BLE001 - a malformed leg is not fatal
                    continue
        return table

    def open_risk_by_strategy_underlying(self) -> dict[tuple[str, str], float]:
        """Open risk keyed by *both* strategy and name.

        The by-underlying view alone conflates sleeves that hold opposite
        exposure. CONVEX buys movement on SPY while CARRY sells it; asking only
        "does the book hold SPY" tells CONVEX it is already positioned when it
        holds nothing at all.
        """
        table: dict[tuple[str, str], float] = {}
        for trade in self.open_trades.values():
            key = (trade.strategy_id, trade.underlying)
            table[key] = table.get(key, 0.0) + trade.max_loss
        return table

    def trades_for(self, strategy_id: str) -> list[OpenTrade]:
        return [t for t in self.open_trades.values() if t.strategy_id == strategy_id]

    def reconcile(
        self,
        broker_symbols: set[str],
        *,
        now: datetime | None = None,
        fill_grace_seconds: float = 600.0,
    ) -> list[str]:
        """Drop trades the broker no longer shows, and report what was dropped.

        Positions disappear for reasons the desk did not initiate — expiry,
        assignment, auto-exercise. Carrying a phantom trade in the ledger would
        keep consuming a strategy's budget forever.
        """
        now = now or datetime.now(timezone.utc)
        vanished = []
        for key, trade in self.open_trades.items():
            if trade.status != "open" or trade.leg_set & broker_symbols:
                continue
            # The order endpoint can report a fill a few seconds before the
            # positions endpoint reflects its legs.  Do not turn that normal
            # propagation race into another phantom close.
            if trade.filled_at:
                try:
                    stamp = datetime.fromisoformat(trade.filled_at.replace("Z", "+00:00"))
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    if (now.astimezone(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds() \
                            <= fill_grace_seconds:
                        continue
                except ValueError:
                    pass
            vanished.append(key)
        for key in vanished:
            log.warning("reconcile: %s no longer at broker, closing in ledger", key)
            self.record_close(key, reason="vanished at broker (expiry/assignment?)")
        return vanished
