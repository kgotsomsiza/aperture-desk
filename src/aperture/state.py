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


@dataclass
class OpenTrade:
    """One structure the desk believes it owns."""

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
    rationale: str = ""
    order_id: str | None = None
    status: str = "pending"  # pending -> open -> closed

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
    open_trades: dict[str, OpenTrade] = field(default_factory=dict)
    closed: list[dict[str, Any]] = field(default_factory=list)
    allocations: dict[str, float] = field(default_factory=dict)

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
            open_trades=trades,
            closed=raw.get("closed") or [],
            allocations=raw.get("allocations") or {},
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "high_water_mark": self.high_water_mark,
            "day_start_equity": self.day_start_equity,
            "day_stamp": self.day_stamp,
            "start_equity": self.start_equity,
            "account_fingerprint": self.account_fingerprint,
            "open_trades": {k: asdict(v) for k, v in self.open_trades.items()},
            "closed": self.closed,
            "allocations": self.allocations,
        }
        # Write to a sibling then replace, so a crash mid-write cannot leave a
        # truncated ledger that the loader would refuse on the next cycle.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    # -- bookkeeping ------------------------------------------------------ #

    def observe_equity(self, equity: float, today: date) -> None:
        stamp = today.isoformat()
        if self.start_equity <= 0:
            self.start_equity = equity
        if self.day_stamp != stamp:
            self.day_stamp = stamp
            self.day_start_equity = equity
        self.high_water_mark = max(self.high_water_mark, equity)

    def record_open(self, trade: OpenTrade) -> None:
        self.open_trades[trade.client_order_id] = trade

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

    def trades_for(self, strategy_id: str) -> list[OpenTrade]:
        return [t for t in self.open_trades.values() if t.strategy_id == strategy_id]

    def reconcile(self, broker_symbols: set[str]) -> list[str]:
        """Drop trades the broker no longer shows, and report what was dropped.

        Positions disappear for reasons the desk did not initiate — expiry,
        assignment, auto-exercise. Carrying a phantom trade in the ledger would
        keep consuming a strategy's budget forever.
        """
        vanished = [
            key
            for key, trade in self.open_trades.items()
            if trade.status == "open" and not (trade.leg_set & broker_symbols)
        ]
        for key in vanished:
            log.warning("reconcile: %s no longer at broker, closing in ledger", key)
            self.record_close(key, reason="vanished at broker (expiry/assignment?)")
        return vanished
