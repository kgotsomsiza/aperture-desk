"""The public snapshot — what the dashboard reads and what judges see.

Written after every cycle and published to static hosting, so the URL outlives
the trading container. When the desk is switched off on 4 September the final
snapshot stays where it is, permanently, at no cost.

**Everything here is public.** The account identifier, API keys and any local
path must never reach this file. That is enforced by construction: the snapshot
is built from an explicit allowlist of fields rather than by serialising whatever
the broker returned.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .alpaca_cli import AlpacaCLI, AlpacaCliError
from .state import DeskState
from .warden import AuditLog

log = logging.getLogger(__name__)

# Fields copied out of a broker position. Anything not named here is dropped,
# which is what keeps account identifiers out of a public file by default.
POSITION_FIELDS = (
    "symbol",
    "qty",
    "avg_entry_price",
    "current_price",
    "market_value",
    "unrealized_pl",
    "unrealized_plpc",
)

AUDIT_FIELDS = ("ts", "event", "strategy", "underlying", "summary", "rationale", "reason")


def _clean(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 400:
        return value[:400]
    return value


@dataclass
class Snapshot:
    state: DeskState
    audit: AuditLog
    cli: AlpacaCLI

    def build(self) -> dict[str, Any]:
        account = self._account()
        equity = float(account.get("equity") or 0.0)
        start = self.state.start_equity or equity

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "equity": round(equity, 2),
            "start_equity": round(start, 2),
            "total_return_pct": round((equity / start - 1) * 100, 3) if start else 0.0,
            "day_pnl_pct": round(self._day_pnl(equity) * 100, 3),
            "high_water_mark": round(self.state.high_water_mark, 2),
            "drawdown_pct": round(self._drawdown(equity) * 100, 3),
            "open_risk": round(sum(self.state.open_risk_by_underlying().values()), 2),
            "positions": self._positions(),
            "open_trades": self._open_trades(),
            "attribution": self._attribution(),
            "equity_curve": self._equity_curve(),
            "recent_decisions": self._decisions(),
            "counts": {
                "open": len(self.state.open_trades),
                "closed": len(self.state.closed),
                "vetoes": len(self.audit.vetoes(limit=1000)),
            },
        }

    # ------------------------------------------------------------------ #

    def _account(self) -> dict[str, Any]:
        try:
            return self.cli.account()
        except AlpacaCliError as exc:
            log.warning("snapshot could not read the account: %s", exc.stderr[:120])
            return {}

    def _day_pnl(self, equity: float) -> float:
        base = self.state.day_start_equity
        return (equity - base) / base if base > 0 else 0.0

    def _drawdown(self, equity: float) -> float:
        peak = self.state.high_water_mark
        return max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

    def _positions(self) -> list[dict[str, Any]]:
        try:
            raw = self.cli.positions()
        except AlpacaCliError:
            return []
        return [{f: _clean(p.get(f)) for f in POSITION_FIELDS} for p in raw]

    def _open_trades(self) -> list[dict[str, Any]]:
        return [
            {
                "strategy": t.strategy_id,
                "underlying": t.underlying,
                "qty": t.qty,
                "net_price": t.net_price,
                "max_loss": round(t.max_loss, 2),
                "opened_at": t.opened_at,
                "rationale": _clean(t.rationale),
                "legs": t.legs,
            }
            for t in self.state.open_trades.values()
        ]

    def _attribution(self) -> list[dict[str, Any]]:
        """Per-strategy scoreboard. This is what the allocator will act on."""
        table: dict[str, dict[str, Any]] = {}
        for trade in self.state.open_trades.values():
            row = table.setdefault(trade.strategy_id, _blank_row(trade.strategy_id))
            row["open"] += 1
            row["risk_at_work"] += trade.max_loss

        for closed in self.state.closed:
            row = table.setdefault(closed["strategy_id"], _blank_row(closed["strategy_id"]))
            row["closed"] += 1
            pnl = closed.get("pnl")
            if pnl is not None:
                row["realized_pnl"] += float(pnl)
                if float(pnl) > 0:
                    row["wins"] += 1

        for row in table.values():
            decided = row["wins"] + max(row["closed"] - row["wins"], 0)
            row["win_rate"] = round(row["wins"] / decided * 100, 1) if decided else None
            row["risk_at_work"] = round(row["risk_at_work"], 2)
            row["realized_pnl"] = round(row["realized_pnl"], 2)
        return sorted(table.values(), key=lambda r: r["strategy"])

    def _equity_curve(self) -> list[dict[str, Any]]:
        try:
            history = self.cli.portfolio_history(period="1W", timeframe="1H")
        except AlpacaCliError:
            return []
        stamps = history.get("timestamp") or []
        values = history.get("equity") or []

        # Alpaca back-fills an account's history with zeros for the period before
        # it existed, and reports profit_loss against base_value across that gap —
        # so a day-old account shows a -100% curve that never happened. Points at
        # or below zero are that artifact, not a drawdown, and are dropped.
        points = [
            {"t": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(), "equity": v}
            for ts, v in zip(stamps, values)
            if isinstance(v, (int, float)) and v > 0
        ]
        if not points:
            log.info("equity curve is empty: the account has no priced history yet")
        return points

    def _decisions(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.audit.tail(limit=limit)
        return [{f: _clean(r.get(f)) for f in AUDIT_FIELDS if r.get(f) is not None} for r in rows]


def _blank_row(strategy_id: str) -> dict[str, Any]:
    return {
        "strategy": strategy_id,
        "open": 0,
        "closed": 0,
        "wins": 0,
        "realized_pnl": 0.0,
        "risk_at_work": 0.0,
        "win_rate": None,
    }


FORBIDDEN_KEYS = ("account_number", "account_id", "id", "api_key", "secret", "key")


def assert_publishable(payload: dict[str, Any]) -> None:
    """Fail loudly rather than publish an identifier.

    The snapshot is built from an allowlist, so this should never fire — which is
    exactly why it is worth asserting. A silent leak into a public file is not
    recoverable by deleting the file afterwards.
    """
    blob = json.dumps(payload).lower()
    for needle in FORBIDDEN_KEYS:
        if f'"{needle}"' in blob:
            raise ValueError(f"snapshot contains a forbidden key: {needle}")
    for marker in ("c:\\\\users", "/home/", "pk", "sk-"):
        if marker in blob and marker in ("c:\\\\users", "/home/"):
            raise ValueError(f"snapshot contains a local path: {marker}")


def write(payload: dict[str, Any], path: Path | str = "public/snapshot.json") -> Path:
    assert_publishable(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path
