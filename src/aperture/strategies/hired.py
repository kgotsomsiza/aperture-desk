"""Live expression of a strategy promoted by the research lab.

The historical simulator selects condors by moneyness because Alpaca does not
publish historical greeks.  This strategy deliberately uses the same parameters
live, so a promoted candidate is the thing that actually trades—not a vaguely
similar hand-written CARRY configuration wearing the candidate's name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..contracts import Right
from ..marketdata import MarketData, Snapshot, for_expiry, select_by_strike_offset
from ..risk import BookState, Proposal
from .base import StrategyConfig, build_iron_condor, credit_to_width_ok


@dataclass
class HiredCondorStrategy:
    record: dict[str, Any]
    config: StrategyConfig = field(init=False)

    def __post_init__(self) -> None:
        spec = self.record.get("spec") or {}
        target = int(spec.get("dte_target", 14))
        self.config = StrategyConfig(
            strategy_id=str(self.record["strategy_id"]),
            sleeve="BALLAST",
            weight=0.0,
            universe=(str(self.record.get("underlying") or "SPY"),),
            # Match the historical selector exactly: any listed expiry from
            # five days out through target + 14, then choose the nearest target.
            min_dte=5,
            max_dte=target + 14,
            min_credit_to_width=float(spec.get("min_credit_to_width", 0.15)),
            take_profit_pct=float(spec.get("take_profit", 0.50)),
            stop_loss_multiple=float(spec.get("stop_multiple", 2.0)),
            slippage=float(spec.get("slippage", 0.05)),
            params={
                "short_pct": float(spec.get("short_pct", 0.04)),
                "width_pct": float(spec.get("width_pct", 0.01)),
                "dte_target": target,
            },
        )

    def propose(self, md: MarketData, book: BookState, budget: float) -> list[Proposal]:
        if not self.config.enabled or budget <= 0:
            return []
        underlying = self.config.universe[0]
        if book.open_risk_by_underlying.get(underlying, 0.0) > 0:
            return []
        proposal = self._condor_for(md, book, underlying, budget)
        return [proposal] if proposal is not None else []

    def _condor_for(
        self, md: MarketData, book: BookState, underlying: str, budget: float
    ) -> Proposal | None:
        spot = md.spot(underlying)
        if spot <= 0:
            return None

        short_pct = float(self.config.params["short_pct"])
        width_pct = float(self.config.params["width_pct"])
        snapshots = md.chain(
            underlying,
            min_dte=self.config.min_dte,
            max_dte=self.config.max_dte,
            strike_band=max(short_pct + width_pct + 0.04, 0.12),
            spot=spot,
        )
        if not snapshots:
            return None

        expiry = self._pick_expiry(snapshots.values(), book.now.date())
        if expiry is None:
            return None
        pool = for_expiry(snapshots.values(), expiry)

        short_put = _nearest(pool, spot * (1 - short_pct), Right.PUT)
        short_call = _nearest(pool, spot * (1 + short_pct), Right.CALL)
        if short_put is None or short_call is None or short_put.strike >= short_call.strike:
            return None

        width = max(spot * width_pct, 1.0)
        long_put = select_by_strike_offset(pool, short_put, width, Right.PUT)
        long_call = select_by_strike_offset(pool, short_call, width, Right.CALL)
        if long_put is None or long_call is None:
            return None

        evidence = self.record.get("backtest") or {}
        rationale = (
            f"research hire {self.config.strategy_id}: {underlying} condor at "
            f"{short_pct:.1%} moneyness / {width_pct:.1%} wings; "
            f"historical edge {float(evidence.get('edge') or 0):+.1%}, "
            f"t={float(evidence.get('t_stat') or 0):.2f} over "
            f"{int(evidence.get('trades') or 0)} trades"
        )
        proposal = build_iron_condor(
            short_put,
            long_put,
            short_call,
            long_call,
            strategy_id=self.config.strategy_id,
            underlying=underlying,
            budget=min(budget, book.equity * self.config.max_trade_loss_pct),
            slippage=self.config.slippage,
            rationale=rationale,
        )
        if proposal is None:
            return None
        return proposal if credit_to_width_ok(proposal, self.config.min_credit_to_width) else None

    def _pick_expiry(self, snapshots, today: date) -> date | None:
        target = int(self.config.params["dte_target"])
        candidates = sorted(
            {
                snap.expiry
                for snap in snapshots
                if self.config.min_dte <= (snap.expiry - today).days <= self.config.max_dte
            }
        )
        return min(candidates, key=lambda expiry: abs((expiry - today).days - target)) \
            if candidates else None


def _nearest(pool: list[Snapshot], target: float, right: Right) -> Snapshot | None:
    candidates = [snap for snap in pool if snap.right is right and snap.is_priceable]
    return min(candidates, key=lambda snap: abs(snap.strike - target)) if candidates else None
