"""CARRY — the ballast engine.

Harvests the variance risk premium: index implied volatility has historically
exceeded subsequently realized volatility, so systematically selling defined-risk
index premium is positive carry. This is the sleeve that pays the bills and
produces the smooth, explainable equity curve. It is not the sleeve that wins the
P&L leaderboard, and it is not trying to be.

Structure: 10-16 delta iron condors on liquid index ETFs, 7-21 DTE, closed at
half the credit or stopped at twice it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..contracts import Right
from ..marketdata import (
    MarketData,
    Snapshot,
    for_expiry,
    select_by_delta,
    select_by_strike_offset,
)
from ..risk import BookState, Proposal
from .base import (
    StrategyConfig,
    build_iron_condor,
    credit_to_width_ok,
)

DEFAULT_CONFIG = StrategyConfig(
    strategy_id="CARRY",
    sleeve="BALLAST",
    weight=0.70,
    universe=("SPY", "QQQ", "IWM"),
    min_dte=7,
    max_dte=21,
    target_short_delta=0.15,
    spread_width=5.0,
    min_credit_to_width=0.15,
    take_profit_pct=0.50,
    stop_loss_multiple=2.0,
)


@dataclass
class CarryStrategy:
    config: StrategyConfig = field(default_factory=lambda: DEFAULT_CONFIG)

    def propose(self, md: MarketData, book: BookState, budget: float) -> list[Proposal]:
        if not self.config.enabled or budget <= 0:
            return []

        per_name = budget / max(len(self.config.universe), 1)
        proposals: list[Proposal] = []

        for underlying in self.config.universe:
            if book.open_risk_by_underlying.get(underlying, 0.0) > 0:
                continue  # one structure per name at a time; the book is not a warehouse
            proposal = self._condor_for(md, underlying, per_name)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    # ------------------------------------------------------------------ #

    def _condor_for(self, md: MarketData, underlying: str, budget: float) -> Proposal | None:
        spot = md.spot(underlying)
        if spot <= 0:
            return None

        snapshots = md.chain(
            underlying,
            min_dte=self.config.min_dte,
            max_dte=self.config.max_dte,
            strike_band=0.15,
            spot=spot,
        )
        if not snapshots:
            return None

        expiry = self._pick_expiry(snapshots.values())
        if expiry is None:
            return None
        pool = for_expiry(snapshots.values(), expiry)

        short_put = select_by_delta(pool, self.config.target_short_delta, Right.PUT)
        short_call = select_by_delta(pool, self.config.target_short_delta, Right.CALL)
        if short_put is None or short_call is None:
            return None
        if short_put.strike >= short_call.strike:
            return None  # the short strikes have crossed; the chain is unusable

        long_put = select_by_strike_offset(pool, short_put, self.config.spread_width, Right.PUT)
        long_call = select_by_strike_offset(pool, short_call, self.config.spread_width, Right.CALL)
        if long_put is None or long_call is None:
            return None

        rationale = (
            f"VRP harvest: {underlying} {expiry:%d %b} iron condor, "
            f"shorts at {abs(short_put.delta or 0):.2f}/{abs(short_call.delta or 0):.2f} delta "
            f"({short_put.strike:g}p / {short_call.strike:g}c), spot {spot:.2f}"
        )

        proposal = build_iron_condor(
            short_put,
            long_put,
            short_call,
            long_call,
            strategy_id=self.config.strategy_id,
            underlying=underlying,
            budget=budget,
            slippage=self.config.slippage,
            rationale=rationale,
        )
        if proposal is None:
            return None
        if not credit_to_width_ok(proposal, self.config.min_credit_to_width):
            return None
        return proposal

    def _pick_expiry(self, snapshots) -> date | None:
        """Prefer the middle of the DTE window: enough theta, not yet gamma-cursed."""
        today = date.today()
        candidates = sorted(
            {
                s.expiry
                for s in snapshots
                if self.config.min_dte <= (s.expiry - today).days <= self.config.max_dte
            }
        )
        if not candidates:
            return None
        target = (self.config.min_dte + self.config.max_dte) / 2
        return min(candidates, key=lambda e: abs((e - today).days - target))


def exit_signal(
    entry_credit: float, current_price: float, config: StrategyConfig
) -> str | None:
    """Whether an open CARRY structure should be closed, and why.

    ``entry_credit`` and ``current_price`` are both in Alpaca's sign convention,
    so a credit spread opens negative and is bought back at a negative price too.
    """
    credit = abs(entry_credit)
    if credit <= 0:
        return None
    cost_to_close = abs(current_price)

    if cost_to_close <= credit * (1 - config.take_profit_pct):
        return f"take profit: closing for {cost_to_close:.2f} against a {credit:.2f} credit"
    if cost_to_close >= credit * config.stop_loss_multiple:
        return f"stop loss: {cost_to_close:.2f} is {config.stop_loss_multiple:g}x the credit"
    return None
