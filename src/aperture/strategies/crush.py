"""CRUSH — the flagship edge.

One sentence: *the market is pricing an X% earnings move; this name has
historically moved Y%; trade the difference.*

Implied volatility into an earnings announcement embeds an event premium that
collapses the moment the news is out. That premium is usually, but not always,
richer than the move the stock actually delivers. So the desk measures both
sides rather than assuming either:

  * **Implied** — the at-the-money straddle on the first expiry after the
    report, with ordinary day-to-day volatility stripped out in variance space,
    leaving the event component alone.
  * **Realized** — the absolute close-to-close move across each of the last
    eight reports.

Rich enough, and it sells a defined-risk iron condor. Cheap enough, and it buys
a strangle instead. In between, it does nothing, which is most of the time.

Every position is opened on the last close before the announcement and closed the
morning after. The desk never holds an event position into a second night.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ..contracts import Right
from ..earnings import EarningsCalendar, EarningsEvent
from ..marketdata import (
    MarketData,
    Snapshot,
    atm_pair,
    compare_moves,
    for_expiry,
    nearest_expiry_after,
    select_by_strike_offset,
)
from ..risk import BookState, Proposal
from .base import (
    StrategyConfig,
    build_iron_condor,
    build_long_strangle,
    credit_to_width_ok,
)

DEFAULT_CONFIG = StrategyConfig(
    strategy_id="CRUSH",
    sleeve="BALLAST",
    weight=0.20,
    min_dte=1,
    max_dte=14,
    spread_width=5.0,
    min_credit_to_width=0.18,
    params={
        # Sell only when implied is at least this multiple of the median realized
        # move. Below ~1.25 the premium does not pay for the gap risk.
        "rich_threshold": 1.25,
        "cheap_threshold": 0.85,
        # Short strikes at this multiple of the implied move: outside what the
        # market itself expects.
        "short_strike_multiple": 1.0,
        "min_history": 4,
    },
)


@dataclass
class CrushStrategy:
    config: StrategyConfig = field(default_factory=lambda: DEFAULT_CONFIG.child())
    calendar: EarningsCalendar = field(default_factory=EarningsCalendar)

    def propose(self, md: MarketData, book: BookState, budget: float) -> list[Proposal]:
        if not self.config.enabled or budget <= 0:
            return []

        today = book.now.date()
        # Only events announcing tonight or tomorrow morning are actionable: the
        # premium has not yet collapsed and the position is a single night old.
        window = self.calendar.upcoming(today, today + timedelta(days=1))
        events = [e for e in window if e.enter_on == today]
        if not events:
            return []

        per_event = min(
            budget / len(events), book.equity * self.config.max_trade_loss_pct
        )
        proposals: list[Proposal] = []
        for event in events:
            if book.open_risk_by_underlying.get(event.symbol, 0.0) > 0:
                continue
            proposal = self._for_event(md, event, per_event, today)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    # ------------------------------------------------------------------ #

    def _for_event(
        self, md: MarketData, event: EarningsEvent, budget: float, today: date
    ) -> Proposal | None:
        spot = md.spot(event.symbol)
        if spot <= 0:
            return None

        snapshots = md.chain(
            event.symbol,
            min_dte=self.config.min_dte,
            max_dte=self.config.max_dte,
            strike_band=0.35,  # earnings moves are large; a tight band misses the wings
            spot=spot,
        )
        if not snapshots:
            return None

        expiry = nearest_expiry_after(snapshots.values(), event.first_session_after)
        if expiry is None:
            return None

        pair = atm_pair(snapshots.values(), spot, expiry)
        if pair is None:
            return None
        call, put = pair

        bars = md.daily_bars(event.symbol)
        history = self.calendar.past_events(event.symbol)
        comparison = compare_moves(
            event.symbol, expiry, call, put, spot, bars, history, asof=today
        )
        if comparison is None:
            return None
        if len(comparison.realized_moves) < int(self.config.params["min_history"]):
            return None

        pool = for_expiry(snapshots.values(), expiry)
        if comparison.richness >= float(self.config.params["rich_threshold"]):
            return self._sell_the_event(pool, comparison, spot, budget)
        if comparison.richness <= float(self.config.params["cheap_threshold"]):
            return self._buy_the_event(pool, comparison, spot, budget)
        return None

    # ------------------------------------------------------------------ #

    def _sell_the_event(
        self, pool: list[Snapshot], comparison, spot: float, budget: float
    ) -> Proposal | None:
        """Iron condor with short strikes outside the market's own expected move."""
        offset = spot * comparison.implied_event_move * float(
            self.config.params["short_strike_multiple"]
        )
        short_put = _nearest_strike(pool, spot - offset, Right.PUT)
        short_call = _nearest_strike(pool, spot + offset, Right.CALL)
        if short_put is None or short_call is None or short_put.strike >= short_call.strike:
            return None

        width = max(self.config.spread_width, round(spot * 0.02, 0))
        long_put = select_by_strike_offset(pool, short_put, width, Right.PUT)
        long_call = select_by_strike_offset(pool, short_call, width, Right.CALL)
        if long_put is None or long_call is None:
            return None

        proposal = build_iron_condor(
            short_put,
            long_put,
            short_call,
            long_call,
            strategy_id=self.config.strategy_id,
            underlying=comparison.underlying,
            budget=budget,
            slippage=self.config.slippage,
            aggression=self.config.aggression,
            rationale=f"SELL event premium - {comparison.explain()}",
        )
        if proposal is None or not credit_to_width_ok(proposal, self.config.min_credit_to_width):
            return None
        return proposal

    def _buy_the_event(
        self, pool: list[Snapshot], comparison, spot: float, budget: float
    ) -> Proposal | None:
        """Long strangle when the market is underpricing what this name does."""
        offset = spot * comparison.implied_event_move
        call = _nearest_strike(pool, spot + offset * 0.5, Right.CALL)
        put = _nearest_strike(pool, spot - offset * 0.5, Right.PUT)
        if call is None or put is None:
            return None
        return build_long_strangle(
            call,
            put,
            strategy_id=self.config.strategy_id,
            underlying=comparison.underlying,
            budget=budget,
            slippage=self.config.slippage,
            aggression=self.config.aggression,
            rationale=f"BUY event premium - {comparison.explain()}",
        )


def _nearest_strike(pool: list[Snapshot], target: float, right: Right) -> Snapshot | None:
    candidates = [s for s in pool if s.right is right and s.is_priceable]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s.strike - target))


def should_close(event: EarningsEvent, today: date) -> bool:
    """CRUSH holds for exactly one night. The premium is gone by the open."""
    return today >= event.first_session_after
