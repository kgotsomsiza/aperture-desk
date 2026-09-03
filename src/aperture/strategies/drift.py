"""DRIFT — post-earnings announcement drift.

The oldest documented anomaly in equity markets: after a large earnings surprise,
prices keep moving in the direction of the surprise for weeks rather than
repricing instantly. Under-reaction, not over-reaction.

The desk expresses it as a **debit vertical** in the direction of the gap rather
than as long stock or a naked long call. Two reasons. A vertical caps the cost at
entry, so a drift that reverses is bounded by the premium paid. And it sells the
expensive further-out-of-the-money leg back to the market, which pays for a
chunk of the position — buying the gap outright means paying elevated
post-earnings volatility on the whole thing.

This is the only strategy with something to trade on day one: the cohort that
reported on 26-27 August is already sitting there when the hackathon opens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Sequence

from ..contracts import Right
from ..earnings import RECENTLY_REPORTED, EarningsCalendar, EarningsEvent
from ..marketdata import MarketData, Snapshot, for_expiry, select_by_delta, _parse_ts
from ..risk import BookState, Proposal
from .base import StrategyConfig, build_vertical, debit_to_width_ok

DEFAULT_CONFIG = StrategyConfig(
    strategy_id="DRIFT",
    sleeve="CONVEX",
    weight=0.15,
    min_dte=14,
    max_dte=45,
    target_short_delta=0.25,  # the leg we sell
    params={
        "long_delta": 0.45,       # the leg we buy, slightly in the money
        "min_gap": 0.03,          # ignore anything under a 3% surprise
        "max_gap": 0.25,          # above this the move is news, not drift
        "max_sessions_since": 5,  # the effect decays; do not chase week-old gaps
        "max_debit_to_width": 0.45,
        "width_pct": 0.05,        # spread width as a fraction of spot
    },
)


def earnings_gap(bars: Sequence[dict[str, Any]], event: EarningsEvent) -> float | None:
    """Signed close-to-close move across the announcement.

    Positive means the stock gapped up. Returns ``None`` when the session that
    carries the gap has not printed yet — which is the correct answer the evening
    of an after-close report, not a reason to guess.
    """
    if not bars:
        return None

    indexed = {}
    for position, bar in enumerate(bars):
        if bar.get("t") and bar.get("c"):
            indexed[_parse_ts(bar["t"]).date()] = position

    session = event.first_session_after
    after = next((d for d in sorted(indexed) if d >= session), None)
    if after is None:
        return None

    index = indexed[after]
    if index == 0:
        return None

    before_close = float(bars[index - 1]["c"])
    after_close = float(bars[index]["c"])
    if before_close <= 0:
        return None
    return after_close / before_close - 1


def sessions_since(bars: Sequence[dict[str, Any]], event: EarningsEvent) -> int:
    """Trading sessions elapsed since the gap session, counted from the bars."""
    session = event.first_session_after
    dates = sorted(
        _parse_ts(b["t"]).date() for b in bars if b.get("t")
    )
    return sum(1 for d in dates if d > session)


@dataclass
class DriftStrategy:
    config: StrategyConfig = field(default_factory=lambda: DEFAULT_CONFIG.child())
    calendar: EarningsCalendar = field(default_factory=EarningsCalendar)
    cohort: tuple[EarningsEvent, ...] = RECENTLY_REPORTED

    def propose(self, md: MarketData, book: BookState, budget: float) -> list[Proposal]:
        if not self.config.enabled or budget <= 0:
            return []

        today = book.now.date()
        candidates = [
            event
            for event in self.cohort
            if event.first_session_after <= today
            and book.open_risk_by_underlying.get(event.symbol, 0.0) == 0.0
        ]
        if not candidates:
            return []

        per_name = min(
            budget / len(candidates), book.equity * self.config.max_trade_loss_pct
        )
        proposals: list[Proposal] = []
        for event in candidates:
            proposal = self._for_event(md, event, per_name, today)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    # ------------------------------------------------------------------ #

    def _for_event(
        self, md: MarketData, event: EarningsEvent, budget: float, today: date
    ) -> Proposal | None:
        bars = md.daily_bars(event.symbol, lookback_days=90)
        gap = earnings_gap(bars, event)
        if gap is None:
            return None

        params = self.config.params
        magnitude = abs(gap)
        if magnitude < float(params["min_gap"]):
            return None
        if magnitude > float(params["max_gap"]):
            # A move this size is a re-rating, not a drift. Different game.
            return None
        if sessions_since(bars, event) > int(params["max_sessions_since"]):
            return None

        spot = md.spot(event.symbol)
        if spot <= 0:
            return None

        bullish = gap > 0
        right = Right.CALL if bullish else Right.PUT
        snapshots = md.chain(
            event.symbol,
            min_dte=self.config.min_dte,
            max_dte=self.config.max_dte,
            option_type="call" if bullish else "put",
            strike_band=0.25,
            spot=spot,
        )
        if not snapshots:
            return None

        expiry = self._pick_expiry(snapshots.values(), today)
        if expiry is None:
            return None
        pool = for_expiry(snapshots.values(), expiry)

        long_leg = select_by_delta(pool, float(params["long_delta"]), right)
        short_leg = select_by_delta(pool, self.config.target_short_delta, right)
        if long_leg is None or short_leg is None:
            return None

        # The bought leg must sit closer to the money than the sold one, or this
        # is a credit spread pointing the wrong way.
        if bullish and not long_leg.strike < short_leg.strike:
            return None
        if not bullish and not long_leg.strike > short_leg.strike:
            return None

        direction = "gapped up" if bullish else "gapped down"
        rationale = (
            f"PEAD: {event.symbol} {direction} {magnitude:.1%} on "
            f"{event.report_date:%d %b}; {'call' if bullish else 'put'} debit spread "
            f"{long_leg.strike:g}/{short_leg.strike:g} expiring {expiry:%d %b}"
        )

        proposal = build_vertical(
            short_leg=short_leg,
            long_leg=long_leg,
            strategy_id=self.config.strategy_id,
            underlying=event.symbol,
            budget=budget,
            slippage=self.config.slippage,
            aggression=self.config.aggression,
            rationale=rationale,
            credit=False,
        )
        if proposal is None:
            return None
        if proposal.net_price <= 0:
            return None  # a debit spread that prices as a credit is a mispriced chain
        if not debit_to_width_ok(proposal, float(params["max_debit_to_width"])):
            return None
        return proposal

    def _pick_expiry(self, snapshots, today: date) -> date | None:
        """Far enough out that the drift has room to happen before theta bites."""
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


def exit_signal(entry_debit: float, current_price: float, config: StrategyConfig) -> str | None:
    """Close a DRIFT spread at a profit target or a stop.

    Both prices use Alpaca's sign convention, so a debit spread opens positive and
    is sold back at a positive price.
    """
    debit = abs(entry_debit)
    if debit <= 0:
        return None
    value = abs(current_price)

    if value >= debit * (1 + config.take_profit_pct):
        return f"take profit: worth {value:.2f} against a {debit:.2f} debit"
    # A positive multiple means "close after this fraction of the debit is
    # left" (2x -> one half).  Zero explicitly disables a mark-based stop for
    # structures such as CONVEX whose maximum loss is already the premium paid.
    if config.stop_loss_multiple > 0:
        floor = debit / config.stop_loss_multiple
        if value <= floor:
            return (
                f"stop loss: worth {value:.2f}, below the {floor:.2f} floor "
                f"for a {debit:.2f} debit"
            )
    return None
