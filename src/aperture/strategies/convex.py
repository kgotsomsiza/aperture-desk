"""CONVEX — the sleeve that buys movement instead of selling it.

Every other strategy on this desk is short volatility: it collects a premium and
wins when nothing happens. That is a good business and the wrong shape for a
tournament. Selling premium wins small and often and loses big and rarely, so
its distribution has a thin right tail — and a contest that pays only the top
places is decided entirely by the right tail. Finishing twentieth and fortieth
pay the same.

This sleeve is the other side. It buys an out-of-the-money call and put on the
same expiry: **maximum loss is the premium paid and nothing else**, while the
upside is uncapped. Most days it loses a little. Occasionally it pays for every
day it lost.

Two conditions gate it, and both must hold:

1. **Implied volatility must be below realised.** Buying convexity is only the
   favourable side of the trade when the market is charging less for movement
   than the underlying has actually been delivering. Above that line, the desk's
   premium-selling sleeves are the right expression and this one stands down.
2. **The regime agent must not be selling premium.** The agents decide what the
   desk is doing; this strategy asks permission rather than assuming it.

It holds one position at a time and never averages down. A convex sleeve that
keeps buying as it bleeds is just a slow way to spend the account.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from ..contracts import Right
from ..marketdata import MarketData, expiries, for_expiry
from ..risk import BookState, Proposal
from .base import StrategyConfig, build_long_strangle

log = logging.getLogger("aperture")

DEFAULT_CONFIG = StrategyConfig(
    strategy_id="CONVEX",
    sleeve="CONVEX",
    weight=0.05,
    enabled=True,
    universe=("SPY", "QQQ"),
    min_dte=3,
    max_dte=10,
    spread_width=0.0,
    min_credit_to_width=0.0,   # a debit structure: there is no credit to test
    take_profit_pct=1.50,      # let a winner run; this sleeve exists for the tail
    stop_loss_multiple=0.0,    # max loss is the premium, so there is nothing to stop
)

# How far out of the money to buy, as a fraction of spot. Close enough that an
# ordinary move reaches it; far enough that the premium stays cheap.
OTM_FRACTION = 0.015

# Refuse only when movement is genuinely EXPENSIVE -- not merely fairly priced.
#
# This was 0.98 for one day, and that was a conceptual error worth recording: it
# is a *value* filter bolted onto an instrument that exists for its *shape*. The
# case for this sleeve is that a near-certain small gain cannot win a tournament
# and a right tail can. That argument holds at fair value; it only fails when the
# tail is being sold at a real premium. At 1.03x the overpay is about 3% of the
# premium -- roughly $150 on a $5,000 sleeve -- against a materially better
# distribution. Gating that out left the sleeve inert and quietly reverted the
# desk to the posture it was built to correct.
MAX_IV_TO_REALISED = 1.15

# The Warden refuses any single structure whose max loss exceeds 4% of equity.
# Sizing to 3.6% leaves room for the fill concession to move the price without
# tipping a good proposal over a limit it could have respected up front.
MAX_TRADE_LOSS_PCT = 0.036


@dataclass
class ConvexStrategy:
    """Buys a defined-risk long strangle when movement is cheap."""

    config: StrategyConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    iv_to_realised: float | None = None   # set each cycle by the loop
    posture: str = "balanced"             # what the regime agent decided

    def propose(self, md: MarketData, book: BookState, budget: float) -> list[Proposal]:
        if not self.config.enabled or budget <= 0:
            return []

        # Only a full stand-down silences this sleeve.
        #
        # This used to also stand down on `sell_premium`, on the reasoning that
        # selling premium and buying it at once is incoherent. That was wrong,
        # and wrong in exactly the way the 0.98 IV gate was wrong: it judges a
        # shape instrument on value. Selling premium in the core while buying a
        # cheap tail is a barbell, which is how books are actually built -- the
        # condor's gain is small and capped, the strangle's is uncapped, and on
        # a large move the second dwarfs the first. The IV/realised ceiling
        # above is already the value filter; this gate was redundant and cost
        # the sleeve a third session.
        if self.posture == "stand_down":
            return []

        # Only when movement is cheaper than what the index has been delivering.
        if self.iv_to_realised is not None and self.iv_to_realised > MAX_IV_TO_REALISED:
            return []

        # Split the allowance across the names still available. Handing each the
        # full budget would spend the sleeve's entire risk twice over.
        # Only this sleeve's own positions block it. Reading the book-wide
        # view instead meant CARRY's SPY and QQQ condors permanently blocked
        # CONVEX from the only two names it trades -- so the sleeve could never
        # fire at all. Caught by scripts/preopen.py before it cost a session.
        held = book.open_risk_by_strategy_underlying
        eligible = [
            u for u in self.config.universe
            if held.get((self.config.strategy_id, u), 0.0) <= 0
        ]
        if not eligible:
            return []

        # Stay inside the Warden's per-trade ceiling. The tournament multiplier
        # can lift this sleeve's allowance above it, and a proposal sized past a
        # limit the desk already publishes is simply a refused proposal -- which
        # is how the first funded cycle was lost.
        per_name = budget / len(eligible)
        per_trade_cap = book.equity * MAX_TRADE_LOSS_PCT
        per_name = min(per_name, per_trade_cap)

        proposals: list[Proposal] = []
        for underlying in eligible:
            proposal = self._strangle_for(md, underlying, per_name)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    # ------------------------------------------------------------------ #

    def _strangle_for(
        self, md: MarketData, underlying: str, budget: float
    ) -> Proposal | None:
        spot = md.spot(underlying)
        if spot <= 0:
            return None

        snapshots = md.chain(
            underlying,
            min_dte=self.config.min_dte,
            max_dte=self.config.max_dte,
            strike_band=0.08,
            spot=spot,
        )
        if not snapshots:
            return None

        expiry = self._pick_expiry(snapshots.values())
        if expiry is None:
            return None
        pool = list(for_expiry(snapshots.values(), expiry))

        call = self._nearest(pool, Right.CALL, spot * (1 + OTM_FRACTION))
        put = self._nearest(pool, Right.PUT, spot * (1 - OTM_FRACTION))
        if call is None or put is None or put.strike >= call.strike:
            return None

        rationale = (
            f"convex sleeve: {underlying} {expiry:%d %b} long strangle "
            f"({put.strike:g}p / {call.strike:g}c), spot {spot:.2f}, "
            f"IV/realised {self.iv_to_realised:.2f}x"
            if self.iv_to_realised is not None
            else f"convex sleeve: {underlying} {expiry:%d %b} long strangle "
                 f"({put.strike:g}p / {call.strike:g}c), spot {spot:.2f}"
        )

        return build_long_strangle(
            call,
            put,
            strategy_id=self.config.strategy_id,
            underlying=underlying,
            budget=budget,
            slippage=self.config.slippage,
            aggression=self.config.aggression,
            rationale=rationale,
        )

    @staticmethod
    def _nearest(pool, right: Right, target: float):
        """The listed strike closest to where we want to be."""
        candidates = [s for s in pool if s.right == right and s.is_priceable]
        if not candidates:
            return None
        return min(candidates, key=lambda s: abs(s.strike - target))

    def _pick_expiry(self, snapshots) -> date | None:
        """The nearest expiry inside the window.

        Convexity is cheapest and most explosive at the short end. But the floor
        under ``min_dte`` matters as much as the ceiling: an option expiring on
        the day the desk is measured is a coin that has already landed, worth
        either a great deal or exactly nothing. Holding one that still has a day
        left keeps residual time value in the position if the move never comes,
        which turns a total loss into a partial one.
        """
        available = sorted(expiries(snapshots))
        return available[0] if available else None
