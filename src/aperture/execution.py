"""Adaptive execution — the desk learns how hard it has to push to get filled.

A limit price is a bet about what the market will meet. Set it too close to mid
and orders expire unfilled; set it too far and every trade gives away edge that
never comes back. The right answer is not knowable in advance: it depends on the
spread, the underlying, the hour, and how badly the market wants the other side.

So the desk measures instead. Each cycle it reads its own recent order outcomes
and moves a single number -- how far through the half-spread it is willing to
reach -- toward whatever the last few orders suggest.

This exists because the alternative was a human watching a dashboard and editing
a constant. On 28 August the desk filled 12 of 31 orders and deployed about two
fifths of the capital it intended to; nobody noticed until the session was over.
An autonomous desk has to notice that itself.

**It only ever adjusts the price it offers.** It cannot change what may be
traded, how large, or against which risk limits. Those belong to the Warden, and
nothing here touches them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

log = logging.getLogger(__name__)

# Floor and ceiling on how far through the half-spread to reach.
#   0.0 = ask for mid and hope
#   1.0 = cross the full half-spread to the far touch
#   >1.0 = pay through the touch, which fills but concedes real money
MIN_AGGRESSION = 0.30
MAX_AGGRESSION = 1.20
DEFAULT_AGGRESSION = 0.60

# Fill rates below this mean the desk is not really trading; above it, it is
# probably paying more than it needs to. The gap between them is deliberate:
# without it the value oscillates every cycle and never settles.
TOO_FEW_FILLS = 0.55
TOO_MANY_FILLS = 0.90

STEP_UP = 0.15    # react quickly to not trading at all
STEP_DOWN = 0.05  # give back edge slowly; being filled is not a problem to fix

TERMINAL_UNFILLED = {"expired", "canceled", "cancelled", "rejected", "done_for_day"}


@dataclass(frozen=True)
class FillReport:
    filled: int
    unfilled: int
    pending: int

    @property
    def decided(self) -> int:
        """Orders whose fate is settled. Pending ones say nothing yet."""
        return self.filled + self.unfilled

    @property
    def rate(self) -> float:
        return self.filled / self.decided if self.decided else 0.0

    def describe(self) -> str:
        if not self.decided:
            return "no settled orders yet"
        return f"{self.filled}/{self.decided} filled ({self.rate:.0%})"


def measure_fills(orders: Sequence[dict[str, Any]], lookback: int = 20) -> FillReport:
    """Fill outcomes for the most recent multi-leg orders.

    Orders still working are counted separately rather than as failures: a limit
    resting for two minutes has not failed, and treating it as a miss would make
    the desk chase its own tail upward within a single session.
    """
    mleg = [o for o in orders if o.get("order_class") == "mleg"][-lookback:]
    filled = unfilled = pending = 0
    for order in mleg:
        status = str(order.get("status", "")).lower()
        if status == "filled":
            filled += 1
        elif status in TERMINAL_UNFILLED:
            unfilled += 1
        else:
            pending += 1
    return FillReport(filled=filled, unfilled=unfilled, pending=pending)


def adapt(current: float, report: FillReport, *, min_sample: int = 6) -> tuple[float, str]:
    """The new aggression, and why it changed.

    Returns the current value unchanged when there is too little evidence. Acting
    on two or three orders would be reading noise, and the cost of reading it
    wrong is paid on every subsequent trade.
    """
    if report.decided < min_sample:
        return current, f"holding at {current:.2f}: {report.describe()}, too few to judge"

    if report.rate < TOO_FEW_FILLS:
        raised = min(current + STEP_UP, MAX_AGGRESSION)
        if raised == current:
            return current, (
                f"already at the {MAX_AGGRESSION:.2f} ceiling with {report.describe()}; "
                "the spreads themselves are the problem, not the price offered"
            )
        return raised, (
            f"{report.describe()} is below {TOO_FEW_FILLS:.0%}, so the desk is not "
            f"deploying what it intends; reaching further: {current:.2f} -> {raised:.2f}"
        )

    if report.rate > TOO_MANY_FILLS:
        lowered = max(current - STEP_DOWN, MIN_AGGRESSION)
        if lowered == current:
            return current, f"at the {MIN_AGGRESSION:.2f} floor with {report.describe()}"
        return lowered, (
            f"{report.describe()} fills easily, so the desk is likely paying more "
            f"than it needs; easing back: {current:.2f} -> {lowered:.2f}"
        )

    return current, f"holding at {current:.2f}: {report.describe()} is healthy"


def clamp(value: float) -> float:
    return max(MIN_AGGRESSION, min(MAX_AGGRESSION, value))
