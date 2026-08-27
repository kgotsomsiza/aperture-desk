"""Strategy interface and the shared structure-building helpers.

A strategy's only job is to *propose*. It never sizes past its budget, never
places an order, and never sees a credential. Everything it returns goes through
the Risk Warden before any of it reaches the broker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..contracts import PositionIntent, Right, Side
from ..marketdata import MarketData, Snapshot
from ..risk import BookState, Leg, Proposal, analyse_payoff


@dataclass
class StrategyConfig:
    """A strategy's identity and its knobs.

    Configs are data, not code, which is what lets the research agent mutate one
    overnight and hand the result to the backtest gate.
    """

    strategy_id: str
    sleeve: str  # "BALLAST" | "CONVEX"
    weight: float  # target fraction of the book
    enabled: bool = True
    universe: tuple[str, ...] = ()
    min_dte: int = 7
    max_dte: int = 21
    target_short_delta: float = 0.15
    spread_width: float = 5.0
    min_credit_to_width: float = 0.15
    take_profit_pct: float = 0.50  # close at 50% of max profit
    stop_loss_multiple: float = 2.0  # exit at 2x the credit received
    slippage: float = 0.05  # price concession per structure, in dollars
    params: dict = field(default_factory=dict)

    def child(self, **overrides) -> "StrategyConfig":
        """A mutated copy — how the research agent proposes a new hire."""
        data = {**self.__dict__, **overrides}
        data["params"] = dict(data.get("params") or {})
        return StrategyConfig(**data)


class Strategy(Protocol):
    config: StrategyConfig

    def propose(self, md: MarketData, book: BookState, budget: float) -> list[Proposal]:
        """Candidate trades. May return an empty list, and usually should."""
        ...


# --------------------------------------------------------------------------- #
# Structure construction
# --------------------------------------------------------------------------- #


def _leg(snapshot: Snapshot, side: Side, ratio: int = 1) -> Leg:
    intent = PositionIntent.BUY_TO_OPEN if side is Side.BUY else PositionIntent.SELL_TO_OPEN
    return Leg(symbol=snapshot.symbol, side=side, ratio=ratio, intent=intent)


def structure_price(legs: Sequence[tuple[Snapshot, Side]]) -> float:
    """Natural mid price of a structure in Alpaca's sign convention.

    Positive is a net debit, negative is a net credit — matching what the API
    expects in ``limit_price`` for an mleg order.
    """
    total = 0.0
    for snapshot, side in legs:
        total += snapshot.mid if side is Side.BUY else -snapshot.mid
    return total


def concede(price: float, slippage: float) -> float:
    """Shift a limit price toward the market to buy fill probability.

    Works in both directions without a branch: a debit (positive) rises, and a
    credit (negative) shrinks toward zero. Both mean "accept a slightly worse
    price", which is what makes a marketable limit fill.
    """
    return round(price + abs(slippage), 2)


def size_to_budget(max_loss_per_unit: float, budget: float, cap: int = 50) -> int:
    """Contracts affordable within a risk budget.

    Deliberately floors rather than rounds: the desk is never allowed to be one
    contract over its allocation because of a rounding rule.
    """
    if max_loss_per_unit <= 0 or budget <= 0:
        return 0
    return max(0, min(cap, int(math.floor(budget / max_loss_per_unit))))


def build_vertical(
    short_leg: Snapshot,
    long_leg: Snapshot,
    *,
    strategy_id: str,
    underlying: str,
    budget: float,
    slippage: float,
    rationale: str,
    credit: bool = True,
) -> Proposal | None:
    """Two-leg vertical spread, sized to the budget."""
    if not (short_leg.is_priceable and long_leg.is_priceable):
        return None
    if short_leg.symbol == long_leg.symbol:
        return None

    pairs = (
        [(short_leg, Side.SELL), (long_leg, Side.BUY)]
        if credit
        else [(long_leg, Side.BUY), (short_leg, Side.SELL)]
    )
    price = concede(structure_price(pairs), slippage)

    probe = Proposal(
        strategy_id=strategy_id,
        underlying=underlying,
        legs=tuple(_leg(s, side) for s, side in pairs),
        qty=1,
        net_price=price,
        rationale=rationale,
    )
    profile = analyse_payoff(probe)
    if not profile.is_defined_risk or not profile.max_loss:
        return None

    qty = size_to_budget(profile.max_loss, budget)
    if qty < 1:
        return None
    return Proposal(
        strategy_id=strategy_id,
        underlying=underlying,
        legs=probe.legs,
        qty=qty,
        net_price=price,
        rationale=rationale,
    )


def build_iron_condor(
    short_put: Snapshot,
    long_put: Snapshot,
    short_call: Snapshot,
    long_call: Snapshot,
    *,
    strategy_id: str,
    underlying: str,
    budget: float,
    slippage: float,
    rationale: str,
) -> Proposal | None:
    """Four-leg iron condor — Alpaca's maximum leg count, and the ballast workhorse."""
    legs = [
        (long_put, Side.BUY),
        (short_put, Side.SELL),
        (short_call, Side.SELL),
        (long_call, Side.BUY),
    ]
    if not all(s.is_priceable for s, _ in legs):
        return None
    if len({s.symbol for s, _ in legs}) != 4:
        return None
    if not (long_put.strike < short_put.strike < short_call.strike < long_call.strike):
        return None

    price = concede(structure_price(legs), slippage)
    probe = Proposal(
        strategy_id=strategy_id,
        underlying=underlying,
        legs=tuple(_leg(s, side) for s, side in legs),
        qty=1,
        net_price=price,
        rationale=rationale,
    )
    profile = analyse_payoff(probe)
    if not profile.is_defined_risk or not profile.max_loss:
        return None

    qty = size_to_budget(profile.max_loss, budget)
    if qty < 1:
        return None
    return Proposal(
        strategy_id=strategy_id,
        underlying=underlying,
        legs=probe.legs,
        qty=qty,
        net_price=price,
        rationale=rationale,
    )


def build_long_strangle(
    call: Snapshot,
    put: Snapshot,
    *,
    strategy_id: str,
    underlying: str,
    budget: float,
    slippage: float,
    rationale: str,
) -> Proposal | None:
    """Long call plus long put — the convex sleeve's basic shape.

    Max loss is the premium paid and nothing else, which is what earns it a place
    alongside the premium-selling ballast.
    """
    if not (call.is_priceable and put.is_priceable):
        return None
    legs = [(call, Side.BUY), (put, Side.BUY)]
    price = concede(structure_price(legs), slippage)
    if price <= 0:
        return None

    per_unit = price * 100
    qty = size_to_budget(per_unit, budget)
    if qty < 1:
        return None
    return Proposal(
        strategy_id=strategy_id,
        underlying=underlying,
        legs=tuple(_leg(s, side) for s, side in legs),
        qty=qty,
        net_price=price,
        rationale=rationale,
    )


def credit_to_width_ok(proposal: Proposal, minimum: float) -> bool:
    """Reject spreads that pay too little for the risk they carry.

    Selling a five-point spread for fifteen cents is a losing trade dressed up as
    a high win rate, and it is the single most common way a premium-selling bot
    bleeds out.
    """
    profile = analyse_payoff(proposal)
    if profile.max_loss is None or profile.max_profit is None:
        return False
    width = profile.max_loss + profile.max_profit
    if width <= 0:
        return False
    return (profile.max_profit / width) >= minimum
