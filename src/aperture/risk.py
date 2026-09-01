"""Deterministic risk engine.

The one rule this module exists to enforce:

    No language model can place an order. LLMs propose; these gates dispose.

Every gate is a pure function of (proposal, market snapshot, book state, limits).
Nothing here calls a network, an LLM, or the clock implicitly — the caller passes
``now`` in. That makes the whole layer unit-testable, which is the point: this is
the code standing between an autonomous agent and a $100k account.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Iterable, Sequence

from .contracts import CONTRACT_MULTIPLIER, PositionIntent, Right, Side, parse_occ

# --------------------------------------------------------------------------- #
# Proposals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Leg:
    """One leg of a proposed structure."""

    symbol: str  # OCC
    side: Side
    ratio: int  # >= 1, and gcd across all legs must be 1 (Alpaca requirement)
    intent: PositionIntent

    @property
    def right(self) -> Right:
        return parse_occ(self.symbol).right

    @property
    def strike(self) -> float:
        return parse_occ(self.symbol).strike

    @property
    def signed_ratio(self) -> int:
        return self.ratio if self.side is Side.BUY else -self.ratio


@dataclass(frozen=True)
class Proposal:
    """A candidate multi-leg options trade, before any gate has seen it."""

    strategy_id: str
    underlying: str
    legs: tuple[Leg, ...]
    qty: int
    # Alpaca's mleg sign convention: positive = net debit, negative = net credit.
    net_price: float
    rationale: str = ""
    multiplier: int = CONTRACT_MULTIPLIER

    @property
    def net_cash(self) -> float:
        """Cash flow at entry. Positive = credit received, negative = debit paid."""
        return -self.net_price * self.qty * self.multiplier


@dataclass(frozen=True)
class LegQuote:
    """Market snapshot for one leg at decision time."""

    symbol: str
    bid: float
    ask: float
    quote_ts: datetime
    open_interest: int = -1  # -1 means the feed did not report it
    volume: int = 0
    bid_size: int = 0
    ask_size: int = 0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct_of_mid(self) -> float:
        return math.inf if self.mid <= 0 else self.spread / self.mid


# --------------------------------------------------------------------------- #
# Payoff geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RiskProfile:
    """Expiry payoff analysis of a structure.

    ``max_loss is None`` means the loss is unbounded to the upside — a naked short
    call somewhere in the structure. That is an automatic rejection, not a number
    to be sized around.
    """

    max_loss: float | None
    max_profit: float | None
    net_cash: float
    is_defined_risk: bool

    @property
    def max_loss_or_inf(self) -> float:
        return math.inf if self.max_loss is None else self.max_loss


def _pnl_at(proposal: Proposal, spot: float) -> float:
    """Structure P&L at expiry for a given underlying price, including premium."""
    intrinsic = 0.0
    for leg in proposal.legs:
        parsed = parse_occ(leg.symbol)
        if parsed.right is Right.CALL:
            value = max(spot - parsed.strike, 0.0)
        else:
            value = max(parsed.strike - spot, 0.0)
        intrinsic += leg.signed_ratio * value

    return intrinsic * proposal.qty * proposal.multiplier + proposal.net_cash


def analyse_payoff(proposal: Proposal) -> RiskProfile:
    """Piecewise-linear expiry payoff, evaluated at every kink.

    The payoff of any combination of European-style options is piecewise linear in
    the underlying with kinks only at the strikes, so the extrema live at 0, at a
    strike, or at infinity. Checking the asymptotic slopes tells us whether either
    tail is unbounded.

    **This model is only valid when every leg shares one expiry.** A calendar or
    diagonal has no single expiry payoff — at the near expiry the far leg still
    carries time value that depends on volatility, not just on spot. Analysed as
    if it were a vertical, a same-strike calendar's legs cancel exactly and the
    structure reports a maximum loss of zero, which is the most dangerous possible
    wrong answer. So a multi-expiry structure is reported as *undefined* risk and
    the gates reject it, rather than being silently mispriced.
    """
    expiries = {parse_occ(leg.symbol).expiry for leg in proposal.legs}
    if len(expiries) > 1:
        return RiskProfile(
            max_loss=None,
            max_profit=None,
            net_cash=proposal.net_cash,
            is_defined_risk=False,
        )

    strikes = sorted({parse_occ(leg.symbol).strike for leg in proposal.legs})

    # Sample the kinks plus a point either side of each, so that flat regions and
    # sign changes between kinks are both captured.
    probes: list[float] = [0.0]
    for strike in strikes:
        probes.extend([max(strike - 1e-6, 0.0), strike, strike + 1e-6])
    probes.append(strikes[-1] * 2 + 100.0)

    values = [_pnl_at(proposal, spot) for spot in probes]

    # Slope as spot -> infinity: only calls still have delta out there.
    call_slope = sum(
        leg.signed_ratio for leg in proposal.legs if parse_occ(leg.symbol).right is Right.CALL
    )
    # Slope as spot -> 0 from above: only puts are live, and each long put gains
    # as spot falls, so a net short put position loses into zero. That loss is
    # bounded (spot cannot go below zero) and is already captured by the probe at 0.
    upside_unbounded_loss = call_slope < 0
    upside_unbounded_profit = call_slope > 0

    max_loss = None if upside_unbounded_loss else max(0.0, -min(values))
    max_profit = None if upside_unbounded_profit else max(0.0, max(values))

    return RiskProfile(
        max_loss=max_loss,
        max_profit=max_profit,
        net_cash=proposal.net_cash,
        is_defined_risk=max_loss is not None,
    )


# --------------------------------------------------------------------------- #
# Book state and limits
# --------------------------------------------------------------------------- #


@dataclass
class BookState:
    equity: float
    high_water_mark: float
    day_start_equity: float
    cash: float
    now: datetime  # timezone-aware, US/Eastern
    open_risk_by_strategy: dict[str, float] = field(default_factory=dict)
    open_risk_by_underlying: dict[str, float] = field(default_factory=dict)
    # Keyed by (strategy_id, underlying). A sleeve that holds no position in
    # a name is not positioned in it, however much the rest of the book holds.
    open_risk_by_strategy_underlying: dict[tuple[str, str], float] = field(
        default_factory=dict
    )

    @property
    def total_open_risk(self) -> float:
        return sum(self.open_risk_by_underlying.values())

    @property
    def drawdown_pct(self) -> float:
        if self.high_water_mark <= 0:
            return 0.0
        return max(0.0, (self.high_water_mark - self.equity) / self.high_water_mark)

    @property
    def day_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (self.equity - self.day_start_equity) / self.day_start_equity


@dataclass(frozen=True)
class RiskLimits:
    """Every number the desk is not allowed to argue with."""

    max_portfolio_loss_pct: float = 0.25
    max_underlying_loss_pct: float = 0.08
    max_single_trade_loss_pct: float = 0.04
    daily_loss_halt_pct: float = 0.03
    drawdown_breaker_pct: float = 0.08
    max_spread_pct_of_mid: float = 0.15
    max_quote_age_s: float = 90.0
    # Alpaca's contracts endpoint returns open_interest as null on this plan, so
    # depth is screened on the signals that ARE populated: traded volume and the
    # size actually quoted on each side. Any one of the three clearing its floor
    # is enough; all three failing means nobody is really making a market here.
    min_open_interest: int = 100
    min_daily_volume: int = 25
    min_quote_size: int = 10
    min_credit_to_width: float = 0.15
    open_blackout_min: int = 10
    close_blackout_min: int = 10
    max_legs: int = 4  # Alpaca hard limit on mleg orders
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class Verdict:
    approved: bool
    profile: RiskProfile
    results: tuple[GateResult, ...]

    @property
    def rejections(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    def audit_line(self) -> str:
        if self.approved:
            return f"APPROVED  max_loss=${self.profile.max_loss_or_inf:,.0f}"
        reasons = "; ".join(f"{r.gate}: {r.reason}" for r in self.rejections)
        return f"VETOED    {reasons}"


def _gcd_of(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = math.gcd(result, abs(value))
    return result


def evaluate(
    proposal: Proposal,
    quotes: Sequence[LegQuote],
    book: BookState,
    limits: RiskLimits = RiskLimits(),
    strategy_budget: float = math.inf,
) -> Verdict:
    """Run every gate. A single failure vetoes the trade.

    Gates are deliberately evaluated in full rather than short-circuited, so the
    audit log records *every* reason a trade was rejected, not just the first.
    """
    profile = analyse_payoff(proposal)
    results: list[GateResult] = []
    by_symbol = {q.symbol: q for q in quotes}

    def check(gate: str, passed: bool, reason: str = "ok") -> None:
        results.append(GateResult(gate, passed, reason))

    # --- structural ------------------------------------------------------- #
    check(
        "leg_count",
        1 <= len(proposal.legs) <= limits.max_legs,
        f"{len(proposal.legs)} legs (Alpaca allows 1-{limits.max_legs})",
    )
    ratios = [leg.ratio for leg in proposal.legs]
    check(
        "ratio_simplified",
        bool(ratios) and _gcd_of(ratios) == 1,
        f"ratio_qty {ratios} must reduce to gcd 1 or Alpaca rejects the order",
    )
    check("qty_positive", proposal.qty >= 1, f"qty={proposal.qty}")

    expiries = {parse_occ(leg.symbol).expiry for leg in proposal.legs}
    check(
        "single_expiry",
        len(expiries) == 1,
        f"legs span {len(expiries)} expiries "
        f"({', '.join(str(e) for e in sorted(expiries))}); the payoff model assumes "
        "one, and Alpaca rejects uncovered calendars in an mleg order anyway",
    )

    # --- defined risk ----------------------------------------------------- #
    check(
        "defined_risk",
        profile.is_defined_risk,
        "loss is unbounded or unanalysable: an uncovered short call, "
        "or legs across multiple expiries",
    )
    max_loss = profile.max_loss_or_inf

    # --- sizing ----------------------------------------------------------- #
    check(
        "single_trade_cap",
        max_loss <= limits.max_single_trade_loss_pct * book.equity,
        f"max loss ${max_loss:,.0f} > "
        f"{limits.max_single_trade_loss_pct:.0%} of ${book.equity:,.0f}",
    )
    strategy_used = book.open_risk_by_strategy.get(proposal.strategy_id, 0.0)
    check(
        "strategy_budget",
        strategy_used + max_loss <= strategy_budget,
        f"{proposal.strategy_id} would use ${strategy_used + max_loss:,.0f} "
        f"of a ${strategy_budget:,.0f} budget",
    )
    underlying_used = book.open_risk_by_underlying.get(proposal.underlying, 0.0)
    check(
        "underlying_concentration",
        underlying_used + max_loss <= limits.max_underlying_loss_pct * book.equity,
        f"{proposal.underlying} would carry ${underlying_used + max_loss:,.0f} "
        f"> {limits.max_underlying_loss_pct:.0%} of equity",
    )
    check(
        "portfolio_cap",
        book.total_open_risk + max_loss <= limits.max_portfolio_loss_pct * book.equity,
        f"book risk would reach ${book.total_open_risk + max_loss:,.0f} "
        f"> {limits.max_portfolio_loss_pct:.0%} of equity",
    )

    # --- circuit breakers ------------------------------------------------- #
    check(
        "daily_loss_halt",
        book.day_pnl_pct > -limits.daily_loss_halt_pct,
        f"day P&L {book.day_pnl_pct:+.2%} breached "
        f"-{limits.daily_loss_halt_pct:.0%}; no new entries today",
    )
    check(
        "drawdown_breaker",
        book.drawdown_pct < limits.drawdown_breaker_pct,
        f"drawdown {book.drawdown_pct:.2%} breached "
        f"{limits.drawdown_breaker_pct:.0%} from high-water mark",
    )

    # --- market microstructure -------------------------------------------- #
    missing = [leg.symbol for leg in proposal.legs if leg.symbol not in by_symbol]
    check("quotes_present", not missing, f"no quote for {missing}")

    for leg in proposal.legs:
        quote = by_symbol.get(leg.symbol)
        if quote is None:
            continue
        check(
            f"liquidity[{leg.symbol}]",
            quote.bid > 0 and quote.spread_pct_of_mid <= limits.max_spread_pct_of_mid,
            f"bid={quote.bid:.2f} spread={quote.spread_pct_of_mid:.1%} "
            f"> {limits.max_spread_pct_of_mid:.0%} of mid",
        )
        depth_signals = (
            quote.open_interest >= limits.min_open_interest,
            quote.volume >= limits.min_daily_volume,
            min(quote.bid_size, quote.ask_size) >= limits.min_quote_size,
        )
        check(
            f"depth[{leg.symbol}]",
            any(depth_signals),
            f"no depth signal clears its floor: OI={quote.open_interest} "
            f"vol={quote.volume} quoted={quote.bid_size}x{quote.ask_size}",
        )
        age = (book.now - quote.quote_ts).total_seconds()
        check(
            f"quote_freshness[{leg.symbol}]",
            age <= limits.max_quote_age_s,
            f"quote is {age:.0f}s old (limit {limits.max_quote_age_s:.0f}s)",
        )

    # --- session windows -------------------------------------------------- #
    check("session", *_session_check(book.now, limits))

    return Verdict(
        approved=all(r.passed for r in results),
        profile=profile,
        results=tuple(results),
    )


def _session_check(now: datetime, limits: RiskLimits) -> tuple[bool, str]:
    """No new entries in the opening or closing blackout."""
    clock = now.time()
    open_until = (
        datetime.combine(now.date(), limits.market_open) + timedelta(minutes=limits.open_blackout_min)
    ).time()
    close_from = (
        datetime.combine(now.date(), limits.market_close) - timedelta(minutes=limits.close_blackout_min)
    ).time()

    if clock < limits.market_open or clock >= limits.market_close:
        return False, f"{clock:%H:%M} is outside regular trading hours"
    if clock < open_until:
        return False, f"{clock:%H:%M} is inside the {limits.open_blackout_min}min opening blackout"
    if clock >= close_from:
        return False, f"{clock:%H:%M} is inside the {limits.close_blackout_min}min closing blackout"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Tournament clock
# --------------------------------------------------------------------------- #


def tournament_risk_multiplier(
    now: datetime,
    deadline: datetime,
    equity: float,
    start_equity: float,
    *,
    target_return: float = 0.05,
) -> float:
    """Scale risk appetite to time remaining and how far ahead or behind we are.

    A seven-day contest is not a career. Finishing 20th and finishing 40th pay the
    same, so the desk is allowed more variance than a real book would take — but
    only through structures whose maximum loss is already bounded by the gates
    above. Ahead and near the deadline, it protects the number instead.

    Returns a multiplier in [0.25, 1.5] applied to position sizing.
    """
    total = max((deadline - now).total_seconds(), 0.0)
    horizon_days = total / 86_400
    progress = (equity - start_equity) / start_equity if start_equity > 0 else 0.0

    # Behind target -> lean in; ahead of target -> bank it.
    shortfall = target_return - progress
    appetite = 1.0 + 1.0 * shortfall / max(target_return, 1e-9)

    # The last day is for protecting the result, not chasing it.
    if horizon_days < 1.0:
        appetite = min(appetite, 0.5 if progress > 0 else 1.0)
    if horizon_days < 0.25:
        appetite = 0.0 if progress > 0 else min(appetite, 0.5)

    return float(min(max(appetite, 0.25 if horizon_days >= 0.25 else 0.0), 1.5))
