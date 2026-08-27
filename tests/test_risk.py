"""Tests for the deterministic risk engine.

These are the tests that matter most in this repo: they are the proof that an
autonomous agent cannot talk its way past the position limits.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from aperture.contracts import PositionIntent, Right, Side, build_occ, parse_occ
from aperture.risk import (
    BookState,
    Leg,
    LegQuote,
    Proposal,
    RiskLimits,
    analyse_payoff,
    evaluate,
    tournament_risk_multiplier,
)

ET = timezone(timedelta(hours=-4))
NOW = datetime(2026, 9, 1, 11, 0, tzinfo=ET)
EXPIRY = date(2026, 9, 18)


def occ(strike: float, right: Right) -> str:
    return build_occ("SPY", EXPIRY, right, strike)


def leg(strike: float, right: Right, side: Side, ratio: int = 1) -> Leg:
    intent = PositionIntent.BUY_TO_OPEN if side is Side.BUY else PositionIntent.SELL_TO_OPEN
    return Leg(symbol=occ(strike, right), side=side, ratio=ratio, intent=intent)


def fresh_quotes(proposal: Proposal, *, now: datetime = NOW, **overrides) -> list[LegQuote]:
    defaults = dict(bid=1.00, ask=1.05, open_interest=5_000, volume=1_000)
    defaults.update(overrides)
    return [LegQuote(symbol=l.symbol, quote_ts=now, **defaults) for l in proposal.legs]


def healthy_book(**overrides) -> BookState:
    defaults = dict(
        equity=100_000.0,
        high_water_mark=100_000.0,
        day_start_equity=100_000.0,
        cash=100_000.0,
        now=NOW,
    )
    defaults.update(overrides)
    return BookState(**defaults)


# --------------------------------------------------------------------------- #
# OCC symbols
# --------------------------------------------------------------------------- #


def test_occ_round_trip():
    symbol = build_occ("AAPL", date(2025, 6, 20), Right.CALL, 200.0)
    assert symbol == "AAPL250620C00200000"
    parsed = parse_occ(symbol)
    assert (parsed.root, parsed.expiry, parsed.right, parsed.strike) == (
        "AAPL",
        date(2025, 6, 20),
        Right.CALL,
        200.0,
    )


def test_occ_handles_fractional_strikes():
    assert build_occ("SPY", date(2026, 9, 18), Right.PUT, 642.5) == "SPY260918P00642500"
    assert parse_occ("SPY260918P00642500").strike == 642.5


def test_occ_rejects_garbage():
    with pytest.raises(ValueError):
        parse_occ("NOT_AN_OPTION")


# --------------------------------------------------------------------------- #
# Payoff geometry
# --------------------------------------------------------------------------- #


def test_long_call_spread_max_loss_is_the_debit():
    # Buy the 640 call, sell the 645 call for a net 2.00 debit, 10 contracts.
    proposal = Proposal(
        strategy_id="DRIFT",
        underlying="SPY",
        legs=(leg(640, Right.CALL, Side.BUY), leg(645, Right.CALL, Side.SELL)),
        qty=10,
        net_price=2.00,  # positive = debit
    )
    profile = analyse_payoff(proposal)
    assert profile.is_defined_risk
    assert profile.max_loss == pytest.approx(2_000.0)  # 2.00 * 10 * 100
    assert profile.max_profit == pytest.approx(3_000.0)  # (5.00 - 2.00) * 10 * 100


def test_short_put_spread_max_loss_is_width_minus_credit():
    # Sell the 630 put, buy the 625 put for a 1.50 credit, 5 contracts.
    proposal = Proposal(
        strategy_id="CARRY",
        underlying="SPY",
        legs=(leg(630, Right.PUT, Side.SELL), leg(625, Right.PUT, Side.BUY)),
        qty=5,
        net_price=-1.50,  # negative = credit
    )
    profile = analyse_payoff(proposal)
    assert profile.max_profit == pytest.approx(750.0)  # credit kept
    assert profile.max_loss == pytest.approx(1_750.0)  # (5.00 - 1.50) * 5 * 100


def test_iron_condor_is_defined_risk_on_both_wings():
    proposal = Proposal(
        strategy_id="CARRY",
        underlying="SPY",
        legs=(
            leg(625, Right.PUT, Side.BUY),
            leg(630, Right.PUT, Side.SELL),
            leg(660, Right.CALL, Side.SELL),
            leg(665, Right.CALL, Side.BUY),
        ),
        qty=4,
        net_price=-1.80,
    )
    profile = analyse_payoff(proposal)
    assert profile.is_defined_risk
    assert profile.max_profit == pytest.approx(720.0)
    assert profile.max_loss == pytest.approx(1_280.0)  # (5.00 - 1.80) * 4 * 100


def test_naked_short_call_is_unbounded():
    proposal = Proposal(
        strategy_id="ROGUE",
        underlying="SPY",
        legs=(leg(660, Right.CALL, Side.SELL),),
        qty=1,
        net_price=-3.00,
    )
    profile = analyse_payoff(proposal)
    assert profile.max_loss is None
    assert not profile.is_defined_risk
    assert profile.max_loss_or_inf == math.inf


def test_short_put_loss_is_bounded_by_a_zero_underlying():
    proposal = Proposal(
        strategy_id="ROGUE",
        underlying="SPY",
        legs=(leg(630, Right.PUT, Side.SELL),),
        qty=1,
        net_price=-5.00,
    )
    profile = analyse_payoff(proposal)
    # Bounded, but enormous: the stock can only go to zero, and that is 62,500.
    assert profile.max_loss == pytest.approx(62_500.0)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def condor(qty: int = 4, net_price: float = -1.80) -> Proposal:
    return Proposal(
        strategy_id="CARRY",
        underlying="SPY",
        legs=(
            leg(625, Right.PUT, Side.BUY),
            leg(630, Right.PUT, Side.SELL),
            leg(660, Right.CALL, Side.SELL),
            leg(665, Right.CALL, Side.BUY),
        ),
        qty=qty,
        net_price=net_price,
    )


def failed_gates(verdict) -> set[str]:
    return {r.gate.split("[")[0] for r in verdict.rejections}


def test_healthy_condor_is_approved():
    proposal = condor()
    verdict = evaluate(proposal, fresh_quotes(proposal), healthy_book())
    assert verdict.approved, verdict.audit_line()
    assert "APPROVED" in verdict.audit_line()


def test_naked_short_call_is_vetoed_for_undefined_risk():
    proposal = Proposal(
        strategy_id="ROGUE",
        underlying="SPY",
        legs=(leg(660, Right.CALL, Side.SELL),),
        qty=1,
        net_price=-3.00,
    )
    verdict = evaluate(proposal, fresh_quotes(proposal), healthy_book())
    assert not verdict.approved
    assert "defined_risk" in failed_gates(verdict)


def test_oversized_trade_is_vetoed():
    # 40 condors is $12,800 of max loss against a 4% ($4,000) single-trade cap.
    proposal = condor(qty=40)
    verdict = evaluate(proposal, fresh_quotes(proposal), healthy_book())
    assert not verdict.approved
    assert "single_trade_cap" in failed_gates(verdict)


def test_underlying_concentration_is_enforced():
    book = healthy_book(open_risk_by_underlying={"SPY": 7_500.0})
    proposal = condor()
    verdict = evaluate(proposal, fresh_quotes(proposal), book)
    assert not verdict.approved
    assert "underlying_concentration" in failed_gates(verdict)


def test_strategy_budget_is_enforced():
    book = healthy_book(open_risk_by_strategy={"CARRY": 9_000.0})
    proposal = condor()
    verdict = evaluate(proposal, fresh_quotes(proposal), book, strategy_budget=10_000.0)
    assert not verdict.approved
    assert "strategy_budget" in failed_gates(verdict)


def test_daily_loss_halt_blocks_new_entries():
    book = healthy_book(equity=96_500.0)  # -3.5% on the day
    proposal = condor()
    verdict = evaluate(proposal, fresh_quotes(proposal), book)
    assert not verdict.approved
    assert "daily_loss_halt" in failed_gates(verdict)


def test_drawdown_breaker_blocks_new_entries():
    book = healthy_book(equity=90_000.0, day_start_equity=90_000.0, high_water_mark=100_000.0)
    proposal = condor()
    verdict = evaluate(proposal, fresh_quotes(proposal), book)
    assert not verdict.approved
    assert "drawdown_breaker" in failed_gates(verdict)


def test_stale_quote_is_vetoed():
    # The free-tier options feed is derived and delayed; this gate is the reason
    # the desk never trades on a snapshot it has been sitting on.
    proposal = condor()
    quotes = fresh_quotes(proposal, now=NOW - timedelta(minutes=15))
    verdict = evaluate(proposal, quotes, healthy_book())
    assert not verdict.approved
    assert "quote_freshness" in failed_gates(verdict)


def test_wide_spread_is_vetoed():
    proposal = condor()
    quotes = fresh_quotes(proposal, bid=0.50, ask=1.50)  # 100% of mid
    verdict = evaluate(proposal, quotes, healthy_book())
    assert not verdict.approved
    assert "liquidity" in failed_gates(verdict)


def test_zero_bid_is_vetoed():
    proposal = condor()
    quotes = fresh_quotes(proposal, bid=0.0, ask=0.05)
    verdict = evaluate(proposal, quotes, healthy_book())
    assert not verdict.approved
    assert "liquidity" in failed_gates(verdict)


def test_thin_open_interest_is_vetoed():
    proposal = condor()
    quotes = fresh_quotes(proposal, open_interest=3)
    verdict = evaluate(proposal, quotes, healthy_book())
    assert not verdict.approved
    assert "open_interest" in failed_gates(verdict)


def test_unsimplified_ratios_are_vetoed():
    # Alpaca rejects legs whose ratio_qty share a common divisor.
    proposal = Proposal(
        strategy_id="CARRY",
        underlying="SPY",
        legs=(leg(630, Right.PUT, Side.SELL, ratio=2), leg(625, Right.PUT, Side.BUY, ratio=2)),
        qty=1,
        net_price=-1.50,
    )
    verdict = evaluate(proposal, fresh_quotes(proposal), healthy_book())
    assert not verdict.approved
    assert "ratio_simplified" in failed_gates(verdict)


def test_more_than_four_legs_is_vetoed():
    legs = (
        leg(620, Right.PUT, Side.BUY),
        leg(625, Right.PUT, Side.SELL),
        leg(630, Right.PUT, Side.SELL),
        leg(660, Right.CALL, Side.SELL),
        leg(665, Right.CALL, Side.BUY),
    )
    proposal = Proposal("CARRY", "SPY", legs, qty=1, net_price=-1.0)
    verdict = evaluate(proposal, fresh_quotes(proposal), healthy_book())
    assert not verdict.approved
    assert "leg_count" in failed_gates(verdict)


@pytest.mark.parametrize("clock", [(9, 35), (15, 55), (8, 0), (16, 30)])
def test_blackout_windows_block_entries(clock):
    book = healthy_book(now=NOW.replace(hour=clock[0], minute=clock[1]))
    proposal = condor()
    quotes = fresh_quotes(proposal, now=book.now)
    verdict = evaluate(proposal, quotes, book)
    assert not verdict.approved
    assert "session" in failed_gates(verdict)


def test_every_rejection_reason_is_recorded_not_just_the_first():
    # An oversized, stale, illiquid trade should report all three, because the
    # audit log is only useful if it is complete.
    proposal = condor(qty=40)
    quotes = fresh_quotes(proposal, bid=0.0, ask=2.0, now=NOW - timedelta(hours=1))
    verdict = evaluate(proposal, quotes, healthy_book())
    assert {"single_trade_cap", "liquidity", "quote_freshness"} <= failed_gates(verdict)


# --------------------------------------------------------------------------- #
# Tournament clock
# --------------------------------------------------------------------------- #


DEADLINE = datetime(2026, 9, 4, 11, 0, tzinfo=ET)


def test_behind_target_early_leans_in():
    now = DEADLINE - timedelta(days=5)
    assert tournament_risk_multiplier(now, DEADLINE, 100_000, 100_000) > 1.0


def test_ahead_of_target_pulls_back():
    now = DEADLINE - timedelta(days=5)
    assert tournament_risk_multiplier(now, DEADLINE, 110_000, 100_000) < 1.0


def test_final_hours_while_ahead_stop_new_risk():
    now = DEADLINE - timedelta(hours=2)
    assert tournament_risk_multiplier(now, DEADLINE, 108_000, 100_000) == 0.0


def test_multiplier_is_always_bounded():
    for days in (7, 3, 1, 0.5, 0.1, 0):
        for equity in (60_000, 100_000, 250_000):
            value = tournament_risk_multiplier(
                DEADLINE - timedelta(days=days), DEADLINE, equity, 100_000
            )
            assert 0.0 <= value <= 1.5


# --------------------------------------------------------------------------- #
# Multi-expiry structures
# --------------------------------------------------------------------------- #


def calendar_proposal() -> Proposal:
    """Same strike, two expiries. Analysed as a vertical, the legs cancel exactly
    and it reports zero risk — the most dangerous possible wrong answer."""
    near = build_occ("SPY", date(2026, 9, 30), Right.PUT, 655)
    far = build_occ("SPY", date(2026, 10, 2), Right.PUT, 655)
    return Proposal(
        strategy_id="PREFLIGHT",
        underlying="SPY",
        legs=(
            Leg(far, Side.SELL, 1, PositionIntent.SELL_TO_OPEN),
            Leg(near, Side.BUY, 1, PositionIntent.BUY_TO_OPEN),
        ),
        qty=1,
        net_price=-0.05,
    )


def test_calendar_is_not_reported_as_zero_risk():
    profile = analyse_payoff(calendar_proposal())
    assert profile.max_loss is None
    assert not profile.is_defined_risk


def test_calendar_is_vetoed_on_both_expiry_and_defined_risk():
    proposal = calendar_proposal()
    verdict = evaluate(proposal, fresh_quotes(proposal), healthy_book())
    assert not verdict.approved
    gates = {r.gate.split("[")[0] for r in verdict.rejections}
    assert "single_expiry" in gates
    assert "defined_risk" in gates


def test_single_expiry_structures_still_pass():
    proposal = condor()
    verdict = evaluate(proposal, fresh_quotes(proposal), healthy_book())
    assert verdict.approved
    assert all(r.gate != "single_expiry" or r.passed for r in verdict.results)
