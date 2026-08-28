"""Tests for the backtest simulator and the promotion gate.

The gate is the part that matters. Generating candidates is trivial; a promotion
only means something if it is hard to earn, so most of these tests are about the
gate refusing.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from aperture.backtest import (
    BacktestResult,
    CondorSpec,
    OptionHistory,
    SimulatedTrade,
    simulate_condors,
)
from aperture.contracts import Right, build_occ
from aperture.research import Candidate, PromotionGate, mutate, run_lab


def trade(pnl: float, risk: float = 1_000.0) -> SimulatedTrade:
    # entry - exit, times -1, times 100 -> pnl. Solve for exit.
    entry = -1.00
    exit_price = entry + pnl / 100.0
    return SimulatedTrade(
        entry=date(2026, 1, 5), exit=date(2026, 1, 12),
        legs=("A", "B", "C", "D"), qty=1,
        entry_price=entry, exit_price=exit_price, max_loss=risk, reason="test",
    )


def result_with(pnls: list[float], risk: float = 1_000.0) -> BacktestResult:
    r = BacktestResult(strategy_id="CANDIDATE")
    r.trades = [trade(p, risk) for p in pnls]
    return r


# --------------------------------------------------------------------------- #
# Result arithmetic
# --------------------------------------------------------------------------- #


def test_pnl_sign_follows_the_entry_convention():
    # Sold for a 1.00 credit, bought back for 0.40: a 0.60 profit on one contract.
    t = SimulatedTrade(
        entry=date(2026, 1, 5), exit=date(2026, 1, 9), legs=("A",), qty=1,
        entry_price=-1.00, exit_price=-0.40, max_loss=400.0, reason="take profit",
    )
    assert t.pnl == pytest.approx(60.0)


def test_a_losing_credit_trade_is_negative():
    t = SimulatedTrade(
        entry=date(2026, 1, 5), exit=date(2026, 1, 9), legs=("A",), qty=1,
        entry_price=-1.00, exit_price=-2.50, max_loss=400.0, reason="stop",
    )
    assert t.pnl == pytest.approx(-150.0)


def test_edge_is_pnl_over_risk_deployed():
    r = result_with([100.0, -50.0, 200.0])
    assert r.total_pnl == pytest.approx(250.0)
    assert r.edge == pytest.approx(250.0 / 3_000.0)


def test_t_stat_needs_a_minimum_sample():
    assert result_with([100.0, 100.0]).t_stat == 0.0


def test_t_stat_rewards_consistency_over_size():
    steady = result_with([60.0, 55.0, 65.0, 58.0, 62.0, 59.0])
    lumpy = result_with([400.0, -300.0, 350.0, -280.0, 300.0, -110.0])
    assert steady.total_pnl < lumpy.total_pnl
    assert steady.t_stat > lumpy.t_stat


def test_max_drawdown_tracks_the_worst_run():
    r = result_with([100.0, 100.0, -300.0, 50.0])
    assert r.max_drawdown == pytest.approx(300.0)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


GOOD = [80.0, 70.0, 90.0, 60.0, 85.0, 75.0, 95.0, 65.0, 88.0, 72.0, 78.0, 82.0]


def test_a_strong_candidate_is_promoted():
    passed, reason = PromotionGate().evaluate(result_with(GOOD), None, candidates_tried=1)
    assert passed, reason
    assert "promoted" in reason


def test_too_few_trades_is_refused():
    passed, reason = PromotionGate().evaluate(result_with(GOOD[:4]), None, 1)
    assert not passed
    assert "trades" in reason


def test_a_thin_edge_is_refused():
    passed, reason = PromotionGate().evaluate(result_with([5.0] * 14), None, 1)
    assert not passed
    assert "below the" in reason


def test_a_lucky_winner_from_a_big_field_is_refused():
    """The garden of forking paths: test twenty mutations and the best looks good
    whether or not any of them work. The bar has to rise with the field."""
    # t = 3.37: comfortably significant on its own, not significant enough to
    # survive being the best of forty.
    modest = result_with([150.0, 10.0, 200.0, -20.0, 180.0, 5.0, 160.0, 20.0,
                          140.0, -10.0, 190.0, 15.0])
    solo_pass, _ = PromotionGate().evaluate(modest, None, candidates_tried=1)
    field_pass, reason = PromotionGate().evaluate(modest, None, candidates_tried=40)
    assert solo_pass
    assert not field_pass
    assert "field of 40" in reason


def test_the_bar_rises_monotonically_with_the_field():
    gate = PromotionGate()
    bars = [gate.required_t(k) for k in (1, 5, 20, 100)]
    assert bars == sorted(bars)
    assert bars[0] == pytest.approx(gate.base_t_stat)


def test_a_marginal_improvement_on_the_incumbent_is_refused():
    incumbent = result_with(GOOD)
    candidate = result_with([p + 1.0 for p in GOOD])
    passed, reason = PromotionGate().evaluate(candidate, incumbent, 1)
    assert not passed
    assert "churn" in reason


def test_a_real_improvement_on_the_incumbent_is_promoted():
    incumbent = result_with([20.0] * 12)
    candidate = result_with(GOOD)
    passed, reason = PromotionGate().evaluate(candidate, incumbent, 1)
    assert passed, reason
    assert "vs incumbent" in reason


def test_a_profitable_but_rough_candidate_is_refused():
    """Significant and profitable, but it gives back 74% of its own profit in one
    run. Clears the t-stat bar and is refused anyway."""
    rough = result_with([200.0] * 16 + [-170.0] * 8)
    assert rough.t_stat > PromotionGate().required_t(1)
    assert rough.edge > PromotionGate().min_edge
    passed, reason = PromotionGate().evaluate(rough, None, 1)
    assert not passed
    assert "drawdown" in reason


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #


def test_mutations_change_exactly_one_parameter():
    base = CondorSpec()
    for candidate in mutate(base, 8, random.Random(7)):
        differences = [
            f for f in ("short_pct", "width_pct", "dte_target", "take_profit", "stop_multiple")
            if getattr(candidate.spec, f) != getattr(base, f)
        ]
        assert len(differences) == 1, f"{candidate.candidate_id} changed {differences}"


def test_mutations_stay_inside_sane_bounds():
    for candidate in mutate(CondorSpec(), 30, random.Random(1)):
        s = candidate.spec
        assert 0.015 <= s.short_pct <= 0.12
        assert 0.005 <= s.width_pct <= 0.05
        assert 5 <= s.dte_target <= 45
        assert 0.25 <= s.take_profit <= 0.85
        assert 1.5 <= s.stop_multiple <= 4.0


def test_mutations_are_distinct_and_described():
    candidates = mutate(CondorSpec(), 6, random.Random(3))
    assert len({c.candidate_id for c in candidates}) == len(candidates)
    assert all("->" in c.mutation for c in candidates)


def test_mutation_is_deterministic_for_a_seed():
    a = [c.mutation for c in mutate(CondorSpec(), 5, random.Random(11))]
    b = [c.mutation for c in mutate(CondorSpec(), 5, random.Random(11))]
    assert a == b


# --------------------------------------------------------------------------- #
# Simulation against a synthetic history
# --------------------------------------------------------------------------- #


def build_history(closes: list[float], expiry: date, strikes: list[float]) -> OptionHistory:
    """A market that sits still, so short premium decays to zero by expiry.

    Time value falls away from the money. Giving every strike the same time value
    makes a condor price at exactly zero -- the short and long legs cancel -- and
    then it is not a credit structure at all.
    """
    history = OptionHistory(underlying="TEST")
    start = date(2026, 1, 5)
    days = [start + timedelta(days=i) for i in range(len(closes))]
    for day, close in zip(days, closes):
        history.spot[day] = close

    for strike in strikes:
        for right in (Right.CALL, Right.PUT):
            symbol = build_occ("TEST", expiry, right, strike)
            table = {}
            for day, close in zip(days, closes):
                remaining = max((expiry - day).days, 0)
                intrinsic = (
                    max(close - strike, 0) if right is Right.CALL else max(strike - close, 0)
                )
                moneyness = abs(strike - close) / close
                decay = max(0.0, 1.0 - moneyness / 0.12)
                table[day] = round(intrinsic + 0.06 * remaining * decay, 2)
            history.bars[symbol] = table
    return history


def test_simulator_produces_trades_on_a_workable_history():
    expiry = date(2026, 2, 20)
    closes = [100.0] * 40
    strikes = [92.0, 94.0, 96.0, 100.0, 104.0, 106.0, 108.0]
    history = build_history(closes, expiry, strikes)

    result = simulate_condors(history, CondorSpec(short_pct=0.04, width_pct=0.02))
    assert result.n >= 1
    # A flat market should let short premium decay into profit.
    assert result.total_pnl > 0


def test_simulator_returns_nothing_when_history_is_too_short():
    history = OptionHistory(underlying="TEST")
    history.spot = {date(2026, 1, 5): 100.0}
    assert simulate_condors(history, CondorSpec()).n == 0


def test_lab_does_nothing_without_history():
    class EmptyCLI:
        def stock_bars(self, *a, **k):
            return {"bars": []}

        def run(self, *a, **k):
            return {}

        def option_bars(self, *a, **k):
            return {}

    report = run_lab(EmptyCLI(), underlying="TEST", seed=1)
    assert report.tested == 0
    assert report.promoted == []


# --------------------------------------------------------------------------- #
# Listing order: the bug class that appeared four times
# --------------------------------------------------------------------------- #


def test_strike_selection_centres_on_the_anchor_not_the_head_of_the_list():
    """Alpaca listings come back ordered. A head slice of a strike-sorted page
    returns only the lowest strikes -- far OTM contracts that never traded and
    therefore have no bars -- so the chain looks empty where it matters."""
    from aperture.backtest import _nearest_strikes

    expiry = date(2026, 2, 20)
    symbols = [build_occ("SPY", expiry, Right.CALL, k) for k in range(600, 760, 5)]
    picked = _nearest_strikes(symbols, anchor=690.0, count=6)
    strikes = sorted(build_occ and __import__(
        "aperture.contracts", fromlist=["parse_occ"]).parse_occ(s).strike for s in picked)

    assert len(picked) == 6
    assert min(strikes) < 690.0 < max(strikes)      # brackets the anchor
    assert all(abs(k - 690.0) <= 20 for k in strikes)
    assert 600.0 not in strikes                      # not the head of the list


def test_target_expiries_spread_across_the_window():
    from aperture.backtest import _target_expiries

    picked = _target_expiries(date(2026, 1, 1), date(2026, 7, 1), 6)
    assert len(picked) == 6
    assert all(d.weekday() == 4 for d in picked)
    assert picked == sorted(picked)
    # Genuinely spread out, not six consecutive Fridays.
    assert (picked[-1] - picked[0]).days > 100
