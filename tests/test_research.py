"""Tests for the backtest simulator and the promotion gate.

The gate is the part that matters. Generating candidates is trivial; a promotion
only means something if it is hard to earn, so most of these tests are about the
gate refusing.
"""

from __future__ import annotations

import random
import json
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
from aperture.research import (
    Candidate,
    PromotionGate,
    hypothesize,
    mutate,
    mutation_pool,
    run_lab,
)


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


def test_candidate_cannot_be_promoted_without_enough_incumbent_evidence():
    candidate = result_with(GOOD)
    thin_incumbent = result_with([20.0] * 4)
    passed, reason = PromotionGate().evaluate(candidate, thin_incumbent, 1)
    assert not passed
    assert "incumbent has only" in reason


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


def test_featherless_prioritises_hypotheses_but_cannot_invent_parameters():
    pool = mutation_pool(CondorSpec())
    chosen = pool[-1]

    class Provider:
        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return json.dumps({
                "selections": [
                    {"candidate_id": "NOT-IN-THE-FAMILY", "thesis": "invalid"},
                    {"candidate_id": chosen.candidate_id, "thesis": "tests tail-risk tolerance"},
                ]
            })

    provider = Provider()
    selected = hypothesize(provider, CondorSpec(), 3, random.Random(9))

    assert len(selected) == 3
    assert selected[0].candidate_id == chosen.candidate_id
    assert selected[0].hypothesis == "tests tail-risk tolerance"
    assert all(candidate.candidate_id in {row.candidate_id for row in pool} for candidate in selected)
    assert provider.calls[0]["tier"] == "fast"


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
    closes = [100.0] * 50  # includes the 20 February expiry
    strikes = [92.0, 94.0, 96.0, 100.0, 104.0, 106.0, 108.0]
    history = build_history(closes, expiry, strikes)

    result = simulate_condors(history, CondorSpec(short_pct=0.04, width_pct=0.02))
    assert result.n >= 1
    # A flat market should let short premium decay into profit.
    assert result.total_pnl > 0


def test_simulator_charges_slippage_adversely_on_both_sides(monkeypatch):
    expiry = date(2026, 2, 20)
    history = build_history(
        [100.0] * 50,  # includes the 20 February expiry
        expiry,
        [92.0, 94.0, 96.0, 100.0, 104.0, 106.0, 108.0],
    )

    mid = simulate_condors(
        history, CondorSpec(short_pct=0.04, width_pct=0.02, slippage=0.0)
    )
    charged = simulate_condors(
        history, CondorSpec(short_pct=0.04, width_pct=0.02, slippage=0.05)
    )

    assert mid.n and charged.n
    # Worse credit at entry is closer to zero; the equivalent holding mark at
    # exit is more negative because buying it back costs more.
    assert charged.trades[0].entry_price == pytest.approx(mid.trades[0].entry_price + 0.05)
    assert charged.trades[0].pnl < mid.trades[0].pnl

    # Isolate the exit sign: a -0.40 holding mark means a +0.40 debit to close.
    # The adverse fill is +0.45, represented in the simulator as -0.45.
    import aperture.backtest as backtest
    start = date(2026, 1, 5)
    small = OptionHistory(underlying="TEST", spot={
        start: 100.0,
        start + timedelta(days=1): 100.0,
    })
    monkeypatch.setattr(backtest, "_price", lambda *args: -0.40)
    _, exit_price, _ = backtest._manage(
        small, ("a", "b", "c", "d"), start, start + timedelta(days=7),
        -1.00, CondorSpec(slippage=0.05),
    )
    assert exit_price == pytest.approx(-0.45)


def test_simulator_counts_each_expiry_only_once_after_an_early_exit(monkeypatch):
    import aperture.backtest as backtest

    start = date(2026, 1, 1)
    expiries = (date(2026, 1, 16), date(2026, 2, 20))
    history = OptionHistory(
        underlying="TEST",
        spot={start + timedelta(days=i): 100.0 for i in range(70)},
    )
    for expiry in expiries:
        for strike, right in (
            (90, Right.PUT), (95, Right.PUT), (105, Right.CALL), (110, Right.CALL)
        ):
            history.bars[build_occ("TEST", expiry, right, strike)] = {start: 1.0}

    def picked(_history, expiry, _spot, _spec, *, day=None):
        return (
            build_occ("TEST", expiry, Right.PUT, 90),
            build_occ("TEST", expiry, Right.PUT, 95),
            build_occ("TEST", expiry, Right.CALL, 105),
            build_occ("TEST", expiry, Right.CALL, 110),
        )

    monkeypatch.setattr(backtest, "_pick_condor", picked)
    monkeypatch.setattr(backtest, "_price", lambda *args: -1.0)
    monkeypatch.setattr(
        backtest,
        "_manage",
        lambda _history, _legs, entry, _expiry, _price, _spec: (
            entry + timedelta(days=1), -0.4, "early profit"
        ),
    )

    result = simulate_condors(history, CondorSpec(dte_target=14, slippage=0.0))
    traded_expiries = [build_occ("TEST", expiry, Right.PUT, 90)[4:10] for expiry in expiries]
    assert result.n == 2
    assert [trade.legs[0][4:10] for trade in result.trades] == traded_expiries


def test_simulator_selects_the_nearest_contracts_that_traded_that_day():
    """An untraded exact strike must not hide a nearby observable structure."""
    import aperture.backtest as backtest

    today = date(2026, 1, 5)
    expiry = date(2026, 1, 23)
    history = OptionHistory(underlying="TEST", spot={today: 100.0})

    # The exact 96/104 short targets are listed but have no bar. Nearby 95/105
    # and their wings did trade, which mirrors a sparse historical option tape.
    for strike, right, price in (
        (93, Right.PUT, 0.15),
        (95, Right.PUT, 0.85),
        (96, Right.PUT, None),
        (104, Right.CALL, None),
        (105, Right.CALL, 0.90),
        (107, Right.CALL, 0.20),
    ):
        symbol = build_occ("TEST", expiry, right, strike)
        history.bars[symbol] = {} if price is None else {today: price}

    legs = backtest._pick_condor(
        history,
        expiry,
        100.0,
        CondorSpec(short_pct=0.04, width_pct=0.02),
        day=today,
    )

    assert legs is not None
    assert [__import__("aperture.contracts", fromlist=["parse_occ"]).parse_occ(s).strike for s in legs] == [93, 95, 105, 107]


def test_simulator_will_not_fake_settlement_beyond_a_slice(monkeypatch):
    import aperture.backtest as backtest

    start = date(2026, 1, 1)
    future_expiry = date(2026, 2, 20)
    history = OptionHistory(
        underlying="TEST",
        spot={start + timedelta(days=i): 100.0 for i in range(20)},
        bars={build_occ("TEST", future_expiry, Right.PUT, 95): {start: 1.0}},
    )
    called = False

    def should_not_pick(*args):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(backtest, "_pick_condor", should_not_pick)
    assert simulate_condors(history, CondorSpec()).n == 0
    assert not called


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


def test_lab_requires_the_selected_candidate_to_survive_a_separate_holdout(monkeypatch):
    import aperture.research as research

    start = date(2025, 1, 1)
    history = OptionHistory(
        underlying="TEST",
        spot={start + timedelta(days=i): 100.0 for i in range(160)},
        bars={
            build_occ("TEST", date(2025, 5, 30), Right.PUT, 95): {
                start + timedelta(days=i): 1.0 for i in range(160)
            }
        },
    )
    candidate = Candidate(
        "CAND-ONE", CondorSpec(width_pct=0.015), "CARRY", "wider wings",
        "reduce tail loss",
    )
    tried = []
    loaded_specs = []

    class Gate:
        def evaluate(self, result, incumbent, candidates_tried):
            tried.append(candidates_tried)
            if result.strategy_id.endswith("/holdout"):
                return False, "failed unseen history"
            return True, "passed selection"

    def fake_load_history(*args, **kwargs):
        loaded_specs.extend(kwargs["universe_specs"])
        return history

    monkeypatch.setattr(research, "load_history", fake_load_history)
    monkeypatch.setattr(
        research, "hypothesize", lambda provider, spec, count, rng: [candidate]
    )
    monkeypatch.setattr(
        research,
        "simulate_condors",
        lambda _history, _spec, strategy_id: BacktestResult(strategy_id=strategy_id),
    )

    report = run_lab(
        object(), underlying="TEST", gate=Gate(), seed=1, prior_trials=11,
        asof=date(2025, 7, 1),
    )

    assert report.promoted == []
    assert report.rejected[0][2] == "holdout: failed unseen history"
    assert report.multiple_testing_trials == 12
    assert tried == [12, 12]
    assert candidate.spec in loaded_specs
    training_end = date.fromisoformat(report.training_window[1])
    validation_start = date.fromisoformat(report.validation_window[0])
    assert (validation_start - training_end).days >= 35


def test_promotion_record_is_stable_and_carries_holdout_evidence():
    from aperture.research import LabReport, promotion_records

    candidate = Candidate("S5-WIDTH", CondorSpec(width_pct=0.015), "CARRY", "wider wings")
    evidence = result_with(GOOD)
    report = LabReport(
        tested=8,
        promoted=[(candidate, evidence, "selection and holdout survived")],
        training_window=("2024-03-01", "2025-12-01"),
        validation_window=("2026-01-05", "2026-07-29"),
    )
    first = promotion_records(report, underlying="SPY")[0]
    second = promotion_records(report, underlying="SPY")[0]

    assert first["strategy_id"] == second["strategy_id"]
    assert first["strategy_id"].startswith("LAB-")
    assert first["status"] == "probation"
    assert first["backtest"]["trades"] == len(GOOD)
    assert first["validation_window"] == report.validation_window


# --------------------------------------------------------------------------- #
# Historical-universe completeness
# --------------------------------------------------------------------------- #


def test_full_candidate_pool_drives_the_required_geometry():
    specs = [CondorSpec(), *(candidate.spec for candidate in mutation_pool(CondorSpec()))]

    assert max(spec.short_pct + spec.width_pct for spec in specs) == pytest.approx(0.065)
    assert max(spec.dte_target + 14 for spec in specs) == 35


def test_strike_window_tracks_actual_eligible_spot_not_one_anchor():
    from aperture.backtest import _strike_window

    expiry = date(2026, 2, 20)
    spot = {
        expiry - timedelta(days=40): 100.0,  # outside every entry window
        expiry - timedelta(days=20): 105.0,
    }
    specs = [CondorSpec(), *(candidate.spec for candidate in mutation_pool(CondorSpec()))]

    low, high = _strike_window(spot, expiry, specs)

    # Mirrors the live strategy's minimum ±12% catalogue band around the spot
    # that was actually eligible for an entry.
    assert low <= 105.0 * 0.88
    assert high == pytest.approx(105.0 * 1.12)


def test_history_loader_keeps_every_contract_in_the_derived_band(monkeypatch):
    import aperture.backtest as backtest

    expiry = date(2026, 2, 20)
    entry_day = expiry - timedelta(days=20)

    class CompleteCLI:
        def __init__(self):
            self.requested_symbols = []
            self.bounds = []

        def stock_bars(self, symbol, start):
            return {"bars": {symbol: [{"t": f"{entry_day}T00:00:00Z", "c": 105.0}]}}

        def run(self, *args):
            option_type = args[args.index("--type") + 1]
            self.bounds.append((
                float(args[args.index("--strike-price-gte") + 1]),
                float(args[args.index("--strike-price-lte") + 1]),
            ))
            right = Right.CALL if option_type == "call" else Right.PUT
            return {
                "option_contracts": [
                    {"symbol": build_occ("SPY", expiry, right, strike)}
                    for strike in range(500, 660)
                ]
            }

        def option_bars(self, symbols, start, page_token=None):
            self.requested_symbols.extend(symbols)
            return {"bars": {}, "next_page_token": None}

    cli = CompleteCLI()
    monkeypatch.setattr(backtest, "_target_expiries", lambda *args: [expiry])
    backtest.load_history(
        cli,
        "SPY",
        start=entry_day,
        end=expiry,
        expiries=1,
        universe_specs=[CondorSpec()],
        cache_dir=None,
    )

    assert len(cli.requested_symbols) == 320
    assert len(set(cli.requested_symbols)) == 320
    assert all(
        low <= 105.0 * 0.88 and high == pytest.approx(105.0 * 1.12)
        for low, high in cli.bounds
    )


def test_target_expiries_spread_across_the_window():
    from aperture.backtest import _target_expiries

    picked = _target_expiries(date(2026, 1, 1), date(2026, 7, 1), 6)
    assert len(picked) == 6
    assert all(d.weekday() == 4 for d in picked)
    assert picked == sorted(picked)
    # Genuinely spread out, not six consecutive Fridays.
    assert (picked[-1] - picked[0]).days > 100


def test_history_cache_key_changes_with_loader_version(monkeypatch):
    import aperture.backtest as backtest

    base_signature = backtest._universe_signature([CondorSpec()])
    args = ("SPY", date(2025, 1, 1), date(2026, 1, 1), 12, base_signature)
    current = backtest._cache_key(*args)
    wider = backtest._universe_signature([CondorSpec(), CondorSpec(short_pct=0.055)])

    assert backtest._cache_key(*args[:-1], wider) != current

    monkeypatch.setattr(backtest, "HISTORY_CACHE_VERSION", 999)

    assert backtest._cache_key(*args) != current


def test_option_history_follows_every_page_and_merges_symbols():
    from aperture.backtest import _all_option_bars

    class PagedCLI:
        def __init__(self):
            self.tokens = []

        def option_bars(self, symbols, start, page_token=None):
            self.tokens.append(page_token)
            if page_token is None:
                return {
                    "bars": {"CALL": [{"t": "2026-01-02T00:00:00Z", "c": 1.0}]},
                    "next_page_token": "page-two",
                }
            return {
                "bars": {
                    "CALL": [{"t": "2026-01-03T00:00:00Z", "c": 0.9}],
                    "PUT": [{"t": "2026-01-02T00:00:00Z", "c": 1.1}],
                },
                "next_page_token": None,
            }

    cli = PagedCLI()
    merged = _all_option_bars(cli, ["CALL", "PUT"], start="2026-01-01")

    assert cli.tokens == [None, "page-two"]
    assert len(merged["CALL"]) == 2
    assert len(merged["PUT"]) == 1


def test_option_history_refuses_repeated_tokens_and_page_overflow():
    from aperture.backtest import _all_option_bars

    class RepeatingCLI:
        def option_bars(self, symbols, start, page_token=None):
            return {"bars": {}, "next_page_token": "same-token"}

    with pytest.raises(RuntimeError, match="repeated"):
        _all_option_bars(RepeatingCLI(), ["A"], start="2026-01-01")

    class EndlessCLI:
        def __init__(self):
            self.page = 0

        def option_bars(self, symbols, start, page_token=None):
            self.page += 1
            return {"bars": {}, "next_page_token": f"page-{self.page}"}

    with pytest.raises(RuntimeError, match="safety bound"):
        _all_option_bars(EndlessCLI(), ["A"], start="2026-01-01", max_pages=3)


def test_alpaca_cli_forwards_the_option_bar_page_token(monkeypatch):
    from aperture.alpaca_cli import AlpacaCLI

    cli = AlpacaCLI.__new__(AlpacaCLI)
    seen = []

    def fake_run(*args):
        seen.extend(args)
        return {"bars": {}}

    monkeypatch.setattr(cli, "run", fake_run)
    cli.option_bars(
        ["SPY260918C00700000"],
        start="2026-08-01",
        page_token="next-page",
    )

    assert seen[seen.index("--page-token") + 1] == "next-page"
