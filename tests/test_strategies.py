"""Tests for the market-data maths, the structure builders and the Warden."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from aperture.contracts import Right, Side, build_occ
from aperture.earnings import EarningsEvent, Timing
from aperture.llm import NullProvider, TokenBudget, ask_json
from aperture.marketdata import (
    Snapshot,
    atm_pair,
    compare_moves,
    implied_total_move,
    parse_snapshot,
    realized_earnings_moves,
    realized_vol,
    select_by_delta,
    select_by_strike_offset,
    strip_diffusive_vol,
)
from aperture.risk import BookState, RiskLimits
from aperture.strategies.base import (
    build_iron_condor,
    concede,
    credit_to_width_ok,
    size_to_budget,
    structure_price,
)
from aperture.strategies.carry import DEFAULT_CONFIG as CARRY_CONFIG
from aperture.strategies.carry import exit_signal
from aperture.warden import AuditLog, RiskWarden

ET = timezone(timedelta(hours=-4))
NOW = datetime(2026, 9, 1, 11, 0, tzinfo=ET)
EXPIRY = date(2026, 9, 18)


def snap(strike: float, right: Right, bid: float, ask: float, delta: float | None = None) -> Snapshot:
    return Snapshot(
        symbol=build_occ("SPY", EXPIRY, right, strike),
        bid=bid,
        ask=ask,
        quote_ts=NOW,
        delta=delta,
        open_interest=5_000,
    )


# --------------------------------------------------------------------------- #
# Snapshot parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        {"latestQuote": {"bp": 1.0, "ap": 1.2, "t": "2026-09-01T15:00:00Z"},
         "greeks": {"delta": -0.15}, "impliedVolatility": 0.22},
        {"latest_quote": {"bid_price": 1.0, "ask_price": 1.2, "timestamp": "2026-09-01T15:00:00Z"},
         "greeks": {"delta": -0.15}, "implied_volatility": 0.22},
    ],
)
def test_snapshot_parses_both_alpaca_spellings(raw):
    parsed = parse_snapshot("SPY260918P00630000", raw)
    assert parsed.bid == 1.0
    assert parsed.ask == 1.2
    assert parsed.delta == -0.15
    assert parsed.implied_volatility == 0.22
    assert parsed.mid == pytest.approx(1.1)


def test_snapshot_without_open_interest_is_marked_unknown():
    parsed = parse_snapshot("SPY260918P00630000", {"latestQuote": {"bp": 1.0, "ap": 1.2}})
    # Unknown stays -1 rather than collapsing to 0, so the depth gate can tell
    # "not reported" apart from "genuinely zero" and fall back to volume/size.
    assert parsed.open_interest == -1
    assert parsed.to_leg_quote().open_interest == -1


def test_zero_bid_contract_is_not_priceable():
    assert not snap(600, Right.PUT, 0.0, 0.05).is_priceable


# --------------------------------------------------------------------------- #
# Implied move
# --------------------------------------------------------------------------- #


def test_implied_total_move_is_the_straddle_over_spot():
    call = snap(650, Right.CALL, 9.8, 10.2)
    put = snap(650, Right.PUT, 9.8, 10.2)
    assert implied_total_move(call, put, 650.0) == pytest.approx(20.0 / 650.0)


def test_stripping_diffusive_vol_reduces_the_event_estimate():
    # A 6% straddle 10 days out, on a name with 30% annual vol, is not a 6% event.
    total = 0.06
    event = strip_diffusive_vol(total, dte=10, baseline_annual_vol=0.30)
    assert 0 < event < total
    # Variance is additive, so the pieces must reconstruct the whole.
    diffusive = 0.30 * math.sqrt(10 / 252)
    assert event**2 + diffusive**2 == pytest.approx(total**2)


def test_stripping_cannot_go_negative():
    # Baseline vol alone exceeds the straddle: the event component is zero, not imaginary.
    assert strip_diffusive_vol(0.01, dte=30, baseline_annual_vol=0.80) == 0.0


def test_realized_earnings_moves_measured_close_to_close():
    bars = [
        {"t": "2026-05-18T00:00:00Z", "c": 100.0},
        {"t": "2026-05-19T00:00:00Z", "c": 110.0},  # +10% on the report
        {"t": "2026-08-17T00:00:00Z", "c": 200.0},
        {"t": "2026-08-18T00:00:00Z", "c": 190.0},  # -5% on the report
    ]
    moves = realized_earnings_moves(bars, [date(2026, 5, 19), date(2026, 8, 18)])
    assert sorted(round(m, 4) for m in moves) == [0.05, 0.10]


def test_realized_earnings_moves_skips_events_without_a_prior_bar():
    bars = [{"t": "2026-05-19T00:00:00Z", "c": 110.0}]
    assert realized_earnings_moves(bars, [date(2026, 5, 19)]) == []


def test_compare_moves_needs_enough_history_to_claim_an_edge():
    bars = [{"t": f"2026-08-{d:02d}T00:00:00Z", "c": 100.0 + d} for d in range(1, 28)]
    result = compare_moves(
        "TEST", EXPIRY,
        snap(650, Right.CALL, 9.8, 10.2), snap(650, Right.PUT, 9.8, 10.2),
        650.0, bars, [date(2026, 8, 5)],  # a single past event
    )
    assert result is None


def test_rich_and_cheap_classification():
    bars = [{"t": f"2026-0{m}-15T00:00:00Z", "c": c}
            for m, c in zip(range(1, 9), [100, 102, 100, 103, 100, 102, 100, 103])]
    events = [date(2026, m, 15) for m in range(2, 9)]
    # A fat straddle against a name that historically moves ~2-3%.
    result = compare_moves(
        "TEST", date(2026, 9, 4),
        snap(100, Right.CALL, 5.0, 5.2), snap(100, Right.PUT, 5.0, 5.2),
        100.0, bars, events, asof=date(2026, 9, 3),
    )
    assert result is not None
    assert result.is_rich
    assert not result.is_cheap
    assert "market prices" in result.explain()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_select_by_delta_picks_the_closest():
    pool = [
        snap(620, Right.PUT, 0.5, 0.6, delta=-0.08),
        snap(630, Right.PUT, 1.0, 1.1, delta=-0.16),
        snap(640, Right.PUT, 2.0, 2.1, delta=-0.30),
    ]
    assert select_by_delta(pool, 0.15, Right.PUT).strike == 630


def test_select_by_delta_ignores_contracts_without_greeks():
    pool = [snap(630, Right.PUT, 1.0, 1.1, delta=None)]
    assert select_by_delta(pool, 0.15, Right.PUT) is None


def test_strike_offset_walks_the_right_direction():
    pool = [snap(s, Right.PUT, 0.5, 0.6) for s in (615, 620, 625, 630)]
    anchor = snap(630, Right.PUT, 1.0, 1.1)
    # A put wing sits *below* the short strike.
    assert select_by_strike_offset(pool, anchor, 5.0, Right.PUT).strike == 625

    calls = [snap(s, Right.CALL, 0.5, 0.6) for s in (660, 665, 670)]
    anchor_call = snap(660, Right.CALL, 1.0, 1.1)
    assert select_by_strike_offset(calls, anchor_call, 5.0, Right.CALL).strike == 665


def test_atm_pair_matches_strikes():
    pool = [snap(645, Right.CALL, 5, 5.2), snap(650, Right.CALL, 3, 3.2),
            snap(645, Right.PUT, 4, 4.2), snap(650, Right.PUT, 6, 6.2)]
    call, put = atm_pair(pool, 649.0, EXPIRY)
    assert call.strike == put.strike == 650


# --------------------------------------------------------------------------- #
# Structure building
# --------------------------------------------------------------------------- #


def test_structure_price_signs_credit_negative():
    legs = [(snap(625, Right.PUT, 0.9, 1.1), Side.BUY),
            (snap(630, Right.PUT, 2.4, 2.6), Side.SELL)]
    assert structure_price(legs) == pytest.approx(1.0 - 2.5)  # -1.50, a credit


@pytest.mark.parametrize("price,expected", [(-1.50, -1.45), (2.00, 2.05), (0.0, 0.05)])
def test_concede_always_moves_toward_the_market(price, expected):
    # Credits shrink, debits grow. Both mean "accept slightly worse to get filled".
    assert concede(price, 0.05) == pytest.approx(expected)


def test_size_to_budget_floors_and_caps():
    assert size_to_budget(320.0, 1_000.0) == 3  # never 3.1 contracts
    assert size_to_budget(320.0, 0.0) == 0
    assert size_to_budget(0.0, 1_000.0) == 0
    assert size_to_budget(1.0, 10_000_000.0) == 50  # hard cap


def test_iron_condor_builds_and_is_defined_risk():
    proposal = build_iron_condor(
        short_put=snap(630, Right.PUT, 2.4, 2.6),
        long_put=snap(625, Right.PUT, 0.9, 1.1),
        short_call=snap(660, Right.CALL, 2.4, 2.6),
        long_call=snap(665, Right.CALL, 0.9, 1.1),
        strategy_id="CARRY", underlying="SPY", budget=5_000.0,
        slippage=0.05, rationale="test",
    )
    assert proposal is not None
    assert proposal.net_price < 0  # a credit
    assert len(proposal.legs) == 4
    assert proposal.qty >= 1


def test_iron_condor_rejects_crossed_strikes():
    assert build_iron_condor(
        short_put=snap(660, Right.PUT, 2.4, 2.6),   # short put above the short call
        long_put=snap(625, Right.PUT, 0.9, 1.1),
        short_call=snap(630, Right.CALL, 2.4, 2.6),
        long_call=snap(665, Right.CALL, 0.9, 1.1),
        strategy_id="CARRY", underlying="SPY", budget=5_000.0,
        slippage=0.05, rationale="test",
    ) is None


def test_iron_condor_rejects_unpriceable_legs():
    assert build_iron_condor(
        short_put=snap(630, Right.PUT, 0.0, 2.6),  # no bid
        long_put=snap(625, Right.PUT, 0.9, 1.1),
        short_call=snap(660, Right.CALL, 2.4, 2.6),
        long_call=snap(665, Right.CALL, 0.9, 1.1),
        strategy_id="CARRY", underlying="SPY", budget=5_000.0,
        slippage=0.05, rationale="test",
    ) is None


def test_credit_to_width_gate_rejects_a_cheap_spread():
    fat = build_iron_condor(
        short_put=snap(630, Right.PUT, 2.4, 2.6), long_put=snap(625, Right.PUT, 0.9, 1.1),
        short_call=snap(660, Right.CALL, 2.4, 2.6), long_call=snap(665, Right.CALL, 0.9, 1.1),
        strategy_id="CARRY", underlying="SPY", budget=5_000.0, slippage=0.05, rationale="",
    )
    thin = build_iron_condor(
        short_put=snap(630, Right.PUT, 1.00, 1.05), long_put=snap(625, Right.PUT, 0.94, 0.99),
        short_call=snap(660, Right.CALL, 1.00, 1.05), long_call=snap(665, Right.CALL, 0.94, 0.99),
        strategy_id="CARRY", underlying="SPY", budget=5_000.0, slippage=0.05, rationale="",
    )
    assert credit_to_width_ok(fat, 0.15)
    assert not credit_to_width_ok(thin, 0.15)


# --------------------------------------------------------------------------- #
# Exits
# --------------------------------------------------------------------------- #


def test_take_profit_at_half_the_credit():
    assert "take profit" in exit_signal(-2.00, -0.90, CARRY_CONFIG)


def test_stop_out_at_twice_the_credit():
    assert "stop loss" in exit_signal(-2.00, -4.20, CARRY_CONFIG)


def test_no_exit_in_between():
    assert exit_signal(-2.00, -1.50, CARRY_CONFIG) is None


# --------------------------------------------------------------------------- #
# Earnings timing
# --------------------------------------------------------------------------- #


def test_after_close_report_moves_the_next_session():
    event = EarningsEvent("PANW", date(2026, 9, 1), Timing.AFTER_CLOSE)
    assert event.enter_on == date(2026, 9, 1)
    assert event.first_session_after == date(2026, 9, 2)


def test_before_open_report_moves_that_same_session():
    event = EarningsEvent("MDT", date(2026, 9, 1), Timing.BEFORE_OPEN)
    assert event.enter_on == date(2026, 8, 31)  # the prior session
    assert event.first_session_after == date(2026, 9, 1)


def test_friday_after_close_skips_the_weekend():
    event = EarningsEvent("X", date(2026, 9, 4), Timing.AFTER_CLOSE)
    assert event.first_session_after == date(2026, 9, 7)  # Monday


# --------------------------------------------------------------------------- #
# Warden
# --------------------------------------------------------------------------- #


def book(**overrides) -> BookState:
    defaults = dict(equity=100_000.0, high_water_mark=100_000.0,
                    day_start_equity=100_000.0, cash=100_000.0, now=NOW)
    defaults.update(overrides)
    return BookState(**defaults)


def condor_proposal():
    return build_iron_condor(
        short_put=snap(630, Right.PUT, 2.4, 2.6), long_put=snap(625, Right.PUT, 0.9, 1.1),
        short_call=snap(660, Right.CALL, 2.4, 2.6), long_call=snap(665, Right.CALL, 0.9, 1.1),
        strategy_id="CARRY", underlying="SPY", budget=3_000.0, slippage=0.05, rationale="test",
    )


def make_warden(tmp_path, **overrides) -> RiskWarden:
    return RiskWarden(
        limits=RiskLimits(),
        audit=AuditLog(path=tmp_path / "audit.jsonl"),
        budgets={"CARRY": 50_000.0},
        **overrides,
    )


def test_warden_logs_approvals_and_vetoes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    warden = make_warden(tmp_path)
    proposal = condor_proposal()
    quotes = [
        Snapshot(symbol=leg.symbol, bid=1.0, ask=1.05, quote_ts=NOW, open_interest=5_000).to_leg_quote()
        for leg in proposal.legs
    ]
    verdict = warden.review(proposal, quotes, book())
    assert verdict.approved
    assert warden.audit.tail()[-1]["event"] == "approval"


def test_warden_veto_is_written_with_reasons(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    warden = make_warden(tmp_path)
    proposal = condor_proposal()
    stale = [
        Snapshot(symbol=leg.symbol, bid=1.0, ask=1.05,
                 quote_ts=NOW - timedelta(hours=2), open_interest=5_000).to_leg_quote()
        for leg in proposal.legs
    ]
    verdict = warden.review(proposal, stale, book())
    assert not verdict.approved
    entry = warden.audit.vetoes()[-1]
    assert entry["event"] == "veto"
    assert any("quote_freshness" in r["gate"] for r in entry["reasons"])


def test_kill_switch_blocks_everything(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    warden = make_warden(tmp_path)
    warden.engage_kill_switch("manual halt")
    proposal = condor_proposal()
    quotes = [
        Snapshot(symbol=leg.symbol, bid=1.0, ask=1.05, quote_ts=NOW, open_interest=5_000).to_leg_quote()
        for leg in proposal.legs
    ]
    assert not warden.review(proposal, quotes, book()).approved
    warden.release_kill_switch()
    assert warden.review(proposal, quotes, book()).approved


def test_breach_detection(tmp_path):
    warden = make_warden(tmp_path)
    assert warden.breached(book()) is None
    assert "drawdown" in warden.breached(book(equity=90_000.0, day_start_equity=90_000.0))
    assert "day P&L" in warden.breached(book(equity=96_000.0))


def test_budget_shrinks_as_the_deadline_approaches_while_ahead(tmp_path):
    deadline = datetime(2026, 9, 4, 11, 0, tzinfo=ET)
    warden = make_warden(tmp_path, deadline=deadline, start_equity=100_000.0)
    early = book(now=deadline - timedelta(days=5), equity=108_000.0)
    late = book(now=deadline - timedelta(hours=2), equity=108_000.0)
    assert warden.budget_for("CARRY", early) > warden.budget_for("CARRY", late)
    assert warden.budget_for("CARRY", late) == 0.0


# --------------------------------------------------------------------------- #
# LLM layer
# --------------------------------------------------------------------------- #


def test_desk_runs_without_a_model():
    provider = NullProvider(canned='{"verdict": "ok"}')
    assert ask_json(provider, system="s", user="u", schema={}, default=None) == {"verdict": "ok"}


def test_bad_model_output_falls_back_to_the_default():
    provider = NullProvider(canned="not json at all")
    assert ask_json(provider, system="s", user="u", schema={}, default={"safe": True}) == {"safe": True}


def test_token_budget_tracks_and_exhausts():
    budget = TokenBudget(fast_daily=100, reasoning_daily=10)
    budget.charge("fast", 60)
    assert budget.remaining("fast") == 40
    budget.charge("fast", 60)
    assert budget.remaining("fast") == 0
    budget.reset()
    assert budget.remaining("fast") == 100


# --------------------------------------------------------------------------- #
# Regression: the earnings move must be measured on the session that carried it
# --------------------------------------------------------------------------- #


# Real PANW data around its 2 June 2026 report (after the close).
# The gap was -5.6% on 3 June. The report date itself moved -1.1%.
PANW_BARS = [
    {"t": "2026-05-29T00:00:00Z", "c": 281.69},
    {"t": "2026-06-01T00:00:00Z", "c": 300.48},
    {"t": "2026-06-02T00:00:00Z", "c": 297.18},   # report date, after close
    {"t": "2026-06-03T00:00:00Z", "c": 280.43},   # <- the gap
    {"t": "2026-06-04T00:00:00Z", "c": 279.25},
]


def test_after_close_report_measures_the_next_session_not_the_report_date():
    """The bug that would have made CRUSH take the wrong side of its own edge.

    Measuring the report date captures the day before the news, which makes every
    event look calm and pushes the desk into selling premium on events that are
    actually underpriced.
    """
    event = EarningsEvent("PANW", date(2026, 6, 2), Timing.AFTER_CLOSE)
    moves = realized_earnings_moves(PANW_BARS, [event])
    assert moves == pytest.approx([abs(280.43 / 297.18 - 1)], rel=1e-6)
    assert moves[0] == pytest.approx(0.0564, abs=0.0005)   # the real -5.6% gap

    # A bare date, lacking timing, lands on the wrong session. That is exactly
    # why the API takes events.
    wrong = realized_earnings_moves(PANW_BARS, [date(2026, 6, 2)])
    assert wrong[0] == pytest.approx(0.0111, abs=0.0005)   # the -1.1% non-event


def test_before_open_report_measures_that_same_session():
    event = EarningsEvent("PANW", date(2026, 6, 3), Timing.BEFORE_OPEN)
    moves = realized_earnings_moves(PANW_BARS, [event])
    assert moves[0] == pytest.approx(0.0564, abs=0.0005)


def test_baseline_vol_excludes_the_gap_sessions():
    """Leave earnings gaps in the baseline and it absorbs the very jumps it is
    measured against, erasing the event component from the implied move."""
    from aperture.marketdata import gap_sessions

    # A calm series drifting ~0.1% a day, with one 8% earnings gap dropped in.
    calm = [
        {"t": f"2026-04-{day:02d}T00:00:00Z", "c": 100.0 + 0.1 * day}
        for day in range(1, 29)
    ]
    gapped = calm + [
        {"t": "2026-06-02T00:00:00Z", "c": 102.8},   # report date, after close
        {"t": "2026-06-03T00:00:00Z", "c": 111.0},   # +8% gap
        {"t": "2026-06-04T00:00:00Z", "c": 111.1},
    ]
    event = EarningsEvent("PANW", date(2026, 6, 2), Timing.AFTER_CLOSE)
    assert gap_sessions([event]) == [date(2026, 6, 3)]

    with_gap = realized_vol(gapped)
    without = realized_vol(gapped, exclude=gap_sessions([event]))
    assert without < with_gap
    # The gap dominates a calm series, so removing it should cut the baseline hard.
    assert without < with_gap / 2


def test_gap_sessions_accepts_bare_dates_too():
    from aperture.marketdata import gap_sessions

    assert gap_sessions([date(2026, 6, 2)]) == [date(2026, 6, 2)]
