"""Tests for the capital allocator.

The allocator is the mechanic the desk is named for, so these tests are mostly
about it *not* over-reacting: four sessions is not enough evidence to justify
large reallocations, and an allocator that swings on three lucky trades is
reading noise.
"""

from __future__ import annotations

import pytest

from aperture.allocator import (
    FIRED,
    FUNDED,
    PROBATION,
    AllocationLimits,
    Allocator,
    StrategyRecord,
    observe,
    summarise,
)
from aperture.state import DeskState, OpenTrade
from aperture.warden import AuditLog

PRIORS = {"CARRY": 0.60, "CRUSH": 0.20, "DRIFT": 0.20}


def rec(strategy_id: str, prior: float, **kw) -> StrategyRecord:
    return StrategyRecord(strategy_id=strategy_id, prior_weight=prior, **kw)


def by_id(allocations):
    return {a.strategy_id: a for a in allocations}


# --------------------------------------------------------------------------- #
# Edge and rates
# --------------------------------------------------------------------------- #


def test_edge_is_return_on_risk_not_on_capital():
    # $200 made off $1,000 risked beats $400 off $10,000.
    small = rec("A", 0.5, closed=3, realized_pnl=200.0, risk_deployed=1_000.0)
    large = rec("B", 0.5, closed=3, realized_pnl=400.0, risk_deployed=10_000.0)
    assert small.edge == pytest.approx(0.20)
    assert large.edge == pytest.approx(0.04)
    assert small.edge > large.edge


def test_edge_is_zero_before_any_risk_is_deployed():
    assert rec("A", 0.5).edge == 0.0


def test_veto_rate_handles_no_proposals():
    assert rec("A", 0.5).veto_rate == 0.0
    assert rec("A", 0.5, proposals=10, vetoes=4).veto_rate == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Shrinkage: the allocator must not believe small samples
# --------------------------------------------------------------------------- #


def test_no_history_leaves_the_designed_weights_alone():
    allocations = Allocator().allocate(
        [rec(k, v) for k, v in PRIORS.items()], equity=100_000.0
    )
    got = by_id(allocations)
    assert got["CARRY"].weight == pytest.approx(0.60, abs=0.01)
    assert got["CRUSH"].weight == pytest.approx(0.20, abs=0.01)
    # Probation means "hired during the run", not "no trades yet". The designed
    # strategies are the baseline, not candidates.
    assert all(a.status == FUNDED for a in allocations)


def test_one_lucky_trade_barely_moves_the_allocation():
    """Three wins is not an edge. An allocator that reallocates hard here is
    reading noise, and in a four-session window that is most of what exists."""
    records = [
        rec("CARRY", 0.60),
        rec("CRUSH", 0.20, closed=1, wins=1, realized_pnl=500.0, risk_deployed=1_000.0),
        rec("DRIFT", 0.20),
    ]
    got = by_id(Allocator().allocate(records, equity=100_000.0))
    # A +50% edge on a single trade should shift CRUSH by a few points, not double it.
    assert 0.20 < got["CRUSH"].weight < 0.30


def test_a_sustained_edge_does_move_capital():
    records = [
        rec("CARRY", 0.60),
        rec("CRUSH", 0.20, closed=12, wins=9, realized_pnl=3_000.0, risk_deployed=10_000.0),
        rec("DRIFT", 0.20),
    ]
    got = by_id(Allocator().allocate(records, equity=100_000.0))
    assert got["CRUSH"].weight > 0.30
    assert got["CRUSH"].status == FUNDED


def test_confidence_grows_with_evidence():
    """The same edge should move the allocation further when better observed."""
    def weight_after(n_closed: int) -> float:
        records = [
            rec("CARRY", 0.60),
            rec("CRUSH", 0.20, closed=n_closed, realized_pnl=0.2 * n_closed * 1_000,
                risk_deployed=n_closed * 1_000.0),
            rec("DRIFT", 0.20),
        ]
        return by_id(Allocator().allocate(records, equity=100_000.0))["CRUSH"].weight

    assert weight_after(2) < weight_after(6) < weight_after(20)


# --------------------------------------------------------------------------- #
# Firing
# --------------------------------------------------------------------------- #


def test_a_persistent_loser_is_fired():
    records = [
        rec("CARRY", 0.60),
        rec("CRUSH", 0.20, closed=6, wins=1, realized_pnl=-4_000.0, risk_deployed=10_000.0),
        rec("DRIFT", 0.20),
    ]
    got = by_id(Allocator().allocate(records, equity=100_000.0))
    assert got["CRUSH"].status == FIRED
    assert got["CRUSH"].budget == 0.0
    assert "lost 40%" in got["CRUSH"].reason


def test_one_bad_trade_is_not_a_firing():
    records = [
        rec("CARRY", 0.60),
        rec("CRUSH", 0.20, closed=1, realized_pnl=-900.0, risk_deployed=1_000.0),
        rec("DRIFT", 0.20),
    ]
    assert by_id(Allocator().allocate(records, equity=100_000.0))["CRUSH"].status != FIRED


def test_a_strategy_that_is_always_vetoed_is_fired_even_without_losses():
    """Being refused permission is a different failure from losing money, and a
    faster one to detect. A strategy whose proposals are always rejected is
    miscalibrated: it keeps asking for trades it is not allowed to make."""
    records = [
        rec("CARRY", 0.60),
        rec("CRUSH", 0.20, proposals=20, vetoes=19),
        rec("DRIFT", 0.20),
    ]
    got = by_id(Allocator().allocate(records, equity=100_000.0))
    assert got["CRUSH"].status == FIRED
    assert "miscalibrated" in got["CRUSH"].reason
    assert got["CRUSH"].budget == 0.0


def test_a_high_veto_rate_on_a_small_sample_is_not_a_firing():
    records = [
        rec("CARRY", 0.60),
        rec("CRUSH", 0.20, proposals=3, vetoes=3),
        rec("DRIFT", 0.20),
    ]
    assert by_id(Allocator().allocate(records, equity=100_000.0))["CRUSH"].status != FIRED


def test_vetoes_penalise_weight_before_they_trigger_a_firing():
    clean = [rec("CARRY", 0.5), rec("CRUSH", 0.5, proposals=10, vetoes=0)]
    noisy = [rec("CARRY", 0.5), rec("CRUSH", 0.5, proposals=10, vetoes=5)]
    assert (
        by_id(Allocator().allocate(noisy, 100_000.0))["CRUSH"].weight
        < by_id(Allocator().allocate(clean, 100_000.0))["CRUSH"].weight
    )


def test_firing_frees_capital_for_the_survivors():
    records = [
        rec("CARRY", 0.40, closed=6, realized_pnl=1_000.0, risk_deployed=8_000.0),
        rec("CRUSH", 0.40, closed=6, realized_pnl=-5_000.0, risk_deployed=10_000.0),
        rec("DRIFT", 0.20, closed=6, realized_pnl=500.0, risk_deployed=6_000.0),
    ]
    got = by_id(Allocator().allocate(records, equity=100_000.0))
    assert got["CRUSH"].status == FIRED
    survivors = [a for a in got.values() if a.is_active]
    assert sum(a.weight for a in survivors) == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_weights_always_sum_to_one_and_budgets_to_the_risk_cap():
    records = [rec(k, v, closed=5, realized_pnl=300.0, risk_deployed=4_000.0)
               for k, v in PRIORS.items()]
    allocations = Allocator().allocate(records, equity=100_000.0)
    assert sum(a.weight for a in allocations) == pytest.approx(1.0, abs=0.01)
    expected = 100_000.0 * AllocationLimits().total_risk_budget_pct
    assert sum(a.budget for a in allocations) == pytest.approx(expected, rel=0.02)


def test_no_strategy_can_own_the_whole_book():
    records = [
        rec("CARRY", 0.60, closed=40, realized_pnl=40_000.0, risk_deployed=40_000.0),
        rec("CRUSH", 0.20),
        rec("DRIFT", 0.20),
    ]
    got = by_id(Allocator().allocate(records, equity=100_000.0))
    assert got["CARRY"].weight <= AllocationLimits().max_weight + 0.01


def test_a_newly_hired_strategy_starts_on_probation():
    records = [
        rec("CARRY", 0.60, closed=10, realized_pnl=1_000.0, risk_deployed=8_000.0),
        rec("CRUSH", 0.20, closed=10, realized_pnl=800.0, risk_deployed=8_000.0),
        rec("S5-INVENTED", 0.0),  # no prior: hired during the run
    ]
    got = by_id(Allocator().allocate(records, equity=100_000.0))
    assert got["S5-INVENTED"].status == PROBATION
    assert got["S5-INVENTED"].weight <= AllocationLimits().probation_weight + 0.01
    assert got["S5-INVENTED"].budget > 0  # funded, but barely


def test_everything_fired_does_not_divide_by_zero():
    records = [
        rec(k, v, closed=8, realized_pnl=-8_000.0, risk_deployed=10_000.0)
        for k, v in PRIORS.items()
    ]
    allocations = Allocator().allocate(records, equity=100_000.0)
    assert all(a.status == FIRED for a in allocations)
    assert all(a.budget == 0.0 for a in allocations)


# --------------------------------------------------------------------------- #
# Observation from the real ledger and audit log
# --------------------------------------------------------------------------- #


def test_observe_reads_both_the_ledger_and_the_veto_log(tmp_path):
    state = DeskState(path=tmp_path / "d.json")
    state.record_open(OpenTrade(
        client_order_id="open1", strategy_id="CARRY", underlying="SPY",
        legs=["SPY260918P00630000"], qty=1, net_price=-1.0, max_loss=400.0,
        opened_at="2026-09-01T15:00:00Z", status="open",
    ))
    state.closed.append({
        "client_order_id": "c1", "strategy_id": "CRUSH", "underlying": "LULU",
        "max_loss": 1_000.0, "pnl": 250.0,
    })
    state.closed.append({
        "client_order_id": "c2", "strategy_id": "CRUSH", "underlying": "PANW",
        "max_loss": 1_000.0, "pnl": -100.0,
    })

    audit = AuditLog(path=tmp_path / "audit.jsonl")
    audit.record("approval", strategy="CARRY")
    audit.record("veto", strategy="DRIFT")
    audit.record("veto", strategy="DRIFT")
    audit.record("closed", strategy="CRUSH")  # not a proposal; must not count

    records = {r.strategy_id: r for r in observe(state, audit, PRIORS)}

    assert records["CARRY"].open_positions == 1
    assert records["CARRY"].proposals == 1 and records["CARRY"].vetoes == 0
    assert records["CRUSH"].closed == 2
    assert records["CRUSH"].wins == 1
    assert records["CRUSH"].realized_pnl == pytest.approx(150.0)
    assert records["CRUSH"].edge == pytest.approx(150.0 / 2_000.0)
    assert records["DRIFT"].proposals == 2 and records["DRIFT"].vetoes == 2
    assert records["DRIFT"].veto_rate == 1.0


def test_observe_picks_up_a_strategy_that_has_no_prior(tmp_path):
    state = DeskState(path=tmp_path / "d.json")
    state.closed.append({"strategy_id": "S5-INVENTED", "max_loss": 500.0, "pnl": 90.0})
    audit = AuditLog(path=tmp_path / "audit.jsonl")

    records = {r.strategy_id: r for r in observe(state, audit, PRIORS)}
    assert "S5-INVENTED" in records
    assert records["S5-INVENTED"].prior_weight == 0.0


def test_summary_is_readable(tmp_path):
    allocations = Allocator().allocate(
        [rec("CARRY", 0.6, closed=8, realized_pnl=900.0, risk_deployed=6_000.0),
         rec("CRUSH", 0.2, closed=6, realized_pnl=-4_000.0, risk_deployed=10_000.0),
         rec("DRIFT", 0.2)],
        equity=100_000.0,
    )
    text = summarise(allocations)
    assert "CARRY" in text and "CRUSH" in text
    assert "x CRUSH" in text  # fired strategies are marked


def test_caps_hold_even_when_every_strategy_is_on_a_bound():
    """The failure the first solver had: normalising a fully-clamped set silently
    violated the caps it had just applied."""
    records = [
        rec("CARRY", 0.60, closed=40, realized_pnl=40_000.0, risk_deployed=40_000.0),
        rec("CRUSH", 0.20, closed=40, realized_pnl=-8_000.0, risk_deployed=40_000.0),
        rec("DRIFT", 0.20, closed=40, realized_pnl=-8_000.0, risk_deployed=40_000.0),
    ]
    got = by_id(Allocator().allocate(records, equity=100_000.0))
    limits = AllocationLimits()
    active = [a for a in got.values() if a.is_active]
    for a in active:
        assert a.weight <= limits.max_weight + 1e-6, f"{a.strategy_id} broke the cap"
        assert a.weight >= limits.min_weight - 1e-6, f"{a.strategy_id} broke the floor"
    assert sum(a.weight for a in active) == pytest.approx(1.0, abs=1e-6)


def test_designed_barbell_survives_untouched_with_no_history():
    """CARRY is meant to be the ballast majority; the cap must not quietly
    rebalance the designed book before a single trade has closed."""
    priors = {"CARRY": 0.643, "CRUSH": 0.179, "DRIFT": 0.178}
    got = by_id(Allocator().allocate([rec(k, v) for k, v in priors.items()], 100_000.0))
    for k, v in priors.items():
        assert got[k].weight == pytest.approx(v, abs=0.01)
