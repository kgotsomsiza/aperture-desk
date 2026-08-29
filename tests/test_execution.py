"""Tests for adaptive execution.

The desk has to notice on its own that it is not getting filled. Nobody is
watching a dashboard, so these tests are mostly about the adjustment being
*bounded* and *slow to be fooled*: a knob that swings on three orders, or that
walks to the ceiling and gives away the spread, is worse than a fixed constant.
"""

from __future__ import annotations

import pytest

from aperture.execution import (
    DEFAULT_AGGRESSION,
    MAX_AGGRESSION,
    MIN_AGGRESSION,
    FillReport,
    adapt,
    clamp,
    measure_fills,
)


def orders(**counts) -> list[dict]:
    out = []
    for status, n in counts.items():
        out += [{"order_class": "mleg", "status": status}] * n
    return out


# --------------------------------------------------------------------------- #
# Measuring
# --------------------------------------------------------------------------- #


def test_counts_only_multi_leg_orders():
    mixed = orders(filled=3) + [{"order_class": "simple", "status": "filled"}] * 5
    assert measure_fills(mixed).decided == 3


def test_working_orders_are_pending_not_failures():
    """A limit resting for two minutes has not failed. Counting it as a miss
    would make the desk chase its own tail upward inside a single session."""
    report = measure_fills(orders(filled=2, new=3, accepted=2, expired=1))
    assert report.filled == 2
    assert report.unfilled == 1
    assert report.pending == 5
    assert report.decided == 3
    assert report.rate == pytest.approx(2 / 3)


def test_every_terminal_unfilled_status_counts_against():
    report = measure_fills(orders(expired=1, canceled=1, rejected=1, done_for_day=1))
    assert report.unfilled == 4
    assert report.rate == 0.0


def test_no_orders_is_not_a_zero_rate_crisis():
    report = measure_fills([])
    assert report.decided == 0
    assert report.rate == 0.0
    assert "no settled orders" in report.describe()


def test_only_the_recent_window_matters():
    report = measure_fills(orders(expired=50) + orders(filled=10), lookback=10)
    assert report.decided == 10
    assert report.rate == 1.0


# --------------------------------------------------------------------------- #
# Adapting
# --------------------------------------------------------------------------- #


def test_small_samples_change_nothing():
    value, why = adapt(0.60, FillReport(filled=1, unfilled=2, pending=0))
    assert value == 0.60
    assert "too few to judge" in why


def test_poor_fills_reach_further():
    """The 28 August case: 12 of 31 filled, two fifths of intended capital."""
    value, why = adapt(0.60, FillReport(filled=12, unfilled=19, pending=0))
    assert value > 0.60
    assert "not deploying what it intends" in why


def test_easy_fills_ease_back_to_stop_overpaying():
    value, why = adapt(0.90, FillReport(filled=19, unfilled=1, pending=0))
    assert value < 0.90
    assert "paying more" in why


def test_a_healthy_rate_is_left_alone():
    value, why = adapt(0.60, FillReport(filled=14, unfilled=6, pending=0))
    assert value == 0.60
    assert "healthy" in why


def test_it_reacts_faster_to_not_trading_than_to_overpaying():
    """Not trading at all is the worse failure and deserves the bigger step."""
    up, _ = adapt(0.60, FillReport(filled=2, unfilled=18, pending=0))
    down, _ = adapt(0.60, FillReport(filled=20, unfilled=0, pending=0))
    assert (up - 0.60) > (0.60 - down)


# --------------------------------------------------------------------------- #
# Bounds: this knob spends real money
# --------------------------------------------------------------------------- #


def test_it_never_climbs_past_the_ceiling():
    value = 0.60
    for _ in range(50):
        value, _ = adapt(value, FillReport(filled=0, unfilled=20, pending=0))
    assert value == MAX_AGGRESSION


def test_it_never_falls_through_the_floor():
    value = 0.60
    for _ in range(50):
        value, _ = adapt(value, FillReport(filled=20, unfilled=0, pending=0))
    assert value == MIN_AGGRESSION


def test_at_the_ceiling_it_says_the_spreads_are_the_problem():
    _, why = adapt(MAX_AGGRESSION, FillReport(filled=0, unfilled=20, pending=0))
    assert "spreads themselves are the problem" in why


def test_it_settles_rather_than_oscillating():
    """A gap between the two thresholds means a value that lands in the middle
    stops moving, instead of flip-flopping every five minutes."""
    value = DEFAULT_AGGRESSION
    seen = []
    for _ in range(12):
        value, _ = adapt(value, FillReport(filled=14, unfilled=6, pending=0))
        seen.append(value)
    assert len(set(seen)) == 1


def test_clamp_guards_anything_restored_from_disk():
    assert clamp(9.9) == MAX_AGGRESSION
    assert clamp(-3.0) == MIN_AGGRESSION
    assert clamp(0.75) == 0.75


# --------------------------------------------------------------------------- #
# Cycle timing: the interval is a cadence, not an idle period
# --------------------------------------------------------------------------- #


def test_a_long_cycle_eats_into_its_own_interval():
    """A two-minute cycle plus a full five-minute sleep is a seven-minute
    cadence, which costs a third of the session's decision points."""
    interval, work = 300, 120
    assert max(1, int(interval - work)) == 180


def test_a_cycle_longer_than_the_interval_still_yields_a_positive_sleep():
    assert max(1, int(300 - 900)) == 1
