"""Tests for the agent decision layer.

These agents decide what the desk trades, so the tests are mostly about what
happens when an agent is wrong, confused, or unreachable. A trading desk that
stops because a model returned malformed JSON is not autonomous, and one that
does whatever a model says is not safe.
"""

from __future__ import annotations

import json

import pytest

from aperture.agents import (
    MAX_CONVICTION,
    MIN_CONVICTION,
    POSTURES,
    TRADEABLE_UNIVERSE,
    call_regime,
    choose_universe,
    rank_proposals,
    red_team,
)
from aperture.llm import NullProvider


class Canned:
    """A provider that returns whatever we hand it."""

    def __init__(self, payload):
        self.payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls = 0

    def complete(self, *, system, user, tier="fast", json_schema=None):
        self.calls += 1
        return self.payload


class Broken:
    def complete(self, *, system, user, tier="fast", json_schema=None):
        raise RuntimeError("provider is down")


MARKET = [
    {"symbol": "SPY", "spot": 770.0, "iv": 0.11, "realised_vol": 0.09, "iv_premium": 1.22},
    {"symbol": "NVDA", "spot": 220.0, "iv": 0.48, "realised_vol": 0.31, "iv_premium": 1.55},
    {"symbol": "TSLA", "spot": 340.0, "iv": 0.55, "realised_vol": 0.50, "iv_premium": 1.10},
]


# --------------------------------------------------------------------------- #
# Scout
# --------------------------------------------------------------------------- #


def test_scout_picks_what_the_agent_chose():
    provider = Canned({"picks": [
        {"symbol": "NVDA", "reason": "IV 1.55x realised"},
        {"symbol": "SPY", "reason": "liquid ballast"},
        {"symbol": "TSLA", "reason": "rich premium"},
    ]})
    choice = choose_universe(provider, MARKET)
    assert choice.symbols == ("NVDA", "SPY", "TSLA")
    assert choice.decided_by == "scout"
    assert "IV 1.55x realised" in choice.explain()


def test_scout_cannot_invent_a_ticker():
    """An agent free to name any symbol eventually names one with a
    two-dollar-wide market, and the desk spends the day being refused."""
    provider = Canned({"picks": [
        {"symbol": "SPY", "reason": "fine"},
        {"symbol": "SCAMCO", "reason": "hallucinated"},
        {"symbol": "NVDA", "reason": "fine"},
    ]})
    choice = choose_universe(provider, MARKET)
    assert "SCAMCO" not in choice.symbols
    assert set(choice.symbols) <= set(TRADEABLE_UNIVERSE)


def test_scout_falls_back_when_the_agent_says_too_little():
    choice = choose_universe(Canned({"picks": [{"symbol": "SPY", "reason": "x"}]}), MARKET)
    assert choice.symbols == ("SPY", "QQQ", "IWM")
    assert "default" in choice.decided_by


def test_scout_survives_a_dead_provider():
    choice = choose_universe(Broken(), MARKET)
    assert choice.symbols == ("SPY", "QQQ", "IWM")


def test_scout_respects_the_cap():
    provider = Canned({"picks": [{"symbol": s, "reason": "r"} for s in TRADEABLE_UNIVERSE]})
    assert len(choose_universe(provider, MARKET, max_names=4).symbols) == 4


def test_no_agent_means_the_designed_universe():
    choice = choose_universe(NullProvider(), MARKET)
    assert choice.symbols == ("SPY", "QQQ", "IWM")


# --------------------------------------------------------------------------- #
# Regime
# --------------------------------------------------------------------------- #


def test_regime_tilts_the_book_toward_premium():
    call = call_regime(
        Canned({"posture": "sell_premium", "confidence": 0.8, "reason": "IV rich"}),
        {"vix": 14},
    )
    assert call.posture == "sell_premium"
    assert call.ballast_tilt > 1.0
    assert call.convex_tilt < 1.0


def test_standing_down_is_available_and_shrinks_everything():
    call = call_regime(
        Canned({"posture": "stand_down", "confidence": 0.9, "reason": "hostile"}),
        {"vix": 42},
    )
    assert call.ballast_tilt < 0.5 and call.convex_tilt < 0.5


def test_a_tilt_can_lean_the_book_but_never_bet_it():
    """Bounded on purpose: an agent may change emphasis, not double the risk."""
    for posture in POSTURES:
        call = call_regime(
            Canned({"posture": posture, "confidence": 1.0, "reason": "r"}), {"vix": 20}
        )
        assert 0.3 <= call.ballast_tilt <= 1.5
        assert 0.3 <= call.convex_tilt <= 1.5


def test_an_unknown_posture_becomes_balanced():
    call = call_regime(
        Canned({"posture": "go_all_in", "confidence": 1.0, "reason": "nonsense"}), {"vix": 20}
    )
    assert call.posture == "balanced"
    assert call.ballast_tilt == 1.0


def test_a_nonsense_confidence_does_not_crash_the_cycle():
    call = call_regime(
        Canned({"posture": "balanced", "confidence": "very", "reason": "r"}), {"vix": 20}
    )
    assert 0.0 <= call.confidence <= 1.0


# --------------------------------------------------------------------------- #
# Red team
# --------------------------------------------------------------------------- #


def test_a_concrete_objection_kills_the_trade():
    verdict = red_team(
        Canned({"kill": True, "severity": 0.9,
                "objection": "earnings land inside the expiry; this is short gamma into an event"}),
        "sell NVDA iron condor", {"dte": 10},
    )
    assert verdict.killed
    assert "earnings" in verdict.objection


def test_vague_unease_does_not_kill_a_trade():
    """A red team that refuses everything is the same as no desk at all."""
    verdict = red_team(
        Canned({"kill": True, "severity": 0.2, "objection": "feels risky"}),
        "sell SPY iron condor", {"dte": 14},
    )
    assert not verdict.killed


def test_a_sound_trade_survives():
    verdict = red_team(
        Canned({"kill": False, "severity": 0.1, "objection": "no material objection"}),
        "sell SPY iron condor", {"dte": 14},
    )
    assert not verdict.killed


def test_red_team_fails_open_so_an_outage_cannot_halt_trading():
    """Its only power is subtraction. A broken agent must not become a veto --
    the Warden is what actually protects the account."""
    assert not red_team(Broken(), "sell SPY condor", {}).killed
    assert not red_team(Canned("not json at all"), "sell SPY condor", {}).killed
    assert not red_team(NullProvider(), "sell SPY condor", {}).killed


# --------------------------------------------------------------------------- #
# Portfolio manager
# --------------------------------------------------------------------------- #


def test_conviction_orders_and_sizes_the_book():
    ranked = rank_proposals(
        Canned({"ranked": [
            {"id": 2, "conviction": 1.0, "reason": "best risk/reward"},
            {"id": 0, "conviction": 0.5, "reason": "crowded"},
        ]}),
        ["a", "b", "c"],
    )
    assert ranked[0].index == 2 and ranked[0].conviction == 1.0
    assert ranked[1].index == 0 and ranked[1].conviction == 0.5


def test_unranked_proposals_still_trade_at_designed_size():
    ranked = rank_proposals(Canned({"ranked": [{"id": 1, "conviction": 0.9, "reason": "r"}]}),
                            ["a", "b", "c"])
    assert {c.index for c in ranked} == {0, 1, 2}
    assert next(c for c in ranked if c.index == 0).conviction == 1.0


def test_conviction_is_clamped_so_an_agent_cannot_upsize_itself():
    ranked = rank_proposals(
        Canned({"ranked": [{"id": 0, "conviction": 9.0, "reason": "very sure"},
                           {"id": 1, "conviction": -4.0, "reason": "negative"}]}),
        ["a", "b"],
    )
    by_index = {c.index: c.conviction for c in ranked}
    assert by_index[0] == MAX_CONVICTION
    assert by_index[1] == MIN_CONVICTION


def test_a_duplicated_id_is_taken_once():
    ranked = rank_proposals(
        Canned({"ranked": [{"id": 0, "conviction": 1.0, "reason": "a"},
                           {"id": 0, "conviction": 0.3, "reason": "again"}]}),
        ["a", "b"],
    )
    assert len([c for c in ranked if c.index == 0]) == 1


def test_a_dead_provider_leaves_every_proposal_at_designed_size():
    ranked = rank_proposals(Broken(), ["a", "b", "c"])
    assert len(ranked) == 3
    assert all(c.conviction == 1.0 for c in ranked)


# --------------------------------------------------------------------------- #
# The kill budget: bounding a miscalibrated agent
# --------------------------------------------------------------------------- #


def _verdicts(*severities):
    from aperture.agents import RedTeamVerdict
    return [RedTeamVerdict(True, f"objection {s}", s, "red team") for s in severities]


def test_an_agent_that_kills_everything_only_gets_half():
    """Observed live: the red team killed 100% of proposals, textbook condors
    included. Prompting helped; only a bound makes it safe."""
    from aperture.agents import apply_kill_budget

    standing = apply_kill_budget(_verdicts(0.9, 0.9, 0.9, 0.9))
    assert sum(standing) == 2


def test_the_budget_spends_itself_on_the_worst_objections_first():
    from aperture.agents import apply_kill_budget
    from aperture.agents import RedTeamVerdict

    verdicts = _verdicts(0.80, 0.99, 0.76, 0.95)
    standing = apply_kill_budget(verdicts)
    killed = {i for i, k in enumerate(standing) if k}
    assert killed == {1, 3}          # the two most severe


def test_a_single_proposal_cannot_be_killed_by_one_agent():
    """With one candidate, half of one rounds to zero -- so a lone proposal
    always reaches the Warden, which is the component that actually decides."""
    from aperture.agents import apply_kill_budget

    assert apply_kill_budget(_verdicts(1.0)) == [False]


def test_nothing_killed_stays_nothing_killed():
    from aperture.agents import RedTeamVerdict, apply_kill_budget

    calm = [RedTeamVerdict(False, "fine", 0.1, "red team")] * 4
    assert apply_kill_budget(calm) == [False] * 4


def test_a_kill_now_needs_real_conviction():
    from aperture.agents import KILL_SEVERITY

    below = red_team(Canned({"kill": True, "severity": KILL_SEVERITY - 0.05,
                             "objection": "unease"}), "sell SPY condor", {})
    at = red_team(Canned({"kill": True, "severity": KILL_SEVERITY,
                          "objection": "earnings inside the expiry"}), "sell SPY condor", {})
    assert not below.killed
    assert at.killed
