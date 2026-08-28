"""The language model may narrate facts, never manufacture them."""

from datetime import date

from aperture.letter import LetterFacts, gather, plain, write
from aperture.state import DeskState, OpenTrade
from aperture.warden import AuditLog


class Provider:
    def __init__(self, answer):
        self.answer = answer

    def complete(self, **kwargs):
        return self.answer


def facts():
    return LetterFacts(
        as_of="2026-08-28",
        equity=97_500.00,
        start_equity=100_000.00,
        day_start_equity=100_000.00,
        open_positions=3,
    )


def test_letter_accepts_only_numbers_present_in_the_fact_block():
    answer = (
        "The desk lost 2.50% on the day and ended with equity of $97,500.00. "
        "No position was hidden and risk remained bounded throughout the session. "
        "The open book contains 3 structures selected by deterministic gates."
    )
    rendered = write(Provider(answer), facts())
    assert answer in rendered


def test_letter_falls_back_if_the_model_invents_or_rounds_a_number():
    answer = (
        "The desk lost 2.50% and claims an unsupported 42% forecast for tomorrow. "
        "That forecast is not part of the deterministic fact block and must never publish."
    )
    rendered = write(Provider(answer), facts())
    assert rendered == plain(facts())
    assert "42%" not in rendered


def test_gather_uses_fills_and_session_tagged_hires_not_submissions(tmp_path):
    state = DeskState(path=tmp_path / "state.json")
    state.start_equity = 100_000.0
    state.day_start_equity = 100_000.0
    for client_id, status in (("filled", "open"), ("waiting", "pending_entry")):
        state.record_open(OpenTrade(
            client_order_id=client_id,
            strategy_id="CARRY",
            underlying="SPY",
            legs=["SPY260918P00630000"],
            qty=1,
            net_price=-1.0,
            max_loss=400.0,
            opened_at="2026-08-28T15:00:00Z",
            status=status,
        ))
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    audit.record(
        "entry_submitted", session="2026-08-28", strategy="CARRY", underlying="SPY",
        qty=1, rationale="not filled",
    )
    audit.record(
        "entry_filled", session="2026-08-28", strategy="CARRY", underlying="SPY",
        qty=1, rationale="filled",
    )
    audit.record(
        "hired", session="2026-08-28", strategy="LAB-1234", underlying="SPY",
        mutation="wider wings", summary="holdout survived",
    )

    collected = gather(state, audit, 99_000.0, today=date(2026, 8, 28))
    assert [row["rationale"] for row in collected.opened_today] == ["filled"]
    assert collected.hires_today[0]["strategy"] == "LAB-1234"
    assert collected.open_positions == 1
