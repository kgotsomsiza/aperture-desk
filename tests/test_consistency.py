"""The propagation checks, run as part of the ordinary suite.

These are not tests of behaviour. They are tests that the project still agrees
with itself -- that a thing added in one place was carried through to the other
places it obliges. They live in the suite rather than in a pre-commit hook or a
checklist because the suite is the one thing everyone already runs.

The failure they exist to prevent actually happened: the agent layer shipped,
its decisions were audited, and the public projection dropped every field
carrying the reasoning while the dashboard had no renderer for the new events.
Nothing failed. The desk traded correctly. The demo page just quietly said less
than it should have, for days.
"""

from __future__ import annotations

import pytest

from aperture import consistency


def _report(findings) -> str:
    return "\n\n".join(str(f) for f in findings)


# --------------------------------------------------------------------------- #
# The checks themselves
# --------------------------------------------------------------------------- #


def test_every_audit_event_is_declared():
    findings = consistency.check_events_are_registered()
    assert not findings, "\n\n" + _report(findings)


def test_public_reasoning_survives_redaction():
    findings = consistency.check_public_events_survive_redaction()
    assert not findings, "\n\n" + _report(findings)


def test_public_events_have_a_renderer():
    findings = consistency.check_public_events_render()
    assert not findings, "\n\n" + _report(findings)


def test_renderers_read_fields_that_exist():
    findings = consistency.check_renderers_read_real_fields()
    assert not findings, "\n\n" + _report(findings)


def test_every_module_is_in_the_repository_map():
    findings = consistency.check_modules_are_documented()
    assert not findings, "\n\n" + _report(findings)


def test_the_agents_are_described_wherever_the_system_is():
    findings = consistency.check_agents_are_documented()
    assert not findings, "\n\n" + _report(findings)


def test_runtime_dependencies_are_in_the_image():
    findings = consistency.check_declared_dependencies_are_installed()
    assert not findings, "\n\n" + _report(findings)


def test_documented_test_counts_match_the_suite(request):
    """Read the real count from pytest itself rather than hard-coding it."""
    total = request.session.testscollected or len(request.session.items)
    if total < consistency.test_function_count():
        pytest.skip("partial run; the documented count is checked on the full suite")
    findings = consistency.check_documented_test_count(total)
    assert not findings, "\n\n" + _report(findings)


# --------------------------------------------------------------------------- #
# The checks have to actually catch things
# --------------------------------------------------------------------------- #


def test_an_undeclared_event_is_caught(monkeypatch):
    monkeypatch.setattr(consistency, "emitted_events",
                        lambda: {"brand_new_decision": {"reason"}})
    findings = consistency.check_events_are_registered()
    assert findings and "brand_new_decision" in findings[0].detail


def test_a_public_event_with_no_renderer_is_caught(monkeypatch):
    """This is the exact bug: audited, published, invisible."""
    monkeypatch.setattr(consistency, "emitted_events",
                        lambda: {"universe": {"symbols", "reasons"}})
    monkeypatch.setattr(consistency, "renderer_field_uses", dict)
    findings = consistency.check_public_events_render()
    assert findings
    assert "Decision recorded" in findings[0].detail


def test_reasoning_stripped_by_the_snapshot_is_caught(monkeypatch):
    monkeypatch.setattr(consistency, "emitted_events",
                        lambda: {"red_team_kill": {"objection"}})
    monkeypatch.setattr(consistency, "renderer_field_uses",
                        lambda: {"red_team_kill": {"objection"}})
    monkeypatch.setattr(consistency, "public_fields", lambda: {"ts", "event"})
    findings = consistency.check_public_events_survive_redaction()
    assert findings and "objection" in findings[0].detail


def test_a_renderer_reading_a_field_nobody_records_is_caught(monkeypatch):
    monkeypatch.setattr(consistency, "emitted_events",
                        lambda: {"regime": {"posture"}})
    monkeypatch.setattr(consistency, "renderer_field_uses",
                        lambda: {"regime": {"posture", "invented_field"}})
    findings = consistency.check_renderers_read_real_fields()
    assert findings and "invented_field" in findings[0].detail


def test_an_internal_event_is_not_required_to_render(monkeypatch):
    """Bookkeeping events carry no argument; a generic line is honest."""
    monkeypatch.setattr(consistency, "emitted_events",
                        lambda: {"entry_filled": {"client_order_id", "qty"}})
    monkeypatch.setattr(consistency, "renderer_field_uses", dict)
    assert not consistency.check_public_events_render()


def test_checks_that_need_missing_files_skip_rather_than_fail(monkeypatch, tmp_path):
    """A clone of the public repo has no submission/ or _private/. The suite
    must pass there, or it stops being run at all."""
    monkeypatch.setattr(consistency, "WRITEUP", tmp_path / "absent.md")
    monkeypatch.setattr(consistency, "HANDOFF", tmp_path / "absent.md")
    consistency.check_agents_are_documented()
    consistency.check_documented_test_count(1)
