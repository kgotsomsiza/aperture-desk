"""Closed-market orchestration: research, hiring, and Featherless explanation."""

from datetime import date, datetime, timezone
from types import SimpleNamespace

from aperture.backtest import BacktestResult, CondorSpec, SimulatedTrade
from aperture.research import Candidate, LabReport
from aperture.runner import Runner
from aperture.state import DeskState
from aperture.warden import AuditLog, RiskWarden


class AccountCLI:
    def account(self):
        return {"equity": "100000.00"}


class RecordingFeatherless:
    label = "featherless"
    fast_model = "Qwen/test-fast"
    reasoning_model = "Kimi/test-reasoning"
    json_mode = "object"

    def __init__(self):
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "The desk finished unchanged. No risk was obscured, and every capital "
            "decision remained subject to the deterministic Warden."
        )


def evidence() -> BacktestResult:
    result = BacktestResult(strategy_id="CAND-WIDTH/holdout")
    result.trades = [
        SimulatedTrade(
            entry=date(2025, 1, 2),
            exit=date(2025, 1, 10),
            legs=("a", "b", "c", "d"),
            qty=1,
            entry_price=-1.0,
            exit_price=-0.4,
            max_loss=400.0,
            reason="test",
        )
    ]
    return result


def test_after_close_hires_persists_and_uses_featherless_once(monkeypatch, tmp_path):
    import aperture.runner as runner_module

    provider = RecordingFeatherless()
    runner = Runner.__new__(Runner)
    runner.cli = AccountCLI()
    runner.provider = provider
    runner.audit = AuditLog(path=tmp_path / "audit.jsonl")

    state = DeskState(path=tmp_path / "desk.json")
    state.start_equity = 100_000.0
    state.day_start_equity = 100_000.0
    state.day_stamp = "2026-08-28"
    state.research_trials = 5
    warden = RiskWarden(audit=runner.audit)

    candidate = Candidate(
        "CAND-WIDTH",
        CondorSpec(width_pct=0.015),
        "CARRY",
        "wider wings",
        "reduce tail sensitivity",
    )
    report = LabReport(
        tested=2,
        promoted=[(candidate, evidence(), "selection and holdout survived")],
        training_window=("2024-03-01", "2025-12-01"),
        validation_window=("2026-01-05", "2026-07-29"),
        multiple_testing_trials=7,
    )
    lab_calls = []

    def fake_lab(*args, **kwargs):
        lab_calls.append((args, kwargs))
        return report

    monkeypatch.setattr(runner_module, "run_lab", fake_lab)
    monkeypatch.setenv("APERTURE_RESEARCH_UNDERLYING", "SPY")

    runner.after_close(state, warden)
    runner.after_close(state, warden)

    assert len(lab_calls) == 1
    assert lab_calls[0][1]["provider"] is provider
    assert lab_calls[0][1]["prior_trials"] == 5
    assert len(provider.calls) == 1
    assert provider.calls[0]["tier"] == "reasoning"

    persisted = DeskState.load(state.path)
    assert persisted.research_trials == 7
    assert persisted.last_research_date == "2026-08-28"
    assert persisted.last_letter_date == "2026-08-28"
    assert persisted.hired_strategies[0]["status"] == "probation"
    assert persisted.hired_strategies[0]["hypothesis"] == "reduce tail sensitivity"
    assert persisted.latest_letter["reasoning"]["vendor"] == "featherless"
    assert persisted.research_history[-1]["reasoning"]["fast_model"] == "Qwen/test-fast"

    events = [row["event"] for row in runner.audit.tail(limit=20)]
    assert "research_complete" in events
    assert "hired" in events
    assert "letter_written" in events


def test_remote_dashboard_failure_cannot_undo_local_snapshot(monkeypatch, tmp_path, caplog):
    import aperture.runner as runner_module

    payload = {"schema_version": 1, "mode": "practice", "equity": 100_000.0}

    class BuiltSnapshot:
        def __init__(self, **kwargs):
            pass

        def build(self):
            return payload

    def remote_failure(_payload):
        raise RuntimeError("dashboard unavailable")

    runner = Runner.__new__(Runner)
    runner.state_path = tmp_path / "desk.json"
    runner.audit = AuditLog(path=tmp_path / "audit.jsonl")
    runner.cli = object()
    runner.args = SimpleNamespace(public=str(tmp_path / "snapshot.json"))

    monkeypatch.setattr(runner_module, "Snapshot", BuiltSnapshot)
    monkeypatch.setattr(runner_module, "publish_remote", remote_failure)

    runner.publish(DeskState(path=runner.state_path))

    assert (tmp_path / "snapshot.json").exists()
    assert "remote snapshot publish failed" in caplog.text


# --------------------------------------------------------------------------- #
# Build provenance
# --------------------------------------------------------------------------- #


def test_a_build_identifies_itself():
    from aperture.version import desk_version

    v = desk_version()
    assert isinstance(v, str) and v


def test_a_stamped_image_reports_what_it_was_built_from(monkeypatch):
    from aperture import version

    version.desk_version.cache_clear()
    monkeypatch.setenv("APERTURE_VERSION", "abc1234")
    assert version.desk_version() == "abc1234"
    version.desk_version.cache_clear()


def test_an_unsubstituted_build_arg_is_not_mistaken_for_a_version(monkeypatch):
    """A Dockerfile default that never got substituted must not be published as
    though it were a real commit."""
    from aperture import version

    version.desk_version.cache_clear()
    monkeypatch.setenv("APERTURE_VERSION", "$GIT_SHA")
    assert version.desk_version() != "$GIT_SHA"
    version.desk_version.cache_clear()


def test_provenance_never_blocks_trading(monkeypatch):
    """Evidence about trading is not a precondition for it."""
    from aperture import version

    version.desk_version.cache_clear()
    monkeypatch.delenv("APERTURE_VERSION", raising=False)
    monkeypatch.setattr(version.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    assert version.desk_version() == version.UNKNOWN
    version.desk_version.cache_clear()
