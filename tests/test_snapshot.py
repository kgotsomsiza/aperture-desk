"""Remote public snapshot transport."""

import json

import httpx
import pytest

from aperture.snapshot import publish_remote


def test_equity_curve_removes_large_external_cash_adjustments():
    from aperture.snapshot import _cash_flow_adjusted_curve

    curve = _cash_flow_adjusted_curve(
        [1_787_923_800, 1_788_183_000, 1_788_186_600, 1_788_446_200],
        [100_000.0, 200_000.0, 200_160.0, 197_630.0],
        base_value=100_000.0,
    )

    assert [point["equity"] for point in curve] == [
        100_000.0,
        100_000.0,
        100_160.0,
        97_630.0,
    ]


def test_public_day_return_prefers_broker_prior_close(tmp_path):
    from aperture.snapshot import Snapshot
    from aperture.state import DeskState
    from aperture.warden import AuditLog

    class Broker:
        def account(self):
            return {"equity": "97_000", "last_equity": "99_000"}

        def positions(self):
            return []

        def portfolio_history(self, period="1W", timeframe="1H"):
            return {"timestamp": [], "equity": [], "base_value": 100_000}

    state = DeskState(path=tmp_path / "desk.json")
    state.start_equity = 100_000.0
    state.day_start_equity = 95_000.0  # stale first-observation value
    state.high_water_mark = 100_000.0
    payload = Snapshot(
        state=state,
        audit=AuditLog(path=tmp_path / "audit.jsonl"),
        cli=Broker(),
    ).build()

    assert payload["day_pnl_pct"] == pytest.approx(-2.02, abs=0.001)


def public_payload():
    return {
        "schema_version": 1,
        "mode": "practice",
        "generated_at": "2026-08-29T00:00:00+00:00",
        "equity": 100_000.0,
    }


def test_remote_publish_is_optional_when_completely_unconfigured(monkeypatch):
    monkeypatch.delenv("APERTURE_SNAPSHOT_URL", raising=False)
    monkeypatch.delenv("APERTURE_PUBLISH_TOKEN", raising=False)
    assert publish_remote(public_payload()) is False


def test_remote_publish_sends_bearer_authenticated_json():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["authorization"] = request.headers["Authorization"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(204)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert publish_remote(
            public_payload(),
            endpoint="https://dashboard.example/api/snapshot",
            token="test-publisher-value",
            client=client,
        )

    assert seen == {
        "method": "POST",
        "authorization": "Bearer test-publisher-value",
        "payload": public_payload(),
    }


def test_remote_publish_rejects_partial_or_insecure_configuration(monkeypatch):
    monkeypatch.setenv("APERTURE_SNAPSHOT_URL", "https://dashboard.example/api/snapshot")
    monkeypatch.delenv("APERTURE_PUBLISH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="requires both"):
        publish_remote(public_payload())

    with pytest.raises(ValueError, match="must use HTTPS"):
        publish_remote(
            public_payload(),
            endpoint="http://dashboard.example/api/snapshot",
            token="test-publisher-value",
        )


def test_remote_publish_allows_plain_http_only_on_loopback():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert publish_remote(
            public_payload(),
            endpoint="http://127.0.0.1:8787/api/snapshot",
            token="test-publisher-value",
            client=client,
        )


def test_remote_http_failure_is_visible_to_the_runner_boundary():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "publisher_unavailable"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            publish_remote(
                public_payload(),
                endpoint="https://dashboard.example/api/snapshot",
                token="test-publisher-value",
                client=client,
            )


# --------------------------------------------------------------------------- #
# The public tape must carry the argument, not just the verdict
# --------------------------------------------------------------------------- #


def test_agent_reasoning_survives_the_public_projection():
    """A judge opening the demo URL should see *why* the desk did things. The
    projection used to keep the event name and drop every field that carried
    the reasoning, so a red team kill arrived with no objection attached."""
    from aperture.snapshot import AUDIT_FIELDS

    for field in ("symbols", "reasons", "posture", "objection", "severity", "confidence"):
        assert field in AUDIT_FIELDS, f"{field} is stripped before it reaches the public"


def test_identity_is_still_never_publishable():
    """Widening the tape must not widen what can leak."""
    from aperture.snapshot import AUDIT_FIELDS, FORBIDDEN_KEYS

    assert not set(AUDIT_FIELDS) & set(FORBIDDEN_KEYS)
    for field in ("account_number", "account_id", "api_key", "secret"):
        assert field not in AUDIT_FIELDS


def test_every_designed_sleeve_appears_on_the_public_roster():
    """CONVEX held 30% of capital and an open position while being invisible on
    the public page, because the roster was a hardcoded three-name tuple."""
    from aperture.loop import PRIOR_WEIGHTS
    from aperture.snapshot import _designed_strategies

    assert set(_designed_strategies()) == set(PRIOR_WEIGHTS)
    assert "CONVEX" in _designed_strategies()
