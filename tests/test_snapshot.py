"""Remote public snapshot transport."""

import json

import httpx
import pytest

from aperture.snapshot import publish_remote


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
