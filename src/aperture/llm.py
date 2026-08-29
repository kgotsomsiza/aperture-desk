"""The reasoning layer, behind one swappable interface.

Two constraints shaped this module.

**No model may place an order.** Everything here returns text or parsed JSON.
Proposals produced from model output are ordinary :class:`Proposal` objects and
go through the Risk Warden exactly like any other. There is deliberately no path
from this file to the broker.

**The provider must be swappable.** The hackathon's technology partners are
announced during the event and partner prizes require integrating partner tech,
so the vendor is one config value rather than an assumption baked through the
codebase. That bet paid: Featherless AI arrived as the technology partner after
this module was written, and supporting it cost one adapter and no changes
anywhere else.

Everything here speaks the OpenAI wire format, which Featherless implements, so
the two differ only in base URL, model names, and how much structured-output
support they actually have.

Two tiers, matched to the free daily allowances: ``fast`` for high-frequency
structured work (news triage, red-team passes, extraction) and ``reasoning`` for
the two jobs that actually need it (strategy invention, the shareholder letter).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

log = logging.getLogger(__name__)

Tier = Literal["fast", "reasoning"]


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    def complete(
        self, *, system: str, user: str, tier: Tier = "fast", json_schema: dict | None = None
    ) -> str: ...


@dataclass
class TokenBudget:
    """Keeps the desk inside the free daily allowance.

    Exceeding it is not an error the desk should discover from a bill, so the
    budget is enforced locally and the reasoning tier degrades to the fast tier
    before it degrades to spending money.
    """

    fast_daily: int = 2_000_000
    reasoning_daily: int = 200_000
    used: dict[str, int] = field(default_factory=lambda: {"fast": 0, "reasoning": 0})

    def remaining(self, tier: Tier) -> int:
        cap = self.fast_daily if tier == "fast" else self.reasoning_daily
        return max(0, cap - self.used.get(tier, 0))

    def charge(self, tier: Tier, tokens: int) -> None:
        self.used[tier] = self.used.get(tier, 0) + tokens

    def reset(self) -> None:
        self.used = {"fast": 0, "reasoning": 0}


TRANSIENT = ("busy", "rate limit", "rate_limit", "timeout", "timed out",
             "temporarily", "503", "502", "overloaded", "try again")


def _error_text(response: Any) -> str:
    """The error message a response is carrying, if any.

    Covers both shapes: an ``error`` attribute the SDK modelled, and the
    ``choices is None`` case that means the body was an error object.
    """
    problem = getattr(response, "error", None)
    if problem:
        if isinstance(problem, dict):
            return str(problem.get("message") or problem)
        return str(getattr(problem, "message", problem))
    if not getattr(response, "choices", None):
        return "response contained no choices"
    return ""


def _is_transient(message: str) -> bool:
    lowered = (message or "").lower()
    return any(marker in lowered for marker in TRANSIENT)


JSON_SCHEMA = "schema"   # OpenAI-style strict json_schema
JSON_OBJECT = "object"   # response_format={"type": "json_object"}
JSON_PROMPT = "prompt"   # ask for JSON in words and parse what comes back


@dataclass
class OpenAICompatibleProvider:
    """Any endpoint speaking the OpenAI chat-completions wire format."""

    api_key: str | None = None
    base_url: str | None = None
    fast_model: str = "gpt-5.4-mini"
    reasoning_model: str = "gpt-5.4"
    json_mode: str = JSON_SCHEMA
    budget: TokenBudget = field(default_factory=TokenBudget)
    # Measured medians are 4-8s. 90s was ten times the tail and, with the
    # reasoning tier's fallback, let one call cost three minutes.
    timeout: float = float(os.environ.get("APERTURE_LLM_TIMEOUT", "30"))
    key_env: str = "OPENAI_API_KEY"
    label: str = "openai"
    max_attempts: int = 3
    retry_backoff: float = 2.0
    _client: Any = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get(self.key_env)
        if not self.api_key:
            raise LLMError(f"{self.key_env} is not set")

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("the `openai` package is not installed") from exc
            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def model_for(self, tier: Tier) -> str:
        return self.reasoning_model if tier == "reasoning" else self.fast_model

    def complete(
        self, *, system: str, user: str, tier: Tier = "fast", json_schema: dict | None = None
    ) -> str:
        # Degrade rather than overspend: a depleted reasoning budget falls back to
        # the fast tier, which has an order of magnitude more headroom.
        if tier == "reasoning" and self.budget.remaining("reasoning") <= 0:
            log.warning("reasoning budget exhausted; falling back to the fast tier")
            tier = "fast"
        if self.budget.remaining(tier) <= 0:
            raise LLMError(f"daily {tier} token budget exhausted")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {"model": self.model_for(tier), "messages": messages}

        if json_schema is not None:
            if self.json_mode == JSON_SCHEMA:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": json_schema, "strict": True},
                }
            elif self.json_mode == JSON_OBJECT:
                kwargs["response_format"] = {"type": "json_object"}
                messages[0]["content"] += f"\n\nReply with JSON matching: {json_schema}"
            else:
                # Smaller open-weight models often support neither response_format
                # nor strict schemas, so the schema goes in the prompt and the
                # answer is salvaged from whatever comes back.
                messages[0]["content"] += (
                    f"\n\nReply with ONLY a JSON object matching this schema, "
                    f"no prose and no code fences: {json_schema}"
                )

        last_error = ""
        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - vendor SDKs raise broadly
                last_error = str(exc)
                if (_is_transient(last_error) and tier == "reasoning"
                        and kwargs["model"] != self.fast_model):
                    log.warning("%s busy; falling back to %s", self.label, self.fast_model)
                    kwargs["model"] = self.fast_model
                    continue
                if not _is_transient(last_error) or attempt == self.max_attempts - 1:
                    raise LLMError(f"completion failed: {exc}") from exc
                time.sleep(self.retry_backoff * (attempt + 1))
                continue

            # Featherless answers a failure with HTTP 200 and an {"error": ...}
            # body, which the OpenAI SDK happily parses into a response object
            # whose `choices` is None. An error wearing a success's clothes is
            # the worst shape of failure, so it is unwrapped explicitly here
            # rather than crashing three lines later on choices[0].
            problem = _error_text(response)
            if problem:
                last_error = problem
                # Featherless reports an unsupported strict json_schema as
                # "model is busy", indistinguishable from real load. Retrying
                # burns forty seconds to arrive at the same answer, so the first
                # failure in schema mode downgrades the provider for good and
                # tries again immediately.
                if json_schema is not None and self.json_mode == JSON_SCHEMA:
                    log.warning(
                        "%s rejected strict json_schema; downgrading to json_object",
                        self.label,
                    )
                    self.json_mode = JSON_OBJECT
                    kwargs["response_format"] = {"type": "json_object"}
                    messages[0]["content"] += f"\n\nReply with JSON matching: {json_schema}"
                    continue
                # Large shared models are busy often enough that "busy" is a
                # normal condition, not an incident. Falling back to the fast
                # model keeps an agent's judgement in play; the alternative is
                # the desk quietly reverting to no opinion at all.
                if (_is_transient(problem) and tier == "reasoning"
                        and kwargs["model"] != self.fast_model):
                    log.warning("%s busy on %s; falling back to %s",
                                self.label, kwargs["model"], self.fast_model)
                    kwargs["model"] = self.fast_model
                    continue
                if not _is_transient(problem) or attempt == self.max_attempts - 1:
                    raise LLMError(f"provider returned an error: {problem}")
                log.warning("%s busy, retrying (%d/%d)", self.label, attempt + 1, self.max_attempts)
                time.sleep(self.retry_backoff * (attempt + 1))
                continue

            usage = getattr(response, "usage", None)
            self.budget.charge(tier, int(getattr(usage, "total_tokens", 0) or 0))
            return response.choices[0].message.content or ""

        raise LLMError(f"completion failed after {self.max_attempts} attempts: {last_error}")


@dataclass
class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI, on the models covered by the free daily tier."""


@dataclass
class FeatherlessProvider(OpenAICompatibleProvider):
    """Featherless AI -- the hackathon's technology partner.

    Serverless inference over open-weight Hugging Face models, OpenAI-compatible
    at ``https://api.featherless.ai/v1``. Structured-output support is not
    documented, so the JSON mode is configurable and probed at startup rather
    than assumed; ``ask_json`` salvages fenced or prose-wrapped replies either way.
    """

    base_url: str = "https://api.featherless.ai/v1"
    # Chosen by measurement, not by size. Qwen3-Next is a mixture-of-experts
    # model with ~3B active parameters, so it answers a JSON classification in
    # about a second -- faster than the dense 7B it replaced. Kimi-K2 was both
    # the quickest and the cleanest writer on the shareholder-letter task.
    fast_model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    reasoning_model: str = "moonshotai/Kimi-K2-Instruct"
    # Measured: Featherless accepts response_format json_object and rejects
    # strict json_schema (reported, confusingly, as "this model is busy").
    json_mode: str = JSON_OBJECT
    key_env: str = "FEATHERLESS_API_KEY"
    label: str = "featherless"

    def available_models(self, limit: int = 20) -> list[str]:
        try:
            return [m.id for m in self.client.models.list().data][:limit]
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"could not list models: {exc}") from exc


@dataclass
class NullProvider:
    """Used in tests and whenever no key is configured.

    The desk must stay tradeable with the reasoning layer switched off entirely —
    the deterministic strategies do not depend on it, and that is the point.
    """

    canned: str = "{}"

    def complete(
        self, *, system: str, user: str, tier: Tier = "fast", json_schema: dict | None = None
    ) -> str:
        return self.canned


def extract_json(raw: str) -> Any:
    """Pull an object out of a reply that may be wrapped in prose or fences.

    Open-weight models routinely return ```json blocks or a sentence of preamble
    even when told not to. Refusing that output would mean discarding answers
    that are perfectly good three characters in.
    """
    text = (raw or "").strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise json.JSONDecodeError("no JSON object found", text, 0)


def ask_json(
    provider: LLMProvider,
    *,
    system: str,
    user: str,
    schema: dict,
    tier: Tier = "fast",
    default: Any = None,
) -> Any:
    """Structured call that never raises into the trading loop.

    A model outage must degrade the desk to its deterministic strategies, not
    stop it trading. Every caller supplies a usable default.
    """
    try:
        raw = provider.complete(system=system, user=user, tier=tier, json_schema=schema)
        return extract_json(raw)
    except Exception as exc:  # noqa: BLE001 - isolation IS this function's job
        # Deliberately broad. This is the boundary between the agent layer and
        # the trading loop, and its whole contract is that nothing crosses it.
        # A narrower catch let a provider raising a plain RuntimeError abort an
        # entire cycle -- so the desk stopped trading because a model misbehaved,
        # which is the opposite of the intended failure mode.
        log.warning("structured call failed, using default: %s", exc)
        return default


def build_provider() -> LLMProvider:
    """Provider selected by environment.

    Featherless first when it is configured: it is the hackathon's technology
    partner, and partner prizes require the partner's technology to actually be
    doing work in the submitted project.
    """
    vendor = os.environ.get("APERTURE_LLM_VENDOR", "featherless").lower()

    if vendor == "none":
        log.info("LLM disabled; running deterministic strategies only")
        return NullProvider()

    # Model overrides are vendor-scoped on purpose. A single generic
    # APERTURE_FAST_MODEL left over from another provider will happily point a
    # Featherless client at an OpenAI model name, and the only symptom is a 404
    # from a provider that was configured correctly in every other respect.
    if vendor == "featherless" and os.environ.get("FEATHERLESS_API_KEY"):
        return FeatherlessProvider(
            fast_model=os.environ.get(
                "APERTURE_FEATHERLESS_FAST", "Qwen/Qwen3-Next-80B-A3B-Instruct"
            ),
            reasoning_model=os.environ.get(
                "APERTURE_FEATHERLESS_REASONING", "moonshotai/Kimi-K2-Instruct"
            ),
            json_mode=os.environ.get("APERTURE_JSON_MODE", JSON_OBJECT),
        )

    if os.environ.get("OPENAI_API_KEY"):
        log.info("falling back to OpenAI")
        return OpenAIProvider(
            fast_model=os.environ.get("APERTURE_OPENAI_FAST", "gpt-5.4-mini"),
            reasoning_model=os.environ.get("APERTURE_OPENAI_REASONING", "gpt-5.4"),
        )

    log.info("no LLM provider configured; running deterministic strategies only")
    return NullProvider()


def provider_info(provider: LLMProvider) -> dict[str, Any]:
    """Public, credential-free evidence of what reasoning layer is active."""
    if isinstance(provider, NullProvider):
        return {"vendor": "none", "fast_model": None, "reasoning_model": None}
    return {
        "vendor": str(getattr(provider, "label", provider.__class__.__name__)).lower(),
        "fast_model": getattr(provider, "fast_model", None),
        "reasoning_model": getattr(provider, "reasoning_model", None),
        "json_mode": getattr(provider, "json_mode", None),
    }
