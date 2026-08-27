"""The reasoning layer, behind one swappable interface.

Two constraints shaped this module.

**No model may place an order.** Everything here returns text or parsed JSON.
Proposals produced from model output are ordinary :class:`Proposal` objects and
go through the Risk Warden exactly like any other. There is deliberately no path
from this file to the broker.

**The provider must be swappable.** The hackathon's technology partners are
announced at kickoff and partner prizes require integrating partner tech, so the
vendor is one config value rather than an assumption baked through the codebase.

Two tiers, matched to the free daily allowances: ``fast`` for high-frequency
structured work (news triage, red-team passes, extraction) and ``reasoning`` for
the two jobs that actually need it (strategy invention, the shareholder letter).
"""

from __future__ import annotations

import json
import logging
import os
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


@dataclass
class OpenAIProvider:
    """OpenAI adapter, defaulted to the models covered by the free daily tier."""

    api_key: str | None = None
    fast_model: str = "gpt-5.4-mini"
    reasoning_model: str = "gpt-5.4"
    budget: TokenBudget = field(default_factory=TokenBudget)
    timeout: float = 60.0
    _client: Any = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("the `openai` package is not installed") from exc
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        return self._client

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

        kwargs: dict[str, Any] = {
            "model": self.reasoning_model if tier == "reasoning" else self.fast_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - vendor SDKs raise broadly
            raise LLMError(f"completion failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        self.budget.charge(tier, int(getattr(usage, "total_tokens", 0) or 0))
        return response.choices[0].message.content or ""


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
        return json.loads(raw)
    except (LLMError, json.JSONDecodeError, TypeError) as exc:
        log.warning("structured call failed, using default: %s", exc)
        return default


def build_provider() -> LLMProvider:
    """Provider selected by environment, defaulting to a no-op."""
    vendor = os.environ.get("APERTURE_LLM_VENDOR", "openai").lower()
    if vendor == "none" or not os.environ.get("OPENAI_API_KEY"):
        log.info("no LLM provider configured; running deterministic strategies only")
        return NullProvider()
    if vendor == "openai":
        return OpenAIProvider(
            fast_model=os.environ.get("APERTURE_FAST_MODEL", "gpt-5.4-mini"),
            reasoning_model=os.environ.get("APERTURE_REASONING_MODEL", "gpt-5.4"),
        )
    raise LLMError(f"unknown LLM vendor: {vendor}")
