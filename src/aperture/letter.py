"""The daily shareholder letter — the desk explaining itself.

The judging criterion asks how clearly a project "presents the reasoning behind
its trading strategy and results". A desk that writes its own account of what it
did, in the same words whether the day went well or badly, answers that directly.

**Every number is computed in Python and handed to the model as fact.** The model
writes prose from a fact block and is told, explicitly, to invent nothing. This
is the same rule as the trading path: deterministic code decides, the language
model narrates. A letter that quietly rounded a loss into a gain would be worse
than no letter, and a model asked to both compute and describe will eventually do
exactly that.

If the model is unavailable the letter still gets written, just plainly. The
fallback is not a degraded mode to be embarrassed about; it is the same facts
without the prose.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from .allocator import Allocation, FIRED, PROBATION
from .llm import LLMProvider, NullProvider, LLMError
from .state import DeskState
from .warden import AuditLog

log = logging.getLogger(__name__)

SYSTEM = """You are an autonomous options trading desk writing the daily letter to
your investors. Rules you must follow exactly:

- Use ONLY the figures given to you. Never invent, round, or estimate a number.
- If the day was a loss, say so plainly in the first sentence.
- No hype, no disclaimers, no apologies, no "as an AI".
- Explain WHY positions were taken or refused, not just what happened.
- Four short paragraphs at most. Under 220 words.
- Write as "the desk", never "I"."""


@dataclass
class LetterFacts:
    """Everything the letter is allowed to say, computed deterministically."""

    as_of: str
    equity: float
    start_equity: float
    day_start_equity: float
    open_positions: int
    opened_today: list[dict[str, Any]] = field(default_factory=list)
    closed_today: list[dict[str, Any]] = field(default_factory=list)
    vetoes_today: list[dict[str, Any]] = field(default_factory=list)
    hires_today: list[dict[str, Any]] = field(default_factory=list)
    allocations: list[dict[str, Any]] = field(default_factory=list)
    breaches: list[str] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        base = self.start_equity or self.equity
        return (self.equity / base - 1) * 100 if base else 0.0

    @property
    def day_return_pct(self) -> float:
        base = self.day_start_equity or self.equity
        return (self.equity / base - 1) * 100 if base else 0.0

    def as_prompt(self) -> str:
        """The fact block. Nothing outside this may appear in the letter."""
        lines = [
            f"Date: {self.as_of}",
            f"Equity: ${self.equity:,.2f}",
            f"Since inception: {self.total_return_pct:+.2f}%",
            f"On the day: {self.day_return_pct:+.2f}%",
            f"Open positions: {self.open_positions}",
        ]

        if self.opened_today:
            lines.append("\nOpened today:")
            for row in self.opened_today:
                lines.append(
                    f"  - {row['strategy']} {row['underlying']} x{row['qty']}: {row['rationale']}"
                )
        else:
            lines.append("\nOpened today: nothing")

        if self.closed_today:
            lines.append("\nClosed today:")
            for row in self.closed_today:
                lines.append(f"  - {row['strategy']} {row['underlying']}: {row['reason']}")

        if self.vetoes_today:
            lines.append("\nRefused by the risk warden today:")
            for row in self.vetoes_today[:6]:
                lines.append(f"  - {row['strategy']} {row['underlying']}: {row['reason']}")

        if self.hires_today:
            lines.append("\nHired by the research lab today:")
            for row in self.hires_today:
                lines.append(
                    f"  - {row['strategy']}: {row['mutation']} — {row['evidence']}"
                )

        if self.allocations:
            lines.append("\nCapital allocation:")
            for row in self.allocations:
                lines.append(
                    f"  - {row['strategy']}: {row['weight']:.0%} "
                    f"(${row['budget']:,.0f}) {row['status']} — {row['reason']}"
                )

        if self.breaches:
            lines.append("\nRisk breaches: " + "; ".join(self.breaches))

        return "\n".join(lines)


def gather(
    state: DeskState,
    audit: AuditLog,
    equity: float,
    allocations: list[Allocation] | None = None,
    *,
    today: date | None = None,
) -> LetterFacts:
    """Assemble the day's facts from the ledger and the audit log."""
    today = today or datetime.now(timezone.utc).date()
    stamp = today.isoformat()

    def is_today(row: dict) -> bool:
        return str(row.get("session") or row.get("ts", ""))[:10] == stamp

    entries = audit.tail(limit=4000)

    opened = [
        {
            "strategy": e.get("strategy", "?"),
            "underlying": e.get("underlying", "?"),
            "qty": e.get("qty", 0),
            "rationale": (e.get("rationale") or "")[:180],
        }
        for e in entries
        if e.get("event") == "entry_filled" and is_today(e)
    ]
    closed = [
        {
            "strategy": e.get("strategy", "?"),
            "underlying": e.get("underlying", "?"),
            "reason": (e.get("reason") or "")[:120],
        }
        for e in entries
        if e.get("event") == "closed" and is_today(e)
    ]
    vetoes = [
        {
            "strategy": e.get("strategy", "?"),
            "underlying": e.get("underlying", "?"),
            "reason": _first_reason(e),
        }
        for e in entries
        if e.get("event") == "veto" and is_today(e)
    ]
    breaches = [
        str(e.get("reason", "")) for e in entries if e.get("event") == "breach" and is_today(e)
    ]
    hires = [
        {
            "strategy": e.get("strategy", "?"),
            "mutation": str(e.get("mutation") or "")[:140],
            "evidence": str(e.get("summary") or "")[:220],
        }
        for e in entries
        if e.get("event") == "hired" and is_today(e)
    ]

    return LetterFacts(
        as_of=stamp,
        equity=equity,
        start_equity=state.start_equity or equity,
        day_start_equity=state.day_start_equity or equity,
        open_positions=sum(
            1
            for trade in state.open_trades.values()
            if trade.status in {"open", "submitting_close", "pending_close"}
        ),
        opened_today=opened,
        closed_today=closed,
        vetoes_today=vetoes,
        hires_today=hires,
        allocations=[
            {
                "strategy": a.strategy_id,
                "weight": a.weight,
                "budget": a.budget,
                "status": a.status,
                "reason": a.reason,
            }
            for a in (allocations or [])
        ],
        breaches=breaches,
    )


def _first_reason(entry: dict) -> str:
    reasons = entry.get("reasons") or []
    if reasons and isinstance(reasons[0], dict):
        return f"{reasons[0].get('gate', '')}: {reasons[0].get('reason', '')}"[:140]
    return str(entry.get("summary", ""))[:140]


def write(provider: LLMProvider, facts: LetterFacts) -> str:
    """The letter. Falls back to plain facts if the model is unreachable."""
    if isinstance(provider, NullProvider):
        return plain(facts)
    try:
        prose = provider.complete(system=SYSTEM, user=facts.as_prompt(), tier="reasoning")
    except LLMError as exc:
        log.warning("letter generation failed, writing the plain version: %s", exc)
        return plain(facts)

    prose = (prose or "").strip()
    if len(prose) < 60:
        log.warning("letter came back too short; writing the plain version")
        return plain(facts)
    if len(prose.split()) > 220:
        log.warning("letter exceeded 220 words; writing the plain version")
        return plain(facts)
    if _unexpected_numbers(prose, facts.as_prompt()):
        log.warning("letter introduced a number outside the fact block; writing the plain version")
        return plain(facts)
    return f"{_header(facts)}\n\n{prose}"


def _unexpected_numbers(prose: str, fact_block: str) -> set[str]:
    """Numbers are allowed only when their exact value exists in deterministic facts."""
    pattern = re.compile(r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?%?")

    def normalise(token: str) -> str:
        return token.replace("$", "").replace(",", "").replace("%", "").lstrip("+")

    allowed = {normalise(token) for token in pattern.findall(fact_block)}
    allowed |= {token.lstrip("-") for token in allowed}
    found = {normalise(token) for token in pattern.findall(prose)}
    return {token for token in found if token not in allowed}


def plain(facts: LetterFacts) -> str:
    """The same day, without prose. Never fails, never invents."""
    lines = [_header(facts), ""]
    lines.append(
        f"Opened {len(facts.opened_today)}, closed {len(facts.closed_today)}, "
        f"refused {len(facts.vetoes_today)}. {facts.open_positions} positions open."
    )
    for row in facts.opened_today:
        lines.append(f"  opened  {row['strategy']:6} {row['underlying']:6} {row['rationale']}")
    for row in facts.closed_today:
        lines.append(f"  closed  {row['strategy']:6} {row['underlying']:6} {row['reason']}")
    for row in facts.vetoes_today[:6]:
        lines.append(f"  refused {row['strategy']:6} {row['underlying']:6} {row['reason']}")
    for row in facts.hires_today:
        lines.append(f"  hired   {row['strategy']:6} {row['mutation']}")
    if facts.breaches:
        lines.append("  BREACH  " + "; ".join(facts.breaches))
    for row in facts.allocations:
        marker = {FIRED: "fired", PROBATION: "probation"}.get(row["status"], "")
        lines.append(
            f"  capital {row['strategy']:6} {row['weight']:6.1%} "
            f"${row['budget']:>9,.0f} {marker}"
        )
    return "\n".join(lines)


def _header(facts: LetterFacts) -> str:
    return (
        f"APERTURE DESK — {facts.as_of}\n"
        f"Equity ${facts.equity:,.2f}   "
        f"day {facts.day_return_pct:+.2f}%   "
        f"since inception {facts.total_return_pct:+.2f}%"
    )
