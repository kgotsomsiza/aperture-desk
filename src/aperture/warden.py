"""The Risk Warden — the only component that can say no, and the only one that can flatten.

The Warden sits between every strategy and the broker. It holds veto authority
over the Portfolio Manager, which means a strategy that is making money can still
be stopped, and an allocator that wants to add risk can still be refused.

Every decision — approvals included — is appended to an audit log. That log is
not a debugging convenience; it is the artifact that makes an autonomous trading
agent accountable, and it is what gets shown on screen when someone asks why the
desk did something.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .risk import (
    BookState,
    LegQuote,
    Proposal,
    RiskLimits,
    Verdict,
    evaluate,
    tournament_risk_multiplier,
)

log = logging.getLogger(__name__)

KILL_SWITCH = Path("KILL_SWITCH")


@dataclass
class AuditLog:
    """Append-only JSONL. One line per decision, never rewritten."""

    path: Path = Path("state/audit.jsonl")

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields) -> dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        return entry

    def tail(self, limit: int = 50, event: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event is None or row.get("event") == event:
                    rows.append(row)
        return rows[-limit:]

    def vetoes(self, limit: int = 50) -> list[dict]:
        return self.tail(limit=limit, event="veto")


@dataclass
class RiskWarden:
    limits: RiskLimits = field(default_factory=RiskLimits)
    audit: AuditLog = field(default_factory=AuditLog)
    deadline: datetime | None = None
    start_equity: float = 100_000.0
    budgets: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Gate
    # ------------------------------------------------------------------ #

    def review(
        self, proposal: Proposal, quotes: Sequence[LegQuote], book: BookState
    ) -> Verdict:
        """Run the gates and record the outcome. This is the only path to an order."""
        if self.halted():
            verdict = _halt_verdict(proposal, "kill switch engaged")
            self._record(proposal, verdict, book)
            return verdict

        verdict = evaluate(
            proposal,
            quotes,
            book,
            self.limits,
            strategy_budget=self.budgets.get(proposal.strategy_id, float("inf")),
        )
        self._record(proposal, verdict, book)
        return verdict

    def _record(self, proposal: Proposal, verdict: Verdict, book: BookState) -> None:
        self.audit.record(
            "approval" if verdict.approved else "veto",
            strategy=proposal.strategy_id,
            underlying=proposal.underlying,
            qty=proposal.qty,
            net_price=proposal.net_price,
            legs=[leg.symbol for leg in proposal.legs],
            max_loss=verdict.profile.max_loss,
            rationale=proposal.rationale,
            equity=round(book.equity, 2),
            summary=verdict.audit_line(),
            reasons=[
                {"gate": r.gate, "reason": r.reason} for r in verdict.rejections
            ],
        )
        if verdict.approved:
            log.info("APPROVED %s %s x%d", proposal.strategy_id, proposal.underlying, proposal.qty)
        else:
            log.warning("VETOED %s %s - %s", proposal.strategy_id, proposal.underlying,
                        verdict.audit_line())

    # ------------------------------------------------------------------ #
    # Sizing
    # ------------------------------------------------------------------ #

    def risk_appetite(self, book: BookState) -> float:
        """Tournament-aware multiplier applied to strategy budgets.

        The gates above are absolute and this never loosens them; it only ever
        scales how much of an already-permitted budget a strategy may use.
        """
        if self.deadline is None:
            return 1.0
        return tournament_risk_multiplier(
            book.now, self.deadline, book.equity, self.start_equity
        )

    def budget_for(self, strategy_id: str, book: BookState) -> float:
        base = self.budgets.get(strategy_id, 0.0)
        used = book.open_risk_by_strategy.get(strategy_id, 0.0)
        return max(0.0, base * self.risk_appetite(book) - used)

    # ------------------------------------------------------------------ #
    # Circuit breakers and the kill switch
    # ------------------------------------------------------------------ #

    def halted(self) -> bool:
        return KILL_SWITCH.exists()

    def engage_kill_switch(self, reason: str) -> None:
        KILL_SWITCH.write_text(f"{datetime.now(timezone.utc).isoformat()} {reason}\n")
        self.audit.record("kill_switch", reason=reason)
        log.critical("KILL SWITCH ENGAGED: %s", reason)

    def release_kill_switch(self) -> None:
        KILL_SWITCH.unlink(missing_ok=True)
        self.audit.record("kill_switch_released")

    def breached(self, book: BookState) -> str | None:
        """Portfolio-level breach that should force de-risking, not just block entries."""
        if book.drawdown_pct >= self.limits.drawdown_breaker_pct:
            return (
                f"drawdown {book.drawdown_pct:.2%} breached "
                f"{self.limits.drawdown_breaker_pct:.0%} from the high-water mark"
            )
        if book.day_pnl_pct <= -self.limits.daily_loss_halt_pct:
            return f"day P&L {book.day_pnl_pct:+.2%} breached the daily loss limit"
        return None


def _halt_verdict(proposal: Proposal, reason: str) -> Verdict:
    from .risk import GateResult, analyse_payoff

    return Verdict(
        approved=False,
        profile=analyse_payoff(proposal),
        results=(GateResult("kill_switch", False, reason),),
    )
