"""The allocator — the desk hires, funds and fires its own strategies.

Capital is the desk's only reward signal. A strategy that earns gets more of it;
one that does not gets less, and eventually none. That is the whole mechanic.

Three ideas make it more than a weighted average of recent returns:

**Shrinkage, because four sessions is not evidence.** The scored window is four
trading days. A strategy might close five trades in that time, and five trades
say almost nothing about an edge. So the allocator holds a prior — the weights
the strategies were designed with — and moves toward observed performance only
in proportion to how much has actually been observed. With the default
half-life, five closed trades move the allocation about a third of the way. A
desk that reallocated hard on three lucky wins would be reading noise, loudly.

**Vetoes are evidence too, and of a different kind.** A strategy whose proposals
are consistently rejected by the Risk Warden is not unlucky — it is miscalibrated.
It keeps asking for trades that are too large, too illiquid, or too concentrated,
which means its own model of what it can do is wrong. Losing money and being
refused permission are different failures, and the second one is diagnosable
without waiting for P&L. Most desks would never think to look at the rejection
log as a performance signal; it is the fastest one available in a four-day window.

**Firing is not the same as losing.** A fired strategy keeps its open positions
and its exits — the desk stops giving it new capital, it does not abandon what it
already holds. Anything else would turn a funding decision into a forced
liquidation at whatever the market happens to be offering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .state import DeskState
from .warden import AuditLog

FUNDED = "funded"
PROBATION = "probation"
FIRED = "fired"


@dataclass
class StrategyRecord:
    """What the desk has actually observed about one strategy."""

    strategy_id: str
    prior_weight: float
    closed: int = 0
    wins: int = 0
    realized_pnl: float = 0.0
    risk_deployed: float = 0.0  # sum of max_loss across closed trades
    proposals: int = 0
    vetoes: int = 0
    open_positions: int = 0

    @property
    def edge(self) -> float:
        """Realized P&L per dollar of risk put at work.

        Return on risk rather than return on capital: a strategy risking $1,000 to
        make $200 is doing better than one risking $10,000 to make $400, and
        allocation should follow the former.
        """
        if self.risk_deployed <= 0:
            return 0.0
        return self.realized_pnl / self.risk_deployed

    @property
    def veto_rate(self) -> float:
        if self.proposals <= 0:
            return 0.0
        return self.vetoes / self.proposals

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.closed if self.closed else None


@dataclass(frozen=True)
class AllocationLimits:
    # Closed trades needed for observation to carry half the weight. Small
    # samples stay anchored to the prior.
    shrinkage_half_life: float = 5.0
    min_weight: float = 0.05          # below this, a strategy is not worth running
    max_weight: float = 0.70          # no strategy owns the whole book
    probation_weight: float = 0.05    # what a newly hired strategy starts with
    probation_trades: int = 3         # closed trades before it is judged normally
    fire_edge: float = -0.25          # lost a quarter of the risk it deployed
    fire_after_trades: int = 4        # ...and has done it often enough to mean it
    fire_veto_rate: float = 0.80      # asks for trades it is never allowed to make
    fire_veto_sample: int = 8         # ...over a meaningful number of proposals
    total_risk_budget_pct: float = 0.28  # of equity, split across everyone


@dataclass
class Allocation:
    strategy_id: str
    weight: float
    budget: float
    status: str
    reason: str

    @property
    def is_active(self) -> bool:
        return self.status != FIRED


@dataclass
class Allocator:
    limits: AllocationLimits = field(default_factory=AllocationLimits)

    def allocate(
        self, records: Sequence[StrategyRecord], equity: float
    ) -> list[Allocation]:
        """Decide each strategy's share of the risk budget."""
        limits = self.limits
        risk_budget = equity * limits.total_risk_budget_pct

        scores: dict[str, float] = {}
        statuses: dict[str, str] = {}
        reasons: dict[str, str] = {}

        for record in records:
            fired, reason = self._fire_check(record)
            if fired:
                scores[record.strategy_id] = 0.0
                statuses[record.strategy_id] = FIRED
                reasons[record.strategy_id] = reason
                continue

            # A strategy hired during the run has no designed weight. Anchoring
            # it to zero would make it permanently unfundable, which would defeat
            # the point of the desk being able to hire at all -- so it is
            # anchored to the probation weight instead and has to earn its way up.
            on_probation = (
                record.prior_weight <= 0.0 and record.closed < limits.probation_trades
            )
            statuses[record.strategy_id] = PROBATION if on_probation else FUNDED
            scores[record.strategy_id] = self._posterior_weight(record)
            reasons[record.strategy_id] = (
                "newly hired; funded on probation until it has a record"
                if on_probation
                else self._explain(record, scores[record.strategy_id])
            )

        weights = _solve_weights(
            scores,
            statuses,
            min_weight=limits.min_weight,
            max_weight=limits.max_weight,
            probation_weight=limits.probation_weight,
        )

        return [
            Allocation(
                strategy_id=strategy_id,
                weight=round(weights.get(strategy_id, 0.0), 4),
                budget=round(risk_budget * weights.get(strategy_id, 0.0), 2),
                status=statuses[strategy_id],
                reason=reasons[strategy_id],
            )
            for strategy_id in scores
        ]

    # ------------------------------------------------------------------ #

    def _posterior_weight(self, record: StrategyRecord) -> float:
        """Prior weight, nudged toward observed edge in proportion to evidence."""
        limits = self.limits
        anchor = record.prior_weight if record.prior_weight > 0 else limits.probation_weight
        confidence = record.closed / (record.closed + limits.shrinkage_half_life)

        # A performance multiplier centred on 1.0. An edge of +20% of risk
        # deployed roughly doubles the weight at full confidence; -20% halves it.
        multiplier = math.exp(3.5 * record.edge * confidence)

        # Being refused permission is its own penalty, independent of P&L.
        if record.proposals >= limits.fire_veto_sample:
            multiplier *= 1.0 - 0.5 * record.veto_rate

        return max(anchor * multiplier, 1e-6)

    def _fire_check(self, record: StrategyRecord) -> tuple[bool, str]:
        limits = self.limits

        if (
            record.closed >= limits.fire_after_trades
            and record.edge <= limits.fire_edge
        ):
            return True, (
                f"fired: lost {abs(record.edge):.0%} of the risk it deployed "
                f"over {record.closed} closed trades"
            )

        if (
            record.proposals >= limits.fire_veto_sample
            and record.veto_rate >= limits.fire_veto_rate
        ):
            return True, (
                f"fired: {record.vetoes} of {record.proposals} proposals vetoed "
                f"({record.veto_rate:.0%}) — it is miscalibrated, not unlucky"
            )

        return False, ""

    def _explain(self, record: StrategyRecord, raw: float) -> str:
        direction = "up" if raw > record.prior_weight else "down" if raw < record.prior_weight else "held"
        if record.closed == 0:
            return f"no closed trades yet; held at its designed weight"
        return (
            f"{direction} on {record.closed} closed trade(s): "
            f"edge {record.edge:+.1%} of risk deployed"
            + (f", {record.veto_rate:.0%} of proposals vetoed" if record.proposals else "")
        )


def _solve_weights(
    scores: dict[str, float],
    statuses: dict[str, str],
    *,
    min_weight: float,
    max_weight: float,
    probation_weight: float,
) -> dict[str, float]:
    """Normalise scores into weights that sum to one and respect every bound.

    Clamping after normalising breaks the sum; normalising after clamping breaks
    the clamps. The fix is to clamp, then push the remaining error only onto
    strategies that still have room to absorb it -- repeatedly, until both hold.
    Normalising a fully-clamped set, as an earlier version did, silently violated
    the caps it had just applied.
    """
    active = {k: v for k, v in scores.items() if statuses[k] != FIRED}
    if not active or sum(active.values()) <= 0:
        return {k: 0.0 for k in scores}

    def ceiling(strategy_id: str) -> float:
        return probation_weight if statuses[strategy_id] == PROBATION else max_weight

    def floor(strategy_id: str) -> float:
        return 0.0 if statuses[strategy_id] == PROBATION else min_weight

    # Infeasible bounds would loop forever; fall back to an even split.
    if sum(floor(k) for k in active) > 1.0 or sum(ceiling(k) for k in active) < 1.0:
        even = 1.0 / len(active)
        return {**{k: 0.0 for k in scores}, **{k: even for k in active}}

    total = sum(active.values())
    weights = {k: v / total for k, v in active.items()}

    for _ in range(50):
        weights = {k: min(max(w, floor(k)), ceiling(k)) for k, w in weights.items()}
        shortfall = 1.0 - sum(weights.values())
        if abs(shortfall) < 1e-9:
            break

        # Room to move, in the direction the correction needs to go.
        room = {
            k: (ceiling(k) - w) if shortfall > 0 else (w - floor(k))
            for k, w in weights.items()
        }
        available = sum(room.values())
        if available <= 1e-12:
            break
        for k in weights:
            weights[k] += shortfall * (room[k] / available)

    return {**{k: 0.0 for k in scores}, **weights}


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #


def observe(
    state: DeskState,
    audit: AuditLog,
    priors: dict[str, float],
    *,
    audit_limit: int = 2000,
) -> list[StrategyRecord]:
    """Build the performance record from the ledger and the audit log.

    The ledger knows what was traded and what it cost. The audit log knows what
    was *proposed* — including everything the Warden refused, which never reaches
    the ledger at all and is invisible to P&L.
    """
    records = {
        strategy_id: StrategyRecord(strategy_id=strategy_id, prior_weight=weight)
        for strategy_id, weight in priors.items()
    }

    def record_for(strategy_id: str) -> StrategyRecord:
        if strategy_id not in records:
            # A strategy the desk hired during the run: no prior, starts on probation.
            records[strategy_id] = StrategyRecord(strategy_id, prior_weight=0.0)
        return records[strategy_id]

    for trade in state.open_trades.values():
        record_for(trade.strategy_id).open_positions += 1

    for closed in state.closed:
        # Accepted, expired or externally-vanished orders are not outcomes.
        # Only a broker-confirmed close with computed P&L is performance
        # evidence for hiring and firing.
        if closed.get("pnl") is None:
            continue
        record = record_for(closed.get("strategy_id", "?"))
        record.closed += 1
        record.risk_deployed += float(closed.get("max_loss") or 0.0)
        pnl = closed.get("pnl")
        record.realized_pnl += float(pnl)
        if float(pnl) > 0:
            record.wins += 1

    # The same economic proposal is often evaluated every five minutes while a
    # quote remains wide.  Keep only its latest decision for the session; a
    # sequence of eight identical vetoes is one miscalibrated idea, not a sample
    # of eight independent ideas.
    decisions: dict[tuple, dict] = {}
    for entry in audit.tail(limit=audit_limit):
        event = entry.get("event")
        if event not in ("approval", "veto"):
            continue
        identity = entry.get("proposal_id") or (
            str(entry.get("ts", ""))[:10],
            entry.get("strategy", "?"),
            entry.get("underlying", "?"),
            tuple(entry.get("legs") or ()),
        )
        decisions[(entry.get("strategy", "?"), identity)] = entry

    for entry in decisions.values():
        event = entry.get("event")
        record = record_for(entry.get("strategy", "?"))
        record.proposals += 1
        if event == "veto":
            record.vetoes += 1

    return list(records.values())


def summarise(allocations: Iterable[Allocation]) -> str:
    """One human-readable block, for the audit log and the shareholder letter."""
    lines = []
    for allocation in sorted(allocations, key=lambda a: -a.weight):
        marker = {FUNDED: " ", PROBATION: "~", FIRED: "x"}[allocation.status]
        lines.append(
            f"  {marker} {allocation.strategy_id:8} {allocation.weight:6.1%} "
            f"${allocation.budget:>9,.0f}  {allocation.reason}"
        )
    return "\n".join(lines)
