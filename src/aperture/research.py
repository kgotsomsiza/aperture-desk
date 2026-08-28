"""The research lab — where the desk invents its own strategies.

It runs when the market is closed, which is the only time there is room for it
and the only time a mistake is free. Each night:

  1. mutate the configs of the strategies currently on the roster,
  2. backtest every candidate against real historical option prices,
  3. put the survivors through a promotion gate,
  4. hand whatever passes to the allocator, which funds it on probation.

**The gate is the whole thing.** Generating candidates is easy and worth nothing;
what makes a promotion meaningful is how hard it is to earn. Three properties
matter:

**It corrects for how many candidates were tried.** Test twenty mutations and the
best one looks good whether or not any of them work — that is the garden of
forking paths, and it is how backtested strategies get built that immediately
lose money. The gate raises its own bar with every additional candidate
evaluated, so a winner drawn from a large field must clear a proportionally
higher hurdle.

**It requires the candidate to beat the incumbent**, not merely to be profitable.
Replacing a working strategy with a marginally different one is churn.

**It distrusts its own inputs.** Every known bias in the simulator flatters
results, so the gate demands an edge comfortably larger than the measurement
error rather than any edge at all.

A night where nothing is promoted is the expected outcome, and a lab that
promotes something every night is not selecting, it is drifting.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Sequence

from .alpaca_cli import AlpacaCLI
from .backtest import BacktestResult, CondorSpec, load_history, simulate_condors
from .llm import LLMProvider, NullProvider, ask_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    spec: CondorSpec
    parent: str
    mutation: str  # what was changed, in words


@dataclass(frozen=True)
class PromotionGate:
    """The bar a candidate must clear to be given real capital."""

    min_trades: int = 10
    min_edge: float = 0.06        # of risk deployed
    base_t_stat: float = 2.0      # before correcting for the number of candidates
    min_improvement: float = 0.03  # must beat the incumbent by this much
    max_drawdown_ratio: float = 0.60  # drawdown, relative to total profit

    def required_t(self, candidates_tried: int) -> float:
        """Raise the bar with the size of the field.

        Searching k candidates and reporting the best is k chances to be fooled.
        Scaling the threshold by sqrt(log k) is the cheap, standard correction --
        crude, but it moves the right way and is honest about why it exists.
        """
        import math

        k = max(candidates_tried, 1)
        return self.base_t_stat * math.sqrt(1.0 + math.log(k))

    def evaluate(
        self,
        candidate: BacktestResult,
        incumbent: BacktestResult | None,
        candidates_tried: int,
    ) -> tuple[bool, str]:
        if candidate.n < self.min_trades:
            return False, f"only {candidate.n} trades; needs {self.min_trades}"

        if candidate.edge < self.min_edge:
            return False, f"edge {candidate.edge:+.1%} below the {self.min_edge:.0%} floor"

        threshold = self.required_t(candidates_tried)
        if candidate.t_stat < threshold:
            return False, (
                f"t={candidate.t_stat:.2f} below {threshold:.2f}, the bar for a field "
                f"of {candidates_tried} candidates"
            )

        if candidate.total_pnl > 0:
            ratio = candidate.max_drawdown / candidate.total_pnl
            if ratio > self.max_drawdown_ratio:
                return False, (
                    f"drawdown ${candidate.max_drawdown:,.0f} is {ratio:.0%} of its "
                    f"own profit; too rough to fund"
                )

        if incumbent is not None and incumbent.n >= self.min_trades:
            lift = candidate.edge - incumbent.edge
            if lift < self.min_improvement:
                return False, (
                    f"beats the incumbent by only {lift:+.1%}; not worth the churn"
                )
            return True, (
                f"promoted: edge {candidate.edge:+.1%} vs incumbent "
                f"{incumbent.edge:+.1%}, t={candidate.t_stat:.2f} over {candidate.n} trades"
            )

        return True, (
            f"promoted: edge {candidate.edge:+.1%}, t={candidate.t_stat:.2f} "
            f"over {candidate.n} trades"
        )


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #

MUTATIONS = (
    ("short_pct", (-0.015, -0.01, 0.01, 0.015), "moved the short strikes"),
    ("width_pct", (-0.005, 0.005, 0.01), "changed the wing width"),
    ("dte_target", (-7, -3, 3, 7), "shifted the tenor"),
    ("take_profit", (-0.15, -0.1, 0.1, 0.15), "adjusted the profit target"),
    ("stop_multiple", (-0.5, 0.5, 1.0), "adjusted the stop"),
)


def mutate(spec: CondorSpec, count: int, rng: random.Random) -> list[Candidate]:
    """Single-parameter perturbations of a working config.

    One knob at a time, deliberately: a candidate that changes four things and
    tests better teaches nothing about which change mattered, and cannot be
    trusted to keep working when one of them stops.
    """
    seen: set[tuple] = set()
    out: list[Candidate] = []

    for index in range(count * 4):
        if len(out) >= count:
            break
        field_name, deltas, description = rng.choice(MUTATIONS)
        delta = rng.choice(deltas)
        current = getattr(spec, field_name)
        value = round(current + delta, 4) if isinstance(current, float) else current + delta

        if not _sane(field_name, value):
            continue
        key = (field_name, value)
        if key in seen:
            continue
        seen.add(key)

        direction = "wider" if delta > 0 else "tighter"
        out.append(
            Candidate(
                candidate_id=f"S{len(out) + 5}-{field_name.upper()[:5]}",
                spec=replace(spec, **{field_name: value}),
                parent="CARRY",
                mutation=f"{description} {direction}: {field_name} {current} -> {value}",
            )
        )
    return out


def _sane(field_name: str, value) -> bool:
    bounds = {
        "short_pct": (0.015, 0.12),
        "width_pct": (0.005, 0.05),
        "dte_target": (5, 45),
        "take_profit": (0.25, 0.85),
        "stop_multiple": (1.5, 4.0),
    }
    low, high = bounds[field_name]
    return low <= value <= high


def narrate(provider: LLMProvider, candidate: Candidate, result: BacktestResult) -> str:
    """One sentence on why this mutation might work, for the audit trail.

    The model explains; it does not decide. Promotion is settled by the gate
    before this is ever called.
    """
    answer = ask_json(
        provider,
        system=(
            "You are a derivatives trader. Explain in one plain sentence why a "
            "change to an iron condor's parameters might improve results. No "
            "hype, no disclaimers."
        ),
        user=f"Change: {candidate.mutation}. Backtest: {result.summary()}",
        schema={
            "type": "object",
            "properties": {"rationale": {"type": "string"}},
            "required": ["rationale"],
            "additionalProperties": False,
        },
        tier="reasoning",
        default={"rationale": candidate.mutation},
    )
    return (answer or {}).get("rationale") or candidate.mutation


# --------------------------------------------------------------------------- #
# The nightly run
# --------------------------------------------------------------------------- #


@dataclass
class LabReport:
    tested: int = 0
    promoted: list[tuple[Candidate, BacktestResult, str]] = field(default_factory=list)
    rejected: list[tuple[Candidate, BacktestResult, str]] = field(default_factory=list)
    incumbent: BacktestResult | None = None

    def summary(self) -> str:
        lines = [f"research lab: {self.tested} candidates tested, {len(self.promoted)} promoted"]
        if self.incumbent:
            lines.append(f"  incumbent  {self.incumbent.summary()}")
        for candidate, result, reason in self.promoted:
            lines.append(f"  HIRED      {candidate.candidate_id}: {reason}")
            lines.append(f"             {candidate.mutation}")
        for candidate, result, reason in self.rejected[:5]:
            lines.append(f"  rejected   {candidate.candidate_id}: {reason}")
        return "\n".join(lines)


def run_lab(
    cli: AlpacaCLI,
    *,
    underlying: str = "SPY",
    baseline: CondorSpec | None = None,
    candidates: int = 8,
    lookback_days: int = 300,
    expiries: int = 40,
    strikes_per_expiry: int = 120,
    gate: PromotionGate | None = None,
    provider: LLMProvider | None = None,
    seed: int | None = None,
    asof: date | None = None,
) -> LabReport:
    """Test a field of mutations against real history and promote what survives."""
    baseline = baseline or CondorSpec()
    gate = gate or PromotionGate()
    provider = provider or NullProvider()
    rng = random.Random(seed)
    asof = asof or date.today()

    start = asof - timedelta(days=lookback_days)
    # Stop short of today: contracts must have expired to be listed and priced.
    end = asof - timedelta(days=30)

    history = load_history(
        cli, underlying, start=start, end=end,
        expiries=expiries, strikes_per_expiry=strikes_per_expiry,
    )
    report = LabReport()
    if not history.bars:
        log.warning("no option history for %s; the lab cannot run tonight", underlying)
        return report

    report.incumbent = simulate_condors(history, baseline, strategy_id="CARRY(incumbent)")
    log.info("incumbent: %s", report.incumbent.summary())

    field_ = mutate(baseline, candidates, rng)
    report.tested = len(field_)

    for candidate in field_:
        result = simulate_condors(history, candidate.spec, strategy_id=candidate.candidate_id)
        passed, reason = gate.evaluate(result, report.incumbent, report.tested)
        if passed:
            reason = f"{reason}. {narrate(provider, candidate, result)}"
            report.promoted.append((candidate, result, reason))
        else:
            report.rejected.append((candidate, result, reason))

    log.info("%s", report.summary())
    return report
