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
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
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
    hypothesis: str = ""  # why the model prioritised it; never a promotion input


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

        if incumbent is not None and incumbent.n < self.min_trades:
            return False, (
                f"incumbent has only {incumbent.n} trades; cannot prove an improvement"
            )

        if incumbent is not None:
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
    pool = mutation_pool(spec)
    rng.shuffle(pool)
    return pool[:count]


def mutation_pool(spec: CondorSpec) -> list[Candidate]:
    """The finite, predeclared hypothesis family the model may choose from.

    Featherless prioritises economically plausible experiments; it does not get
    to invent arbitrary parameters after seeing results.  Keeping the family
    finite makes the multiple-testing correction know what was searched.
    """
    out: list[Candidate] = []
    for field_name, deltas, description in MUTATIONS:
        current = getattr(spec, field_name)
        for delta in deltas:
            value = round(current + delta, 4) if isinstance(current, float) else current + delta
            if not _sane(field_name, value):
                continue
            direction = "wider" if delta > 0 else "tighter"
            encoded = str(value).replace("-", "M").replace(".", "P")
            out.append(
                Candidate(
                    candidate_id=f"CAND-{field_name.upper()[:5]}-{encoded}",
                    spec=replace(spec, **{field_name: value}),
                    parent="CARRY",
                    mutation=(
                        f"{description} {direction}: {field_name} {current} -> {value}"
                    ),
                )
            )
    return out


def hypothesize(
    provider: LLMProvider,
    spec: CondorSpec,
    count: int,
    rng: random.Random,
) -> list[Candidate]:
    """Let the reasoning layer prioritise safe one-knob experiments.

    This is the model's substantive research job.  It sees the candidate family
    and market mechanics, but no backtest outcomes.  Python validates its IDs,
    fills any missing choices deterministically, and the statistical gate alone
    decides whether a hypothesis is hired.
    """
    fallback = mutate(spec, count, rng)
    if isinstance(provider, NullProvider) or not fallback:
        return fallback

    pool = mutation_pool(spec)
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    choices = "\n".join(
        f"- {candidate.candidate_id}: {candidate.mutation}" for candidate in pool
    )
    default = {
        "selections": [
            {
                "candidate_id": candidate.candidate_id,
                "thesis": "deterministic fallback selection",
            }
            for candidate in fallback
        ]
    }
    answer = ask_json(
        provider,
        system=(
            "You are the hypothesis-selection layer for a defined-risk options desk. "
            "Prioritise experiments for economic plausibility only. You have no "
            "performance data and must choose only IDs from the supplied list. "
            "The statistical gate, not you, decides promotion."
        ),
        user=(
            f"Choose exactly {min(count, len(pool))} distinct one-parameter iron-condor "
            f"experiments. Give one short falsifiable thesis for each.\n\n{choices}"
        ),
        schema={
            "type": "object",
            "properties": {
                "selections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "thesis": {"type": "string"},
                        },
                        "required": ["candidate_id", "thesis"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["selections"],
            "additionalProperties": False,
        },
        tier="fast",
        default=default,
    )

    selected: list[Candidate] = []
    seen: set[str] = set()
    rows = answer.get("selections", []) if isinstance(answer, dict) else []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        candidate = by_id.get(candidate_id)
        if candidate is None or candidate_id in seen:
            continue
        thesis = " ".join(str(row.get("thesis") or "").split())[:240]
        selected.append(replace(candidate, hypothesis=thesis))
        seen.add(candidate_id)
        if len(selected) >= count:
            break

    # Invalid, duplicate, or incomplete model output cannot reduce the nightly
    # experiment count.  Seeded fallback choices make replay deterministic.
    for candidate in fallback:
        if len(selected) >= count:
            break
        if candidate.candidate_id not in seen:
            selected.append(replace(candidate, hypothesis="deterministic fallback selection"))
            seen.add(candidate.candidate_id)
    return selected


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
        user=(
            f"Change: {candidate.mutation}. Initial hypothesis: "
            f"{candidate.hypothesis or 'not supplied'}. Backtest: {result.summary()}"
        ),
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
    training_window: tuple[str, str] | None = None
    validation_window: tuple[str, str] | None = None
    multiple_testing_trials: int = 0

    def summary(self) -> str:
        lines = [f"research lab: {self.tested} candidates tested, {len(self.promoted)} promoted"]
        if self.incumbent:
            lines.append(f"  incumbent  {self.incumbent.summary()}")
        if self.training_window and self.validation_window:
            lines.append(
                f"  selected on {self.training_window[0]}..{self.training_window[1]}, "
                f"validated on {self.validation_window[0]}..{self.validation_window[1]}"
            )
        if self.multiple_testing_trials:
            lines.append(
                f"  promotion bar accounts for {self.multiple_testing_trials} cumulative trials"
            )
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
    gate: PromotionGate | None = None,
    provider: LLMProvider | None = None,
    seed: int | None = None,
    asof: date | None = None,
    holdout_fraction: float = 0.25,
    embargo_days: int = 35,
    prior_trials: int = 0,
) -> LabReport:
    """Select mutations on history and promote only those that survive a holdout.

    The chronological holdout is separated from selection by an embargo longer
    than the traded tenor.  That keeps an option structure opened in the training
    window from leaking its later marks into validation.
    """
    baseline = baseline or CondorSpec()
    gate = gate or PromotionGate()
    provider = provider or NullProvider()
    rng = random.Random(seed)
    asof = asof or date.today()

    start = asof - timedelta(days=lookback_days)
    # Stop short of today: contracts must have expired to be listed and priced.
    end = asof - timedelta(days=30)

    # Select hypotheses before loading option history so the loader can derive a
    # complete contract universe for exactly those geometries. The model still
    # sees no outcomes; it only prioritises the predeclared candidate family.
    field_ = hypothesize(provider, baseline, candidates, rng)

    history = load_history(
        cli, underlying, start=start, end=end,
        expiries=expiries,
        universe_specs=[baseline, *(candidate.spec for candidate in field_)],
    )
    report = LabReport()
    if not history.bars:
        log.warning("no option history for %s; the lab cannot run tonight", underlying)
        return report

    sessions = history.sessions()
    if len(sessions) < 20:
        log.warning("history has only %d sessions; not enough for a holdout", len(sessions))
        return report
    split_index = min(max(int(len(sessions) * (1 - holdout_fraction)), 1), len(sessions) - 1)
    validation_start = sessions[split_index]
    training_end = validation_start - timedelta(days=embargo_days)
    training = history.between(end=training_end)
    validation = history.between(start=validation_start)
    if len(training.sessions()) < 10 or len(validation.sessions()) < 10:
        log.warning("history cannot support a chronological holdout")
        return report

    report.training_window = (str(training.sessions()[0]), str(training.sessions()[-1]))
    report.validation_window = (str(validation.sessions()[0]), str(validation.sessions()[-1]))
    report.incumbent = simulate_condors(
        training, baseline, strategy_id="CARRY(incumbent/train)"
    )
    validation_incumbent = simulate_condors(
        validation, baseline, strategy_id="CARRY(incumbent/holdout)"
    )
    log.info("incumbent: %s", report.incumbent.summary())

    report.tested = len(field_)
    report.multiple_testing_trials = max(prior_trials, 0) + report.tested

    for candidate in field_:
        selected = simulate_condors(
            training, candidate.spec, strategy_id=f"{candidate.candidate_id}/train"
        )
        passed, reason = gate.evaluate(
            selected, report.incumbent, report.multiple_testing_trials
        )
        if not passed:
            report.rejected.append((candidate, selected, f"selection: {reason}"))
            continue

        held_out = simulate_condors(
            validation, candidate.spec, strategy_id=f"{candidate.candidate_id}/holdout"
        )
        passed, holdout_reason = gate.evaluate(
            held_out, validation_incumbent, report.multiple_testing_trials
        )
        if not passed:
            report.rejected.append((candidate, held_out, f"holdout: {holdout_reason}"))
            continue

        reason = (
            f"selection survived ({reason}); holdout survived ({holdout_reason}). "
            f"{narrate(provider, candidate, held_out)}"
        )
        report.promoted.append((candidate, held_out, reason))

    log.info("%s", report.summary())
    return report


def promotion_records(
    report: LabReport, *, underlying: str, hired_at: datetime | None = None
) -> list[dict]:
    """Stable, serialisable roster rows for candidates the gate actually hired."""
    hired_at = hired_at or datetime.now(timezone.utc)
    rows = []
    for candidate, result, reason in report.promoted:
        spec = asdict(candidate.spec)
        identity = json.dumps(
            {"underlying": underlying, "spec": spec}, sort_keys=True, separators=(",", ":")
        )
        strategy_id = f"LAB-{hashlib.sha256(identity.encode()).hexdigest()[:8].upper()}"
        rows.append({
            "strategy_id": strategy_id,
            "candidate_id": candidate.candidate_id,
            "parent": candidate.parent,
            "underlying": underlying,
            "mutation": candidate.mutation,
            "hypothesis": candidate.hypothesis,
            "spec": spec,
            "backtest": {
                "trades": result.n,
                "wins": result.wins,
                "edge": round(result.edge, 6),
                "t_stat": round(result.t_stat, 4),
                "total_pnl": round(result.total_pnl, 2),
                "max_drawdown": round(result.max_drawdown, 2),
                "diagnostics": dict(result.diagnostics),
            },
            "reason": reason,
            "status": "probation",
            "hired_at": hired_at.isoformat(),
            "training_window": report.training_window,
            "validation_window": report.validation_window,
            "multiple_testing_trials": report.multiple_testing_trials,
        })
    return rows
