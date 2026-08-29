"""Checks that a change in one place was carried through to the others.

This module exists because of a specific, repeated failure. The agent layer was
built, the agents made real decisions, and those decisions were audited -- but
the public projection stripped every field carrying the *reasoning*, and the
dashboard had no renderer for the new events. For days the demo URL showed a
tape that looked like a deterministic system, and nobody noticed, because
nothing failed. Every test passed. The desk traded correctly. The only symptom
was a page quietly saying less than it should have.

That is the failure mode this guards: **the parts of the project that must
change together are in different languages, different directories, and
different deploy targets, so nothing forces them to move at the same time.**

The rule these checks encode is: *if adding a thing in one place obliges you to
touch another place, that obligation should be executable.* A comment asking
future maintainers to remember is not executable. A failing test is.

Run standalone for a readable report:

    python -m aperture.consistency

The same checks run inside the normal suite (`tests/test_consistency.py`), so a
missed propagation fails `pytest` rather than waiting to be noticed in review.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Where things live
# --------------------------------------------------------------------------- #

PKG = Path(__file__).resolve().parent
REPO = PKG.parent.parent
APP_JS = REPO / "dashboard" / "site" / "app.js"
README = REPO / "README.md"

# Outside the repo: present when working locally, absent in the container and
# in any clone of the public repo. Checks that need them skip rather than fail.
WRITEUP = REPO / "submission" / "One-Page-Writeup.md"
HANDOFF = REPO.parent / "_private" / "AGENT_HANDOFF.md"


# --------------------------------------------------------------------------- #
# The audit event registry
# --------------------------------------------------------------------------- #

# Every event the desk can record has to be declared here as either:
#
#   "public"   -- it belongs on the evidence surface a judge or the public sees.
#                 Its distinguishing fields must survive snapshot redaction AND
#                 the dashboard must know how to render it.
#   "internal" -- operator diagnostics. It may reach the tape, but it carries no
#                 reasoning a reader needs, so a generic rendering is correct.
#
# Adding an audit event without adding it here fails the suite. That is the
# point: the failure arrives at the moment the event is written, not weeks later
# when someone happens to look at the page.
EVENT_REGISTRY: dict[str, str] = {
    # --- what the agents decided: the reason this desk is interesting --------
    "universe": "public",                 # which names the scout chose, and why
    "regime": "public",                   # the posture the desk took for the day
    "red_team_kill": "public",            # the objection that removed a trade
    "red_team_budget_spent": "public",    # proposals that went unchallenged
    "mcp_research": "public",             # what the agents looked at
    # --- the desk reasoning about itself ------------------------------------
    "execution_adapted": "public",        # noticing it was not getting filled
    "allocation": "public",               # capital moving on evidence
    "hired": "public",                    # a researched strategy earning capital
    "research_complete": "public",        # a night of the lab, promoted or not
    # --- risk authority ------------------------------------------------------
    "breach": "public",                   # a circuit breaker firing
    "kill_switch": "public",
    "kill_switch_released": "public",
    # --- execution bookkeeping: real, but it carries no argument to read -----
    "entry_reserved": "internal",
    "entry_submitted": "internal",
    "entry_filled": "internal",
    "entry_unfilled": "internal",
    "entry_abandoned": "internal",
    "entry_cancel_requested": "internal",
    "entry_cancel_failed": "internal",
    "entry_submission_uncertain": "internal",
    "close_reserved": "internal",
    "close_submitted": "internal",
    "close_unfilled": "internal",
    "close_submission_uncertain": "internal",
    "submission_recovered": "internal",
    "submission_recovery_failed": "internal",
    "fill_price_missing": "internal",
    "order_sync_error": "internal",
    "reconcile": "internal",
    "position_mismatch": "internal",
    "cycle_error": "internal",
    "letter_written": "internal",
    "letter_error": "internal",
    "research_error": "internal",
}

# Fields every event may carry; they say *what happened*, not *why*. An event
# whose meaning lives only in a field outside this set needs its own renderer,
# or it arrives on the page as "Decision recorded."
GENERIC_FIELDS = frozenset({"ts", "event", "strategy", "underlying", "summary",
                            "rationale", "reason"})


@dataclass(frozen=True)
class Finding:
    check: str
    detail: str
    fix: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.detail}\n    -> {self.fix}"


# --------------------------------------------------------------------------- #
# Reading the code rather than trusting a list
# --------------------------------------------------------------------------- #


def emitted_events() -> dict[str, set[str]]:
    """Every `*.audit.record("name", **fields)` in the package, by AST.

    Regex was tried first and quietly missed the multi-line calls -- including
    ``red_team_kill``, the single most important event on the tape.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(PKG.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "record"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            name = node.args[0].value
            if not isinstance(name, str):
                continue
            fields = {kw.arg for kw in node.keywords if kw.arg}
            found.setdefault(name, set()).update(fields)
    return found


def public_fields() -> set[str]:
    """The snapshot's allowlist, read from the source of truth."""
    from .snapshot import AUDIT_FIELDS

    return set(AUDIT_FIELDS)


def rendered_events() -> set[str]:
    """Events the dashboard knows how to describe."""
    if not APP_JS.exists():
        return set()
    return set(re.findall(r'case "([a-z_]+)":', APP_JS.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #


def test_function_count() -> int:
    """How many test functions exist on disk, by AST.

    Used only to tell a full-suite run from a subset run: pytest's collected
    count exceeds this when parametrised cases expand, and falls far below it
    when someone runs a single file.
    """
    tests_dir = REPO / "tests"
    if not tests_dir.exists():
        return 0
    total = 0
    for path in tests_dir.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        total += sum(
            1 for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    return total


def check_events_are_registered() -> list[Finding]:
    findings = []
    emitted = emitted_events()
    for event in sorted(set(emitted) - set(EVENT_REGISTRY)):
        findings.append(Finding(
            "event-registry",
            f"the desk records {event!r} but it is not declared",
            "add it to EVENT_REGISTRY in consistency.py as 'public' or 'internal'",
        ))
    return findings


def renderer_field_uses() -> dict[str, set[str]]:
    """Which ``decision.<field>`` each ``case "event":`` branch reads.

    This is what makes the check a chain rather than a name match: a renderer
    reading a field the snapshot strips silently renders ``undefined``.
    """
    if not APP_JS.exists():
        return {}
    text = APP_JS.read_text(encoding="utf-8")
    block = text[text.find("function decisionText("):]
    block = block[: block.find("\nfunction ")] or block
    uses: dict[str, set[str]] = {}
    parts = re.split(r'case "([a-z_]+)":', block)
    for name, body in zip(parts[1::2], parts[2::2]):
        uses.setdefault(name, set()).update(re.findall(r"decision\.([a-zA-Z_]+)", body))
    return uses


def check_public_events_survive_redaction() -> list[Finding]:
    """A public event whose reasoning is filtered out is worse than no event:
    the page shows that something happened and hides what it was."""
    findings = []
    allowed = public_fields()
    uses = renderer_field_uses()
    for event, fields in sorted(emitted_events().items()):
        if EVENT_REGISTRY.get(event) != "public":
            continue
        # Only the fields the dashboard actually reads have to survive. An
        # event may record more for the operator's audit trail than it shows.
        needed = (uses.get(event, set()) & fields) or set()
        lost = sorted(f for f in needed - allowed if f not in GENERIC_FIELDS)
        if lost:
            findings.append(Finding(
                "snapshot-redaction",
                f"the dashboard reads {', '.join(lost)} from {event!r}, "
                "but the public projection drops those fields",
                "add them to AUDIT_FIELDS in snapshot.py",
            ))
    return findings


def check_renderers_read_real_fields() -> list[Finding]:
    """A renderer reading a field nothing records is dead code that silently
    falls through to the generic text."""
    findings = []
    emitted = emitted_events()
    for event, used in sorted(renderer_field_uses().items()):
        if event not in emitted:
            findings.append(Finding(
                "dashboard-renderer",
                f"app.js renders {event!r} but nothing records that event",
                "remove the case, or record the event in the trading path",
            ))
            continue
        unknown = sorted(used - emitted[event] - GENERIC_FIELDS)
        if unknown:
            findings.append(Finding(
                "dashboard-renderer",
                f"app.js reads {', '.join(unknown)} from {event!r}, "
                "which that event never records",
                f"record the field, or stop reading it in decisionText()",
            ))
    return findings


def check_public_events_render() -> list[Finding]:
    """The exact miss this module was written for."""
    findings = []
    if not APP_JS.exists():
        return findings
    renderers = set(renderer_field_uses())
    for event, fields in sorted(emitted_events().items()):
        if EVENT_REGISTRY.get(event) != "public":
            continue
        if fields <= GENERIC_FIELDS:
            continue  # a generic rendering already says everything it knows
        if event not in renderers:
            findings.append(Finding(
                "dashboard-renderer",
                f"{event!r} reaches the public tape but has no renderer, "
                'so it displays as "Decision recorded."',
                f'add a case "{event}" to decisionText() in dashboard/site/app.js',
            ))
    return findings


def check_modules_are_documented() -> list[Finding]:
    """Every module should be findable from the README's repository map."""
    findings = []
    if not README.exists():
        return findings
    text = README.read_text(encoding="utf-8")
    for path in sorted(PKG.glob("*.py")):
        if path.stem in {"__init__", "__main__", "consistency"}:
            continue
        if path.name not in text:
            findings.append(Finding(
                "module-docs",
                f"{path.name} is not in the README repository map",
                "add a one-line entry under '## Repository map' in README.md",
            ))
    return findings


def check_agents_are_documented() -> list[Finding]:
    """The agents are the premise of the project; a doc that omits one is
    describing a system that does not exist."""
    findings = []
    roles = {
        "choose_universe": ("scout",),
        "call_regime": ("regime",),
        "red_team": ("red team", "red-team"),
        "rank_proposals": ("portfolio manager", "conviction"),
    }
    targets = [("README.md", README), ("One-Page-Writeup.md", WRITEUP)]
    for label, path in targets:
        if not path.exists():
            continue  # not present in a public clone; nothing to check
        text = path.read_text(encoding="utf-8").lower()
        for func, aliases in roles.items():
            if not any(alias in text for alias in aliases):
                findings.append(Finding(
                    "agent-docs",
                    f"{label} never mentions the agent behind {func}()",
                    f"describe it -- one of: {', '.join(aliases)}",
                ))
    return findings


def check_documented_test_count(actual: int | None = None) -> list[Finding]:
    """Docs quote a test count; a stale one reads as carelessness to a judge."""
    findings = []
    if actual is None:
        return findings
    for label, path in [("README.md", README), ("One-Page-Writeup.md", WRITEUP),
                        ("AGENT_HANDOFF.md", HANDOFF)]:
        if not path.exists():
            continue
        for claimed in re.findall(r"(\d+)\s+(?:Python\s+)?tests", path.read_text(encoding="utf-8")):
            if int(claimed) != actual:
                findings.append(Finding(
                    "test-count",
                    f"{label} claims {claimed} tests; the suite has {actual}",
                    f"update the number in {label}",
                ))
    return findings


def check_declared_dependencies_are_installed() -> list[Finding]:
    """A dependency that runs locally but is missing from the image fails only
    in production -- which for this project means during the scored window."""
    findings = []
    pyproject = REPO / "pyproject.toml"
    dockerfile = REPO / "Dockerfile"
    if not (pyproject.exists() and dockerfile.exists()):
        return findings
    block = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject.read_text(encoding="utf-8"), re.S)
    if not block:
        return findings
    docker = dockerfile.read_text(encoding="utf-8")
    for dep in re.findall(r'"([A-Za-z0-9_.-]+)', block.group(1)):
        if dep.lower() not in docker.lower():
            findings.append(Finding(
                "docker-deps",
                f"{dep} is a runtime dependency but is not installed in the Dockerfile",
                f"add {dep} to the pip install layer in Dockerfile",
            ))
    return findings


ALL_CHECKS = (
    check_events_are_registered,
    check_public_events_survive_redaction,
    check_public_events_render,
    check_renderers_read_real_fields,
    check_modules_are_documented,
    check_agents_are_documented,
    check_declared_dependencies_are_installed,
)


def run_all(test_count: int | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for check in ALL_CHECKS:
        findings.extend(check())
    findings.extend(check_documented_test_count(test_count))
    return findings


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="check the project agrees with itself")
    parser.add_argument("--tests", type=int, default=None,
                        help="current test count, to verify what the docs claim")
    args = parser.parse_args(argv)

    findings = run_all(args.tests)
    print("APERTURE CONSISTENCY")
    print("=" * 66)
    if not findings:
        print("\nEverything that must move together has moved together.\n")
        return 0
    print(f"\n{len(findings)} thing(s) did not get carried through:\n")
    for finding in findings:
        print(f"  {finding}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
