"""Run the production promotion gate over Alpaca's full option-history window.

This is a reproducible research artifact, not a second more-permissive backtest.
It uses the same Featherless hypothesis selector, chronological holdout, 35-day
embargo, adverse slippage, incumbent comparison, and multiple-testing correction
as the nightly desk.  The JSON report is runtime evidence and remains private.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aperture.alpaca_cli import AlpacaCLI  # noqa: E402
from aperture.llm import build_provider, provider_info  # noqa: E402
from aperture.research import promotion_records, run_lab  # noqa: E402


log = logging.getLogger("long_backtest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Aperture's held-out research lab")
    parser.add_argument("--underlying", default="SPY")
    parser.add_argument("--candidates", type=int, default=18)
    parser.add_argument("--lookback", type=int, default=900)
    parser.add_argument("--expiries", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--prior-trials", type=int, default=0)
    parser.add_argument("--out", default="state/long_backtest.json")
    args = parser.parse_args(argv)

    provider = build_provider()
    report = run_lab(
        AlpacaCLI(),
        underlying=args.underlying,
        candidates=args.candidates,
        lookback_days=args.lookback,
        expiries=args.expiries,
        provider=provider,
        seed=args.seed,
        asof=date.today(),
        prior_trials=args.prior_trials,
    )

    rejected = [
        {
            "candidate": candidate.candidate_id,
            "mutation": candidate.mutation,
            "hypothesis": candidate.hypothesis,
            "spec": asdict(candidate.spec),
            "trades": result.n,
            "edge": round(result.edge, 6),
            "t_stat": round(result.t_stat, 4),
            "diagnostics": dict(result.diagnostics),
            "reason": reason,
        }
        for candidate, result, reason in report.rejected
    ]
    payload = {
        "as_of": date.today().isoformat(),
        "underlying": args.underlying,
        "provider": provider_info(provider),
        "tested": report.tested,
        "cumulative_trials": report.multiple_testing_trials,
        "training_window": report.training_window,
        "validation_window": report.validation_window,
        "incumbent": report.incumbent.summary() if report.incumbent else None,
        "promoted": promotion_records(report, underlying=args.underlying),
        "rejected": rejected,
        "summary": report.summary(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(out)

    log.info("%s", report.summary())
    log.info("wrote %s", out)
    return 0 if report.tested else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    raise SystemExit(main())
