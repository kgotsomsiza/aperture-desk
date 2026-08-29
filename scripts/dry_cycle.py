"""Run one complete decision cycle without submitting anything.

The runner refuses to think while the market is shut, which is correct for
trading and unhelpful for verification: it means the full path -- research,
agents, strategies, red team, portfolio manager, Warden -- had never once been
exercised end to end before the first scored session.

This drives `run_cycle` directly with `dry_run=True`. Everything runs except the
order submission. Use it to prove the desk can think before trusting it to act.

    PYTHONPATH=src python scripts/dry_cycle.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, "src")

from aperture.alpaca_cli import AlpacaCLI
from aperture.llm import build_provider
from aperture.loop import build_book, build_strategies, run_cycle
from aperture.marketdata import MarketData
from aperture.state import DeskState, audit_path_for
from aperture.warden import AuditLog, RiskWarden
from aperture.risk import RiskLimits


class OpenMarketCLI(AlpacaCLI):
    """The real CLI, with one lie: it says the market is open.

    `run_cycle` returns immediately when the clock is shut, which is right for
    trading and means the decision path cannot be rehearsed. Only `clock()` is
    overridden -- account, positions, chains and quotes are all live. Paired
    with `dry_run=True` nothing can be submitted: the guard in
    `_submit_if_approved` returns before the reservation and before any call to
    `submit_mleg`.
    """

    def clock(self) -> dict:
        real = super().clock()
        return {**real, "is_open": True, "simulated": True}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="  %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ledger = DeskState.load()
    audit = AuditLog(audit_path_for(ledger.path))
    cli = OpenMarketCLI()
    md = MarketData(cli)

    print("=" * 70)
    print("DRY CYCLE -- every decision runs, nothing is submitted")
    print("  clock is simulated open; all other data is live")
    print("=" * 70)

    book = build_book(cli, ledger, datetime.now(timezone.utc))
    warden = RiskWarden(
        limits=RiskLimits(),
        audit=audit,
        start_equity=ledger.start_equity or book.equity,
        budgets={"CARRY": book.equity * 0.18,
                 "CRUSH": book.equity * 0.05,
                 "DRIFT": book.equity * 0.05},
    )
    strategies = build_strategies(ledger)
    provider = build_provider()
    print(f"provider   : {type(provider).__name__}")
    print(f"equity     : ${book.equity:,.2f}")
    print(f"strategies : {[type(s).__name__ for s in strategies]}\n")

    started = time.perf_counter()
    summary = run_cycle(
        cli, md, warden, ledger, strategies,
        dry_run=True, provider=provider,
    )
    elapsed = time.perf_counter() - started

    print("\n" + "=" * 70)
    print(f"CYCLE COMPLETED in {elapsed:.1f}s")
    print("=" * 70)
    for key in ("research_source", "posture", "universe", "approved", "vetoed",
                "red_team_kills", "submitted", "closed", "fill_rate", "aggression"):
        if key in summary:
            print(f"  {key:18} {summary[key]}")

    decided = summary.get("posture") or summary.get("universe")
    print()
    if decided:
        print("  The agents decided. The path works.")
    else:
        print("  WARNING: no agent decision reached the summary -- investigate")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
