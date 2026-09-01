"""Would each strategy actually trade today? Ask before the bell, not after.

This exists because of a specific failure. `CONVEX` shipped with a gate that only
bought volatility below 0.98x realised. It was tested at 0.94x -- the ratio that
morning -- and passed. Overnight the ratio moved to 1.03x, the gate closed, and
the sleeve sat inert through the one session all week that actually moved. The
code was correct, the tests passed, the container was healthy, and the strategy
did nothing. Nothing anywhere reported a problem, because nothing *was* wrong in
the sense any existing check understood.

The general shape: **a strategy that silently declines to act looks identical to
a strategy with nothing to do.** Every other check on this desk verifies that
things work when they fire. This one asks whether they would fire at all, under
the conditions that exist right now, and makes each one say why not.

Run it before the open. It places no orders and touches no ledger.

    PYTHONPATH=src python scripts/preopen.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")

from aperture.alpaca_cli import AlpacaCLI
from aperture.loop import _market_conditions, build_book, build_strategies
from aperture.marketdata import MarketData
from aperture.risk import analyse_payoff
from aperture.state import DeskState


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="  %(levelname)s %(message)s")
    # Must read the SAME ledger the desk trades from. Against an empty local
    # ledger every strategy looks willing, because the "already holding this
    # name" checks all pass -- which would make this check confidently wrong.
    ledger = sys.argv[1] if len(sys.argv) > 1 else "state/judged.json"
    cli = AlpacaCLI()
    md = MarketData(cli)
    state = DeskState.load(ledger)
    book = build_book(cli, state, datetime.now(timezone.utc))
    conditions = _market_conditions(md, state, book)

    print("=" * 68)
    print("PRE-OPEN: would each strategy actually trade under today's market?")
    print("=" * 68)
    print(f"  ledger      {ledger}  ({len(state.open_trades)} open trades)")
    print(f"  equity      ${book.equity:,.2f}")
    print(f"  open risk   ${book.total_open_risk:,.2f}")
    print(f"  IV/realised {conditions.get('implied / realised', 'n/a')}")
    print()

    strategies = build_strategies(state)
    silent: list[str] = []

    for strategy in strategies:
        sid = strategy.config.strategy_id

        # Hand each strategy the inputs the live loop would hand it, so this
        # measures the real decision rather than a hypothetical one.
        if hasattr(strategy, "posture"):
            strategy.posture = "balanced"
        if hasattr(strategy, "iv_to_realised"):
            strategy.iv_to_realised = conditions.get("iv_to_realised")

        budget = 5_000.0
        try:
            proposals = strategy.propose(md, book, budget)
        except Exception as exc:  # noqa: BLE001 - a broken strategy is a finding
            print(f"  {sid:8} RAISED {type(exc).__name__}: {str(exc)[:60]}")
            silent.append(sid)
            continue

        if not proposals:
            print(f"  {sid:8} would propose NOTHING")
            silent.append(sid)
            continue

        for proposal in proposals:
            profile = analyse_payoff(proposal)
            loss = profile.max_loss
            print(f"  {sid:8} {proposal.underlying:5} qty {proposal.qty:3d} "
                  f"@ {proposal.net_price:6.2f}  max loss "
                  f"{('$%.0f' % loss) if loss is not None else 'UNBOUNDED'}")

    print()
    if silent:
        print(f"  SILENT TODAY: {', '.join(silent)}")
        print()
        print("  A silent strategy is not automatically wrong -- CRUSH is meant to")
        print("  be quiet on a day with no event. But each one should be silent for")
        print("  a reason you can name. If you cannot name it, that is the finding.")
    else:
        print("  Every funded strategy would act on today's market.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
