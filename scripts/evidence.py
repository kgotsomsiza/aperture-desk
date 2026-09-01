"""Does the incumbent strategy actually make money? Pool enough trades to know.

The desk's own research says CARRY earns +5.5% on risk with **t = 1.02** over
**13 trades**. Thirteen observations cannot distinguish a real edge from luck;
t = 1.02 is what noise looks like. Every decision the desk makes rests on an
evidence base too thin to support it, and no amount of live trading over four
days will fix that.

The constraint is not the design -- the backtest samples up to 120 expiries. It
is that free-tier option history is sparse, so most sampled expiries yield no
usable chain. One underlying therefore produces ~13 trades over 20 months.

The fix is more underlyings, not more time. A t-statistic scales with the square
root of the sample, so pooling four liquid index ETFs turns t = 1.02 into
t ~= 2 **if the edge is real** -- and shows it collapsing toward zero if it is
not. Either answer is worth more than the one we have.

This runs the identical incumbent spec across several underlyings and pools the
trades. It touches no trading path and can run while the market is open.

    PYTHONPATH=src python scripts/evidence.py
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date, timedelta

sys.path.insert(0, "src")

from aperture.alpaca_cli import AlpacaCLI
from aperture.backtest import CondorSpec, load_history, simulate_condors

UNDERLYINGS = ("SPY", "QQQ", "IWM", "DIA")
LOOKBACK_DAYS = 900
EXPIRIES = 120


def t_stat(pnls: list[float]) -> float:
    """How many standard errors the mean sits from zero."""
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    sd = math.sqrt(var)
    return (mean / (sd / math.sqrt(n))) if sd > 0 else 0.0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="  %(levelname)s %(message)s")
    cli = AlpacaCLI()
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    spec = CondorSpec()  # the incumbent, unchanged

    print("=" * 72)
    print("IS THE EDGE REAL? pooling the incumbent across liquid index ETFs")
    print(f"window {start} -> {end}")
    print("=" * 72)

    pooled_pnl: list[float] = []
    pooled_risk = 0.0
    rows = []

    for symbol in UNDERLYINGS:
        try:
            history = load_history(
                cli, symbol, start=start, end=end,
                expiries=EXPIRIES, universe_specs=[spec],
            )
            result = simulate_condors(history, spec, strategy_id=f"CARRY/{symbol}")
        except Exception as exc:  # noqa: BLE001 - one name must not stop the study
            print(f"  {symbol:5} FAILED: {type(exc).__name__} {str(exc)[:70]}")
            continue

        pnls = [t.pnl for t in result.trades]
        risk = result.total_risk
        edge = (result.total_pnl / risk) if risk else 0.0
        rows.append((symbol, result.n, result.wins, edge, t_stat(pnls)))
        pooled_pnl += pnls
        pooled_risk += risk
        print(f"  {symbol:5} {result.n:4d} trades  {result.wins:3d} wins  "
              f"edge {edge:+6.1%}  t={t_stat(pnls):5.2f}")

    print("-" * 72)
    n = len(pooled_pnl)
    if n < 2:
        print("  not enough pooled trades to say anything")
        return 1
    edge = (sum(pooled_pnl) / pooled_risk) if pooled_risk else 0.0
    t = t_stat(pooled_pnl)
    wins = sum(1 for p in pooled_pnl if p > 0)
    print(f"  POOLED {n:4d} trades  {wins:3d} wins  edge {edge:+6.1%}  t={t:5.2f}")
    print()
    print(f"  single-underlying baseline was: 13 trades, edge +5.5%, t=1.02")
    print()
    if t >= 2.0:
        print("  VERDICT: the edge survives a real sample. t >= 2 -- this is")
        print("           evidence, not an anecdote.")
    elif t >= 1.5:
        print("  VERDICT: suggestive but short of conventional significance.")
        print("           Directionally supportive; still not proof.")
    else:
        print("  VERDICT: the edge does NOT survive a larger sample. What looked")
        print("           like +5.5% on 13 trades washes out. The honest reading is")
        print("           that this strategy has no demonstrated edge, and sizing")
        print("           it as though it does would be a mistake.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
