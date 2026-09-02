"""Which configuration does the evidence actually support?

The evidence study answered "does the incumbent work" (yes: +7.8% of risk over
126 trades, t=3.24). This asks the next question: is the incumbent the *best*
configuration the data supports, or merely the first one we tried?

Method matters here more than the answer. Sweeping parameters and reporting the
winner is how people fool themselves -- test twenty variants and one will look
excellent by chance. So:

* every candidate is scored on a **training window only**;
* the winner is then checked once on an **embargoed holdout** it never touched;
* the significance bar rises with the number of variants tried, via the
  sqrt(1 + log k) correction the desk's promotion gate already uses;
* results are pooled across four underlyings, so a configuration has to work in
  more than one market rather than fitting one name's quirks.

A sweep that finds nothing is a real result. The incumbent is already known to
work; the bar for replacing it is evidence that something else works *better*.

    PYTHONPATH=src python scripts/sweep.py
"""

from __future__ import annotations

import itertools
import logging
import math
import pathlib
import pickle
import sys
from dataclasses import replace
from datetime import date, timedelta

sys.path.insert(0, "src")

from aperture.backtest import CondorSpec, OptionHistory, simulate_condors

CACHE = pathlib.Path("state/history")
SPLIT = date(2025, 10, 1)
EMBARGO_DAYS = 35


def t_stat(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return (mean / (sd / math.sqrt(n))) if sd > 0 else 0.0


def window(src: OptionHistory, lo: date, hi: date) -> OptionHistory:
    spot = {d: v for d, v in src.spot.items() if lo <= d <= hi}
    bars = {}
    for sym, series in src.bars.items():
        kept = {d: v for d, v in series.items() if lo <= d <= hi}
        if kept:
            bars[sym] = kept
    out = OptionHistory(underlying=src.underlying, spot=spot, bars=bars)
    object.__setattr__(out, "_contracts_index", {})
    return out


def load_all() -> dict[str, OptionHistory]:
    """Every cached underlying, largest cache per symbol."""
    best: dict[str, pathlib.Path] = {}
    for path in CACHE.glob("*.pkl"):
        symbol = path.name.split("-")[0]
        if symbol not in best or path.stat().st_size > best[symbol].stat().st_size:
            best[symbol] = path
    out = {}
    for symbol, path in sorted(best.items()):
        try:
            history = pickle.loads(path.read_bytes())
            object.__setattr__(history, "_contracts_index", {})
            out[symbol] = history
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol}: cache unreadable ({type(exc).__name__})")
    return out


def score(histories: dict[str, OptionHistory], spec: CondorSpec,
          lo: date, hi: date) -> tuple[int, float, float]:
    """Pooled trades, edge on risk, and t across every underlying."""
    pnls: list[float] = []
    risk = 0.0
    for history in histories.values():
        result = simulate_condors(window(history, lo, hi), spec, strategy_id="S")
        pnls += [t.pnl for t in result.trades]
        risk += result.total_risk
    edge = (sum(pnls) / risk) if risk else 0.0
    return len(pnls), edge, t_stat(pnls)


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    histories = load_all()
    if not histories:
        print("  no cached history; run scripts/evidence.py first")
        return 1

    start = min(min(h.spot) for h in histories.values())
    end = max(max(h.spot) for h in histories.values())
    hold_from = SPLIT + timedelta(days=EMBARGO_DAYS)

    print("=" * 78)
    print("WHICH CONFIGURATION DOES THE EVIDENCE SUPPORT?")
    print(f"underlyings {', '.join(histories)}   window {start} -> {end}")
    print(f"train {start}..{SPLIT}   embargo {EMBARGO_DAYS}d   holdout {hold_from}..{end}")
    print("=" * 78)

    base = CondorSpec()
    n, edge, t = score(histories, base, start, SPLIT)
    print(f"\n  INCUMBENT (train): {n:4d} trades  edge {edge:+6.1%}  t={t:5.2f}\n")

    # A deliberately small, pre-declared grid. Three knobs, three values each:
    # far enough apart to mean something, few enough to keep the multiple-
    # testing penalty honest.
    grid = list(itertools.product(
        (0.030, 0.040, 0.050),        # short_pct: how far out the shorts sit
        (0.15, 0.18, 0.22),           # min_credit_to_width: the price floor
        (10, 14, 21),                 # dte_target
    ))
    print(f"  sweeping {len(grid)} configurations on the TRAINING window only")
    print(f"  {'short':>6} {'floor':>6} {'dte':>4}  {'trades':>7} {'edge':>8} {'t':>6}")
    print("  " + "-" * 44)

    rows = []
    for short_pct, floor, dte in grid:
        spec = replace(base, short_pct=short_pct, min_credit_to_width=floor,
                       dte_target=dte)
        n, edge, t = score(histories, spec, start, SPLIT)
        rows.append((t, n, edge, spec, short_pct, floor, dte))
        flag = "  <-- incumbent" if (short_pct, floor, dte) == (0.04, 0.15, 14) else ""
        print(f"  {short_pct:6.3f} {floor:6.2f} {dte:4d}  {n:7d} {edge:+8.1%} {t:6.2f}{flag}")

    # Only configurations with enough observations to mean anything.
    viable = [r for r in rows if r[1] >= 30]
    if not viable:
        print("\n  nothing cleared the 30-trade minimum. No candidate to promote.")
        return 0

    viable.sort(reverse=True)
    best_t, best_n, best_edge, best_spec, sp, fl, dte = viable[0]

    # The bar rises with how many variants were tried.
    k = len(grid)
    bar = 2.0 * math.sqrt(1 + math.log(k))
    print(f"\n  best on training : short {sp:.3f} floor {fl:.2f} dte {dte}")
    print(f"                     {best_n} trades  edge {best_edge:+.1%}  t={best_t:.2f}")
    print(f"  bar after {k} trials: t >= {bar:.2f}  (2.0 x sqrt(1+ln k))")

    if best_t < bar:
        print("\n  VERDICT: the best variant does not clear the multiple-testing bar.")
        print("           Sweeping 27 configurations will always surface a winner;")
        print("           this one is not distinguishable from that. Keep the")
        print("           incumbent -- it is already known to work.")
        return 0

    hn, hedge, ht = score(histories, best_spec, hold_from, end)
    print(f"\n  HOLDOUT (never touched): {hn} trades  edge {hedge:+.1%}  t={ht:.2f}")
    if hn >= 20 and hedge > 0 and ht >= 1.5:
        print("\n  VERDICT: clears the trials bar AND survives an embargoed holdout.")
        print("           This is a real candidate to replace the incumbent.")
    else:
        print("\n  VERDICT: cleared training, failed the holdout. This is what")
        print("           overfitting looks like from the inside. Keep the incumbent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
