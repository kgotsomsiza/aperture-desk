"""Plain-English status -- `python -m aperture.status`.

The desk already emits a running log, an append-only audit trail, a ledger and a
public snapshot. Between them they hold everything, and none of them answers the
question a person actually asks: *what is it doing, and is it working?*

This reads all of those and says so in words. It is read-only: it never places,
cancels or modifies anything, so it is always safe to run, including while the
runner is mid-cycle.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .alpaca_cli import AlpacaCLI, AlpacaCliError
from .state import DeskState, audit_path_for
from .warden import AuditLog

BAR = "-" * 66


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def heading(title: str) -> None:
    print(f"\n{title}\n{BAR}")


# --------------------------------------------------------------------------- #


def account_section(cli: AlpacaCLI, state: DeskState) -> float:
    heading("THE MONEY")
    try:
        account = cli.account()
    except AlpacaCliError as exc:
        print(f"  could not read the account: {exc.stderr[:120]}")
        return 0.0

    equity = float(account.get("equity") or 0)
    cash = float(account.get("cash") or 0)
    start = state.start_equity or equity
    day_start = state.day_start_equity or equity
    peak = state.high_water_mark or equity

    print(f"  Equity now          {money(equity)}")
    print(f"  Cash                {money(cash)}")
    if start:
        print(f"  Since it started    {pct((equity / start - 1) * 100)}   (from {money(start)})")
    if day_start:
        print(f"  Today               {pct((equity / day_start - 1) * 100)}")
    if peak and equity < peak:
        print(f"  Below its best by   {pct(-(peak - equity) / peak * 100)}   (peak {money(peak)})")
    return equity


def fill_section(cli: AlpacaCLI) -> None:
    """Did the orders it placed actually get filled?

    The single most important operational number in the first hour. A desk whose
    orders expire unfilled is not trading, however busy its log looks.
    """
    heading("DID ITS ORDERS ACTUALLY FILL?")
    try:
        orders = [o for o in cli.orders(status="all") if o.get("order_class") == "mleg"]
    except AlpacaCliError as exc:
        print(f"  could not read orders: {exc.stderr[:120]}")
        return

    if not orders:
        print("  No multi-leg orders yet.")
        return

    tally = Counter(o.get("status", "?") for o in orders)
    filled = tally.get("filled", 0)
    total = len(orders)
    rate = filled / total * 100 if total else 0.0

    print(f"  Orders placed       {total}")
    for status, count in tally.most_common():
        print(f"    {status:<16}{count}")
    print(f"  Fill rate           {rate:.0f}%")

    if rate < 55 and total >= 5:
        print()
        print("  ! Below 55%. The desk is asking for prices the market is not")
        print("    meeting, so it is deploying less capital than it intends.")
        print("    Fix: raise `aggression` in strategies/base.concede.")
    elif total >= 5:
        print("  ok Healthy. Most of what it decides to do actually happens.")


def positions_section(cli: AlpacaCLI, state: DeskState) -> None:
    heading("WHAT IT IS HOLDING")
    try:
        broker = cli.positions()
    except AlpacaCliError as exc:
        print(f"  could not read positions: {exc.stderr[:120]}")
        return

    if not state.open_trades and not broker:
        print("  Nothing open. Either it has not started, the market is shut,")
        print("  or every candidate it looked at failed a risk check.")
        return

    unrealized = defaultdict(float)
    for position in broker:
        unrealized[position.get("symbol", "")] = float(position.get("unrealized_pl") or 0)

    for trade in state.open_trades.values():
        pnl = sum(unrealized.get(leg, 0.0) for leg in trade.legs)
        shape = "credit received" if trade.net_price < 0 else "paid up front"
        print(f"  {trade.strategy_id:<7}{trade.underlying:<6} x{trade.qty}")
        print(f"    {trade.rationale[:62]}")
        print(f"    worst case {money(trade.max_loss)} | {shape} "
              f"{money(abs(trade.net_price) * trade.qty * 100)} | now {money(pnl)}")

    total_risk = sum(t.max_loss for t in state.open_trades.values())
    print(f"\n  Most it can lose on everything open: {money(total_risk)}")
    print(f"  Broker shows {len(broker)} option legs across "
          f"{len(state.open_trades)} structures.")


def capital_section(state: DeskState, audit: AuditLog) -> None:
    heading("WHERE THE CAPITAL IS")
    rows = audit.tail(limit=4000, event="allocation")
    if not rows:
        print("  No reallocation yet -- still on its designed weights.")
        print("  CARRY 64% | CRUSH 18% | DRIFT 18%")
        return

    latest = rows[-1]
    weights = latest.get("weights") or {}
    budgets = latest.get("budgets") or {}
    reasons = latest.get("reasons") or {}
    for name in sorted(weights, key=lambda k: -weights[k]):
        state_word = "FIRED" if weights[name] == 0 else ""
        print(f"  {name:<8}{weights[name]:>6.1%}  {money(budgets.get(name, 0)):>12}  {state_word}")
        if reasons.get(name):
            print(f"           {reasons[name][:58]}")


def decisions_section(audit: AuditLog, limit: int) -> None:
    heading(f"ITS LAST {limit} DECISIONS")
    rows = [r for r in audit.tail(limit=2000)
            if r.get("event") in {"approval", "veto", "submitted", "entry_filled", "closed"}]
    if not rows:
        print("  Nothing decided yet.")
        return

    words = {
        "approval": "allowed",
        "veto": "REFUSED",
        "submitted": "ordered",
        "entry_filled": "FILLED",
        "closed": "closed",
    }
    for row in rows[-limit:]:
        when = str(row.get("ts", ""))[11:16]
        event = words.get(row.get("event", ""), row.get("event", ""))
        who = f"{row.get('strategy', '?')} {row.get('underlying', '')}".strip()
        print(f"  {when}  {event:<8} {who}")
        if row.get("event") == "veto":
            for reason in (row.get("reasons") or [])[:2]:
                print(f"           why: {reason.get('gate')} -- {str(reason.get('reason'))[:52]}")
        elif row.get("rationale"):
            print(f"           {str(row['rationale'])[:58]}")


def veto_summary(audit: AuditLog) -> None:
    heading("WHY IT SAID NO")
    vetoes = audit.tail(limit=4000, event="veto")
    if not vetoes:
        print("  It has not refused anything yet.")
        return
    gates = Counter()
    for row in vetoes:
        for reason in row.get("reasons") or []:
            gates[str(reason.get("gate", "?")).split("[")[0]] += 1
    print(f"  {len(vetoes)} refusals. The gates doing the work:")
    for gate, count in gates.most_common(6):
        print(f"    {count:>4}  {gate}")
    print("\n  Refusals are the system working. A desk that never refuses")
    print("  anything is not checking.")


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plain-English desk status")
    parser.add_argument("--state", default="state/judged.json")
    parser.add_argument("--audit", default=None, help="defaults to <state dir>/audit.jsonl")
    parser.add_argument("--decisions", type=int, default=10)
    args = parser.parse_args(argv)

    state_path = Path(args.state)
    audit_path = Path(args.audit) if args.audit else audit_path_for(state_path)

    try:
        state = DeskState.load(state_path)
    except RuntimeError as exc:
        print(f"ledger unreadable: {exc}")
        return 1

    audit = AuditLog(path=audit_path)
    cli = AlpacaCLI()

    now = datetime.now(timezone.utc).astimezone()
    print(f"\nAPERTURE DESK | {now:%a %d %b %H:%M %Z}")
    print(f"ledger {state_path}   audit {audit_path}")

    try:
        clock = cli.clock()
        print("market OPEN" if clock.get("is_open")
              else f"market closed | next open {clock.get('next_open')}")
    except AlpacaCliError:
        pass

    account_section(cli, state)
    fill_section(cli)
    positions_section(cli, state)
    capital_section(state, audit)
    veto_summary(audit)
    decisions_section(audit, args.decisions)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
