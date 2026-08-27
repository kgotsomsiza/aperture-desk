"""The desk loop — `python -m aperture.loop`.

One cycle, in order:

  1. Read the account and reconcile the ledger against what the broker shows.
  2. Check the circuit breakers *before* anything else, because a breached desk
     de-risks rather than trades.
  3. Manage exits on what is already open.
  4. Ask each funded strategy for proposals.
  5. Put every proposal through the Risk Warden.
  6. Submit only what the Warden approved.

The ordering is deliberate. Exits run before entries so that capital freed this
cycle is available this cycle, and so a desk at its risk limit can still get
*out* of things when it cannot get into them.

Run it on a schedule during market hours. It is idempotent: every order carries a
deterministic client_order_id, so a cycle that runs twice does not double up.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from .alpaca_cli import AlpacaCLI, AlpacaCliError, idempotency_key
from .contracts import Side
from .earnings import EarningsCalendar
from .identity import WrongAccountError, check as check_account
from .marketdata import MarketData, Snapshot
from .contracts import PositionIntent
from .risk import BookState, Leg, Proposal, RiskLimits, analyse_payoff
from .state import DeskState, OpenTrade
from .strategies import carry, crush, drift
from .strategies.base import Strategy, structure_price
from .warden import AuditLog, RiskWarden

ET = ZoneInfo("America/New_York")
log = logging.getLogger("aperture")

# The submission deadline: 4 Sep 2026, 17:00 SAST = 11:00 ET. The desk uses this
# to scale its own risk appetite as the contest runs out.
DEADLINE = datetime(2026, 9, 4, 11, 0, tzinfo=ET)


def build_strategies() -> list[Strategy]:
    calendar = EarningsCalendar()
    return [
        carry.CarryStrategy(),
        crush.CrushStrategy(calendar=calendar),
        drift.DriftStrategy(calendar=calendar),
    ]


def build_book(
    cli: AlpacaCLI, state: DeskState, now: datetime, *, require_expected: bool = False
) -> BookState:
    account = cli.account()

    # Identity before anything else. Every number below is meaningless, and
    # every order dangerous, if this is not the account we think it is.
    state.account_fingerprint = check_account(
        account,
        recorded=state.account_fingerprint or None,
        require_expected=require_expected,
    )

    equity = float(account.get("equity") or 0.0)
    state.observe_equity(equity, now.date())

    return BookState(
        equity=equity,
        high_water_mark=state.high_water_mark,
        day_start_equity=state.day_start_equity or equity,
        cash=float(account.get("cash") or 0.0),
        now=now,
        open_risk_by_strategy=state.open_risk_by_strategy(),
        open_risk_by_underlying=state.open_risk_by_underlying(),
    )


def current_structure_price(md: MarketData, trade: OpenTrade) -> float | None:
    """What it would cost to close this structure right now.

    Returned in Alpaca's sign convention so it compares directly against the
    entry price: a credit spread opens negative and closes negative.
    """
    snaps = md.snapshots_for(trade.legs, trade.underlying)
    if len(snaps) != len(trade.legs):
        return None

    if not trade.leg_sides:
        return None  # a ledger entry without sides cannot be priced; do not guess

    pairs: list[tuple[Snapshot, Side]] = []
    for symbol in trade.legs:
        snap = snaps[symbol]
        if not snap.is_priceable:
            return None
        # Closing reverses every leg, but the structure's *price* is quoted the
        # same way it was opened, so the original sides are what we sum.
        pairs.append((snap, Side(trade.leg_sides[symbol])))
    return structure_price(pairs)


# --------------------------------------------------------------------------- #
# Cycle
# --------------------------------------------------------------------------- #


def run_cycle(
    cli: AlpacaCLI,
    md: MarketData,
    warden: RiskWarden,
    state: DeskState,
    strategies: Sequence[Strategy],
    *,
    dry_run: bool = False,
    require_expected: bool = False,
) -> dict:
    now = datetime.now(ET)
    summary = {"submitted": 0, "approved": 0, "vetoed": 0, "closed": 0, "breached": None}

    clock = cli.clock()
    if not clock.get("is_open"):
        log.info("market closed; next open %s", clock.get("next_open"))
        return summary

    book = build_book(cli, state, now, require_expected=require_expected)
    log.info(
        "equity $%s | drawdown %.2f%% | day %+.2f%% | open risk $%s",
        f"{book.equity:,.0f}",
        book.drawdown_pct * 100,
        book.day_pnl_pct * 100,
        f"{book.total_open_risk:,.0f}",
    )

    # 1. Reconcile the ledger against the broker before trusting any of it.
    broker_symbols = {p.get("symbol") for p in cli.positions()}
    for vanished in state.reconcile(broker_symbols):
        warden.audit.record("reconcile", client_order_id=vanished)

    # 2. Circuit breakers before anything else.
    breach = warden.breached(book)
    if breach:
        summary["breached"] = breach
        warden.audit.record("breach", reason=breach, equity=book.equity)
        log.critical("BREACH: %s", breach)
        summary["closed"] += _derisk(cli, md, state, warden, dry_run=dry_run)
        state.save()
        return summary

    # 3. Exits before entries: free capital this cycle, and always allow an exit.
    summary["closed"] += manage_exits(cli, md, state, warden, dry_run=dry_run)

    # 4-6. Propose, gate, submit.
    book = build_book(cli, state, now)  # refresh after any exits
    for strategy in strategies:
        budget = warden.budget_for(strategy.config.strategy_id, book)
        if budget <= 0:
            continue
        try:
            proposals = strategy.propose(md, book, budget)
        except AlpacaCliError as exc:
            log.warning("%s could not propose: %s", strategy.config.strategy_id, exc.stderr[:120])
            continue

        for proposal in proposals:
            outcome = _submit_if_approved(cli, md, warden, state, book, proposal, dry_run=dry_run)
            if outcome == "vetoed":
                summary["vetoed"] += 1
                continue
            summary["approved"] += 1
            if outcome == "submitted":
                summary["submitted"] += 1
                book = build_book(cli, state, now)  # each fill changes the risk picture

    state.save()
    return summary


def _submit_if_approved(
    cli: AlpacaCLI,
    md: MarketData,
    warden: RiskWarden,
    state: DeskState,
    book: BookState,
    proposal: Proposal,
    *,
    dry_run: bool,
) -> str:
    """Returns "vetoed", "duplicate", "dry_run", "failed" or "submitted".

    A distinct value per outcome, because "approved but not sent" and "the Warden
    said no" are opposite results and collapsing them into one boolean makes the
    cycle summary lie about what the desk decided.
    """
    symbols = [leg.symbol for leg in proposal.legs]
    quotes = md.leg_quotes(symbols, proposal.underlying)

    verdict = warden.review(proposal, quotes, book)
    if not verdict.approved:
        return "vetoed"

    key = idempotency_key(proposal)
    if key in state.open_trades:
        log.info("skipping %s: already in the ledger", key)
        return "duplicate"

    if dry_run:
        log.info("[dry-run] would submit %s", proposal.rationale)
        return "dry_run"

    try:
        order = cli.submit_mleg(proposal, client_order_id=key)
    except AlpacaCliError as exc:
        warden.audit.record(
            "submit_failed",
            strategy=proposal.strategy_id,
            underlying=proposal.underlying,
            error=exc.stderr[:300],
        )
        log.error("submit failed for %s: %s", proposal.underlying, exc.stderr[:200])
        return "failed"

    trade = OpenTrade(
        client_order_id=key,
        strategy_id=proposal.strategy_id,
        underlying=proposal.underlying,
        legs=symbols,
        qty=proposal.qty,
        net_price=proposal.net_price,
        max_loss=verdict.profile.max_loss_or_inf,
        opened_at=datetime.now(timezone.utc).isoformat(),
        rationale=proposal.rationale,
        order_id=(order or {}).get("id"),
        status="open",
        leg_sides={leg.symbol: leg.side.value for leg in proposal.legs},
    )
    state.record_open(trade)
    warden.audit.record(
        "submitted",
        client_order_id=key,
        order_id=trade.order_id,
        strategy=proposal.strategy_id,
        underlying=proposal.underlying,
        qty=proposal.qty,
        net_price=proposal.net_price,
        rationale=proposal.rationale,
    )
    log.info("SUBMITTED %s", proposal.rationale)
    return "submitted"


def manage_exits(
    cli: AlpacaCLI,
    md: MarketData,
    state: DeskState,
    warden: RiskWarden,
    *,
    dry_run: bool = False,
) -> int:
    closed = 0
    for trade in list(state.open_trades.values()):
        if trade.status != "open":
            continue
        price = current_structure_price(md, trade)
        if price is None:
            continue

        config = _config_for(trade.strategy_id)
        if trade.net_price < 0:
            reason = carry.exit_signal(trade.net_price, price, config)
        else:
            reason = drift.exit_signal(trade.net_price, price, config)
        if reason is None:
            continue

        if dry_run:
            log.info("[dry-run] would close %s: %s", trade.underlying, reason)
            continue
        if _close_trade(cli, md, state, warden, trade, reason):
            closed += 1
    return closed


def _derisk(
    cli: AlpacaCLI, md: MarketData, state: DeskState, warden: RiskWarden, *, dry_run: bool
) -> int:
    """On a breach, flatten the convex sleeve and leave the ballast alone.

    The ballast is what recovers a drawdown; the convex sleeve is what caused it.
    """
    closed = 0
    for trade in list(state.open_trades.values()):
        if _config_for(trade.strategy_id).sleeve != "CONVEX":
            continue
        if dry_run:
            log.info("[dry-run] would de-risk %s", trade.underlying)
            continue
        if _close_trade(cli, md, state, warden, trade, "circuit breaker de-risk"):
            closed += 1
    return closed


def build_closing_proposal(trade: OpenTrade, price: float) -> Proposal:
    """The mirror of an open structure, as one order.

    Every leg reverses: what was bought is sold to close, what was sold is bought
    to close. The price mirrors too — a structure opened for a credit is closed
    for a debit — so the closing limit is the negation of what the structure is
    worth right now.
    """
    legs = []
    for symbol in trade.legs:
        opened_side = Side(trade.leg_sides[symbol])
        if opened_side is Side.BUY:
            legs.append(Leg(symbol, Side.SELL, 1, PositionIntent.SELL_TO_CLOSE))
        else:
            legs.append(Leg(symbol, Side.BUY, 1, PositionIntent.BUY_TO_CLOSE))

    return Proposal(
        strategy_id=trade.strategy_id,
        underlying=trade.underlying,
        legs=tuple(legs),
        qty=trade.qty,
        net_price=round(-price + 0.05, 2),  # concede toward the market to get filled
        rationale=f"closing {trade.client_order_id}",
    )


def _close_trade(
    cli: AlpacaCLI,
    md: MarketData,
    state: DeskState,
    warden: RiskWarden,
    trade: OpenTrade,
    reason: str,
) -> bool:
    """Close a structure as ONE multi-leg order.

    Closing leg by leg is how a defined-risk position turns into an undefined one.
    Alpaca fills each single-leg close independently, and the short legs are the
    easy ones to fill: lose the race and the desk is left holding naked shorts
    with the hedge already sold. An mleg close fills all legs or none.

    Verified the hard way on 27 Aug: `position close-all` on a live iron condor
    closed both shorts and left both longs open.
    """
    price = current_structure_price(md, trade)
    if price is None:
        log.warning("cannot price %s to close; will retry next cycle", trade.underlying)
        return False

    proposal = build_closing_proposal(trade, price)
    try:
        cli.submit_mleg(proposal, client_order_id=f"close-{trade.client_order_id}"[:128])
    except AlpacaCliError as exc:
        warden.audit.record(
            "close_failed", client_order_id=trade.client_order_id, error=exc.stderr[:300]
        )
        log.error("CLOSE FAILED on %s - %s", trade.underlying, exc.stderr[:160])
        return False

    state.record_close(trade.client_order_id, reason)
    warden.audit.record(
        "closed",
        client_order_id=trade.client_order_id,
        strategy=trade.strategy_id,
        underlying=trade.underlying,
        reason=reason,
    )
    log.info("CLOSED %s - %s", trade.underlying, reason)
    return True


def _config_for(strategy_id: str):
    return {
        "CARRY": carry.DEFAULT_CONFIG,
        "CRUSH": crush.DEFAULT_CONFIG,
        "DRIFT": drift.DEFAULT_CONFIG,
    }.get(strategy_id, carry.DEFAULT_CONFIG)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one desk cycle")
    parser.add_argument("--dry-run", action="store_true", help="decide, gate and log, but never submit")
    parser.add_argument("--state", default="state/desk.json")
    parser.add_argument("--feed", default="indicative")
    parser.add_argument("--equity", type=float, default=100_000.0, help="starting equity")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    cli = AlpacaCLI()
    md = MarketData(cli=cli, feed=args.feed)
    state = DeskState.load(args.state)
    warden = RiskWarden(
        limits=RiskLimits(),
        audit=AuditLog(path=Path(args.state).parent / "audit.jsonl"),
        deadline=DEADLINE,
        start_equity=state.start_equity or args.equity,
        budgets=_budgets(state.start_equity or args.equity),
    )

    if warden.halted():
        log.critical("kill switch is engaged; no trading this cycle")
        return 2

    summary = run_cycle(cli, md, warden, state, build_strategies(), dry_run=args.dry_run)
    log.info(
        "cycle complete: %d approved (%d submitted), %d vetoed, %d closed%s",
        summary["approved"],
        summary["submitted"],
        summary["vetoed"],
        summary["closed"],
        f" | BREACH: {summary['breached']}" if summary["breached"] else "",
    )
    return 1 if summary["breached"] else 0


def _budgets(equity: float) -> dict[str, float]:
    """The barbell, expressed as maximum loss each strategy may have at risk."""
    return {
        "CARRY": equity * 0.18,
        "CRUSH": equity * 0.05,
        "DRIFT": equity * 0.05,
    }


if __name__ == "__main__":
    sys.exit(main())
