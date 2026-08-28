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

from .allocator import Allocator, observe, summarise
from .alpaca_cli import AlpacaCLI, AlpacaCliError, idempotency_key
from .contracts import Side, parse_occ
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

# Official timeline, per Alpaca's published guidelines.
#
# The scored window is FOUR sessions, not six, and it ends at the *opening bell*
# on 4 September rather than at the submission deadline. Judging takes a snapshot
# of total account equity at that moment -- equity, not cash, so open positions
# are marked to market and count.
SCORING_OPEN = datetime(2026, 8, 31, 9, 30, tzinfo=ET)
SCORING_CLOSE = datetime(2026, 9, 4, 9, 30, tzinfo=ET)
DEADLINE = SCORING_CLOSE  # what the tournament clock scales against

# Everything is closed during Thursday afternoon -- the last full session before
# the snapshot, leaving two hours of liquid market to get out in.
#
# The snapshot lands one hour after the September jobs report and moments after
# the opening bell, when option marks are widest and least reliable. A position
# held through that is a bet on a macro print, priced at the worst quotes of the
# week. Being flat converts the result into cash, which cannot be re-marked.
FLATTEN_FROM = datetime(2026, 9, 3, 14, 0, tzinfo=ET)

TERMINAL_ORDER_STATUSES = {"canceled", "expired", "rejected"}


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
    summary = {"submitted": 0, "approved": 0, "vetoed": 0, "closed": 0,
               "breached": None, "flattening": False, "fired": []}

    clock = cli.clock()
    if not clock.get("is_open"):
        log.info("market closed; next open %s", clock.get("next_open"))
        return summary

    book = build_book(cli, state, now, require_expected=require_expected)

    # Submission acceptance is not execution.  Resolve pending entries and
    # closes against the parent mleg order before the ledger is compared with
    # positions or any strategy is allowed to spend the reserved risk again.
    lifecycle = sync_order_lifecycle(cli, state, warden)
    summary["closed"] += lifecycle["closed"]

    # Refresh the state-backed part of the book after fill reconciliation.
    book.open_risk_by_strategy = state.open_risk_by_strategy()
    book.open_risk_by_underlying = state.open_risk_by_underlying()
    log.info(
        "equity $%s | drawdown %.2f%% | day %+.2f%% | open risk $%s",
        f"{book.equity:,.0f}",
        book.drawdown_pct * 100,
        book.day_pnl_pct * 100,
        f"{book.total_open_risk:,.0f}",
    )

    # 1. Reconcile the ledger against the broker before trusting any of it.
    broker_positions = cli.positions()
    broker_symbols = {p.get("symbol") for p in broker_positions}
    for vanished in state.reconcile(broker_symbols):
        warden.audit.record("reconcile", client_order_id=vanished)

    mismatches = position_mismatches(state, broker_positions, now=now)
    if mismatches:
        detail = "; ".join(mismatches[:8])
        summary["breached"] = f"broker/ledger position mismatch: {detail}"
        warden.audit.record("position_mismatch", reason=detail, count=len(mismatches))
        warden.engage_kill_switch(summary["breached"])
        state.save()
        return summary

    # 2. Endgame: past the flatten point, the only job is to be in cash.
    if now >= FLATTEN_FROM:
        summary["flattening"] = True
        log.warning(
            "endgame: past %s, closing everything and opening nothing",
            FLATTEN_FROM.strftime("%d %b %H:%M ET"),
        )
        _cancel_pending_entries(cli, state, warden, dry_run=dry_run)
        summary["closed"] += _flatten(cli, md, state, warden, dry_run=dry_run)
        state.save()
        return summary

    # 3. Circuit breakers.
    breach = warden.breached(book)
    if breach:
        summary["breached"] = breach
        warden.audit.record("breach", reason=breach, equity=book.equity)
        log.critical("BREACH: %s", breach)
        _cancel_pending_entries(cli, state, warden, dry_run=dry_run)
        summary["closed"] += _derisk(cli, md, state, warden, dry_run=dry_run)
        state.save()
        return summary

    # 4. Exits before entries: free capital this cycle, and always allow an exit.
    summary["closed"] += manage_exits(cli, md, state, warden, dry_run=dry_run)

    # 5. Reallocate. Capital is the desk's only reward signal, so this runs
    #    before anyone is asked for proposals -- a fired strategy should not get
    #    the chance to propose, and a promoted one should feel it this cycle.
    allocations = Allocator().allocate(
        observe(state, warden.audit, PRIOR_WEIGHTS), book.equity
    )
    warden.budgets = {a.strategy_id: a.budget for a in allocations}
    fired = {a.strategy_id for a in allocations if not a.is_active}
    summary["fired"] = sorted(fired)

    previous = state.allocations or {}
    current = {a.strategy_id: a.weight for a in allocations}
    if current != previous:
        log.info("allocation changed:\n%s", summarise(allocations))
        warden.audit.record(
            "allocation",
            weights=current,
            budgets={a.strategy_id: a.budget for a in allocations},
            reasons={a.strategy_id: a.reason for a in allocations},
        )
        state.allocations = current

    # 6-8. Propose, gate, submit.
    book = build_book(cli, state, now)  # refresh after any exits
    for strategy in strategies:
        if strategy.config.strategy_id in fired:
            continue  # defunded: it keeps its open positions and its exits, not new capital
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
    # A strategy gets one live intent per underlying.  The broker can hold a
    # day limit for nearly an hour before filling it; repricing that proposal on
    # every five-minute cycle otherwise creates several orders that can all fill
    # later, long after the ledger has forgotten the first one.
    duplicate = next(
        (
            trade for trade in state.open_trades.values()
            if trade.strategy_id == proposal.strategy_id
            and trade.underlying == proposal.underlying
            and trade.status in {
                "submitting_entry", "pending_entry", "open",
                "submitting_close", "pending_close",
            }
        ),
        None,
    )
    if duplicate is not None:
        log.info(
            "skipping duplicate intent: %s already has %s %s",
            proposal.strategy_id, duplicate.status, proposal.underlying,
        )
        return "duplicate"

    symbols = [leg.symbol for leg in proposal.legs]
    quotes = md.leg_quotes(symbols, proposal.underlying)

    verdict = warden.review(proposal, quotes, book)
    if not verdict.approved:
        return "vetoed"

    # The day salt keeps a legitimate retry on a later session from colliding
    # with yesterday's expired client id.  Within a session it stays stable.
    key = idempotency_key(proposal, salt=book.now.date().isoformat())
    if key in state.open_trades:
        log.info("skipping %s: already in the ledger", key)
        return "duplicate"

    if dry_run:
        log.info("[dry-run] would submit %s", proposal.rationale)
        return "dry_run"

    # Write-ahead reservation.  If the process dies after the HTTP request
    # reaches Alpaca but before its response reaches us, the next process sees
    # this intent and recovers it by the same client id instead of submitting a
    # second order.
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
        order_id=None,
        status="submitting_entry",
        leg_sides={leg.symbol: leg.side.value for leg in proposal.legs},
        leg_ratios={leg.symbol: leg.ratio for leg in proposal.legs},
    )
    state.record_open(trade)
    state.save()
    warden.audit.record(
        "entry_reserved",
        client_order_id=key,
        strategy=proposal.strategy_id,
        underlying=proposal.underlying,
        qty=proposal.qty,
        net_price=proposal.net_price,
        rationale=proposal.rationale,
    )

    try:
        order = cli.submit_mleg(proposal, client_order_id=key)
    except AlpacaCliError as exc:
        # Deliberately keep the reservation.  The failure may be a timeout after
        # acceptance; lifecycle recovery first queries this exact client id.
        warden.audit.record(
            "entry_submission_uncertain",
            client_order_id=key,
            strategy=proposal.strategy_id,
            underlying=proposal.underlying,
            error=exc.stderr[:300],
        )
        log.error("entry submission uncertain for %s: %s", proposal.underlying, exc.stderr[:200])
        return "failed"

    trade.order_id = (order or {}).get("id")
    trade.status = "pending_entry"
    state.save()
    warden.audit.record(
        "entry_submitted",
        client_order_id=key,
        order_id=trade.order_id,
        strategy=proposal.strategy_id,
        underlying=proposal.underlying,
        qty=proposal.qty,
        net_price=proposal.net_price,
        rationale=proposal.rationale,
    )
    log.info("ENTRY SUBMITTED (awaiting fill) %s", proposal.rationale)
    return "submitted"


def _order_for_trade(cli: AlpacaCLI, trade: OpenTrade, *, closing: bool = False) -> dict:
    order_id = trade.close_order_id if closing else trade.order_id
    client_id = trade.close_client_order_id if closing else trade.client_order_id
    if order_id:
        return cli.order(order_id)
    return cli.order_by_client_id(client_id)


def _order_not_found(exc: AlpacaCliError) -> bool:
    text = exc.stderr.lower()
    return "40410000" in text or ("404" in text and "order not found" in text)


def _filled_qty(order: dict) -> int:
    try:
        return max(int(float(order.get("filled_qty") or 0)), 0)
    except (TypeError, ValueError):
        return 0


def _filled_price(order: dict) -> float | None:
    try:
        return float(order.get("filled_avg_price"))
    except (TypeError, ValueError):
        pass

    # Parent mleg fills normally carry a signed average.  If that field is ever
    # absent, reconstruct the signed net from the nested leg fills rather than
    # pretending the submitted limit was the execution price.
    total = 0.0
    legs = order.get("legs") or []
    if not legs:
        return None
    for leg in legs:
        try:
            price = float(leg.get("filled_avg_price"))
            ratio = int(float(leg.get("ratio_qty") or 1))
        except (TypeError, ValueError):
            return None
        side = str(leg.get("side") or "").lower()
        if side not in {"buy", "sell"}:
            return None
        total += price * ratio if side == "buy" else -price * ratio
    return round(total, 4)


def _filled_entry_max_loss(trade: OpenTrade, qty: int, net_price: float) -> float:
    """Recompute risk from the actual fill, not the submitted limit."""
    legs = []
    for symbol in trade.legs:
        side = Side(trade.leg_sides[symbol])
        intent = (
            PositionIntent.BUY_TO_OPEN if side is Side.BUY
            else PositionIntent.SELL_TO_OPEN
        )
        legs.append(Leg(symbol, side, trade.leg_ratios.get(symbol, 1), intent))
    proposal = Proposal(
        strategy_id=trade.strategy_id,
        underlying=trade.underlying,
        legs=tuple(legs),
        qty=qty,
        net_price=net_price,
        rationale=trade.rationale,
    )
    return analyse_payoff(proposal).max_loss_or_inf


def _entry_proposal_from_trade(trade: OpenTrade) -> Proposal:
    legs = []
    for symbol in trade.legs:
        side = Side(trade.leg_sides[symbol])
        intent = (
            PositionIntent.BUY_TO_OPEN if side is Side.BUY
            else PositionIntent.SELL_TO_OPEN
        )
        legs.append(Leg(symbol, side, trade.leg_ratios.get(symbol, 1), intent))
    return Proposal(
        strategy_id=trade.strategy_id,
        underlying=trade.underlying,
        legs=tuple(legs),
        qty=trade.qty,
        net_price=trade.net_price,
        rationale=trade.rationale,
    )


def _pending_close_proposal(trade: OpenTrade) -> Proposal:
    if trade.close_limit_price is None:
        raise ValueError("pending close has no persisted limit price")
    legs = []
    for symbol in trade.legs:
        opened_side = Side(trade.leg_sides[symbol])
        side = Side.SELL if opened_side is Side.BUY else Side.BUY
        intent = (
            PositionIntent.SELL_TO_CLOSE if side is Side.SELL
            else PositionIntent.BUY_TO_CLOSE
        )
        legs.append(Leg(symbol, side, trade.leg_ratios.get(symbol, 1), intent))
    return Proposal(
        strategy_id=trade.strategy_id,
        underlying=trade.underlying,
        legs=tuple(legs),
        qty=trade.qty,
        net_price=trade.close_limit_price,
        rationale=f"closing {trade.client_order_id}",
    )


def _recover_reserved_submission(
    cli: AlpacaCLI, state: DeskState, trade: OpenTrade, warden: RiskWarden
) -> bool:
    closing = trade.status == "submitting_close"
    if warden.halted() and not closing:
        state.discard_pending(trade.client_order_id)
        state.save()
        warden.audit.record(
            "entry_abandoned",
            client_order_id=trade.client_order_id,
            strategy=trade.strategy_id,
            underlying=trade.underlying,
            reason="kill switch engaged before broker accepted reserved entry",
        )
        return True
    try:
        proposal = _pending_close_proposal(trade) if closing else _entry_proposal_from_trade(trade)
        client_id = trade.close_client_order_id if closing else trade.client_order_id
        order = cli.submit_mleg(proposal, client_order_id=client_id)
    except (AlpacaCliError, ValueError) as exc:
        detail = exc.stderr[:300] if isinstance(exc, AlpacaCliError) else str(exc)
        warden.audit.record(
            "submission_recovery_failed",
            client_order_id=(trade.close_client_order_id if closing else trade.client_order_id),
            strategy=trade.strategy_id,
            underlying=trade.underlying,
            error=detail,
        )
        return False

    if closing:
        trade.close_order_id = (order or {}).get("id")
        trade.status = "pending_close"
    else:
        trade.order_id = (order or {}).get("id")
        trade.status = "pending_entry"
    state.save()
    warden.audit.record(
        "submission_recovered",
        client_order_id=(trade.close_client_order_id if closing else trade.client_order_id),
        order_id=(order or {}).get("id"),
        strategy=trade.strategy_id,
        underlying=trade.underlying,
        side="close" if closing else "entry",
    )
    return True


def sync_order_lifecycle(
    cli: AlpacaCLI,
    state: DeskState,
    warden: RiskWarden,
    *,
    allow_submission_recovery: bool = True,
) -> dict[str, int]:
    """Resolve pending parent orders into fills or terminal non-events.

    Errors keep the reservation in place.  That is conservative on purpose: an
    unknown order must continue consuming its full risk budget until the broker
    can tell us whether it filled.
    """
    summary = {"entries_filled": 0, "entries_unfilled": 0, "closed": 0}
    for trade in list(state.open_trades.values()):
        if trade.status not in {
            "submitting_entry", "pending_entry", "submitting_close", "pending_close"
        }:
            continue
        closing = trade.status in {"submitting_close", "pending_close"}
        try:
            order = _order_for_trade(cli, trade, closing=closing)
        except AlpacaCliError as exc:
            if (
                allow_submission_recovery
                and trade.status in {"submitting_entry", "submitting_close"}
                and _order_not_found(exc)
            ):
                _recover_reserved_submission(cli, state, trade, warden)
                continue
            warden.audit.record(
                "order_sync_error",
                client_order_id=(
                    trade.close_client_order_id if closing else trade.client_order_id
                ),
                error=exc.stderr[:300],
            )
            log.warning("could not reconcile order for %s: %s", trade.underlying, exc.stderr[:120])
            continue

        status = str(order.get("status") or "").lower()
        filled_qty = _filled_qty(order)
        terminal = status == "filled" or status in TERMINAL_ORDER_STATUSES
        if not terminal:
            continue

        if trade.status in {"submitting_entry", "pending_entry"}:
            if filled_qty <= 0:
                state.discard_pending(trade.client_order_id)
                summary["entries_unfilled"] += 1
                warden.audit.record(
                    "entry_unfilled",
                    client_order_id=trade.client_order_id,
                    strategy=trade.strategy_id,
                    underlying=trade.underlying,
                    order_status=status,
                )
                state.save()
                continue

            fill_price = _filled_price(order)
            if fill_price is None:
                warden.audit.record(
                    "fill_price_missing",
                    client_order_id=trade.client_order_id,
                    strategy=trade.strategy_id,
                    underlying=trade.underlying,
                    order_status=status,
                )
                continue
            max_loss = _filled_entry_max_loss(trade, filled_qty, fill_price)
            state.confirm_entry(
                trade.client_order_id,
                qty=filled_qty,
                net_price=fill_price,
                max_loss=max_loss,
                filled_at=order.get("filled_at"),
            )
            state.save()
            summary["entries_filled"] += 1
            warden.audit.record(
                "entry_filled",
                client_order_id=trade.client_order_id,
                order_id=trade.order_id,
                strategy=trade.strategy_id,
                underlying=trade.underlying,
                qty=filled_qty,
                net_price=fill_price,
                max_loss=max_loss,
            )
            log.info("ENTRY FILLED %s x%d @ %+.2f", trade.underlying, filled_qty, fill_price)
            continue

        # A close can also finish partially.  Record only broker-confirmed units
        # and return any remainder to open management.
        if filled_qty <= 0:
            reason = trade.close_reason or "close"
            close_client_id = trade.close_client_order_id
            state.reopen_after_unfilled_close(trade.client_order_id)
            state.save()
            warden.audit.record(
                "close_unfilled",
                client_order_id=trade.client_order_id,
                close_client_order_id=close_client_id,
                strategy=trade.strategy_id,
                underlying=trade.underlying,
                order_status=status,
                reason=reason,
            )
            continue

        close_price = _filled_price(order)
        if close_price is None:
            warden.audit.record(
                "fill_price_missing",
                client_order_id=trade.close_client_order_id,
                strategy=trade.strategy_id,
                underlying=trade.underlying,
                order_status=status,
            )
            continue
        close_qty = min(filled_qty, trade.qty)
        pnl = round(-(trade.net_price + close_price) * close_qty * 100, 2)
        reason = trade.close_reason or "close filled"
        closed_row = state.record_filled_close(
            trade.client_order_id,
            qty=close_qty,
            reason=reason,
            pnl=pnl,
            close_price=close_price,
            closed_at=order.get("filled_at"),
        )
        state.save()
        event = "closed" if closed_row is not None else "partial_close"
        if closed_row is not None:
            summary["closed"] += 1
        warden.audit.record(
            event,
            client_order_id=trade.client_order_id,
            strategy=trade.strategy_id,
            underlying=trade.underlying,
            qty=close_qty,
            close_price=close_price,
            pnl=pnl,
            reason=reason,
        )
        log.info("CLOSE FILLED %s x%d | P&L $%+.2f", trade.underlying, close_qty, pnl)

    return summary


def emergency_flatten_cycle(
    cli: AlpacaCLI,
    md: MarketData,
    warden: RiskWarden,
    state: DeskState,
    *,
    dry_run: bool = False,
    require_expected: bool = False,
) -> dict[str, int]:
    """Cancel new risk and work every known structure toward a confirmed close.

    This is the operational meaning of the kill switch.  It runs repeatedly
    while the market is open: pending closes are reconciled first, unfilled
    entries are canceled, and only broker-confirmed fills leave the ledger.
    """
    now = datetime.now(ET)
    build_book(cli, state, now, require_expected=require_expected)
    lifecycle = sync_order_lifecycle(cli, state, warden)

    broker_positions = cli.positions()
    broker_symbols = {p.get("symbol") for p in broker_positions}
    for vanished in state.reconcile(broker_symbols):
        warden.audit.record("reconcile", client_order_id=vanished)

    _cancel_pending_entries(cli, state, warden, dry_run=dry_run)
    before = sum(1 for trade in state.open_trades.values() if trade.status == "open")
    _flatten(cli, md, state, warden, dry_run=dry_run)
    after = sum(1 for trade in state.open_trades.values() if trade.status == "open")
    state.save()
    return {
        "closed": lifecycle["closed"],
        "close_submitted": max(before - after, 0),
        "remaining": len(state.open_trades),
    }


def _recent_fill(trade: OpenTrade, now: datetime, grace_seconds: float = 600.0) -> bool:
    if not trade.filled_at:
        return False
    try:
        stamp = datetime.fromisoformat(trade.filled_at.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    return (now.astimezone(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds() \
        <= grace_seconds


def position_mismatches(
    state: DeskState,
    broker_positions: Sequence[dict],
    *,
    now: datetime | None = None,
) -> list[str]:
    """Compare signed per-leg quantities once no order is in flight.

    Symbol presence alone cannot catch a ledger that says 215/200 x2 while the
    broker owns 215/195 x1.  Pending orders and just-filled positions are allowed
    a short propagation window; settled structures must match exactly.
    """
    now = now or datetime.now(timezone.utc)
    expected: dict[str, float] = {}
    flexible: set[str] = set()

    for trade in state.open_trades.values():
        in_flight = trade.status in {
            "submitting_entry", "pending_entry", "submitting_close", "pending_close"
        }
        if in_flight or _recent_fill(trade, now):
            flexible.update(trade.legs)
        if trade.status in {"open", "submitting_close", "pending_close"}:
            for symbol in trade.legs:
                ratio = trade.leg_ratios.get(symbol, 1)
                sign = 1 if Side(trade.leg_sides[symbol]) is Side.BUY else -1
                expected[symbol] = expected.get(symbol, 0.0) + sign * trade.qty * ratio

    actual: dict[str, float] = {}
    for position in broker_positions:
        symbol = str(position.get("symbol") or "")
        try:
            parse_occ(symbol)
            qty = float(position.get("qty") or 0.0)
        except (TypeError, ValueError):
            continue
        actual[symbol] = actual.get(symbol, 0.0) + qty

    issues = []
    for symbol in sorted(set(expected) | set(actual)):
        if symbol in flexible:
            continue
        wanted = expected.get(symbol, 0.0)
        found = actual.get(symbol, 0.0)
        if abs(wanted - found) > 1e-9:
            issues.append(f"{symbol} ledger {wanted:+g}, broker {found:+g}")
    return issues


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

        reason = exit_reason(trade, price)
        if reason is None:
            continue

        if dry_run:
            log.info("[dry-run] would close %s: %s", trade.underlying, reason)
            continue
        _close_trade(cli, md, state, warden, trade, reason)
    return closed


def exit_reason(trade: OpenTrade, current_price: float, *, today=None) -> str | None:
    """Apply the owning strategy's exit contract, not the trade's price sign."""
    if trade.strategy_id == "CRUSH":
        today = today or datetime.now(ET).date()
        raw = trade.filled_at or trade.opened_at
        try:
            opened = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            opened_day = opened.astimezone(ET).date()
        except (TypeError, ValueError):
            # An unreadable timestamp is not permission to hold an event trade
            # indefinitely.  Surface it as an immediate, explicit exit.
            return "CRUSH entry timestamp invalid; fail-safe close"
        return "earnings window elapsed; CRUSH holds one night" if today > opened_day else None

    config = _config_for(trade.strategy_id)
    if trade.net_price < 0:
        return carry.exit_signal(trade.net_price, current_price, config)
    return drift.exit_signal(trade.net_price, current_price, config)


def _flatten(
    cli: AlpacaCLI, md: MarketData, state: DeskState, warden: RiskWarden, *, dry_run: bool
) -> int:
    """Close everything, regardless of sleeve or P&L.

    Distinct from de-risking: a breach keeps the ballast because that is what
    recovers a drawdown, whereas the endgame wants no marks at all.
    """
    closed = 0
    for trade in list(state.open_trades.values()):
        if trade.status != "open":
            continue
        if dry_run:
            log.info("[dry-run] would flatten %s", trade.underlying)
            continue
        _close_trade(cli, md, state, warden, trade, "endgame flatten")
    return closed


def _derisk(
    cli: AlpacaCLI, md: MarketData, state: DeskState, warden: RiskWarden, *, dry_run: bool
) -> int:
    """On a breach, flatten the convex sleeve and leave the ballast alone.

    The ballast is what recovers a drawdown; the convex sleeve is what caused it.
    """
    closed = 0
    for trade in list(state.open_trades.values()):
        if trade.status != "open":
            continue
        if _config_for(trade.strategy_id).sleeve != "CONVEX":
            continue
        if dry_run:
            log.info("[dry-run] would de-risk %s", trade.underlying)
            continue
        _close_trade(cli, md, state, warden, trade, "circuit breaker de-risk")
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
            legs.append(Leg(
                symbol, Side.SELL, trade.leg_ratios.get(symbol, 1),
                PositionIntent.SELL_TO_CLOSE,
            ))
        else:
            legs.append(Leg(
                symbol, Side.BUY, trade.leg_ratios.get(symbol, 1),
                PositionIntent.BUY_TO_CLOSE,
            ))

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
    if trade.status != "open":
        return False

    price = current_structure_price(md, trade)
    if price is None:
        log.warning("cannot price %s to close; will retry next cycle", trade.underlying)
        return False

    proposal = build_closing_proposal(trade, price)
    close_client_id = f"close-{trade.client_order_id}-{trade.close_attempts + 1}"[:128]

    # The close gets the same write-ahead treatment as an entry.  A process
    # crash can delay a close, but it cannot erase a possibly-live close order
    # from the ledger and send a second one on restart.
    state.mark_close_pending(
        trade.client_order_id,
        order_id=None,
        close_client_order_id=close_client_id,
        reason=reason,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        limit_price=proposal.net_price,
    )
    state.save()
    warden.audit.record(
        "close_reserved",
        client_order_id=trade.client_order_id,
        close_client_order_id=close_client_id,
        strategy=trade.strategy_id,
        underlying=trade.underlying,
        reason=reason,
    )

    try:
        order = cli.submit_mleg(proposal, client_order_id=close_client_id)
    except AlpacaCliError as exc:
        warden.audit.record(
            "close_submission_uncertain",
            client_order_id=trade.client_order_id,
            close_client_order_id=close_client_id,
            error=exc.stderr[:300],
        )
        log.error("CLOSE SUBMISSION UNCERTAIN on %s - %s", trade.underlying, exc.stderr[:160])
        return False

    state.confirm_close_submission(
        trade.client_order_id,
        order_id=(order or {}).get("id"),
        limit_price=proposal.net_price,
    )
    state.save()
    warden.audit.record(
        "close_submitted",
        client_order_id=trade.client_order_id,
        close_client_order_id=close_client_id,
        order_id=(order or {}).get("id"),
        strategy=trade.strategy_id,
        underlying=trade.underlying,
        reason=reason,
    )
    log.info("CLOSE SUBMITTED (awaiting fill) %s - %s", trade.underlying, reason)
    return True


def _cancel_pending_entries(
    cli: AlpacaCLI,
    state: DeskState,
    warden: RiskWarden,
    *,
    dry_run: bool,
) -> int:
    """Withdraw every not-yet-filled entry when the desk may no longer add risk."""
    requested = 0
    for trade in list(state.open_trades.values()):
        if trade.status not in {"submitting_entry", "pending_entry"}:
            continue
        if dry_run:
            log.info("[dry-run] would cancel pending entry %s", trade.underlying)
            continue
        if not trade.order_id:
            # Keep the risk reservation.  Without an id, the next lifecycle poll
            # uses the client id; silently dropping it would be the unsafe move.
            log.warning("cannot cancel %s yet: broker order id is unavailable", trade.underlying)
            continue
        try:
            cli.cancel_order(trade.order_id)
        except AlpacaCliError as exc:
            # The usual race is that it filled between the poll and the cancel.
            # Leave it pending so the next sync resolves the truth.
            warden.audit.record(
                "entry_cancel_failed",
                client_order_id=trade.client_order_id,
                strategy=trade.strategy_id,
                underlying=trade.underlying,
                error=exc.stderr[:300],
            )
            continue
        requested += 1
        warden.audit.record(
            "entry_cancel_requested",
            client_order_id=trade.client_order_id,
            strategy=trade.strategy_id,
            underlying=trade.underlying,
        )
    return requested


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
        clock = cli.clock()
        if not clock.get("is_open"):
            sync_order_lifecycle(cli, state, warden, allow_submission_recovery=False)
            log.critical("kill switch engaged; market closed, flatten resumes next open")
            return 2
        result = emergency_flatten_cycle(
            cli, md, warden, state, dry_run=args.dry_run, require_expected=not args.dry_run
        )
        log.critical(
            "kill switch flatten: %d confirmed closed, %d close orders submitted, %d remaining",
            result["closed"], result["close_submitted"], result["remaining"],
        )
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


# The designed barbell, as shares of the total risk budget. These are the
# allocator's prior, not a fixed schedule: from here the desk funds what works.
PRIOR_WEIGHTS = {"CARRY": 0.64, "CRUSH": 0.18, "DRIFT": 0.18}


def _budgets(equity: float) -> dict[str, float]:
    """Designed weights, before the allocator has observed anything."""
    from .allocator import AllocationLimits

    budget = equity * AllocationLimits().total_risk_budget_pct
    return {k: budget * w for k, w in PRIOR_WEIGHTS.items()}


if __name__ == "__main__":
    sys.exit(main())
