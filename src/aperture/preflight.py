"""Environment preflight — `python -m aperture.preflight`.

Answers, in one run, every question the desk's design depends on:

  * Does the CLI authenticate, and is this really a paper account?
  * Is options level 3 (multi-leg) actually enabled?
  * Which options data feed does this account's plan allow — opra or indicative?
  * Do chain snapshots carry greeks and implied volatility?
  * How stale are the quotes in practice?
  * What request body does `--legs` actually produce?
  * Is there enough historical option data behind us for the backtest gate?

Run it before the first trade of every session. `--live-test` additionally sends
one real 1-lot defined-risk spread and then closes it, which is the only way to
confirm the fill path end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .alpaca_cli import AlpacaCLI, AlpacaCliError
from .contracts import Right, build_occ, parse_occ
from .identity import WrongAccountError, check as check_identity
from .loop import build_closing_proposal, current_structure_price
from .marketdata import MarketData
from .risk import Leg, PositionIntent, Proposal, Side, analyse_payoff
from .state import OpenTrade

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_ICON = {PASS: "[PASS]", FAIL: "[FAIL]", WARN: "[WARN]", INFO: "[INFO]"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = "") -> None:
        self.rows.append((status, check, detail))
        print(f"{_ICON[status]:8} {check:38} {detail}")

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)


def _spot(cli: AlpacaCLI, symbol: str = "SPY") -> float:
    payload = cli.latest_stock_quote([symbol])
    quote = ((payload or {}).get("quotes") or {}).get(symbol) or {}
    bid, ask = float(quote.get("bp", 0) or 0), float(quote.get("ap", 0) or 0)
    return (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0


def _nearest_the_money(
    chain: dict[str, Any], spot: float
) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    """Sample near the money, never whichever contract happens to come first.

    Chains arrive ordered by strike, so taking the first entry samples the
    deepest in-the-money contract on the board — where implied volatility is
    numerically unstable and often simply absent. Probing there reports "no IV
    available" for an account that has perfectly good IV at the strikes anyone
    would actually trade.
    """
    snapshots = (chain or {}).get("snapshots") or {}
    if not snapshots:
        return None, None
    if spot <= 0:
        symbol = next(iter(snapshots))
        return symbol, snapshots[symbol]
    symbol = min(snapshots, key=lambda s: abs(parse_occ(s).strike - spot))
    return symbol, snapshots[symbol]


def check_cli(cli: AlpacaCLI, report: Report) -> None:
    version = cli.run("version", parse=False).strip()
    report.add(PASS, "alpaca CLI on PATH", f"{version} at {cli.binary}")


def check_account(
    cli: AlpacaCLI, report: Report, *, require_fresh_equity: bool = True
) -> dict[str, Any]:
    try:
        account = cli.account()
    except AlpacaCliError as exc:
        report.add(FAIL, "authentication", f"{exc.stderr[:160]}")
        raise SystemExit(1) from exc

    # Never print the account number: it is a submission-form value, not a log value.
    report.add(PASS, "authentication", "credentials accepted")

    try:
        check_identity(account, recorded=None, require_expected=True)
    except WrongAccountError as exc:
        report.add(FAIL, "expected account identity", str(exc))
    else:
        report.add(PASS, "expected account identity", "named account matches credentials")

    equity = float(account.get("equity", 0))
    fresh = abs(equity - 100_000.0) <= 0.01
    equity_ok = fresh if require_fresh_equity else equity > 0
    report.add(
        PASS if equity_ok else FAIL,
        "equity",
        f"${equity:,.2f}"
        + (
            ""
            if fresh or not require_fresh_equity
            else "  <-- fresh judged account must be exactly $100,000.00"
        ),
    )

    level = account.get("options_trading_level")
    report.add(
        PASS if str(level) == "3" else FAIL,
        "options trading level",
        f"level {level} (need 3 for multi-leg)",
    )
    report.add(
        INFO,
        "options buying power",
        f"${float(account.get('options_buying_power', 0)):,.2f}",
    )

    for flag in ("trading_blocked", "account_blocked", "trade_suspended_by_user"):
        if account.get(flag):
            report.add(FAIL, f"account flag {flag}", "blocked")
    return account


def check_clock(cli: AlpacaCLI, report: Report) -> None:
    clock = cli.clock()
    state = "OPEN" if clock.get("is_open") else "CLOSED"
    report.add(INFO, "market clock", f"{state}, next open {clock.get('next_open')}")


def check_data_feed(cli: AlpacaCLI, report: Report) -> str:
    """Determine which options feed this plan allows. Returns the usable feed."""
    usable = None
    for feed in ("opra", "indicative"):
        try:
            chain = cli.option_chain("SPY", feed=feed, limit=5)
            count = len((chain or {}).get("snapshots") or {})
            if count:
                report.add(PASS, f"options feed '{feed}'", f"{count} snapshots returned")
                usable = usable or feed
            else:
                report.add(WARN, f"options feed '{feed}'", "accepted but returned nothing")
        except AlpacaCliError as exc:
            report.add(WARN, f"options feed '{feed}'", f"unavailable: {exc.stderr[:100]}")

    if usable is None:
        report.add(FAIL, "options data", "no usable feed — the desk cannot price anything")
        raise SystemExit(1)
    report.add(INFO, "feed selected", usable)
    return usable


def check_greeks_and_staleness(cli: AlpacaCLI, report: Report, feed: str) -> float:
    # Look ~30 days out: greeks are undefined at 0DTE (days-to-expiry in the
    # Black-Scholes denominator), so a near-dated probe would be a false negative.
    start = (date.today() + timedelta(days=21)).isoformat()
    end = (date.today() + timedelta(days=45)).isoformat()
    spot = _spot(cli)
    report.add(PASS if spot > 0 else WARN, "underlying spot (IEX)", f"SPY {spot:,.2f}")

    chain = cli.option_chain(
        "SPY", feed=feed, expiration_gte=start, expiration_lte=end,
        strike_gte=round(spot * 0.9, 2) if spot else None,
        strike_lte=round(spot * 1.1, 2) if spot else None,
        limit=1000,
    )
    snapshots = (chain or {}).get("snapshots") or {}
    symbol, snap = _nearest_the_money(chain, spot)
    if snap is None:
        report.add(FAIL, "chain snapshot", "no contracts returned for a 21-45 DTE window")
        return spot

    report.add(INFO, "chain size near the money", f"{len(snapshots)} contracts within +/-10%")

    greeks = snap.get("greeks") or {}
    have = [k for k in ("delta", "gamma", "theta", "vega", "rho") if greeks.get(k) is not None]
    report.add(
        PASS if len(have) >= 4 else WARN,
        "greeks in snapshot",
        f"{symbol}: {', '.join(have) or 'none'}",
    )
    iv = snap.get("impliedVolatility", snap.get("implied_volatility"))
    with_iv = sum(
        1 for r in snapshots.values()
        if r.get("impliedVolatility", r.get("implied_volatility")) is not None
    )
    report.add(
        PASS if with_iv else WARN,
        "implied volatility",
        (f"{iv:.4f} at the money; " if isinstance(iv, (int, float)) else "absent at the money; ")
        + f"{with_iv}/{len(snapshots)} contracts carry IV",
    )

    quote = snap.get("latestQuote") or snap.get("latest_quote") or {}
    if quote.get("t"):
        ts = datetime.fromisoformat(str(quote["t"]).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        report.add(
            PASS if age < 300 else WARN,
            "quote staleness",
            f"{age:,.0f}s old — size the risk.max_quote_age_s gate above this",
        )
        bid, ask = float(quote.get("bp", 0) or 0), float(quote.get("ap", 0) or 0)
        mid = (bid + ask) / 2
        report.add(
            INFO,
            "sample quote",
            f"{symbol} {bid:.2f} x {ask:.2f}"
            + (f"  spread {((ask - bid) / mid):.1%} of mid" if mid > 0 else ""),
        )
    return spot


def check_historical_options(cli: AlpacaCLI, report: Report, feed: str, spot: float) -> None:
    """The backtest gate is only credible if the history is actually there."""
    start = (date.today() + timedelta(days=21)).isoformat()
    end = (date.today() + timedelta(days=45)).isoformat()
    chain = cli.option_chain(
        "SPY", feed=feed, expiration_gte=start, expiration_lte=end, option_type="call",
        strike_gte=round(spot * 0.95, 2) if spot else None,
        strike_lte=round(spot * 1.05, 2) if spot else None,
        limit=1000,
    )
    symbol, _ = _nearest_the_money(chain, spot)
    if symbol is None:
        report.add(WARN, "historical option bars", "no contract to probe with")
        return
    try:
        bars = cli.option_bars([symbol], start=(date.today() - timedelta(days=60)).isoformat())
        series = ((bars or {}).get("bars") or {}).get(symbol) or []
        report.add(
            PASS if series else WARN,
            "historical option bars",
            f"{len(series)} daily bars for {symbol}",
        )
    except AlpacaCliError as exc:
        report.add(WARN, "historical option bars", exc.stderr[:120])


def build_probe_spread(cli: AlpacaCLI, feed: str, spot: float) -> Proposal | None:
    """A deliberately far-OTM 1-lot put credit spread on SPY, for path testing."""
    if spot <= 0:
        return None

    chain = cli.option_chain(
        "SPY",
        feed=feed,
        expiration_gte=(date.today() + timedelta(days=21)).isoformat(),
        expiration_lte=(date.today() + timedelta(days=45)).isoformat(),
        option_type="put",
        strike_gte=round(spot * 0.75, 2),
        strike_lte=round(spot * 0.85, 2),
        limit=1000,
    )
    symbols = sorted((chain or {}).get("snapshots") or {})
    if len(symbols) < 2:
        return None

    # Group by expiry FIRST. Sorting the whole chain by strike alone happily
    # pairs two same-strike contracts from different expiries, which is a
    # calendar, not a vertical - and the payoff model cannot analyse one.
    by_expiry: dict[Any, list] = {}
    for symbol in symbols:
        parsed = parse_occ(symbol)
        by_expiry.setdefault(parsed.expiry, []).append(parsed)

    usable = [legs for legs in by_expiry.values() if len(legs) >= 2]
    if not usable:
        return None
    legs = sorted(max(usable, key=len), key=lambda p: p.strike)

    # Two adjacent strikes within that one expiry: sell the higher, buy the lower.
    short_leg, long_leg = legs[-1], legs[-2]

    return Proposal(
        strategy_id="PREFLIGHT",
        underlying="SPY",
        legs=(
            Leg(str(short_leg), Side.SELL, 1, PositionIntent.SELL_TO_OPEN),
            Leg(str(long_leg), Side.BUY, 1, PositionIntent.BUY_TO_OPEN),
        ),
        qty=1,
        net_price=-0.05,  # negative = net credit
        rationale="preflight path test only",
    )


def check_order_path(
    cli: AlpacaCLI, report: Report, feed: str, spot: float, live_test: bool
) -> None:
    proposal = build_probe_spread(cli, feed, spot)
    if proposal is None:
        report.add(WARN, "mleg order path", "could not build a probe spread from the chain")
        return

    profile = analyse_payoff(proposal)
    report.add(
        PASS if profile.is_defined_risk else FAIL,
        "probe spread risk",
        f"max loss ${profile.max_loss_or_inf:,.0f} on "
        f"{' / '.join(l.symbol for l in proposal.legs)}",
    )

    try:
        body = cli.submit_mleg(proposal, dry_run=True)
        report.add(PASS, "mleg --dry-run", "request body accepted by the CLI")
        print(json.dumps(body, indent=2)[:900] if isinstance(body, (dict, list)) else str(body)[:900])
    except AlpacaCliError as exc:
        report.add(FAIL, "mleg --dry-run", exc.stderr[:200])
        return

    if not live_test:
        report.add(INFO, "live order test", "skipped (pass --live-test to send one real order)")
        return

    client_id = f"preflight-entry-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    try:
        order = cli.submit_mleg(proposal, client_order_id=client_id)
        order_id = (order or {}).get("id", "?")
        report.add(PASS, "live mleg submit", f"accepted, order {order_id}")
    except AlpacaCliError as exc:
        report.add(FAIL, "live mleg submit", exc.stderr[:200])
        return

    if order_id == "?":
        report.add(FAIL, "probe cleanup", "accepted order returned no id; inspect Alpaca manually")
        return

    try:
        final = _wait_order(cli, order_id, timeout_s=3.0)
        if str(final.get("status") or "").lower() not in {
            "filled", "canceled", "expired", "rejected"
        }:
            cli.cancel_order(order_id)
            final = _wait_order(cli, order_id, timeout_s=5.0)
    except AlpacaCliError as exc:
        # A cancel often races a fill. Re-read once before deciding whether a
        # structure exists that must be closed.
        report.add(WARN, "probe order cancel", exc.stderr[:120])
        try:
            final = cli.order(order_id)
        except AlpacaCliError as read_exc:
            report.add(FAIL, "probe cleanup", f"cannot determine order state: {read_exc.stderr[:120]}")
            return

    filled_qty = int(float(final.get("filled_qty") or 0))
    if filled_qty <= 0:
        status = str(final.get("status") or "unknown").lower()
        report.add(
            PASS if status in {"canceled", "expired", "rejected"} else FAIL,
            "probe order cancelled",
            f"terminal status {status}; no position opened",
        )
        return

    _close_filled_probe(cli, report, feed, proposal, final, filled_qty)
    if not report.failed:
        report.add(
            INFO,
            "sign convention",
            "check the fill: a negative limit_price must book as a CREDIT",
        )


def _wait_order(
    cli: AlpacaCLI, order_id: str, *, timeout_s: float, poll_s: float = 0.5
) -> dict[str, Any]:
    """Poll a parent order until terminal, returning the latest known payload."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    latest: dict[str, Any] = {}
    while True:
        latest = cli.order(order_id)
        if str(latest.get("status") or "").lower() in {
            "filled", "canceled", "expired", "rejected"
        }:
            return latest
        if time.monotonic() >= deadline:
            return latest
        time.sleep(max(poll_s, 0.05))


def _close_filled_probe(
    cli: AlpacaCLI,
    report: Report,
    feed: str,
    opening: Proposal,
    order: dict[str, Any],
    filled_qty: int,
) -> None:
    """Close the probe as one structure; never expose a naked intermediate leg."""
    trade = OpenTrade(
        client_order_id=str(order.get("client_order_id") or "preflight-entry"),
        strategy_id="PREFLIGHT",
        underlying=opening.underlying,
        legs=[leg.symbol for leg in opening.legs],
        qty=filled_qty,
        net_price=float(order.get("filled_avg_price") or opening.net_price),
        max_loss=analyse_payoff(opening).max_loss_or_inf,
        opened_at=str(order.get("filled_at") or datetime.now(timezone.utc).isoformat()),
        leg_sides={leg.symbol: leg.side.value for leg in opening.legs},
        leg_ratios={leg.symbol: leg.ratio for leg in opening.legs},
        status="open",
    )
    price = current_structure_price(MarketData(cli=cli, feed=feed), trade)
    if price is None:
        report.add(
            FAIL,
            "probe mleg close",
            "filled spread could not be priced; close it manually as one multi-leg order",
        )
        return

    closing = build_closing_proposal(trade, price)
    close_client_id = f"preflight-close-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    try:
        close_order = cli.submit_mleg(closing, client_order_id=close_client_id)
        close_id = str((close_order or {}).get("id") or "")
        if not close_id:
            report.add(FAIL, "probe mleg close", "accepted close returned no order id")
            return
        final = _wait_order(cli, close_id, timeout_s=20.0)
    except AlpacaCliError as exc:
        report.add(
            FAIL,
            "probe mleg close",
            f"{exc.stderr[:140]} — close manually as one multi-leg order",
        )
        return

    closed_qty = int(float(final.get("filled_qty") or 0))
    if closed_qty >= filled_qty:
        report.add(PASS, "probe mleg close", f"all {filled_qty} spread(s) closed atomically")
        return
    report.add(
        FAIL,
        "probe mleg close",
        f"close order {close_id} is {final.get('status', 'unknown')} with "
        f"{closed_qty}/{filled_qty} filled; leave it working and do not start the desk",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aperture desk preflight")
    parser.add_argument(
        "--live-test",
        action="store_true",
        help="also send one real 1-lot defined-risk spread to prove the fill path",
    )
    parser.add_argument("--binary", default="alpaca", help="path to the alpaca CLI")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an already-traded account; relax only the fresh $100,000 equity check",
    )
    args = parser.parse_args(argv)

    report = Report()
    cli = AlpacaCLI(binary=args.binary)

    check_cli(cli, report)
    check_account(cli, report, require_fresh_equity=not args.resume)
    check_clock(cli, report)
    feed = check_data_feed(cli, report)
    spot = check_greeks_and_staleness(cli, report, feed)
    check_historical_options(cli, report, feed, spot)
    check_order_path(cli, report, feed, spot, args.live_test)

    print()
    print("FAILED — fix the above before trading." if report.failed else "All critical checks passed.")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
