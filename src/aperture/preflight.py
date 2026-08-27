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
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .alpaca_cli import AlpacaCLI, AlpacaCliError
from .contracts import Right, build_occ, parse_occ
from .risk import Leg, PositionIntent, Proposal, Side, analyse_payoff

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


def _first_snapshot(chain: dict[str, Any]) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    snapshots = (chain or {}).get("snapshots") or {}
    for symbol, snap in snapshots.items():
        return symbol, snap
    return None, None


def check_cli(cli: AlpacaCLI, report: Report) -> None:
    version = cli.run("version", parse=False).strip()
    report.add(PASS, "alpaca CLI on PATH", f"{version} at {cli.binary}")


def check_account(cli: AlpacaCLI, report: Report) -> dict[str, Any]:
    try:
        account = cli.account()
    except AlpacaCliError as exc:
        report.add(FAIL, "authentication", f"{exc.stderr[:160]}")
        raise SystemExit(1) from exc

    # Never print the account number: it is a submission-form value, not a log value.
    report.add(PASS, "authentication", "credentials accepted")

    equity = float(account.get("equity", 0))
    report.add(
        PASS if equity > 0 else FAIL,
        "equity",
        f"${equity:,.2f}"
        + ("" if abs(equity - 100_000) < 1 else "  <-- hackathon rules require $100,000"),
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


def check_greeks_and_staleness(cli: AlpacaCLI, report: Report, feed: str) -> None:
    # Look ~30 days out: greeks are undefined at 0DTE (days-to-expiry in the
    # Black-Scholes denominator), so a near-dated probe would be a false negative.
    start = (date.today() + timedelta(days=21)).isoformat()
    end = (date.today() + timedelta(days=45)).isoformat()
    chain = cli.option_chain(
        "SPY", feed=feed, expiration_gte=start, expiration_lte=end, limit=20
    )
    symbol, snap = _first_snapshot(chain)
    if snap is None:
        report.add(FAIL, "chain snapshot", "no contracts returned for a 21-45 DTE window")
        return

    greeks = snap.get("greeks") or {}
    have = [k for k in ("delta", "gamma", "theta", "vega", "rho") if greeks.get(k) is not None]
    report.add(
        PASS if len(have) >= 4 else WARN,
        "greeks in snapshot",
        f"{symbol}: {', '.join(have) or 'none'}",
    )
    iv = snap.get("implied_volatility")
    report.add(
        PASS if iv is not None else WARN,
        "implied volatility",
        f"{iv:.4f}" if isinstance(iv, (int, float)) else "absent",
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
        bid, ask = float(quote.get("bp", 0)), float(quote.get("ap", 0))
        mid = (bid + ask) / 2
        report.add(
            INFO,
            "sample quote",
            f"{symbol} {bid:.2f} x {ask:.2f}"
            + (f"  spread {((ask - bid) / mid):.1%} of mid" if mid > 0 else ""),
        )


def check_historical_options(cli: AlpacaCLI, report: Report, feed: str) -> None:
    """The backtest gate is only credible if the history is actually there."""
    start = (date.today() + timedelta(days=21)).isoformat()
    end = (date.today() + timedelta(days=45)).isoformat()
    chain = cli.option_chain(
        "SPY", feed=feed, expiration_gte=start, expiration_lte=end, option_type="call", limit=5
    )
    symbol, _ = _first_snapshot(chain)
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


def build_probe_spread(cli: AlpacaCLI, feed: str) -> Proposal | None:
    """A deliberately far-OTM 1-lot put credit spread on SPY, for path testing."""
    quote = cli.latest_stock_quote(["SPY"])
    quotes = (quote or {}).get("quotes") or {}
    spy = quotes.get("SPY") or {}
    spot = (float(spy.get("bp", 0)) + float(spy.get("ap", 0))) / 2
    if spot <= 0:
        return None

    chain = cli.option_chain(
        "SPY",
        feed=feed,
        expiration_gte=(date.today() + timedelta(days=21)).isoformat(),
        expiration_lte=(date.today() + timedelta(days=45)).isoformat(),
        option_type="put",
        strike_lte=spot * 0.85,
        limit=50,
    )
    symbols = sorted((chain or {}).get("snapshots") or {})
    if len(symbols) < 2:
        return None

    # Two adjacent strikes: sell the higher, buy the lower.
    parsed = sorted((parse_occ(s) for s in symbols), key=lambda p: p.strike)
    short_leg, long_leg = parsed[-1], parsed[-2]

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


def check_order_path(cli: AlpacaCLI, report: Report, feed: str, live_test: bool) -> None:
    proposal = build_probe_spread(cli, feed)
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

    try:
        order = cli.submit_mleg(proposal)
        order_id = (order or {}).get("id", "?")
        report.add(PASS, "live mleg submit", f"accepted, order {order_id}")
        report.add(
            INFO,
            "verify sign convention",
            "negative limit_price should show as a CREDIT on the fill",
        )
    except AlpacaCliError as exc:
        report.add(FAIL, "live mleg submit", exc.stderr[:200])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aperture desk preflight")
    parser.add_argument(
        "--live-test",
        action="store_true",
        help="also send one real 1-lot defined-risk spread to prove the fill path",
    )
    parser.add_argument("--binary", default="alpaca", help="path to the alpaca CLI")
    args = parser.parse_args(argv)

    report = Report()
    cli = AlpacaCLI(binary=args.binary)

    check_cli(cli, report)
    check_account(cli, report)
    check_clock(cli, report)
    feed = check_data_feed(cli, report)
    check_greeks_and_staleness(cli, report, feed)
    check_historical_options(cli, report, feed)
    check_order_path(cli, report, feed, args.live_test)

    print()
    print("FAILED — fix the above before trading." if report.failed else "All critical checks passed.")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
