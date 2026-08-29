"""Typed wrapper around the Alpaca CLI — the desk's execution spine.

Why the CLI and not the SDK for execution: the CLI is built for exactly this
shape of workload. It emits structured JSON, returns meaningful exit codes,
retries rate limits on its own, takes ``--client-order-id`` for idempotency and
``--dry-run`` for a free pre-trade check, and it needs no language model in the
loop to move money. The strategy and research layers are the desk's brain; this
is their deliberately narrow set of hands.

Verified against alpaca CLI v0.0.13, whose ``order submit`` exposes ``--legs``
and ``--order-class mleg`` natively (the docs site does not mention this).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

from .risk import Proposal

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_AUTH = 2


class AlpacaCliError(RuntimeError):
    def __init__(self, argv: Sequence[str], returncode: int, stderr: str):
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr.strip()
        redacted = " ".join(argv)
        super().__init__(f"`alpaca {redacted}` exited {returncode}: {self.stderr}")


@dataclass
class AlpacaCLI:
    """Thin, synchronous shell over the `alpaca` binary."""

    binary: str = "alpaca"
    timeout: int = 30
    paper: bool = True

    def __post_init__(self) -> None:
        resolved = shutil.which(self.binary)
        if resolved is None:
            raise FileNotFoundError(
                f"{self.binary!r} not on PATH. Install with: "
                "go install github.com/alpacahq/cli/cmd/alpaca@latest"
            )
        self.binary = resolved

    # -- plumbing --------------------------------------------------------- #

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Paper is the CLI default; be explicit rather than trusting the default.
        env["ALPACA_LIVE_TRADE"] = "false" if self.paper else "true"
        return env

    def run(self, *args: str, parse: bool = True) -> Any:
        argv = [self.binary, *args, "--quiet"]
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=self._env(),
        )
        if proc.returncode != EXIT_OK:
            raise AlpacaCliError(args, proc.returncode, proc.stderr)
        if not parse:
            return proc.stdout
        stdout = proc.stdout.strip()
        if not stdout:
            return None
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise AlpacaCliError(args, EXIT_OK, f"non-JSON output: {stdout[:400]}") from exc

    # -- account ---------------------------------------------------------- #

    def account(self) -> dict[str, Any]:
        return self.run("account", "get")

    def account_config(self) -> dict[str, Any]:
        return self.run("account", "config", "get")

    def clock(self) -> dict[str, Any]:
        return self.run("clock")

    def positions(self) -> list[dict[str, Any]]:
        return self.run("position", "list") or []

    def orders(self, status: str = "open") -> list[dict[str, Any]]:
        return self.run("order", "list", "--status", status) or []

    def order(self, order_id: str) -> dict[str, Any]:
        """Return one order, including its fill state.

        Submission acceptance is not a fill.  The desk polls the parent mleg
        order before it promotes a pending ledger reservation into a position.
        """
        return self.run("order", "get", "--order-id", order_id, "--nested") or {}

    def order_by_client_id(self, client_order_id: str) -> dict[str, Any]:
        """Recover an order when Alpaca accepted it without returning an id."""
        return self.run(
            "order", "get-by-client-id", "--client-order-id", client_order_id
        ) or {}

    def portfolio_history(self, period: str = "1W", timeframe: str = "1H") -> dict[str, Any]:
        return self.run("account", "portfolio", "--period", period, "--timeframe", timeframe)

    # -- options market data ---------------------------------------------- #

    def option_chain(
        self,
        underlying: str,
        *,
        feed: str = "indicative",
        expiration_date: str | None = None,
        expiration_gte: str | None = None,
        expiration_lte: str | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        limit: int = 1000,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """One page of a live chain snapshot, with greeks where Alpaca computes them.

        ``feed`` defaults to ``indicative`` rather than the CLI's own ``opra``
        default: OPRA needs a signed agreement and a paid data plan, and an
        account without one gets a hard 403 rather than a fallback.

        ``limit`` is a hard truncation applied *after* the strike filter, not a
        page size that preserves the filtered range — a low limit silently
        returns only the lowest strikes and makes the chain look like it stops
        far below the money. Callers should paginate; see ``MarketData.chain``.
        """
        args = ["data", "option", "chain", "--underlying-symbol", underlying,
                "--feed", feed, "--limit", str(limit)]
        for flag, value in (
            ("--expiration-date", expiration_date),
            ("--expiration-date-gte", expiration_gte),
            ("--expiration-date-lte", expiration_lte),
            ("--strike-price-gte", strike_gte),
            ("--strike-price-lte", strike_lte),
            ("--type", option_type),
            ("--page-token", page_token),
        ):
            if value is not None:
                args += [flag, str(value)]
        return self.run(*args)

    def option_snapshot(self, symbols: Sequence[str], *, feed: str = "indicative") -> dict[str, Any]:
        return self.run(
            "data", "option", "snapshot", "--symbols", ",".join(symbols), "--feed", feed
        )

    def option_bars(
        self,
        symbols: Sequence[str],
        start: str,
        *,
        timeframe: str = "1Day",
        limit: int = 1000,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """One historical option-bar page.

        Alpaca's limit applies to total data points, not to each symbol.  A
        multi-symbol response commonly ends midway through the call side of a
        chain, so research callers must follow ``next_page_token`` until it is
        absent rather than treating this page as a complete history.
        """
        args = [
            "data", "option", "bars",
            "--symbols", ",".join(symbols),
            "--start", start,
            "--timeframe", timeframe,
            "--limit", str(limit),
        ]
        if page_token is not None:
            args += ["--page-token", page_token]
        return self.run(*args)

    def stock_bars(
        self, symbol: str, start: str, *, timeframe: str = "1Day", limit: int = 1000
    ) -> dict[str, Any]:
        """Historical stock bars for ONE symbol.

        The CLI is inconsistent here: ``data bars`` takes a singular ``--symbol``
        while ``data latest-quotes`` takes a comma-separated ``--symbols``. Passing
        the wrong one fails with "unknown flag", not a validation message.
        """
        return self.run(
            "data", "bars",
            "--symbol", symbol,
            "--start", start,
            "--timeframe", timeframe,
            "--limit", str(limit),
        )

    def latest_stock_quote(self, symbols: Sequence[str]) -> dict[str, Any]:
        return self.run("data", "latest-quotes", "--symbols", ",".join(symbols))

    # -- execution -------------------------------------------------------- #

    def submit_mleg(
        self,
        proposal: Proposal,
        *,
        client_order_id: str | None = None,
        time_in_force: str = "day",
        dry_run: bool = False,
    ) -> Any:
        """Submit a multi-leg options order.

        Alpaca's sign convention for ``limit_price`` on an mleg order: positive is
        a net debit, negative is a net credit. ``Proposal.net_price`` already uses
        that convention, so it passes straight through.
        """
        legs = [
            {
                "symbol": leg.symbol,
                "side": leg.side.value,
                "ratio_qty": str(leg.ratio),
                "position_intent": leg.intent.value,
            }
            for leg in proposal.legs
        ]
        args = [
            "order", "submit",
            "--order-class", "mleg",
            "--qty", str(proposal.qty),
            "--type", "limit",
            "--limit-price", f"{proposal.net_price:.2f}",
            "--time-in-force", time_in_force,
            "--legs", json.dumps(legs, separators=(",", ":")),
            "--client-order-id", client_order_id or idempotency_key(proposal),
        ]
        if dry_run:
            args.append("--dry-run")
        return self.run(*args)

    def cancel_order(self, order_id: str) -> Any:
        return self.run("order", "cancel", "--order-id", order_id)

    def close_position(self, symbol: str) -> Any:
        return self.run("position", "close", "--symbol", symbol)

    def close_all_positions(self) -> Any:
        """The kill switch's last step. Irreversible — callers must mean it."""
        return self.run("position", "close-all")


def idempotency_key(proposal: Proposal, *, salt: str = "") -> str:
    """Deterministic client_order_id, so a retry can never double-fill.

    Two calls describing the same structure at the same size produce the same id,
    and Alpaca rejects the duplicate rather than opening a second position.
    """
    payload = "|".join(
        [
            proposal.strategy_id,
            proposal.underlying,
            str(proposal.qty),
            f"{proposal.net_price:.2f}",
            *(f"{leg.symbol}:{leg.side.value}:{leg.ratio}" for leg in proposal.legs),
            salt,
        ]
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"aperture-{proposal.strategy_id.lower()}-{digest}"
