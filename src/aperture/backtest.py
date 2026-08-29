"""Historical simulation of defined-risk option structures.

This is what stands behind the claim that the desk *invented* a strategy rather
than merely generated one. A candidate that has not been tested against real
prices is a guess with a config file.

**What this simulator does and does not know**, stated plainly because a backtest
that overstates itself is worse than none:

  * **Daily bars only.** Entries and exits happen at closes. Intraday paths are
    invisible, so a structure that touched its stop and recovered inside a
    session is recorded as never having touched it. This flatters stops.
  * **No historical greeks.** Alpaca serves greeks on live snapshots, not on
    history, so strikes are chosen by *moneyness* rather than delta. A 15-delta
    short is approximated as a fixed percentage out of the money, which is close
    at the tenors traded here and wrong in a volatility spike.
  * **Last-trade closes, with a fixed adverse concession.** Alpaca historical
    option bars do not contain quote mids. Cross-leg timestamps can differ, so
    even this conservative adjustment cannot prove the structure was fillable.
  * **Survivorship is not an issue** (contracts do not disappear) **but liquidity
    is**: a strike that never traded still has bars, and the simulator cannot
    tell that nobody would have filled you there.

Every one of those biases points the same way — toward flattering results. The
promotion gate in ``research.py`` is calibrated with that in mind.
"""

from __future__ import annotations

import hashlib
import logging
import math
import pickle
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from .alpaca_cli import AlpacaCLI, AlpacaCliError
from .contracts import Right, parse_occ
from .marketdata import _parse_ts

log = logging.getLogger(__name__)

CONTRACT_MULTIPLIER = 100
HISTORY_CACHE_VERSION = 3
HISTORY_MIN_STRIKE_BAND = 0.12
HISTORY_STRIKE_BUFFER = 0.04


@dataclass
class HistoricalBar:
    day: date
    close: float


@dataclass
class OptionHistory:
    """Daily closes for every contract we could load, indexed for fast lookup."""

    underlying: str
    spot: dict[date, float] = field(default_factory=dict)
    bars: dict[str, dict[date, float]] = field(default_factory=dict)
    _contracts_index: dict[tuple[date, Right], list[str]] = field(
        default_factory=dict, init=False, repr=False
    )

    def price(self, symbol: str, day: date) -> float | None:
        return self.bars.get(symbol, {}).get(day)

    def sessions(self) -> list[date]:
        return sorted(self.spot)

    def contracts_for(self, expiry: date, right: Right) -> list[str]:
        # Simulation asks this twice per session and once per candidate. Build
        # the immutable catalogue index once instead of reparsing every OCC
        # symbol hundreds of millions of times during a full-window sweep.
        if not self._contracts_index and self.bars:
            for symbol in self.bars:
                try:
                    parsed = parse_occ(symbol)
                except ValueError:
                    continue
                self._contracts_index.setdefault((parsed.expiry, parsed.right), []).append(
                    symbol
                )
            for symbols in self._contracts_index.values():
                symbols.sort(key=lambda value: parse_occ(value).strike)
        return self._contracts_index.get((expiry, right), [])

    def between(self, start: date | None = None, end: date | None = None) -> "OptionHistory":
        """Chronological slice used to keep selection and validation separate."""
        def inside(day: date) -> bool:
            return (start is None or day >= start) and (end is None or day <= end)

        return OptionHistory(
            underlying=self.underlying,
            spot={day: value for day, value in self.spot.items() if inside(day)},
            bars={
                symbol: {day: value for day, value in table.items() if inside(day)}
                for symbol, table in self.bars.items()
                if any(inside(day) for day in table)
            },
        )


@dataclass
class SimulatedTrade:
    entry: date
    exit: date
    legs: tuple[str, ...]
    qty: int
    entry_price: float  # Alpaca convention: + debit, - credit
    exit_price: float
    max_loss: float
    reason: str

    @property
    def pnl(self) -> float:
        # Opened at entry_price, closed at exit_price, both quoted the same way.
        return (self.entry_price - self.exit_price) * self.qty * CONTRACT_MULTIPLIER * -1


@dataclass
class BacktestResult:
    strategy_id: str
    trades: list[SimulatedTrade] = field(default_factory=list)
    diagnostics: dict[str, int] = field(default_factory=dict)

    def note(self, outcome: str, count: int = 1) -> None:
        """Count why an otherwise eligible simulation step did not become a trade."""
        self.diagnostics[outcome] = self.diagnostics.get(outcome, 0) + count

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_risk(self) -> float:
        return sum(t.max_loss for t in self.trades)

    @property
    def edge(self) -> float:
        """Mean P&L per dollar of risk, the same unit the allocator scores on."""
        return self.total_pnl / self.total_risk if self.total_risk > 0 else 0.0

    @property
    def returns(self) -> list[float]:
        return [t.pnl / t.max_loss for t in self.trades if t.max_loss > 0]

    @property
    def t_stat(self) -> float:
        """How much of the edge survives its own noise.

        With a handful of trades this is the only number worth trusting, and even
        then only as a filter against obvious luck.
        """
        series = self.returns
        if len(series) < 3:
            return 0.0
        spread = statistics.stdev(series)
        if spread <= 0:
            return 0.0
        return statistics.mean(series) / (spread / math.sqrt(len(series)))

    @property
    def max_drawdown(self) -> float:
        peak = running = 0.0
        worst = 0.0
        for trade in self.trades:
            running += trade.pnl
            peak = max(peak, running)
            worst = min(worst, running - peak)
        return abs(worst)

    def summary(self) -> str:
        if not self.n:
            return f"{self.strategy_id}: no trades"
        return (
            f"{self.strategy_id}: {self.n} trades, {self.wins}/{self.n} wins, "
            f"edge {self.edge:+.1%} of risk, t={self.t_stat:.2f}, "
            f"maxDD ${self.max_drawdown:,.0f}"
        )


# --------------------------------------------------------------------------- #
# Loading history
# --------------------------------------------------------------------------- #


def _target_expiries(start: date, end: date, count: int) -> list[date]:
    """Fridays spread evenly across the window.

    Sampling deliberately, rather than taking whatever the contract listing
    returns first. The listing comes back in expiry order, so a naive cap yields
    hundreds of contracts that all expire on the same early date -- every one of
    them with a single bar, and nothing for a walk-forward to walk through.
    """
    fridays = []
    day = start
    while day <= end:
        if day.weekday() == 4:
            fridays.append(day)
        day += timedelta(days=1)
    if not fridays:
        return []
    step = max(len(fridays) // max(count, 1), 1)
    return fridays[::step][:count]


def _universe_signature(specs: Sequence["CondorSpec"]) -> str:
    rows = sorted({
        (
            float(spec.short_pct),
            float(spec.width_pct),
            int(spec.dte_target),
        )
        for spec in specs
    })
    return ";".join(f"{short:.6f},{width:.6f},{dte}" for short, width, dte in rows)


def _strike_window(
    spot: dict[date, float],
    expiry: date,
    specs: Sequence["CondorSpec"],
) -> tuple[float, float] | None:
    """Exact historical catalogue window needed by the selected strategies.

    The live hired strategy requests at least a twelve-percent band around spot.
    History mirrors that rule over every session on which any selected spec could
    enter. This makes candidate geometry observable without relying on a magic
    number of strikes or on spot from one arbitrary anchor session.
    """
    lows: list[float] = []
    highs: list[float] = []
    for spec in specs:
        strike_band = max(
            float(spec.short_pct) + float(spec.width_pct) + HISTORY_STRIKE_BUFFER,
            HISTORY_MIN_STRIKE_BAND,
        )
        max_dte = int(spec.dte_target) + 14
        for day, price in spot.items():
            dte = (expiry - day).days
            if 5 <= dte <= max_dte:
                lows.append(price * (1 - strike_band))
                highs.append(price * (1 + strike_band))
    if not lows:
        return None
    # Round outwards: a catalogue bound must never trim the boundary strike.
    return math.floor(min(lows) * 100) / 100, math.ceil(max(highs) * 100) / 100


def _cache_key(
    underlying: str,
    start: date,
    end: date,
    expiries: int,
    universe_signature: str,
) -> str:
    # The loader's semantics are part of the cache identity. Version 2 follows
    # Alpaca's option-bar pagination; reusing a version-1 file would silently
    # preserve the truncated chain that this loader is designed to prevent.
    raw = (
        f"v{HISTORY_CACHE_VERSION}|{underlying}|{start}|{end}|{expiries}|"
        f"{universe_signature}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_history(
    cli: AlpacaCLI,
    underlying: str,
    *,
    start: date,
    end: date,
    expiries: int = 10,
    universe_specs: Sequence["CondorSpec"] | None = None,
    cache_dir: Path | str | None = "state/history",
) -> OptionHistory:
    """Fetch underlying and option daily bars for a window.

    Contracts are pulled one expiry at a time so that each has dense coverage
    across its whole life, which is what a walk-forward simulation needs. Only
    resolved windows work: contract discovery uses the expired listing.

    The contract universe is derived from the selected strategy geometries and
    every actual spot session on which they could enter. It mirrors the live
    strategy's strike-band rule, and retains every listed contract in that band;
    a strategy is never rejected merely because an arbitrary strike count hid
    its wings.

    The fetch is thousands of API calls and takes minutes, while the window it
    describes is entirely in the past and cannot change. It is cached on disk so
    an interrupted sweep does not pay for the same history twice.
    """
    specs = list(universe_specs or [CondorSpec()])
    universe_signature = _universe_signature(specs)
    cache_path = None
    if cache_dir is not None:
        key = _cache_key(underlying, start, end, expiries, universe_signature)
        cache_path = Path(cache_dir) / f"{underlying}-{key}.pkl"
        if cache_path.exists():
            try:
                history = pickle.loads(cache_path.read_bytes())
                log.info(
                    "reusing cached history: %d sessions, %d contracts (%s)",
                    len(history.spot), len(history.bars), cache_path.name,
                )
                return history
            except Exception as exc:  # noqa: BLE001 - a bad cache is never fatal
                log.warning("cached history unreadable, refetching: %s", exc)

    history = OptionHistory(underlying=underlying)

    for bar in _bars(cli.stock_bars(underlying, start=start.isoformat()), underlying):
        day = _parse_ts(bar.get("t")).date()
        if start <= day <= end and bar.get("c"):
            history.spot[day] = float(bar["c"])

    if not history.spot:
        log.warning("no underlying history for %s", underlying)
        return history

    sessions = history.sessions()
    for expiry in _target_expiries(start, end, expiries):
        strike_window = _strike_window(history.spot, expiry, specs)
        if strike_window is None:
            continue
        strike_low, strike_high = strike_window

        # Calls and puts are requested separately. The listing is ordered by
        # symbol, so "C" sorts ahead of "P" and any head-slice of a combined
        # response returns nothing but calls -- which silently yields a chain
        # with no put side and a condor that can never be built.
        symbols: list[str] = []
        for option_type in ("call", "put"):
            try:
                payload = cli.run(
                    "option", "contracts",
                    "--underlying-symbols", underlying,
                    "--expiration-date", expiry.isoformat(),
                    "--strike-price-gte", f"{strike_low:.2f}",
                    "--strike-price-lte", f"{strike_high:.2f}",
                    "--type", option_type,
                    "--status", "inactive",
                    "--limit", "10000",
                )
            except AlpacaCliError as exc:
                log.warning("%s contracts for %s failed: %s", option_type, expiry, exc.stderr[:100])
                continue
            found = [
                c["symbol"]
                for c in (payload or {}).get("option_contracts") or []
                if c.get("symbol")
            ]
            symbols.extend(found)

        if not symbols:
            continue

        bars_from = (expiry - timedelta(days=60)).isoformat()
        for chunk in (symbols[i:i + 100] for i in range(0, len(symbols), 100)):
            try:
                bars_by_symbol = _all_option_bars(cli, chunk, start=bars_from)
            except AlpacaCliError as exc:
                log.warning("option bars failed: %s", exc.stderr[:100])
                raise
            for symbol, series in bars_by_symbol.items():
                table = history.bars.setdefault(symbol, {})
                for bar in series or []:
                    if bar.get("c"):
                        table[_parse_ts(bar.get("t")).date()] = float(bar["c"])

    if cache_path is not None and history.bars:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(history))
        tmp.replace(cache_path)
        log.info("history cached to %s", cache_path)

    dense = sum(1 for t in history.bars.values() if len(t) >= 10)
    log.info(
        "%s: %d sessions, %d contracts (%d with 10+ bars), %d expiries",
        underlying, len(history.spot), len(history.bars), dense,
        len({parse_occ(s).expiry for s in history.bars}),
    )
    return history


MAX_OPTION_BAR_PAGES = 100


def _all_option_bars(
    cli: AlpacaCLI,
    symbols: Sequence[str],
    *,
    start: str,
    max_pages: int = MAX_OPTION_BAR_PAGES,
) -> dict[str, list[dict[str, Any]]]:
    """Read every option-bar page or fail rather than backtest a half-chain."""
    merged: dict[str, list[dict[str, Any]]] = {}
    page_token: str | None = None
    seen_tokens: set[str] = set()

    for _ in range(max_pages):
        payload = cli.option_bars(symbols, start=start, page_token=page_token) or {}
        for symbol, series in (payload.get("bars") or {}).items():
            merged.setdefault(symbol, []).extend(series or [])

        next_token = payload.get("next_page_token")
        if not next_token:
            return merged
        next_token = str(next_token)
        if next_token in seen_tokens:
            raise RuntimeError("Alpaca option-bar pagination repeated a page token")
        seen_tokens.add(next_token)
        page_token = next_token

    raise RuntimeError(
        f"Alpaca option-bar history exceeded the {max_pages}-page safety bound"
    )


def _bars(payload: dict[str, Any] | None, symbol: str) -> list[dict[str, Any]]:
    bars = (payload or {}).get("bars")
    if isinstance(bars, list):
        return bars
    if isinstance(bars, dict):
        return bars.get(symbol) or []
    return []


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CondorSpec:
    """A short iron condor described by moneyness rather than delta.

    Delta is unavailable historically, so the short strikes sit a fixed
    percentage out of the money. At 7-21 days that tracks a 10-20 delta short
    reasonably well, and diverges when volatility moves sharply.
    """

    short_pct: float = 0.04     # short strikes this far OTM
    width_pct: float = 0.01     # wing width, as a fraction of spot
    dte_target: int = 14
    take_profit: float = 0.50
    stop_multiple: float = 2.0
    slippage: float = 0.05
    min_credit_to_width: float = 0.15


def simulate_condors(
    history: OptionHistory, spec: CondorSpec, *, strategy_id: str = "CANDIDATE"
) -> BacktestResult:
    """Walk forward, opening at most one condor per expiry and managing it to a rule.

    Treating an early exit and a re-entry into the same expiry as independent
    observations inflates both the sample size and the t-statistic.  The lab is
    deliberately stricter: one expiry is one opportunity, even if a profit
    target is reached quickly.
    """
    result = BacktestResult(strategy_id=strategy_id)
    sessions = history.sessions()
    result.diagnostics["sessions"] = len(sessions)
    if len(sessions) < 10:
        result.note("insufficient_sessions")
        return result

    # A chronological slice can still contain bars for a contract whose expiry
    # lies beyond the slice.  Such a contract cannot be settled without looking
    # into the future, so it is ineligible here.  This is essential at the
    # training/holdout boundary.
    expiries = sorted(
        expiry
        for expiry in {parse_occ(s).expiry for s in history.bars}
        if expiry <= sessions[-1]
    )
    open_until: date | None = None

    for today in sessions:
        if open_until and today <= open_until:
            result.note("position_overlap")
            continue

        spot = history.spot[today]
        expiry = _nearest_expiry(expiries, today, spec.dte_target)
        if expiry is None:
            result.note("no_eligible_expiry")
            continue

        legs = _pick_condor(history, expiry, spot, spec, day=today)
        if legs is None:
            priceable_puts = sum(
                history.price(symbol, today) is not None
                for symbol in history.contracts_for(expiry, Right.PUT)
            )
            priceable_calls = sum(
                history.price(symbol, today) is not None
                for symbol in history.contracts_for(expiry, Right.CALL)
            )
            result.note(
                "missing_priceable_side"
                if min(priceable_puts, priceable_calls) < 2
                else "invalid_strike_geometry"
            )
            continue

        entry = _price(history, legs, today)
        if entry is None:
            result.note("incomplete_entry_price")
            continue
        if entry >= 0:
            result.note("not_a_credit")
            continue  # a condor that is not a credit is not this structure

        # A credit fill worse than the observed composite mark is closer to
        # zero: -1.00 becomes -0.95.  Charging this at entry as well as exit is
        # essential; otherwise every simulated round trip gets a free fill.
        entry += spec.slippage
        if entry >= 0:
            result.note("slippage_erased_credit")
            continue

        strikes = [parse_occ(symbol).strike for symbol in legs]
        width = max(strikes[1] - strikes[0], strikes[3] - strikes[2])
        if width <= 0 or (-entry / width) < spec.min_credit_to_width:
            result.note("credit_below_floor")
            continue

        max_loss = _max_loss(legs, entry)
        if max_loss <= 0:
            result.note("invalid_max_loss")
            continue

        exit_day, exit_price, reason = _manage(history, legs, today, expiry, entry, spec)
        if exit_day is None:
            result.note("no_exit_data")
            continue

        result.trades.append(
            SimulatedTrade(
                entry=today, exit=exit_day, legs=legs, qty=1,
                entry_price=entry, exit_price=exit_price,
                max_loss=max_loss, reason=reason,
            )
        )
        result.note("trades_opened")
        open_until = expiry

    return result


def _nearest_expiry(expiries: Sequence[date], today: date, target: int) -> date | None:
    candidates = [e for e in expiries if 5 <= (e - today).days <= target + 14]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e - today).days - target))


def _pick_condor(
    history: OptionHistory,
    expiry: date,
    spot: float,
    spec: CondorSpec,
    *,
    day: date | None = None,
) -> tuple[str, ...] | None:
    puts = history.contracts_for(expiry, Right.PUT)
    calls = history.contracts_for(expiry, Right.CALL)
    # The expired-contract catalogue includes every listed strike, including
    # contracts that did not trade on the candidate entry session.  Selecting
    # the mathematically nearest strike from that full catalogue and then giving
    # up when it has no bar discards a perfectly observable nearby structure.
    # Live CARRY selects from priceable snapshots; the historical analogue is to
    # select from contracts that actually printed a bar that day.
    if day is not None:
        puts = [symbol for symbol in puts if history.price(symbol, day) is not None]
        calls = [symbol for symbol in calls if history.price(symbol, day) is not None]
    if len(puts) < 2 or len(calls) < 2:
        return None

    width = max(spot * spec.width_pct, 1.0)
    short_put = _nearest(puts, spot * (1 - spec.short_pct))
    short_call = _nearest(calls, spot * (1 + spec.short_pct))
    if short_put is None or short_call is None:
        return None
    # Anchor protective wings to the strikes that were actually selected.  This
    # mirrors the live strategy when the ideal moneyness falls between listings.
    long_put = _nearest(puts, parse_occ(short_put).strike - width)
    long_call = _nearest(calls, parse_occ(short_call).strike + width)
    if long_put is None or long_call is None:
        return None

    strikes = [parse_occ(s).strike for s in (long_put, short_put, short_call, long_call)]
    if not strikes[0] < strikes[1] < strikes[2] < strikes[3]:
        return None
    return (long_put, short_put, short_call, long_call)


def _nearest(symbols: Sequence[str], target: float) -> str | None:
    return min(symbols, key=lambda s: abs(parse_occ(s).strike - target)) if symbols else None


def _price(history: OptionHistory, legs: Sequence[str], day: date) -> float | None:
    """Structure price on a day: long legs add, short legs subtract."""
    long_put, short_put, short_call, long_call = legs
    parts = {}
    for symbol in legs:
        value = history.price(symbol, day)
        if value is None:
            return None
        parts[symbol] = value
    return (
        parts[long_put] - parts[short_put] - parts[short_call] + parts[long_call]
    )


def _max_loss(legs: Sequence[str], entry_price: float) -> float:
    long_put, short_put, short_call, long_call = (parse_occ(s).strike for s in legs)
    width = max(short_put - long_put, long_call - short_call)
    credit = -entry_price
    return max((width - credit) * CONTRACT_MULTIPLIER, 0.0)


def _manage(
    history: OptionHistory,
    legs: Sequence[str],
    entry_day: date,
    expiry: date,
    entry_price: float,
    spec: CondorSpec,
) -> tuple[date | None, float, str]:
    """Hold until the profit target, the stop, or expiry -- whichever comes first."""
    credit = abs(entry_price)
    for day in history.sessions():
        if day <= entry_day:
            continue
        if day > expiry:
            break
        current = _price(history, legs, day)
        if current is None:
            continue
        cost_to_close = abs(current)

        if cost_to_close <= credit * (1 - spec.take_profit):
            # ``current`` uses the original holding orientation (negative for a
            # short condor).  Paying an adverse close makes that equivalent mark
            # *more* negative: -0.50 becomes -0.55, not -0.45.
            return day, current - spec.slippage, "take profit"
        if cost_to_close >= credit * spec.stop_multiple:
            return day, current - spec.slippage, "stop"

    # Fell through to expiry: settle at intrinsic against the final spot.
    final = max((d for d in history.spot if d <= expiry), default=None)
    if final is None:
        return None, 0.0, "no data"
    return expiry, _intrinsic(legs, history.spot[final]), "expiry"


def _intrinsic(legs: Sequence[str], spot: float) -> float:
    long_put, short_put, short_call, long_call = legs
    value = 0.0
    for symbol, sign in ((long_put, 1), (short_put, -1), (short_call, -1), (long_call, 1)):
        parsed = parse_occ(symbol)
        if parsed.right is Right.CALL:
            value += sign * max(spot - parsed.strike, 0.0)
        else:
            value += sign * max(parsed.strike - spot, 0.0)
    return value
