"""Market data layer — chains, greeks, and the implied-move calculation.

The free data plan shapes everything here. Options quotes are the *indicative*
feed (derived, not OPRA) and option trades are delayed 15 minutes, so nothing in
the desk may depend on sub-minute option pricing. Signals are therefore built
from the underlying (real-time IEX is free) and from the implied-vol surface,
and every position is held for hours to days rather than seconds.

Alpaca is inconsistent about camelCase vs snake_case across endpoints, so every
field read goes through :func:`_pick`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .alpaca_cli import AlpacaCLI, AlpacaCliError
from .contracts import Right, parse_occ
from .risk import LegQuote

TRADING_DAYS_PER_YEAR = 252


def _pick(obj: dict[str, Any] | None, *names: str, default: Any = None) -> Any:
    """First present key out of several spellings."""
    if not obj:
        return default
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    return default


def _bars_of(payload: dict[str, Any] | None, symbol: str) -> list[dict[str, Any]]:
    """Normalise the two shapes Alpaca returns for bars.

    ``data bars --symbol X`` (singular) returns ``{"bars": [...], "symbol": "X"}``
    while the plural multi-symbol endpoints return ``{"bars": {"X": [...]}}``.
    Assuming either one silently yields an empty series against the other.
    """
    bars = (payload or {}).get("bars")
    if isinstance(bars, list):
        return bars
    if isinstance(bars, dict):
        return bars.get(symbol) or []
    return []


def _parse_ts(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Snapshot:
    """One option contract as the desk sees it at decision time."""

    symbol: str
    bid: float
    ask: float
    quote_ts: datetime
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    open_interest: int = -1  # -1 means "not reported" -- Alpaca returns null here
    volume: int = 0
    bid_size: int = 0
    ask_size: int = 0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def right(self) -> Right:
        return parse_occ(self.symbol).right

    @property
    def strike(self) -> float:
        return parse_occ(self.symbol).strike

    @property
    def expiry(self) -> date:
        return parse_occ(self.symbol).expiry

    @property
    def is_priceable(self) -> bool:
        return self.bid > 0 and self.ask > self.bid

    def to_leg_quote(self) -> LegQuote:
        return LegQuote(
            symbol=self.symbol,
            bid=self.bid,
            ask=self.ask,
            quote_ts=self.quote_ts,
            open_interest=self.open_interest,
            volume=self.volume,
            bid_size=self.bid_size,
            ask_size=self.ask_size,
        )


def parse_snapshot(symbol: str, raw: dict[str, Any]) -> Snapshot:
    quote = _pick(raw, "latestQuote", "latest_quote", default={}) or {}
    greeks = _pick(raw, "greeks", default={}) or {}
    daily = _pick(raw, "dailyBar", "daily_bar", default={}) or {}

    return Snapshot(
        symbol=symbol,
        bid=float(_pick(quote, "bp", "bid_price", default=0.0) or 0.0),
        ask=float(_pick(quote, "ap", "ask_price", default=0.0) or 0.0),
        quote_ts=_parse_ts(_pick(quote, "t", "timestamp")),
        implied_volatility=_pick(raw, "impliedVolatility", "implied_volatility"),
        delta=_pick(greeks, "delta"),
        gamma=_pick(greeks, "gamma"),
        theta=_pick(greeks, "theta"),
        vega=_pick(greeks, "vega"),
        open_interest=int(_pick(raw, "openInterest", "open_interest", default=-1) or -1),
        volume=int(_pick(daily, "v", "volume", default=0) or 0),
        bid_size=int(_pick(quote, "bs", "bid_size", default=0) or 0),
        ask_size=int(_pick(quote, "as", "ask_size", default=0) or 0),
    )


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


@dataclass
class MarketData:
    """Everything the strategies are allowed to know about the market."""

    cli: AlpacaCLI
    feed: str = "indicative"
    _oi_cache: dict[str, dict[str, int]] = None  # underlying -> {symbol: open_interest}

    def __post_init__(self) -> None:
        if self._oi_cache is None:
            self._oi_cache = {}

    def spot(self, symbol: str) -> float:
        """Mid of the real-time underlying quote. IEX on the free plan."""
        payload = self.cli.latest_stock_quote([symbol])
        quote = ((payload or {}).get("quotes") or {}).get(symbol) or {}
        bid = float(_pick(quote, "bp", "bid_price", default=0.0) or 0.0)
        ask = float(_pick(quote, "ap", "ask_price", default=0.0) or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        # IEX can be one-sided outside of active trading; fall back to the last daily close.
        series = _bars_of(
            self.cli.stock_bars(symbol, start=(date.today() - timedelta(days=7)).isoformat()),
            symbol,
        )
        return float(series[-1]["c"]) if series else 0.0

    def open_interest(self, underlying: str) -> dict[str, int]:
        """Open interest per contract, from the contracts endpoint.

        Chain snapshots do not reliably carry open interest, but the liquidity
        gate needs it, so it is fetched separately and cached per session.
        """
        if underlying in self._oi_cache:
            return self._oi_cache[underlying]

        table: dict[str, int] = {}
        try:
            payload = self.cli.run(
                "option", "contracts", "--underlying-symbols", underlying, "--limit", "10000"
            )
            for contract in (payload or {}).get("option_contracts") or []:
                symbol = contract.get("symbol")
                if symbol:
                    table[symbol] = int(contract.get("open_interest") or 0)
        except AlpacaCliError:
            # Leave the table empty: unknown OI stays unknown, and the gate rejects.
            pass

        self._oi_cache[underlying] = table
        return table

    def chain(
        self,
        underlying: str,
        *,
        min_dte: int = 7,
        max_dte: int = 45,
        option_type: str | None = None,
        strike_band: float | None = None,
        spot: float | None = None,
        limit: int = 1000,
        max_pages: int = 6,
    ) -> dict[str, Snapshot]:
        """Live chain, filtered to a DTE window and optionally a strike band.

        Paginates. The API's ``limit`` truncates the filtered result rather than
        windowing it, so a single under-sized page returns only the lowest strikes
        and the chain appears to end far below the money — which silently produces
        nonsense strike selection rather than an error.
        """
        today = date.today()
        kwargs: dict[str, Any] = {
            "feed": self.feed,
            "expiration_gte": (today + timedelta(days=min_dte)).isoformat(),
            "expiration_lte": (today + timedelta(days=max_dte)).isoformat(),
            "limit": limit,
        }
        if option_type:
            kwargs["option_type"] = option_type
        if strike_band is not None:
            reference = spot if spot is not None else self.spot(underlying)
            if reference > 0:
                kwargs["strike_gte"] = round(reference * (1 - strike_band), 2)
                kwargs["strike_lte"] = round(reference * (1 + strike_band), 2)

        raw_snapshots: dict[str, Any] = {}
        page_token: str | None = None
        for _ in range(max_pages):
            payload = self.cli.option_chain(underlying, page_token=page_token, **kwargs)
            raw_snapshots.update((payload or {}).get("snapshots") or {})
            page_token = (payload or {}).get("next_page_token")
            if not page_token:
                break

        oi_table = self.open_interest(underlying)
        snapshots: dict[str, Snapshot] = {}
        for symbol, raw in raw_snapshots.items():
            snap = parse_snapshot(symbol, raw)
            if snap.open_interest < 0 and symbol in oi_table:
                snap = Snapshot(**{**snap.__dict__, "open_interest": oi_table[symbol]})
            snapshots[symbol] = snap
        return snapshots

    def snapshots_for(self, symbols: Sequence[str], underlying: str) -> dict[str, Snapshot]:
        """Fresh snapshots for specific contracts, with open interest merged in.

        Used at the moment of decision: the chain that produced a proposal may be
        seconds old by the time the Warden sees it, and the staleness gate is only
        meaningful against quotes fetched now.
        """
        if not symbols:
            return {}
        payload = self.cli.option_snapshot(list(symbols), feed=self.feed)
        oi_table = self.open_interest(underlying)

        result: dict[str, Snapshot] = {}
        for symbol, raw in ((payload or {}).get("snapshots") or {}).items():
            snap = parse_snapshot(symbol, raw)
            if snap.open_interest < 0 and symbol in oi_table:
                snap = Snapshot(**{**snap.__dict__, "open_interest": oi_table[symbol]})
            result[symbol] = snap
        return result

    def leg_quotes(self, symbols: Sequence[str], underlying: str) -> list[LegQuote]:
        snaps = self.snapshots_for(symbols, underlying)
        return [snaps[s].to_leg_quote() for s in symbols if s in snaps]

    def daily_bars(self, symbol: str, lookback_days: int = 800) -> list[dict[str, Any]]:
        start = (date.today() - timedelta(days=lookback_days)).isoformat()
        return _bars_of(self.cli.stock_bars(symbol, start=start), symbol)


# --------------------------------------------------------------------------- #
# Selection helpers
# --------------------------------------------------------------------------- #


def expiries(snapshots: Iterable[Snapshot]) -> list[date]:
    return sorted({s.expiry for s in snapshots})


def for_expiry(snapshots: Iterable[Snapshot], expiry: date) -> list[Snapshot]:
    return [s for s in snapshots if s.expiry == expiry]


def nearest_expiry_after(snapshots: Iterable[Snapshot], cutoff: date) -> date | None:
    candidates = [e for e in expiries(snapshots) if e >= cutoff]
    return candidates[0] if candidates else None


def select_by_delta(
    snapshots: Iterable[Snapshot], target_delta: float, right: Right
) -> Snapshot | None:
    """Contract whose |delta| is closest to the target. Requires greeks."""
    pool = [
        s
        for s in snapshots
        if s.right is right and s.delta is not None and s.is_priceable
    ]
    if not pool:
        return None
    return min(pool, key=lambda s: abs(abs(s.delta) - abs(target_delta)))


def select_by_strike_offset(
    snapshots: Iterable[Snapshot], anchor: Snapshot, width: float, right: Right
) -> Snapshot | None:
    """The contract ``width`` points further out of the money than ``anchor``.

    Used to build the protective wing when greeks are unavailable, and to keep
    spread widths predictable when they are.
    """
    target = anchor.strike - width if right is Right.PUT else anchor.strike + width
    pool = [s for s in snapshots if s.right is right and s.is_priceable]
    if not pool:
        return None
    return min(pool, key=lambda s: abs(s.strike - target))


def atm_pair(snapshots: Iterable[Snapshot], spot: float, expiry: date) -> tuple[Snapshot, Snapshot] | None:
    """The at-the-money call and put for one expiry."""
    same = for_expiry(snapshots, expiry)
    calls = [s for s in same if s.right is Right.CALL and s.is_priceable]
    puts = [s for s in same if s.right is Right.PUT and s.is_priceable]
    if not calls or not puts:
        return None
    call = min(calls, key=lambda s: abs(s.strike - spot))
    put = min(puts, key=lambda s: abs(s.strike - call.strike))
    return call, put


# --------------------------------------------------------------------------- #
# The implied-move edge
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MoveComparison:
    """The market's priced move against what this name actually does.

    The whole CRUSH strategy is one sentence: *the market is pricing X%, this
    name has historically moved Y%, so sell the difference when X is rich enough.*
    """

    underlying: str
    expiry: date
    implied_total_move: float  # straddle / spot, all sources of vol to expiry
    implied_event_move: float  # the earnings component alone
    realized_moves: tuple[float, ...]
    median_realized: float
    richness: float  # implied_event / median_realized

    @property
    def is_rich(self) -> bool:
        return self.richness >= 1.25

    @property
    def is_cheap(self) -> bool:
        return self.richness <= 0.85

    def explain(self) -> str:
        return (
            f"{self.underlying}: market prices a {self.implied_event_move:.1%} earnings move; "
            f"last {len(self.realized_moves)} averaged {self.median_realized:.1%} "
            f"({self.richness:.2f}x)"
        )


def implied_total_move(call: Snapshot, put: Snapshot, spot: float) -> float:
    """Expected absolute move to expiry, as a fraction of spot.

    An at-the-forward straddle is worth E[|S_T - K|], so dividing by spot gives
    the expected absolute *return* directly — which is exactly the quantity the
    realized earnings moves below are measured in. No fudge factor required.
    """
    if spot <= 0:
        return 0.0
    return (call.mid + put.mid) / spot


def strip_diffusive_vol(total_move: float, dte: int, baseline_annual_vol: float) -> float:
    """Remove ordinary day-to-day vol, leaving the earnings jump.

    A straddle on the expiry that contains earnings prices both the event and the
    remaining ordinary trading days. Variance is additive, so:

        total_variance = event_variance + baseline_variance * (dte / 252)

    Without this the desk would systematically overestimate how rich an event is
    whenever the nearest expiry is a week or more out.
    """
    if total_move <= 0:
        return 0.0
    diffusive = baseline_annual_vol * math.sqrt(max(dte, 0) / TRADING_DAYS_PER_YEAR)
    return math.sqrt(max(total_move**2 - diffusive**2, 0.0))


def realized_vol(bars: Sequence[dict[str, Any]], window: int = 60) -> float:
    """Annualised close-to-close volatility."""
    closes = [float(b["c"]) for b in bars[-(window + 1):] if b.get("c")]
    if len(closes) < 10:
        return 0.0
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0]
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)


def realized_earnings_moves(
    bars: Sequence[dict[str, Any]], event_dates: Sequence[date], max_events: int = 8
) -> list[float]:
    """Absolute overnight move across each past earnings date.

    Measured close-before to close-after, which is how the move is quoted and how
    the straddle prices it.
    """
    by_date: dict[date, int] = {}
    for index, bar in enumerate(bars):
        stamp = _parse_ts(bar.get("t")).date()
        by_date[stamp] = index

    ordered = sorted(by_date)
    moves: list[float] = []
    for event in sorted(event_dates, reverse=True)[:max_events]:
        after = next((d for d in ordered if d >= event), None)
        if after is None:
            continue
        index = by_date[after]
        if index == 0:
            continue
        prior_close = float(bars[index - 1]["c"])
        event_close = float(bars[index]["c"])
        if prior_close > 0:
            moves.append(abs(event_close / prior_close - 1))
    return moves


def compare_moves(
    underlying: str,
    expiry: date,
    call: Snapshot,
    put: Snapshot,
    spot: float,
    bars: Sequence[dict[str, Any]],
    event_dates: Sequence[date],
    *,
    asof: date | None = None,
) -> MoveComparison | None:
    """Assemble the full implied-versus-realized picture for one earnings event."""
    moves = realized_earnings_moves(bars, event_dates)
    if len(moves) < 4:
        return None  # too little history to claim an edge

    asof = asof or date.today()
    total = implied_total_move(call, put, spot)
    event_move = strip_diffusive_vol(total, (expiry - asof).days, realized_vol(bars))
    median = statistics.median(moves)
    if median <= 0 or event_move <= 0:
        return None

    return MoveComparison(
        underlying=underlying,
        expiry=expiry,
        implied_total_move=total,
        implied_event_move=event_move,
        realized_moves=tuple(moves),
        median_realized=median,
        richness=event_move / median,
    )
