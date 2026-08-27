"""Earnings calendar.

Alpaca's corporate-actions endpoint covers dividends and splits, not earnings
dates, so this is the one place the desk reaches outside Alpaca for data. It
degrades in three steps — a curated table, then yfinance, then nothing — because
a missing calendar must mean "CRUSH proposes no trades", never "CRUSH guesses".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

log = logging.getLogger(__name__)


class Timing(str, Enum):
    BEFORE_OPEN = "bmo"
    AFTER_CLOSE = "amc"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EarningsEvent:
    symbol: str
    report_date: date
    timing: Timing = Timing.UNKNOWN

    @property
    def first_session_after(self) -> date:
        """The session that carries the gap.

        A company reporting after the close moves the *next* session; one
        reporting before the open moves that same session. Getting this backwards
        puts the trade on a day late, which is the whole edge gone.
        """
        if self.timing is Timing.BEFORE_OPEN:
            return self.report_date
        return _next_weekday(self.report_date)

    @property
    def enter_on(self) -> date:
        """Last session that closes before the announcement."""
        if self.timing is Timing.BEFORE_OPEN:
            return _prev_weekday(self.report_date)
        return self.report_date


def _next_weekday(day: date) -> date:
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _prev_weekday(day: date) -> date:
    prev = day - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


# Curated for the hackathon window. Verified dates override anything a library
# guesses, because a wrong earnings date is worse than no earnings date.
# VERIFY EACH OF THESE against the company IR page before trading it.
HACKATHON_WINDOW: tuple[EarningsEvent, ...] = (
    EarningsEvent("PANW", date(2026, 9, 1), Timing.AFTER_CLOSE),
    EarningsEvent("MDT", date(2026, 9, 1), Timing.BEFORE_OPEN),
)

# Names that reported just before the window opens: no CRUSH trade left in them,
# but they are the DRIFT strategy's starting cohort on day one.
RECENTLY_REPORTED: tuple[EarningsEvent, ...] = (
    EarningsEvent("NVDA", date(2026, 8, 26), Timing.AFTER_CLOSE),
    EarningsEvent("CRM", date(2026, 8, 26), Timing.AFTER_CLOSE),
    EarningsEvent("CRWD", date(2026, 8, 26), Timing.AFTER_CLOSE),
    EarningsEvent("SNPS", date(2026, 8, 26), Timing.AFTER_CLOSE),
    EarningsEvent("MRVL", date(2026, 8, 27), Timing.AFTER_CLOSE),
    EarningsEvent("ADSK", date(2026, 8, 27), Timing.AFTER_CLOSE),
    EarningsEvent("WDAY", date(2026, 8, 27), Timing.AFTER_CLOSE),
)


@dataclass
class EarningsCalendar:
    """Upcoming and historical earnings dates, curated table first."""

    curated: tuple[EarningsEvent, ...] = HACKATHON_WINDOW
    use_yfinance: bool = True

    def upcoming(self, start: date, end: date) -> list[EarningsEvent]:
        events = [e for e in self.curated if start <= e.report_date <= end]
        return sorted(events, key=lambda e: e.report_date)

    def upcoming_for(self, symbol: str, start: date, end: date) -> EarningsEvent | None:
        for event in self.upcoming(start, end):
            if event.symbol == symbol:
                return event
        if not self.use_yfinance:
            return None
        for day in self._yf_dates(symbol, future=True):
            if start <= day <= end:
                return EarningsEvent(symbol, day, Timing.UNKNOWN)
        return None

    def past_dates(self, symbol: str, limit: int = 12) -> list[date]:
        """Historical report dates, newest first — the input to realized moves."""
        curated = [e.report_date for e in RECENTLY_REPORTED if e.symbol == symbol]
        return sorted(set(curated + self._yf_dates(symbol, future=False)), reverse=True)[:limit]

    # ------------------------------------------------------------------ #

    def _yf_dates(self, symbol: str, *, future: bool) -> list[date]:
        if not self.use_yfinance:
            return []
        try:
            import yfinance  # imported lazily: the desk runs without it
        except ImportError:
            log.debug("yfinance not installed; earnings history unavailable for %s", symbol)
            return []

        try:
            frame = yfinance.Ticker(symbol).get_earnings_dates(limit=24)
        except Exception as exc:  # noqa: BLE001 - a data vendor failing is not fatal
            log.warning("earnings lookup failed for %s: %s", symbol, exc)
            return []

        if frame is None or frame.empty:
            return []

        today = date.today()
        days = [stamp.date() for stamp in frame.index.to_pydatetime()]
        return sorted(d for d in days if (d > today) == future)
