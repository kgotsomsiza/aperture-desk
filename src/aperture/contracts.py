"""OCC option symbol construction and parsing.

Alpaca uses unpadded-root OCC symbols, e.g. ``AAPL250620C00200000`` for the
AAPL 2025-06-20 200.00 call. Everything in the desk speaks this format, so the
parsing lives in exactly one place and is unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum

CONTRACT_MULTIPLIER = 100

_OCC_RE = re.compile(
    r"^(?P<root>[A-Z]{1,6})"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<right>[CP])"
    r"(?P<strike>\d{8})$"
)


class Right(str, Enum):
    CALL = "C"
    PUT = "P"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class PositionIntent(str, Enum):
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_CLOSE = "sell_to_close"


@dataclass(frozen=True)
class OptionSymbol:
    root: str
    expiry: date
    right: Right
    strike: float

    def __str__(self) -> str:
        return build_occ(self.root, self.expiry, self.right, self.strike)


def build_occ(root: str, expiry: date, right: Right | str, strike: float) -> str:
    """Build an OCC symbol. Strike is encoded as strike * 1000, zero-padded to 8."""
    root = root.upper().strip()
    if not root.isalpha() or not 1 <= len(root) <= 6:
        raise ValueError(f"invalid root symbol: {root!r}")

    right = Right(right if isinstance(right, Right) else str(right).upper()[:1])

    # Strikes are quoted in 1/1000ths. Round rather than truncate so that
    # floats like 202.49999999 do not silently become the 202.499 strike.
    strike_milli = round(strike * 1000)
    if strike_milli <= 0 or strike_milli > 99_999_999:
        raise ValueError(f"strike out of representable range: {strike}")

    return f"{root}{expiry:%y%m%d}{right.value}{strike_milli:08d}"


def parse_occ(symbol: str) -> OptionSymbol:
    """Parse an OCC symbol. Raises ValueError on anything that is not one."""
    match = _OCC_RE.match(symbol.upper().strip())
    if match is None:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")

    return OptionSymbol(
        root=match["root"],
        expiry=date(2000 + int(match["yy"]), int(match["mm"]), int(match["dd"])),
        right=Right(match["right"]),
        strike=int(match["strike"]) / 1000,
    )


def is_option_symbol(symbol: str) -> bool:
    return _OCC_RE.match(symbol.upper().strip()) is not None


def underlying_of(symbol: str) -> str:
    """Root symbol for an option, or the symbol itself for an equity."""
    return parse_occ(symbol).root if is_option_symbol(symbol) else symbol.upper().strip()


def dte(symbol_or_expiry: str | date, asof: date) -> int:
    """Calendar days to expiry. Negative once expired."""
    expiry = (
        parse_occ(symbol_or_expiry).expiry
        if isinstance(symbol_or_expiry, str)
        else symbol_or_expiry
    )
    return (expiry - asof).days
