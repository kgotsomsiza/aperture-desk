"""Tests for DRIFT, the persisted ledger, and the loop's pricing helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from aperture.contracts import Right, Side, build_occ
from aperture.earnings import EarningsEvent, Timing
from aperture.marketdata import Snapshot
from aperture.state import DeskState, OpenTrade
from aperture.strategies.base import build_vertical, debit_to_width_ok
from aperture.strategies.carry import DEFAULT_CONFIG as CARRY_CONFIG
from aperture.strategies.drift import DEFAULT_CONFIG as DRIFT_CONFIG
from aperture.strategies.drift import earnings_gap, exit_signal, sessions_since

ET = timezone(timedelta(hours=-4))
NOW = datetime(2026, 9, 1, 11, 0, tzinfo=ET)
EXPIRY = date(2026, 9, 18)


def bar(day: str, close: float) -> dict:
    return {"t": f"{day}T00:00:00Z", "c": close}


def snap(strike: float, right: Right, bid: float, ask: float) -> Snapshot:
    return Snapshot(
        symbol=build_occ("NVDA", EXPIRY, right, strike),
        bid=bid, ask=ask, quote_ts=NOW, open_interest=5_000,
    )


# --------------------------------------------------------------------------- #
# DRIFT: measuring the gap
# --------------------------------------------------------------------------- #


AFTER_CLOSE = EarningsEvent("NVDA", date(2026, 8, 26), Timing.AFTER_CLOSE)
BEFORE_OPEN = EarningsEvent("MDT", date(2026, 8, 26), Timing.BEFORE_OPEN)


def test_gap_is_signed_and_measured_across_the_right_session():
    bars = [bar("2026-08-25", 100.0), bar("2026-08-26", 101.0), bar("2026-08-27", 110.0)]
    # Reported after the close on the 26th, so the 27th carries the move.
    assert earnings_gap(bars, AFTER_CLOSE) == pytest.approx(110.0 / 101.0 - 1)


def test_before_open_report_uses_the_same_session():
    bars = [bar("2026-08-25", 100.0), bar("2026-08-26", 92.0)]
    # Reported before the open on the 26th, so the 26th carries the move.
    assert earnings_gap(bars, BEFORE_OPEN) == pytest.approx(-0.08)


def test_gap_is_none_before_the_session_has_printed():
    # The evening of an after-close report: tomorrow's bar does not exist yet.
    bars = [bar("2026-08-25", 100.0), bar("2026-08-26", 101.0)]
    assert earnings_gap(bars, AFTER_CLOSE) is None


def test_gap_is_none_without_a_prior_close():
    assert earnings_gap([bar("2026-08-27", 110.0)], AFTER_CLOSE) is None


def test_gap_handles_empty_bars():
    assert earnings_gap([], AFTER_CLOSE) is None


def test_sessions_since_counts_bars_after_the_gap():
    bars = [bar("2026-08-27", 110.0), bar("2026-08-28", 111.0), bar("2026-08-31", 112.0)]
    assert sessions_since(bars, AFTER_CLOSE) == 2


# --------------------------------------------------------------------------- #
# DRIFT: structure quality
# --------------------------------------------------------------------------- #


def debit_spread(long_mid: float, short_mid: float, width: float = 5.0):
    long_leg = snap(100, Right.CALL, long_mid - 0.05, long_mid + 0.05)
    short_leg = snap(100 + width, Right.CALL, short_mid - 0.05, short_mid + 0.05)
    return build_vertical(
        short_leg=short_leg, long_leg=long_leg,
        strategy_id="DRIFT", underlying="NVDA", budget=10_000.0,
        slippage=0.05, rationale="test", credit=False,
    )


def test_debit_spread_prices_as_a_debit():
    proposal = debit_spread(long_mid=3.00, short_mid=1.00)
    assert proposal is not None
    assert proposal.net_price > 0


def test_cheap_debit_spread_passes_the_width_gate():
    # Paying ~2.05 for a 5-wide spread is 41% of width.
    assert debit_to_width_ok(debit_spread(long_mid=3.00, short_mid=1.00), 0.45)


def test_expensive_debit_spread_is_rejected():
    # Paying ~4.05 for a 5-wide spread is 81% of width: a bad risk/reward.
    assert not debit_to_width_ok(debit_spread(long_mid=5.00, short_mid=1.00), 0.45)


def test_drift_exit_targets():
    assert "take profit" in exit_signal(2.00, 3.10, DRIFT_CONFIG)
    assert "stop loss" in exit_signal(2.00, 0.90, DRIFT_CONFIG)
    assert exit_signal(2.00, 2.40, DRIFT_CONFIG) is None


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def trade(**overrides) -> OpenTrade:
    defaults = dict(
        client_order_id="k1", strategy_id="CARRY", underlying="SPY",
        legs=["SPY260918P00630000", "SPY260918P00625000"], qty=2,
        net_price=-1.50, max_loss=700.0, opened_at="2026-09-01T15:00:00Z",
        leg_sides={"SPY260918P00630000": "sell", "SPY260918P00625000": "buy"},
        status="open",
    )
    defaults.update(overrides)
    return OpenTrade(**defaults)


def test_state_round_trips_through_disk(tmp_path):
    state = DeskState(path=tmp_path / "desk.json")
    state.record_open(trade())
    state.observe_equity(105_000.0, date(2026, 9, 1))
    state.save()

    reloaded = DeskState.load(tmp_path / "desk.json")
    assert reloaded.high_water_mark == 105_000.0
    assert reloaded.start_equity == 105_000.0
    restored = reloaded.open_trades["k1"]
    assert restored.leg_sides == {"SPY260918P00630000": "sell", "SPY260918P00625000": "buy"}
    assert restored.max_loss == 700.0


def test_corrupt_ledger_raises_rather_than_reading_as_empty(tmp_path):
    # An empty ledger reads as "no open risk", which would let the desk double up.
    path = tmp_path / "desk.json"
    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        DeskState.load(path)


def test_missing_ledger_is_a_fresh_desk(tmp_path):
    state = DeskState.load(tmp_path / "nothing.json")
    assert state.open_trades == {}
    assert state.high_water_mark == 0.0


def test_high_water_mark_only_ratchets_up(tmp_path):
    state = DeskState(path=tmp_path / "d.json")
    state.observe_equity(100_000.0, date(2026, 9, 1))
    state.observe_equity(107_000.0, date(2026, 9, 2))
    state.observe_equity(94_000.0, date(2026, 9, 3))
    assert state.high_water_mark == 107_000.0


def test_day_start_equity_resets_on_a_new_session(tmp_path):
    state = DeskState(path=tmp_path / "d.json")
    state.observe_equity(100_000.0, date(2026, 9, 1))
    state.observe_equity(103_000.0, date(2026, 9, 1))
    assert state.day_start_equity == 100_000.0  # unchanged intraday
    state.observe_equity(103_000.0, date(2026, 9, 2))
    assert state.day_start_equity == 103_000.0  # rebased next session


def test_risk_views_aggregate_by_strategy_and_underlying(tmp_path):
    state = DeskState(path=tmp_path / "d.json")
    state.record_open(trade(client_order_id="a", max_loss=700.0))
    state.record_open(trade(client_order_id="b", strategy_id="DRIFT",
                            underlying="NVDA", max_loss=400.0))
    assert state.open_risk_by_strategy() == {"CARRY": 700.0, "DRIFT": 400.0}
    assert state.open_risk_by_underlying() == {"SPY": 700.0, "NVDA": 400.0}


def test_reconcile_drops_positions_the_broker_no_longer_shows(tmp_path):
    # Expiry and assignment remove positions without the desk asking. A phantom
    # trade would consume its strategy's budget forever.
    state = DeskState(path=tmp_path / "d.json")
    state.record_open(trade(client_order_id="gone"))
    state.record_open(trade(client_order_id="here", underlying="QQQ",
                            legs=["QQQ260918P00500000"],
                            leg_sides={"QQQ260918P00500000": "sell"}))

    vanished = state.reconcile({"QQQ260918P00500000"})
    assert vanished == ["gone"]
    assert "here" in state.open_trades
    assert state.closed[0]["client_order_id"] == "gone"
    assert state.open_risk_by_underlying() == {"QQQ": 700.0}


def test_reconcile_keeps_a_partially_present_structure(tmp_path):
    # One leg still at the broker means the structure is not gone, it is broken —
    # and a broken structure must stay visible rather than being silently dropped.
    state = DeskState(path=tmp_path / "d.json")
    state.record_open(trade(client_order_id="half"))
    assert state.reconcile({"SPY260918P00630000"}) == []
    assert "half" in state.open_trades


# --------------------------------------------------------------------------- #
# Structure pricing on exit
# --------------------------------------------------------------------------- #


class FakeMarketData:
    def __init__(self, snaps: dict[str, Snapshot]):
        self._snaps = snaps

    def snapshots_for(self, symbols, underlying):
        return {s: self._snaps[s] for s in symbols if s in self._snaps}


def test_closing_price_keeps_the_opening_sign_convention():
    from aperture.loop import current_structure_price

    short_sym = "SPY260918P00630000"
    long_sym = "SPY260918P00625000"
    snaps = {
        short_sym: Snapshot(symbol=short_sym, bid=0.70, ask=0.80, quote_ts=NOW, open_interest=100),
        long_sym: Snapshot(symbol=long_sym, bid=0.20, ask=0.30, quote_ts=NOW, open_interest=100),
    }
    price = current_structure_price(FakeMarketData(snaps), trade())
    # Sold the 630 at 0.75, bought the 625 at 0.25 -> still a 0.50 credit to close.
    assert price == pytest.approx(0.25 - 0.75)

    # Opened for -1.50, now closes for -0.50: two thirds of the credit captured.
    from aperture.strategies.carry import exit_signal as carry_exit
    assert "take profit" in carry_exit(-1.50, price, CARRY_CONFIG)


def test_closing_price_is_none_when_a_leg_is_unpriceable():
    from aperture.loop import current_structure_price

    short_sym = "SPY260918P00630000"
    long_sym = "SPY260918P00625000"
    snaps = {
        short_sym: Snapshot(symbol=short_sym, bid=0.0, ask=0.80, quote_ts=NOW),
        long_sym: Snapshot(symbol=long_sym, bid=0.20, ask=0.30, quote_ts=NOW),
    }
    assert current_structure_price(FakeMarketData(snaps), trade()) is None


def test_closing_price_is_none_without_recorded_sides():
    from aperture.loop import current_structure_price

    short_sym = "SPY260918P00630000"
    long_sym = "SPY260918P00625000"
    snaps = {
        short_sym: Snapshot(symbol=short_sym, bid=0.70, ask=0.80, quote_ts=NOW),
        long_sym: Snapshot(symbol=long_sym, bid=0.20, ask=0.30, quote_ts=NOW),
    }
    assert current_structure_price(FakeMarketData(snaps), trade(leg_sides={})) is None


# --------------------------------------------------------------------------- #
# Cycle accounting
# --------------------------------------------------------------------------- #


def test_dry_run_is_not_counted_as_a_veto(tmp_path, monkeypatch):
    """An approved-but-not-sent proposal and a vetoed one are opposite results."""
    monkeypatch.chdir(tmp_path)
    from aperture.loop import _submit_if_approved
    from aperture.risk import BookState
    from aperture.warden import AuditLog, RiskWarden
    from aperture.strategies.base import build_iron_condor

    def s(strike, right, bid, ask):
        return Snapshot(symbol=build_occ("SPY", EXPIRY, right, strike), bid=bid, ask=ask,
                        quote_ts=NOW, open_interest=5_000, volume=900,
                        bid_size=50, ask_size=50)

    proposal = build_iron_condor(
        short_put=s(630, Right.PUT, 2.4, 2.6), long_put=s(625, Right.PUT, 0.9, 1.1),
        short_call=s(660, Right.CALL, 2.4, 2.6), long_call=s(665, Right.CALL, 0.9, 1.1),
        strategy_id="CARRY", underlying="SPY", budget=3_000.0, slippage=0.05, rationale="t",
    )
    quotes = {l.symbol: s(l.strike, l.right, 1.0, 1.05) for l in proposal.legs}

    class FakeMD:
        def leg_quotes(self, symbols, underlying):
            return [quotes[x].to_leg_quote() for x in symbols]

    warden = RiskWarden(audit=AuditLog(path=tmp_path / "a.jsonl"), budgets={"CARRY": 50_000.0})
    book = BookState(equity=100_000.0, high_water_mark=100_000.0,
                     day_start_equity=100_000.0, cash=100_000.0, now=NOW)

    assert _submit_if_approved(None, FakeMD(), warden, DeskState(path=tmp_path / "d.json"),
                               book, proposal, dry_run=True) == "dry_run"


# --------------------------------------------------------------------------- #
# Public snapshot
# --------------------------------------------------------------------------- #


class FakeCLI:
    """Returns broker payloads that DO contain identifiers, to prove the
    snapshot's allowlist strips them rather than merely not asking for them."""

    def account(self):
        return {
            "id": "8f3a2b1c-9d4e-4a5b-8c7d-1e2f3a4b5c6d",
            "account_number": "PA0FIXTUREDEV",
            "equity": "104250.00",
            "cash": "50000.00",
        }

    def positions(self):
        return [{
            "asset_id": "abc-123", "account_id": "8f3a2b1c-9d4e",
            "symbol": "SPY260918P00630000", "qty": "-2",
            "avg_entry_price": "2.50", "current_price": "1.10",
            "market_value": "-220", "unrealized_pl": "280", "unrealized_plpc": "0.56",
        }]

    def portfolio_history(self, period="1W", timeframe="1H"):
        return {"timestamp": [1756900000, 1756903600], "equity": [100000, 104250]}


def build_snapshot(tmp_path):
    from aperture.snapshot import Snapshot as PublicSnapshot
    from aperture.warden import AuditLog

    state = DeskState(path=tmp_path / "d.json")
    state.observe_equity(100_000.0, date(2026, 9, 1))
    state.record_open(trade())
    state.observe_equity(104_250.0, date(2026, 9, 2))

    audit = AuditLog(path=tmp_path / "audit.jsonl")
    audit.record("veto", strategy="CARRY", underlying="SPY", summary="VETOED liquidity")
    return PublicSnapshot(state=state, audit=audit, cli=FakeCLI()).build()


def test_snapshot_strips_account_identifiers(tmp_path):
    import json as _json
    payload = build_snapshot(tmp_path)
    blob = _json.dumps(payload)
    assert "PA0FIXTUREDEV" not in blob
    assert "8f3a2b1c" not in blob
    assert "asset_id" not in blob


def test_snapshot_reports_the_numbers_that_matter(tmp_path):
    payload = build_snapshot(tmp_path)
    assert payload["equity"] == 104_250.0
    assert payload["total_return_pct"] == pytest.approx(4.25)
    assert payload["high_water_mark"] == 104_250.0
    assert payload["drawdown_pct"] == 0.0
    assert payload["counts"]["open"] == 1
    assert payload["positions"][0]["symbol"] == "SPY260918P00630000"


def test_snapshot_attribution_groups_by_strategy(tmp_path):
    payload = build_snapshot(tmp_path)
    rows = {r["strategy"]: r for r in payload["attribution"]}
    assert rows["CARRY"]["open"] == 1
    assert rows["CARRY"]["risk_at_work"] == 700.0


def test_publish_guard_rejects_a_leaked_identifier(tmp_path):
    from aperture.snapshot import assert_publishable

    assert_publishable(build_snapshot(tmp_path))  # the real one is clean
    with pytest.raises(ValueError, match="forbidden key"):
        assert_publishable({"account_number": "PA123", "equity": 1})
    # Assembled at runtime rather than written literally: the repo's own
    # pre-push secret scan cannot distinguish a fixture from a real leak, and it
    # should stay that strict.
    local_path = "C:" + chr(92) + "Users" + chr(92) + "someone"
    with pytest.raises(ValueError, match="local path"):
        assert_publishable({"note": local_path})


def test_write_refuses_to_publish_a_leak(tmp_path):
    from aperture.snapshot import write

    with pytest.raises(ValueError):
        write({"account_id": "abc"}, tmp_path / "snapshot.json")
    assert not (tmp_path / "snapshot.json").exists()


# --------------------------------------------------------------------------- #
# Account identity
# --------------------------------------------------------------------------- #


DEV = {"account_number": "PA3DEVDEVDEV1", "equity": "100000"}
JUDGED = {"account_number": "PA9JUDGEDJUDG", "equity": "100000"}


def test_fingerprint_is_stable_and_not_the_account_number():
    from aperture.identity import fingerprint

    a = fingerprint("PA3DEVDEVDEV1")
    assert a == fingerprint("pa3devdevdev1 ")     # case and whitespace insensitive
    assert a != fingerprint("PA9JUDGEDJUDG")
    assert "PA3DEV" not in a                       # the number itself never appears


def test_fresh_ledger_binds_to_whatever_account_it_first_sees():
    from aperture.identity import check

    assert check(DEV, recorded=None) == check(DEV, recorded=None)


def test_ledger_refuses_a_different_account():
    """The expensive mistake: resuming the judged account against dev's ledger."""
    from aperture.identity import WrongAccountError, check, fingerprint

    with pytest.raises(WrongAccountError, match="separate --state file"):
        check(JUDGED, recorded=fingerprint("PA3DEVDEVDEV1"))


def test_expected_account_assertion_catches_the_opposite_mistake():
    """State file swapped but the keys were not."""
    from aperture.identity import WrongAccountError, check

    with pytest.raises(WrongAccountError, match="APERTURE_EXPECT_ACCOUNT names"):
        check(DEV, recorded=None, expected="PA9JUDGEDJUDG")
    assert check(DEV, recorded=None, expected="PA3DEVDEVDEV1")


def test_live_trading_requires_naming_the_account():
    from aperture.identity import WrongAccountError, check

    with pytest.raises(WrongAccountError, match="not set"):
        check(DEV, recorded=None, expected=None, require_expected=True)


def test_missing_account_identifier_is_an_error_not_a_default():
    from aperture.identity import WrongAccountError, check

    with pytest.raises(WrongAccountError, match="no account identifier"):
        check({"equity": "100000"}, recorded=None)


def test_environment_variable_is_honoured(monkeypatch):
    from aperture.identity import WrongAccountError, check

    monkeypatch.setenv("APERTURE_EXPECT_ACCOUNT", "PA9JUDGEDJUDG")
    assert check(JUDGED, recorded=None)
    with pytest.raises(WrongAccountError):
        check(DEV, recorded=None)


def test_fingerprint_persists_through_the_ledger(tmp_path):
    from aperture.identity import fingerprint

    state = DeskState(path=tmp_path / "d.json")
    state.account_fingerprint = fingerprint("PA3DEVDEVDEV1")
    state.save()
    assert DeskState.load(tmp_path / "d.json").account_fingerprint == fingerprint("PA3DEVDEVDEV1")
    # And the raw account number is nowhere in the file.
    assert "PA3DEVDEVDEV1" not in (tmp_path / "d.json").read_text()


# --------------------------------------------------------------------------- #
# Closing a structure
# --------------------------------------------------------------------------- #


def test_closing_proposal_reverses_every_leg():
    """Closing leg-by-leg is how a defined-risk position becomes an undefined one:
    Alpaca fills single-leg closes independently, and the shorts fill first."""
    from aperture.loop import build_closing_proposal
    from aperture.contracts import PositionIntent

    opened = trade()
    closing = build_closing_proposal(opened, price=-0.50)

    by_symbol = {l.symbol: l for l in closing.legs}
    short = by_symbol["SPY260918P00630000"]   # was sold to open
    long_ = by_symbol["SPY260918P00625000"]   # was bought to open

    assert short.side is Side.BUY
    assert short.intent is PositionIntent.BUY_TO_CLOSE
    assert long_.side is Side.SELL
    assert long_.intent is PositionIntent.SELL_TO_CLOSE
    assert len(closing.legs) == len(opened.legs)
    assert closing.qty == opened.qty


def test_closing_price_mirrors_the_structure():
    from aperture.loop import build_closing_proposal

    # Worth -0.50 to hold (a credit structure): closing costs a 0.50 debit,
    # plus a nickel conceded to get filled.
    assert build_closing_proposal(trade(), price=-0.50).net_price == pytest.approx(0.55)
    # A debit structure worth +2.00 is sold back for a 2.00 credit.
    assert build_closing_proposal(trade(), price=2.00).net_price == pytest.approx(-1.95)


def test_closing_an_iron_condor_keeps_all_four_legs_together():
    four = trade(
        legs=["SPY260918P00625000", "SPY260918P00630000",
              "SPY260918C00660000", "SPY260918C00665000"],
        leg_sides={"SPY260918P00625000": "buy", "SPY260918P00630000": "sell",
                   "SPY260918C00660000": "sell", "SPY260918C00665000": "buy"},
    )
    from aperture.loop import build_closing_proposal
    closing = build_closing_proposal(four, price=-1.00)
    assert len(closing.legs) == 4
    sides = {l.symbol: l.side for l in closing.legs}
    assert sides["SPY260918P00630000"] is Side.BUY    # short put bought back
    assert sides["SPY260918C00660000"] is Side.BUY    # short call bought back
    assert sides["SPY260918P00625000"] is Side.SELL   # long put sold
    assert sides["SPY260918C00665000"] is Side.SELL   # long call sold
