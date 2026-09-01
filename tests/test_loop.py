"""Tests for DRIFT, the persisted ledger, and the loop's pricing helpers."""

from __future__ import annotations

import pathlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
    state.record_open(trade(exit_policy={"take_profit_pct": 0.61}))
    state.observe_equity(105_000.0, date(2026, 9, 1))
    state.hired_strategies = [{"strategy_id": "LAB-1234", "status": "probation"}]
    state.research_history = [{"session": "2026-08-28", "tested": 8}]
    state.research_trials = 8
    state.last_research_date = "2026-08-28"
    state.latest_letter = {"as_of": "2026-08-28", "text": "facts"}
    state.last_letter_date = "2026-08-28"
    state.save()

    reloaded = DeskState.load(tmp_path / "desk.json")
    assert reloaded.high_water_mark == 105_000.0
    assert reloaded.start_equity == 105_000.0
    restored = reloaded.open_trades["k1"]
    assert restored.leg_sides == {"SPY260918P00630000": "sell", "SPY260918P00625000": "buy"}
    assert restored.max_loss == 700.0
    assert restored.exit_policy == {"take_profit_pct": 0.61}
    assert reloaded.hired_strategies[0]["strategy_id"] == "LAB-1234"
    assert reloaded.research_trials == 8
    assert reloaded.research_history[-1]["tested"] == 8
    assert reloaded.latest_letter["text"] == "facts"


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


def test_promoted_research_strategy_joins_the_live_roster_on_probation(tmp_path):
    from aperture.loop import _priors, build_strategies

    state = DeskState(path=tmp_path / "desk.json")
    state.hired_strategies = [
        {
            "strategy_id": "LAB-ACTIVE",
            "status": "probation",
            "underlying": "SPY",
            "spec": {"short_pct": 0.04, "width_pct": 0.01, "dte_target": 14},
        },
        {
            "strategy_id": "LAB-FIRED",
            "status": "fired",
            "underlying": "SPY",
            "spec": {},
        },
    ]

    roster = {strategy.config.strategy_id for strategy in build_strategies(state)}
    assert roster == {"CARRY", "CRUSH", "DRIFT", "CONVEX", "LAB-ACTIVE"}
    assert _priors(state)["LAB-ACTIVE"] == 0.0
    assert "LAB-FIRED" not in _priors(state)


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


def test_position_reconciliation_compares_signed_leg_quantities(tmp_path):
    from aperture.loop import position_mismatches

    state = DeskState(path=tmp_path / "d.json")
    state.record_open(trade())
    correct = [
        {"symbol": "SPY260918P00630000", "qty": "-2"},
        {"symbol": "SPY260918P00625000", "qty": "2"},
    ]
    assert position_mismatches(state, correct, now=NOW) == []

    wrong = [
        {"symbol": "SPY260918P00630000", "qty": "-4"},
        {"symbol": "SPY260918P00620000", "qty": "4"},
    ]
    issues = position_mismatches(state, wrong, now=NOW)
    assert any("broker -4" in issue for issue in issues)
    assert any("broker +0" in issue for issue in issues)
    assert any("00620000" in issue for issue in issues)


def test_just_filled_position_gets_a_propagation_grace_period(tmp_path):
    from aperture.loop import position_mismatches

    state = DeskState(path=tmp_path / "d.json")
    state.record_open(trade(filled_at=NOW.astimezone(timezone.utc).isoformat()))
    assert position_mismatches(state, [], now=NOW) == []

    later = NOW + timedelta(minutes=11)
    assert position_mismatches(state, [], now=later)


# --------------------------------------------------------------------------- #
# Broker-confirmed order lifecycle
# --------------------------------------------------------------------------- #


class OrderCLI:
    def __init__(self, payload):
        self.payload = payload

    def order(self, order_id):
        return dict(self.payload)

    def order_by_client_id(self, client_order_id):
        return dict(self.payload)


def sync_pending(tmp_path, pending, payload):
    from aperture.loop import sync_order_lifecycle
    from aperture.warden import AuditLog, RiskWarden

    state = DeskState(path=tmp_path / "d.json")
    state.record_open(pending)
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    result = sync_order_lifecycle(OrderCLI(payload), state, RiskWarden(audit=audit))
    return state, audit, result


def test_accepted_entry_stays_pending_and_reserves_risk(tmp_path):
    pending = trade(status="pending_entry", order_id="o1")
    state, _, result = sync_pending(
        tmp_path, pending,
        {"status": "accepted", "filled_qty": "0", "limit_price": "-1.50"},
    )

    assert state.open_trades["k1"].status == "pending_entry"
    assert state.open_risk_by_strategy() == {"CARRY": 700.0}
    assert state.closed == []
    assert result == {"entries_filled": 0, "entries_unfilled": 0, "closed": 0}


def test_expired_unfilled_entry_is_not_scored_as_a_closed_trade(tmp_path):
    pending = trade(status="pending_entry", order_id="o1")
    state, audit, result = sync_pending(
        tmp_path, pending,
        {"status": "expired", "filled_qty": "0", "limit_price": "-1.50"},
    )

    assert state.open_trades == {}
    assert state.closed == []
    assert result["entries_unfilled"] == 1
    assert audit.tail()[-1]["event"] == "entry_unfilled"


def test_filled_entry_uses_actual_price_and_fill_time(tmp_path):
    pending = trade(status="pending_entry", order_id="o1", rationale="why this trade exists")
    state, audit, result = sync_pending(
        tmp_path, pending,
        {
            "status": "filled", "filled_qty": "2", "limit_price": "-1.50",
            "filled_avg_price": "-1.60", "filled_at": "2026-09-01T15:05:00Z",
        },
    )

    filled = state.open_trades["k1"]
    assert filled.status == "open"
    assert filled.net_price == pytest.approx(-1.60)
    assert filled.max_loss == pytest.approx(680.0)
    assert filled.opened_at == "2026-09-01T15:05:00Z"
    assert result["entries_filled"] == 1
    assert audit.tail()[-1]["event"] == "entry_filled"
    assert audit.tail()[-1]["rationale"] == "why this trade exists"


def test_expired_partial_entry_keeps_only_the_filled_quantity(tmp_path):
    pending = trade(status="pending_entry", order_id="o1")
    state, _, _ = sync_pending(
        tmp_path, pending,
        {"status": "expired", "filled_qty": "1", "filled_avg_price": "-1.60"},
    )

    filled = state.open_trades["k1"]
    assert filled.status == "open"
    assert filled.qty == 1
    assert filled.max_loss == pytest.approx(340.0)


def test_nested_leg_fills_reconstruct_a_missing_parent_fill_price(tmp_path):
    pending = trade(status="pending_entry", order_id="o1")
    state, _, _ = sync_pending(
        tmp_path, pending,
        {
            "status": "filled", "filled_qty": "2",
            "legs": [
                {"side": "sell", "filled_avg_price": "2.00", "ratio_qty": "1"},
                {"side": "buy", "filled_avg_price": "0.40", "ratio_qty": "1"},
            ],
        },
    )
    assert state.open_trades["k1"].net_price == pytest.approx(-1.60)


def test_fill_without_any_execution_price_stays_reserved(tmp_path):
    pending = trade(status="pending_entry", order_id="o1")
    state, audit, result = sync_pending(
        tmp_path, pending, {"status": "filled", "filled_qty": "2"}
    )
    assert state.open_trades["k1"].status == "pending_entry"
    assert state.closed == []
    assert result["entries_filled"] == 0
    assert audit.tail()[-1]["event"] == "fill_price_missing"


def test_write_ahead_entry_recovers_by_the_same_client_id(tmp_path):
    from aperture.alpaca_cli import AlpacaCliError
    from aperture.loop import sync_order_lifecycle
    from aperture.warden import AuditLog, RiskWarden

    class RecoverCLI:
        def __init__(self):
            self.submitted = []

        def order_by_client_id(self, client_order_id):
            raise AlpacaCliError(
                ["order", "get-by-client-id"], 1,
                '{"code":40410000,"status":404,"error":"order not found"}',
            )

        def submit_mleg(self, proposal, client_order_id=None):
            self.submitted.append(client_order_id)
            return {"id": "recovered-order"}

    state = DeskState(path=tmp_path / "d.json")
    state.record_open(trade(status="submitting_entry", order_id=None))
    state.save()
    cli = RecoverCLI()
    sync_order_lifecycle(
        cli, state, RiskWarden(audit=AuditLog(path=tmp_path / "audit.jsonl"))
    )

    assert cli.submitted == ["k1"]
    assert state.open_trades["k1"].status == "pending_entry"
    assert state.open_trades["k1"].order_id == "recovered-order"
    assert DeskState.load(state.path).open_trades["k1"].order_id == "recovered-order"


def test_filled_close_records_realized_pnl_only_after_confirmation(tmp_path):
    pending = trade(status="pending_close", order_id="open1", close_order_id="close1",
                    close_client_order_id="close-k1-1", close_reason="take profit")
    state, audit, result = sync_pending(
        tmp_path, pending,
        {
            "status": "filled", "filled_qty": "2", "filled_avg_price": "0.50",
            "filled_at": "2026-09-02T15:00:00Z",
        },
    )

    assert state.open_trades == {}
    assert result["closed"] == 1
    assert state.closed[0]["status"] == "closed"
    assert state.closed[0]["pnl"] == pytest.approx(200.0)
    assert state.closed[0]["close_price"] == pytest.approx(0.50)
    assert audit.tail()[-1]["event"] == "closed"


def test_unfilled_close_reopens_the_position(tmp_path):
    pending = trade(status="pending_close", order_id="open1", close_order_id="close1",
                    close_client_order_id="close-k1-1", close_reason="take profit")
    state, audit, result = sync_pending(
        tmp_path, pending,
        {"status": "expired", "filled_qty": "0", "limit_price": "0.50"},
    )

    reopened = state.open_trades["k1"]
    assert reopened.status == "open"
    assert reopened.close_order_id is None
    assert state.closed == []
    assert result["closed"] == 0
    assert audit.tail()[-1]["event"] == "close_unfilled"


def test_partial_close_reduces_open_quantity_and_risk(tmp_path):
    pending = trade(status="pending_close", order_id="open1", close_order_id="close1",
                    close_client_order_id="close-k1-1", close_reason="take profit")
    state, _, _ = sync_pending(
        tmp_path, pending,
        {"status": "canceled", "filled_qty": "1", "filled_avg_price": "0.50"},
    )

    # A partial execution is not a second independent "trade" for the
    # allocator.  It accumulates on the original until that trade is flat.
    assert state.closed == []
    remainder = state.open_trades["k1"]
    assert remainder.status == "open"
    assert remainder.qty == 1
    assert remainder.max_loss == pytest.approx(350.0)
    assert remainder.partial_close_qty == 1
    assert remainder.partial_close_pnl == pytest.approx(100.0)
    assert remainder.partial_close_risk == pytest.approx(350.0)

    state.mark_close_pending(
        "k1", order_id="close2", close_client_order_id="close-k1-2",
        reason="take profit", submitted_at="2026-09-02T15:05:00Z", limit_price=0.50,
    )
    state.confirm_close_submission("k1", order_id="close2", limit_price=0.50)

    from aperture.loop import sync_order_lifecycle
    from aperture.warden import AuditLog, RiskWarden
    sync_order_lifecycle(
        OrderCLI({"status": "filled", "filled_qty": "1", "filled_avg_price": "0.50"}),
        state,
        RiskWarden(audit=AuditLog(path=tmp_path / "second.jsonl")),
    )
    assert state.open_trades == {}
    assert len(state.closed) == 1
    assert state.closed[0]["qty"] == 2
    assert state.closed[0]["max_loss"] == pytest.approx(700.0)
    assert state.closed[0]["pnl"] == pytest.approx(200.0)


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


# A synthetic account number, never a real one. The fixture has to look like a
# genuine Alpaca identifier for the redaction test to be meaningful, but using
# an actual account number would publish the very thing the test asserts is
# never published -- which is exactly what happened once and had to be purged.
FAKE_ACCOUNT_NUMBER = "PA0FIXTURESNP"
FIXTURE_ACCOUNT_UUID = "8f3a2b1c-9d4e-4a5b-8c7d-1e2f3a4b5c6d"
FIXTURE_SHORT_UUID = "8f3a2b1c-9d4e"


class FakeCLI:
    """Returns broker payloads that DO contain identifiers, to prove the
    snapshot's allowlist strips them rather than merely not asking for them."""

    def account(self):
        return {
            "id": FIXTURE_ACCOUNT_UUID,
            "account_number": FAKE_ACCOUNT_NUMBER,
            "equity": "104250.00",
            "cash": "50000.00",
        }

    def positions(self):
        return [{
            "asset_id": "abc-123", "account_id": FIXTURE_SHORT_UUID,
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
    assert FAKE_ACCOUNT_NUMBER not in blob
    assert FIXTURE_SHORT_UUID not in blob
    assert "asset_id" not in blob


def test_snapshot_reports_the_numbers_that_matter(tmp_path, monkeypatch):
    monkeypatch.setenv("APERTURE_PUBLIC_MODE", "scoring")
    payload = build_snapshot(tmp_path)
    assert payload["schema_version"] == 1
    assert payload["mode"] == "scoring"
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


def test_snapshot_exposes_hiring_research_and_reasoning_without_private_state(tmp_path):
    from aperture.snapshot import Snapshot as PublicSnapshot, assert_publishable
    from aperture.warden import AuditLog

    state = DeskState(path=tmp_path / "d.json")
    state.start_equity = 100_000.0
    state.allocations = {"CARRY": 0.60, "CRUSH": 0.18, "DRIFT": 0.17, "LAB-1234": 0.05}
    state.hired_strategies = [{
        "strategy_id": "LAB-1234",
        "status": "probation",
        "weight": 0.05,
        "hired_at": "2026-08-28T20:00:00Z",
        "mutation": "wider wings",
        "backtest": {"trades": 12, "edge": 0.08, "t_stat": 4.2},
        "reason": "selection and holdout survived",
    }]
    state.research_history = [{
        "session": "2026-08-28",
        "tested": 8,
        "cumulative_trials": 16,
        "promoted": ["LAB-1234"],
        "reasoning": {"vendor": "featherless", "fast_model": "Qwen/test"},
    }]
    state.latest_letter = {
        "as_of": "2026-08-28",
        "text": "The desk reported only deterministic facts.",
        "reasoning": {"vendor": "featherless", "reasoning_model": "Kimi/test"},
    }
    payload = PublicSnapshot(
        state=state,
        audit=AuditLog(path=tmp_path / "audit.jsonl"),
        cli=FakeCLI(),
    ).build()

    hired = next(row for row in payload["roster"] if row["strategy"] == "LAB-1234")
    assert hired["status"] == "probation"
    assert hired["evidence"]["trades"] == 12
    assert payload["research"]["cumulative_trials"] == 16
    assert payload["research"]["reasoning"]["vendor"] == "featherless"
    assert payload["shareholder_letter"]["reasoning"]["reasoning_model"] == "Kimi/test"
    assert_publishable(payload)


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


DEV = {"account_number": "PA0FIXTUREDEV", "equity": "100000"}
JUDGED = {"account_number": "PA0FIXTUREJDG", "equity": "100000"}


def test_fingerprint_is_stable_and_not_the_account_number():
    from aperture.identity import fingerprint

    a = fingerprint("PA0FIXTUREDEV")
    assert a == fingerprint("pa0fixturedev ")     # case and whitespace insensitive
    assert a != fingerprint("PA0FIXTUREJDG")
    assert "PA3DEV" not in a                       # the number itself never appears


def test_fresh_ledger_binds_to_whatever_account_it_first_sees():
    from aperture.identity import check

    assert check(DEV, recorded=None) == check(DEV, recorded=None)


def test_ledger_refuses_a_different_account():
    """The expensive mistake: resuming the judged account against dev's ledger."""
    from aperture.identity import WrongAccountError, check, fingerprint

    with pytest.raises(WrongAccountError, match="separate --state file"):
        check(JUDGED, recorded=fingerprint("PA0FIXTUREDEV"))


def test_expected_account_assertion_catches_the_opposite_mistake():
    """State file swapped but the keys were not."""
    from aperture.identity import WrongAccountError, check

    with pytest.raises(WrongAccountError, match="APERTURE_EXPECT_ACCOUNT names"):
        check(DEV, recorded=None, expected="PA0FIXTUREJDG")
    assert check(DEV, recorded=None, expected="PA0FIXTUREDEV")


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

    monkeypatch.setenv("APERTURE_EXPECT_ACCOUNT", "PA0FIXTUREJDG")
    assert check(JUDGED, recorded=None)
    with pytest.raises(WrongAccountError):
        check(DEV, recorded=None)


def test_fingerprint_persists_through_the_ledger(tmp_path):
    from aperture.identity import fingerprint

    state = DeskState(path=tmp_path / "d.json")
    state.account_fingerprint = fingerprint("PA0FIXTUREDEV")
    state.save()
    assert DeskState.load(tmp_path / "d.json").account_fingerprint == fingerprint("PA0FIXTUREDEV")
    # And the raw account number is nowhere in the file.
    assert "PA0FIXTUREDEV" not in (tmp_path / "d.json").read_text()


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


def test_crush_exit_is_next_session_regardless_of_credit_or_debit():
    from aperture.loop import exit_reason

    opened = "2026-09-01T19:55:00Z"  # 15:55 ET
    credit = trade(strategy_id="CRUSH", net_price=-1.50, filled_at=opened)
    debit = trade(strategy_id="CRUSH", net_price=2.00, filled_at=opened)

    assert exit_reason(credit, -0.90, today=date(2026, 9, 1)) is None
    assert "one night" in exit_reason(credit, -0.90, today=date(2026, 9, 2))
    assert "one night" in exit_reason(debit, 1.50, today=date(2026, 9, 2))


def test_crush_invalid_timestamp_fails_safe_to_close():
    from aperture.loop import exit_reason

    broken = trade(strategy_id="CRUSH", filled_at="not-a-time")
    assert "fail-safe" in exit_reason(broken, -1.0, today=date(2026, 9, 1))


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


def test_close_is_persisted_before_an_uncertain_submission(tmp_path):
    from aperture.alpaca_cli import AlpacaCliError
    from aperture.loop import _close_trade
    from aperture.warden import AuditLog, RiskWarden

    short_sym = "SPY260918P00630000"
    long_sym = "SPY260918P00625000"
    md = FakeMarketData({
        short_sym: Snapshot(
            symbol=short_sym, bid=0.70, ask=0.80, quote_ts=NOW, open_interest=100
        ),
        long_sym: Snapshot(
            symbol=long_sym, bid=0.20, ask=0.30, quote_ts=NOW, open_interest=100
        ),
    })

    class TimeoutCLI:
        def submit_mleg(self, proposal, client_order_id=None):
            raise AlpacaCliError(["order", "submit"], 1, "request timed out")

    state = DeskState(path=tmp_path / "desk.json")
    state.record_open(trade())
    warden = RiskWarden(audit=AuditLog(path=tmp_path / "audit.jsonl"))

    assert not _close_trade(TimeoutCLI(), md, state, warden, state.open_trades["k1"], "test")
    reserved = DeskState.load(state.path).open_trades["k1"]
    assert reserved.status == "submitting_close"
    assert reserved.close_client_order_id == "close-k1-1"
    assert reserved.close_limit_price == pytest.approx(0.55)


# --------------------------------------------------------------------------- #
# Official scoring window
# --------------------------------------------------------------------------- #


def test_scoring_window_matches_alpacas_published_timeline():
    """The judged moment is stated exactly: "total equity as of EOD Thursday
    Sep 3rd". The window nominally runs to Friday's opening bell, but aiming the
    desk at Friday would have it opening positions on Thursday morning as though
    another session remained."""
    from aperture.loop import SCORING_CLOSE, SCORING_OPEN, DEADLINE

    assert (SCORING_OPEN.month, SCORING_OPEN.day, SCORING_OPEN.hour, SCORING_OPEN.minute) == (8, 31, 9, 30)
    assert (SCORING_CLOSE.month, SCORING_CLOSE.day) == (9, 3)
    assert (SCORING_CLOSE.hour, SCORING_CLOSE.minute) == (16, 0)  # Thursday's close
    assert DEADLINE == SCORING_CLOSE


def test_the_desk_does_not_liquidate_into_the_measurement():
    """Judging uses total equity, not cash. Crossing the spread on every open
    structure would convert a mid-market mark into slightly less cash, for
    certain, to remove an uncertainty that is unbiased and smaller."""
    import aperture.loop as loop

    assert not hasattr(loop, "FLATTEN_FROM")
    source = (pathlib.Path(loop.__file__)).read_text(encoding="utf-8")
    endgame = source[source.index("# 2. Past the measurement"):source.index("# 3. Circuit breakers")]
    assert "_flatten(" not in endgame


def test_risk_appetite_throttles_itself_toward_the_measurement():
    """With no blanket liquidation, the tournament clock *is* the endgame: it
    has to stop the desk opening positions on its own."""
    from aperture.loop import SCORING_CLOSE, SCORING_OPEN
    from aperture.risk import tournament_risk_multiplier

    early = tournament_risk_multiplier(SCORING_OPEN, SCORING_CLOSE, 100_000, 100_000)
    late = tournament_risk_multiplier(
        SCORING_CLOSE - timedelta(hours=20), SCORING_CLOSE, 106_000, 100_000
    )
    assert late <= 0.5
    assert late < early

    # In the final hours, being ahead means opening nothing at all.
    final = tournament_risk_multiplier(
        SCORING_CLOSE - timedelta(hours=2), SCORING_CLOSE, 106_000, 100_000
    )
    assert final == 0.0


def test_being_behind_near_the_end_still_allows_bounded_variance():
    """Finishing 20th and finishing 40th pay the same, so a desk that is behind
    should not protect a result it does not have -- but only through structures
    the Warden has already bounded."""
    from aperture.loop import SCORING_CLOSE
    from aperture.risk import tournament_risk_multiplier

    behind = tournament_risk_multiplier(
        SCORING_CLOSE - timedelta(hours=2), SCORING_CLOSE, 97_000, 100_000
    )
    assert 0.0 < behind <= 0.5


def test_tournament_clock_still_leans_in_early_in_the_window():
    from aperture.loop import SCORING_CLOSE, SCORING_OPEN
    from aperture.risk import tournament_risk_multiplier

    assert tournament_risk_multiplier(SCORING_OPEN, SCORING_CLOSE, 100_000, 100_000) > 1.0


# --------------------------------------------------------------------------- #
# Audit trails belong to the ledger they describe
# --------------------------------------------------------------------------- #


def test_each_ledger_gets_its_own_audit_trail():
    """A shared audit.jsonl merges accounts. The allocator reads that file for
    its veto-rate firing signal, so decisions made while testing on a throwaway
    account would fire a strategy on the scored account before it had traded."""
    from aperture.state import audit_path_for

    judged = audit_path_for("state/judged.json")
    dev = audit_path_for("state/dev.json")
    assert judged.name == "judged.audit.jsonl"
    assert dev.name == "dev.audit.jsonl"
    assert judged != dev
    assert judged.parent == dev.parent == Path("state")


def test_audit_path_accepts_a_path_or_a_string():
    from aperture.state import audit_path_for

    assert audit_path_for(Path("a/b/book.json")) == Path("a/b/book.audit.jsonl")
    assert audit_path_for("a/b/book.json") == Path("a/b/book.audit.jsonl")


def test_runner_and_status_agree_on_the_audit_location():
    """If they disagree, the status tool reports on a file nothing writes to."""
    import inspect
    from aperture import runner, status
    from aperture.state import audit_path_for

    assert "audit_path_for" in inspect.getsource(runner.Runner.__init__)
    assert "audit_path_for" in inspect.getsource(status.main)
    assert audit_path_for("state/judged.json").name == "judged.audit.jsonl"
