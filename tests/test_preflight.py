"""Launch-time checks for the dedicated judged account."""

from datetime import date

from aperture.contracts import PositionIntent, Right, Side, build_occ
from aperture.preflight import Report, _close_filled_probe, check_account
from aperture.risk import Leg, Proposal


class AccountCLI:
    def __init__(self, equity="100000.00", number="PA9JUDGEDJUDG"):
        self.payload = {
            "account_number": number,
            "equity": equity,
            "options_trading_level": 3,
            "options_buying_power": "100000.00",
            "trading_blocked": False,
            "account_blocked": False,
            "trade_suspended_by_user": False,
        }

    def account(self):
        return dict(self.payload)


def test_launch_preflight_requires_exact_fresh_equity(monkeypatch):
    monkeypatch.setenv("APERTURE_EXPECT_ACCOUNT", "PA9JUDGEDJUDG")
    report = Report()
    check_account(AccountCLI(equity="99999.98"), report)
    assert report.failed
    assert any(check == "equity" and status == "FAIL" for status, check, _ in report.rows)


def test_resume_relaxes_only_the_fresh_equity_check(monkeypatch):
    monkeypatch.setenv("APERTURE_EXPECT_ACCOUNT", "PA9JUDGEDJUDG")
    report = Report()
    check_account(AccountCLI(equity="97495.46"), report, require_fresh_equity=False)
    assert not report.failed


def test_preflight_requires_the_named_account_to_match(monkeypatch):
    monkeypatch.setenv("APERTURE_EXPECT_ACCOUNT", "PA9JUDGEDJUDG")
    report = Report()
    check_account(AccountCLI(number="PA3DEVDEVDEV1"), report)
    assert report.failed
    assert any(
        check == "expected account identity" and status == "FAIL"
        for status, check, _ in report.rows
    )


def test_preflight_fails_when_expected_account_is_not_named(monkeypatch):
    monkeypatch.delenv("APERTURE_EXPECT_ACCOUNT", raising=False)
    report = Report()
    check_account(AccountCLI(), report)
    assert report.failed


def test_filled_probe_is_closed_as_one_atomic_mleg_never_leg_by_leg(monkeypatch):
    import aperture.preflight as preflight

    expiry = date(2026, 9, 18)
    short = build_occ("SPY", expiry, Right.PUT, 500)
    long = build_occ("SPY", expiry, Right.PUT, 495)
    opening = Proposal(
        strategy_id="PREFLIGHT",
        underlying="SPY",
        legs=(
            Leg(short, Side.SELL, 1, PositionIntent.SELL_TO_OPEN),
            Leg(long, Side.BUY, 1, PositionIntent.BUY_TO_OPEN),
        ),
        qty=1,
        net_price=-0.50,
        rationale="probe",
    )

    class CLI:
        def __init__(self):
            self.proposals = []

        def submit_mleg(self, proposal, client_order_id=None):
            self.proposals.append(proposal)
            return {"id": "close-parent"}

        def close_position(self, symbol):
            raise AssertionError("single-leg cleanup must never be called")

    cli = CLI()
    monkeypatch.setattr(preflight, "current_structure_price", lambda md, trade: -0.20)
    monkeypatch.setattr(
        preflight,
        "_wait_order",
        lambda cli, order_id, timeout_s: {"status": "filled", "filled_qty": "1"},
    )
    report = Report()

    _close_filled_probe(
        cli,
        report,
        "indicative",
        opening,
        {
            "client_order_id": "preflight-entry",
            "filled_avg_price": "-0.50",
            "filled_at": "2026-09-01T15:00:00Z",
        },
        1,
    )

    assert not report.failed
    assert len(cli.proposals) == 1
    closing = cli.proposals[0]
    assert len(closing.legs) == 2
    assert {leg.intent for leg in closing.legs} == {
        PositionIntent.BUY_TO_CLOSE,
        PositionIntent.SELL_TO_CLOSE,
    }
