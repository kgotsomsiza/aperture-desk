"""Launch-time checks for the dedicated judged account."""

from aperture.preflight import Report, check_account


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
