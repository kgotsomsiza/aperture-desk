"""Account identity guard.

The single most expensive mistake available this week is pointing the desk at the
wrong account: trading the throwaway while believing it is the judged one, or
resuming the judged account against a ledger built from the throwaway. Both look
completely normal in the logs.

Two independent protections, because they fail in different ways:

  * **Fingerprint** — the ledger records a hash of the account it was created
    against, and refuses to load against a different one. Automatic, needs no
    configuration, and catches a key swap where the state file was not swapped.
  * **Assertion** — ``APERTURE_EXPECT_ACCOUNT`` names the account the desk is
    supposed to be trading. Catches the opposite mistake, where the state file
    was swapped but the keys were not.

Only a hash is stored. The account number is a submission-form value and must
never appear in a file that could be committed or published.
"""

from __future__ import annotations

import hashlib
import logging
import os

log = logging.getLogger(__name__)


class WrongAccountError(RuntimeError):
    """Raised when the live account is not the one the desk expected."""


def fingerprint(account_number: str) -> str:
    """Short, stable, non-reversible identifier for an account."""
    return hashlib.sha256(account_number.strip().upper().encode()).hexdigest()[:16]


def resolve_account_number(account: dict) -> str:
    number = str(account.get("account_number") or account.get("id") or "").strip()
    if not number:
        raise WrongAccountError("the broker returned no account identifier")
    return number


def check(
    account: dict,
    *,
    recorded: str | None,
    expected: str | None = None,
    require_expected: bool = False,
) -> str:
    """Verify identity and return the fingerprint to record.

    ``recorded`` is the fingerprint already in the ledger, or None for a fresh
    desk. ``expected`` is the account number the operator asserts, usually from
    ``APERTURE_EXPECT_ACCOUNT``.
    """
    number = resolve_account_number(account)
    actual = fingerprint(number)

    if expected is None:
        expected = os.environ.get("APERTURE_EXPECT_ACCOUNT") or None

    if expected:
        if fingerprint(expected) != actual:
            raise WrongAccountError(
                "the live account is not the one APERTURE_EXPECT_ACCOUNT names. "
                "Check which keys are loaded before trading."
            )
    elif require_expected:
        raise WrongAccountError(
            "APERTURE_EXPECT_ACCOUNT is not set. Live trading requires naming the "
            "account explicitly, so a stale key file cannot quietly trade the wrong book."
        )

    if recorded and recorded != actual:
        raise WrongAccountError(
            f"this ledger was created against account {recorded}, but the live "
            f"account is {actual}. Use a separate --state file per account rather "
            "than resuming one book against another."
        )

    if not recorded:
        log.info("ledger bound to account fingerprint %s", actual)
    return actual
