"""The always-on runner — `python -m aperture.runner`.

Turns the single-cycle loop into an autonomous desk. It runs unattended for the
whole contest, so its first duty is simply to stay alive: a data vendor hiccup,
a rate limit, or a malformed chain must never be the reason the desk stops
trading for the rest of the week.

Three rules it follows:

  * **Never die on an exception.** Cycle failures are logged, counted, and the
    runner sleeps and tries again. Only a repeated, unbroken failure streak
    escalates to the kill switch, because a desk failing every cycle is more
    dangerous than one that has stopped.
  * **Sleep through the close.** It asks Alpaca when the next open is rather
    than assuming a calendar, so holidays and half-days need no special casing.
  * **Publish after every cycle**, including failed ones, so the dashboard shows
    what is actually happening rather than freezing at the last success.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .alpaca_cli import AlpacaCLI, AlpacaCliError
from .allocator import Allocator, observe
from .identity import WrongAccountError
from .letter import gather as gather_letter, write as write_letter
from .llm import build_provider, provider_info
from .loop import (
    DEADLINE,
    _priors,
    build_strategies,
    emergency_flatten_cycle,
    run_cycle,
    sync_order_lifecycle,
)
from .marketdata import MarketData
from .research import promotion_records, run_lab
from .risk import RiskLimits
from .snapshot import Snapshot, publish_remote, write
from .state import DeskState, audit_path_for
from .warden import AuditLog, RiskWarden

log = logging.getLogger("aperture.runner")

MAX_SLEEP = 900  # never sleep more than 15 minutes, so a kill switch lands quickly


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stopping = False
        self.consecutive_failures = 0

        self.cli = AlpacaCLI()
        self.md = MarketData(cli=self.cli, feed=args.feed)
        self.state_path = Path(args.state)
        self.audit = AuditLog(path=audit_path_for(self.state_path))
        self.provider = build_provider()

        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, signum, _frame) -> None:
        # Finish the cycle in flight rather than abandoning a half-placed order.
        log.warning("signal %s received; stopping after this cycle", signum)
        self.stopping = True

    # ------------------------------------------------------------------ #

    def run(self) -> int:
        log.info(
            "runner starting | feed=%s | state=%s | deadline=%s | dry_run=%s",
            self.args.feed, self.state_path, DEADLINE.isoformat(), self.args.dry_run,
        )
        while not self.stopping:
            if datetime.now(timezone.utc) >= DEADLINE.astimezone(timezone.utc):
                log.warning("deadline passed; the desk is done trading")
                self.publish()
                return 0

            started = time.monotonic()
            slept = self.tick()
            if self.stopping:
                break
            # --interval means "a decision every N seconds", not "N seconds of
            # idling after however long the work took". A cycle costs about two
            # minutes, so sleeping the full interval on top of it stretched the
            # real cadence to seven minutes and cost a third of the session's
            # decision points. Backoffs and until-open waits are left alone --
            # those are computed at the end of the cycle and are already correct.
            if slept == self.args.interval:
                slept = max(1, int(slept - (time.monotonic() - started)))
            self.sleep(slept)

        self.publish()
        log.info("runner stopped cleanly")
        return 0

    def tick(self) -> int:
        """One cycle. Returns the seconds to wait before the next one."""
        state = None
        try:
            state = DeskState.load(self.state_path)
            warden = RiskWarden(
                limits=RiskLimits(),
                audit=self.audit,
                deadline=DEADLINE,
                start_equity=state.start_equity or self.args.equity,
                budgets=_budgets(state.start_equity or self.args.equity),
            )

            clock = self.cli.clock()
            # Orders expire and late fills settle even after the bell.  Keep the
            # ledger truthful while closed, but never recover a not-yet-accepted
            # submission until the market is open again.
            sync_order_lifecycle(
                self.cli,
                state,
                warden,
                allow_submission_recovery=bool(clock.get("is_open")),
            )

            if warden.halted():
                if not clock.get("is_open"):
                    log.critical("kill switch engaged; market closed, flatten resumes next open")
                    return self._until_open(clock)
                result = emergency_flatten_cycle(
                    self.cli,
                    self.md,
                    warden,
                    state,
                    dry_run=self.args.dry_run,
                    require_expected=not self.args.dry_run,
                )
                log.critical(
                    "kill switch flatten: %d confirmed closed, %d close orders submitted, "
                    "%d remaining",
                    result["closed"], result["close_submitted"], result["remaining"],
                )
                self.consecutive_failures = 0
                return self.args.interval

            if not clock.get("is_open"):
                self.after_close(state, warden)
                return self._until_open(clock)

            summary = run_cycle(
                self.cli, self.md, warden, state, build_strategies(state),
                dry_run=self.args.dry_run,
                require_expected=not self.args.dry_run,
                provider=self.provider,
            )
            log.info(
                "cycle: %d approved (%d submitted), %d vetoed, %d closed%s",
                summary["approved"], summary["submitted"], summary["vetoed"],
                summary["closed"],
                (f" | FIRED: {','.join(summary['fired'])}" if summary.get("fired") else "")
                + (" | FLATTENING" if summary.get("flattening") else "")
                + (f" | BREACH: {summary['breached']}" if summary["breached"] else ""),
            )
            self.consecutive_failures = 0
            return self.args.interval

        except WrongAccountError:
            # Never retry this. Staying alive is the runner's job, but not while
            # aimed at the wrong account.
            log.critical("WRONG ACCOUNT - refusing to trade", exc_info=True)
            self.stopping = True
            raise

        except Exception as exc:  # noqa: BLE001 - staying alive is the whole job
            self.consecutive_failures += 1
            log.exception("cycle failed (%d in a row): %s", self.consecutive_failures, exc)
            self.audit.record(
                "cycle_error", error=str(exc)[:400], streak=self.consecutive_failures
            )
            if self.consecutive_failures >= self.args.max_failures:
                # Something is systematically wrong. Stop trading rather than keep
                # firing orders built on whatever is broken.
                RiskWarden(audit=self.audit).engage_kill_switch(
                    f"{self.consecutive_failures} consecutive cycle failures"
                )
            return min(self.args.interval * self.consecutive_failures, MAX_SLEEP)

        finally:
            self.publish(state)

    def after_close(self, state: DeskState, warden: RiskWarden) -> None:
        """Run the desk's research and explanation jobs once per completed session."""
        session = state.day_stamp
        if not session:
            return
        self._run_research(state, session)
        self._write_shareholder_letter(state, warden, session)

    def _run_research(self, state: DeskState, session: str) -> None:
        if state.last_research_date == session:
            return
        try:
            report = run_lab(
                self.cli,
                underlying=os.environ.get("APERTURE_RESEARCH_UNDERLYING", "SPY"),
                candidates=int(os.environ.get("APERTURE_RESEARCH_CANDIDATES", "8")),
                lookback_days=int(os.environ.get("APERTURE_RESEARCH_LOOKBACK", "900")),
                expiries=int(os.environ.get("APERTURE_RESEARCH_EXPIRIES", "120")),
                provider=self.provider,
                seed=int(session.replace("-", "")),
                prior_trials=state.research_trials,
            )
        except Exception as exc:  # noqa: BLE001 - a closed-market lab cannot stop the desk
            log.exception("research lab failed: %s", exc)
            self.audit.record("research_error", session=session, error=str(exc)[:400])
            return

        underlying = os.environ.get("APERTURE_RESEARCH_UNDERLYING", "SPY")
        records = promotion_records(report, underlying=underlying)
        known = {str(row.get("strategy_id")) for row in state.hired_strategies}
        hired = [row for row in records if row["strategy_id"] not in known]
        state.hired_strategies.extend(hired)
        state.research_trials += report.tested
        state.research_history.append({
            "session": session,
            "tested": report.tested,
            "cumulative_trials": report.multiple_testing_trials,
            "promoted": [row["strategy_id"] for row in hired],
            "training_window": report.training_window,
            "validation_window": report.validation_window,
            "reasoning": provider_info(self.provider),
            "summary": report.summary(),
        })
        state.last_research_date = session
        state.save()
        self.audit.record(
            "research_complete",
            session=session,
            tested=report.tested,
            cumulative_trials=report.multiple_testing_trials,
            promoted=[row["strategy_id"] for row in hired],
            training_window=report.training_window,
            validation_window=report.validation_window,
        )
        for row in hired:
            self.audit.record(
                "hired",
                session=session,
                strategy=row["strategy_id"],
                underlying=row["underlying"],
                summary=row["reason"],
                mutation=row["mutation"],
                hypothesis=row.get("hypothesis"),
                backtest=row["backtest"],
            )
        log.info("%s", report.summary())

    def _write_shareholder_letter(
        self, state: DeskState, warden: RiskWarden, session: str
    ) -> None:
        if state.last_letter_date == session:
            return
        try:
            account = self.cli.account()
            equity = float(account.get("equity") or 0.0)
            allocations = Allocator().allocate(
                observe(state, warden.audit, _priors(state)), equity
            )
            facts = gather_letter(
                state,
                warden.audit,
                equity,
                allocations,
                today=datetime.fromisoformat(session).date(),
            )
            text = write_letter(self.provider, facts)
        except Exception as exc:  # noqa: BLE001 - narration can never stop trading
            log.exception("shareholder letter failed: %s", exc)
            self.audit.record("letter_error", session=session, error=str(exc)[:400])
            return

        state.latest_letter = {
            "as_of": session,
            "text": text,
            "reasoning": provider_info(self.provider),
        }
        state.last_letter_date = session
        state.save()
        self.audit.record(
            "letter_written",
            session=session,
            provider=provider_info(self.provider),
        )
        log.info("shareholder letter written with %s", provider_info(self.provider)["vendor"])

    def publish(self, state: DeskState | None = None) -> None:
        try:
            state = state or DeskState.load(self.state_path)
            payload = Snapshot(state=state, audit=self.audit, cli=self.cli).build()
            write(payload, self.args.public)
        except Exception as exc:  # noqa: BLE001 - publishing must never stop trading
            log.warning("local snapshot publish failed: %s", exc)
            return

        try:
            if publish_remote(payload):
                log.info("public snapshot accepted by dashboard")
        except Exception as exc:  # noqa: BLE001 - a dashboard can never stop trading
            log.warning("remote snapshot publish failed: %s", exc)

    def _until_open(self, clock: dict) -> int:
        nxt = clock.get("next_open")
        if not nxt:
            return MAX_SLEEP
        try:
            opens = datetime.fromisoformat(str(nxt))
        except ValueError:
            return MAX_SLEEP
        seconds = (opens - datetime.now(opens.tzinfo)).total_seconds()
        log.info("market closed; next open %s (%.1f h)", nxt, max(seconds, 0) / 3600)
        return max(int(seconds), 60)

    def sleep(self, seconds: int) -> None:
        """Sleep in slices so a stop signal is honoured promptly."""
        remaining = max(min(seconds, MAX_SLEEP), 1)
        while remaining > 0 and not self.stopping:
            nap = min(remaining, 5)
            time.sleep(nap)
            remaining -= nap


def _budgets(equity: float) -> dict[str, float]:
    return {"CARRY": equity * 0.18, "CRUSH": equity * 0.05, "DRIFT": equity * 0.05}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the desk continuously")
    parser.add_argument("--interval", type=int, default=300, help="seconds between cycles")
    parser.add_argument("--state", default="state/desk.json")
    parser.add_argument("--public", default="public/snapshot.json")
    parser.add_argument("--feed", default="indicative")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--max-failures", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true", help="one cycle, then exit")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    runner = Runner(args)
    if args.once:
        runner.tick()
        return 0
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
