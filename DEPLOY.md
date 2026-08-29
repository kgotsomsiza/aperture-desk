# Deployment runbook

The trading loop and public desk have deliberately separate credentials, failure domains, and lifetimes:

- a private container runs the Alpaca cycle and owns the persistent ledger;
- a Cloudflare Worker accepts only a redacted, bounded snapshot and serves the static public desk from KV-backed state.

The dashboard can fail without stopping trading, and the trading host can be destroyed after the event while the final evidence remains public.

## 1. Deploy the public desk

From `dashboard/`:

```bash
npm ci
npm test
npm run check
npx wrangler login
npx wrangler kv namespace create SNAPSHOTS
```

Put the returned namespace ID in `dashboard/wrangler.jsonc`. Then create a long random publishing token in a password manager and enter it interactively:

```bash
npx wrangler secret put PUBLISH_TOKEN
npm run deploy
```

The current production URL is [aperture-desk.kgotsonceba.workers.dev](https://aperture-desk.kgotsonceba.workers.dev).

The Worker routes `/api/*` before static assets. `GET /api/snapshot` is public and no-store; `POST /api/snapshot` requires the bearer token, enforces a 512 KiB streaming body limit, validates the public schema and mode, rejects forbidden/private keys and local paths recursively, and writes only validated JSON to KV.

## 2. Configure the trading host

Build from `Dockerfile`. It installs the Alpaca CLI in a Go stage and copies only the binary into a non-root Python runtime.

Set these as host-managed secrets or variables—never repository files:

```text
ALPACA_API_KEY
ALPACA_SECRET_KEY
APERTURE_EXPECT_ACCOUNT
ALPACA_LIVE_TRADE=false
APERTURE_OPTIONS_FEED=indicative

APERTURE_LLM_VENDOR=featherless
FEATHERLESS_API_KEY
APERTURE_FEATHERLESS_FAST=Qwen/Qwen3-Next-80B-A3B-Instruct
APERTURE_FEATHERLESS_REASONING=moonshotai/Kimi-K2-Instruct

APERTURE_SNAPSHOT_URL=https://aperture-desk.kgotsonceba.workers.dev/api/snapshot
APERTURE_PUBLISH_TOKEN
APERTURE_PUBLIC_MODE=practice
```

`APERTURE_PUBLISH_TOKEN` must match the Worker's `PUBLISH_TOKEN`. Use the platform secret UI so the value does not enter shell history or logs.

Mount a persistent volume at `/app/state`. The ledger records fill-confirmed structures, pending reservations, per-strategy attribution, hired strategies, and the account fingerprint. Losing the volume does not change broker positions, but it removes the evidence Aperture needs to reconcile, allocate, and manage them safely.

## 3. Prove the environment before the event

Run against the exact deployed image and dedicated paper account:

```bash
python -m aperture.preflight
python -m aperture.runner --dry-run --once --state state/preflight.json
```

Review the preflight output for:

- paper account identity and the expected-account guard;
- options level 3 / multi-leg permission;
- available option-data feed, Greeks, and quote freshness;
- historical option-data access;
- Alpaca CLI multi-leg dry-run acceptance.

Only when the account is otherwise flat and the operator intends to send a paper order, run:

```bash
python -m aperture.preflight --live-test
```

That command submits one real 1-lot paper spread, waits for fill state, closes it, and verifies the entry/exit path. Use a fresh throwaway state file; never reuse its ledger for the scored account.

## 4. Start the private runner

The container command is:

```bash
python -m aperture.runner --interval 300 --state state/desk.json
```

The included `railway.json` and `fly.toml` are worker configurations; neither exposes an HTTP service. On Fly.io, create and mount `desk_state` at `/app/state` before deployment. On Railway, attach a volume at the same path.

Start the scored account from a new state file and a flat broker book. Do not copy practice state into it. The first cycle binds the ledger to a hash of `APERTURE_EXPECT_ACCOUNT`.

## 5. Public modes

The dashboard label is driven by `APERTURE_PUBLIC_MODE`:

- `practice` — rehearsals before the official window;
- `scoring` — the judged account during the official window;
- `final` — the immutable final snapshot after flattening.

Change the host variable, restart the worker safely, and confirm the public label after each transition. If no live snapshot is available, the site falls back to clearly labeled demo data rather than presenting it as performance.

## 6. Monitoring and recovery

Monitor container health, cycle logs, the append-only audit, and the public snapshot timestamp. The runner is designed to recover accepted orders by deterministic client-order ID and to distinguish pending, filled, canceled, and rejected parent multi-leg states.

On a systematic error, the runner stops new entries and engages the kill switch after its failure threshold. Publishing errors are logged but never count as trading-cycle failures.

To request a stop, create `KILL_SWITCH` in the container working directory. The Warden will cancel open orders and flatten through confirmed close orders while the market is open; it resumes flattening on the next open if necessary. Do not terminate the container until the broker book and pending ledger are both empty.

## 7. Tournament endgame

Aperture's clock stops all new risk and begins flattening on **Thursday, 3 September 2026 at 14:00 ET**, ahead of the Friday equity snapshot. After the broker is flat:

1. verify positions and open orders are both empty;
2. set `APERTURE_PUBLIC_MODE=final`;
3. run one last publish cycle;
4. verify the public desk timestamp and final label;
5. back up the private ledger and audit log outside the public repository;
6. stop and remove the trading container, leaving the Cloudflare Worker online.

Never publish account numbers, API keys, bearer tokens, raw broker payloads, private state, or audit files.
