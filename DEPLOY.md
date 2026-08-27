# Deploying the desk

The trading loop and the dashboard have different lifespans, so they are hosted
separately. The loop runs for the length of the contest and is then deleted; the
dashboard has to stay reachable long after, so it lives on static hosting where
that costs nothing.

## Trading loop — a container

Runs `aperture.runner`, which cycles every five minutes while the market is open
and sleeps until the next open when it is not.

Required environment variables (set them in the host's dashboard, never in this
repository):

    ALPACA_API_KEY
    ALPACA_SECRET_KEY
    ALPACA_LIVE_TRADE=false
    APERTURE_OPTIONS_FEED=indicative
    OPENAI_API_KEY

A persistent volume must be mounted at `/app/state`. The desk keeps its own
ledger there — which strategy opened which structure, and what its maximum loss
was at entry. Losing it does not lose money, but it does lose the desk's ability
to enforce per-strategy budgets or attribute P&L, and the next cycle would treat
an already-open book as empty.

### Railway

Connect the repository, add the variables above, add a volume at `/app/state`.
`railway.json` selects the Dockerfile and sets the start command.

### Fly.io

`fly launch --no-deploy`, then `fly secrets set ...`, then
`fly volumes create desk_state --size 1`, then `fly deploy`. `fly.toml`
deliberately declares no HTTP service: this is a worker, and exposing a port
invites the platform to cycle machines mid-trade.

## Dashboard — static hosting

The runner writes `public/snapshot.json` after every cycle. That file plus the
dashboard page are the entire deployment: no server, no database, no runtime.

Publish `public/` to any static host. When the trading container is destroyed at
the end of the contest, the last snapshot stays live at the same URL.

## Shutting down

    # stop trading
    touch KILL_SWITCH        # inside the container, or commit a restart with it present

    # then, once positions are closed and the final snapshot is published
    railway down             # or: fly apps destroy aperture-desk

The dashboard is unaffected and stays up.
