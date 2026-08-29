# The desk needs both runtimes: the Alpaca CLI is a Go binary and is the
# execution spine, while the strategies and risk engine are Python.
FROM golang:1.24-bookworm AS cli
# Pinned, not @latest. The desk's whole execution path was verified against
# v0.0.13 -- the mleg --legs serialisation, the singular/plural --symbol
# inconsistency, the exit codes. An unpinned build silently picked up
# v0.0.14 mid-project and failed on a Go version bump; had it built, the
# container would have been running a CLI nothing was tested against.
ARG ALPACA_CLI_VERSION=v0.0.13
RUN go install github.com/alpacahq/cli/cmd/alpaca@${ALPACA_CLI_VERSION}

FROM python:3.12-slim-bookworm
COPY --from=cli /go/bin/alpaca /usr/local/bin/alpaca

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ALPACA_LIVE_TRADE=false

# uv provides uvx, which launches Alpaca's MCP server for agent research.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml ./
RUN pip install --no-cache-dir mcp httpx numpy pandas pydantic \
    python-dotenv rich scipy openai yfinance

COPY src/ ./src/

# Run as a non-root user: this container holds trading credentials.
RUN useradd --create-home --uid 10001 desk \
    && mkdir -p /app/state /app/public \
    && chown -R desk:desk /app
USER desk

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s \
    CMD alpaca clock --quiet > /dev/null || exit 1

CMD ["python", "-m", "aperture.runner", "--interval", "300"]
