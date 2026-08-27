# The desk needs both runtimes: the Alpaca CLI is a Go binary and is the
# execution spine, while the strategies and risk engine are Python.
FROM golang:1.23-bookworm AS cli
RUN go install github.com/alpacahq/cli/cmd/alpaca@latest

FROM python:3.12-slim-bookworm
COPY --from=cli /go/bin/alpaca /usr/local/bin/alpaca

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ALPACA_LIVE_TRADE=false

COPY pyproject.toml ./
RUN pip install --no-cache-dir alpaca-py httpx numpy pandas pydantic \
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
