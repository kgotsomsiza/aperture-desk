import { SELF, env } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";

const ENDPOINT = "https://aperture.test/api/snapshot";
const TOKEN = "unit-test-publish-token";

function snapshot(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: 1,
    mode: "practice",
    generated_at: "2026-08-28T20:00:00+00:00",
    equity: 100_250,
    start_equity: 100_000,
    total_return_pct: 0.25,
    day_pnl_pct: 0.1,
    high_water_mark: 100_300,
    drawdown_pct: 0.05,
    open_risk: 3_500,
    positions: [],
    open_trades: [],
    attribution: [],
    equity_curve: [],
    recent_decisions: [],
    roster: [],
    research: null,
    shareholder_letter: null,
    counts: { open: 0, pending: 0, closed: 0, vetoes: 0 },
    ...overrides,
  };
}

async function publish(payload: unknown, token = TOKEN): Promise<Response> {
  return SELF.fetch(ENDPOINT, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

describe("snapshot API", () => {
  beforeEach(async () => {
    await env.SNAPSHOTS.delete("snapshot:latest");
  });

  it("reports an honest unavailable state before the first publish", async () => {
    const response = await SELF.fetch(ENDPOINT);
    expect(response.status).toBe(503);
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    await expect(response.json()).resolves.toEqual({ error: "snapshot_unavailable" });
  });

  it("requires the publisher bearer token", async () => {
    const response = await publish(snapshot(), "wrong-value");
    expect(response.status).toBe(401);
    expect(await env.SNAPSHOTS.get("snapshot:latest")).toBeNull();
  });

  it("persists an authenticated, public-safe snapshot", async () => {
    const payload = snapshot();
    expect((await publish(payload)).status).toBe(204);

    const response = await SELF.fetch(ENDPOINT);
    expect(response.status).toBe(200);
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    await expect(response.json()).resolves.toEqual(payload);

    const head = await SELF.fetch(ENDPOINT, { method: "HEAD" });
    expect(head.status).toBe(200);
    expect(await head.text()).toBe("");
  });

  it("rejects private identifiers even with valid authentication", async () => {
    const response = await publish(snapshot({ account_number: "not-public" }));
    expect(response.status).toBe(422);
    expect(await env.SNAPSHOTS.get("snapshot:latest")).toBeNull();
  });

  it("rejects malformed schema, media type, and oversized bodies", async () => {
    expect((await publish(snapshot({ equity: "100250" }))).status).toBe(422);

    const wrongType = await SELF.fetch(ENDPOINT, {
      method: "POST",
      headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "text/plain" },
      body: "{}",
    });
    expect(wrongType.status).toBe(415);

    const tooLarge = await SELF.fetch(ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${TOKEN}`,
        "Content-Type": "application/json",
        "Content-Length": String(512 * 1024 + 1),
      },
      body: "{}",
    });
    expect(tooLarge.status).toBe(413);
  });

  it("does not expose unrelated Worker routes", async () => {
    const response = await SELF.fetch("https://aperture.test/api/private");
    expect(response.status).toBe(404);

    const method = await SELF.fetch(ENDPOINT, { method: "DELETE" });
    expect(method.status).toBe(405);
    expect(method.headers.get("Allow")).toBe("GET, HEAD, POST");
  });
});
