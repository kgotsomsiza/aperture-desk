const SNAPSHOT_KEY = "snapshot:latest";
const MAX_BODY_BYTES = 512 * 1024;

interface Bindings extends Env {
  PUBLISH_TOKEN?: string;
}

const FORBIDDEN_FIELDS = new Set([
  "account_id",
  "account_number",
  "api_key",
  "id",
  "key",
  "secret",
]);

const API_HEADERS = {
  "Cache-Control": "no-store",
  "Content-Type": "application/json; charset=utf-8",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
} as const;

class PayloadTooLarge extends Error {}

function json(body: unknown, status = 200, extraHeaders?: HeadersInit): Response {
  const headers = new Headers(API_HEADERS);
  if (extraHeaders) {
    new Headers(extraHeaders).forEach((value, key) => headers.set(key, value));
  }
  return new Response(JSON.stringify(body), { status, headers });
}

async function tokensMatch(provided: string, expected: string): Promise<boolean> {
  const encoder = new TextEncoder();
  const [providedHash, expectedHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(provided)),
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
  ]);
  return crypto.subtle.timingSafeEqual(providedHash, expectedHash);
}

async function authorized(request: Request, env: Bindings): Promise<boolean> {
  if (!env.PUBLISH_TOKEN) return false;
  const header = request.headers.get("Authorization") ?? "";
  if (!header.startsWith("Bearer ")) return false;
  const provided = header.slice("Bearer ".length);
  return provided.length > 0 && tokensMatch(provided, env.PUBLISH_TOKEN);
}

async function readLimitedBody(request: Request): Promise<string> {
  const declared = Number(request.headers.get("Content-Length") ?? 0);
  if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) {
    throw new PayloadTooLarge();
  }
  if (!request.body) return "";

  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let total = 0;
  let body = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_BODY_BYTES) {
      await reader.cancel();
      throw new PayloadTooLarge();
    }
    body += decoder.decode(value, { stream: true });
  }
  return body + decoder.decode();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function containsPrivateData(root: unknown): boolean {
  const pending: unknown[] = [root];
  let visited = 0;

  while (pending.length > 0) {
    if (++visited > 20_000) return true;
    const value = pending.pop();
    if (typeof value === "string") {
      const lowered = value.toLowerCase();
      if (lowered.includes("c:\\users\\") || lowered.includes("/home/")) return true;
      continue;
    }
    if (Array.isArray(value)) {
      pending.push(...value);
      continue;
    }
    if (!isRecord(value)) continue;
    for (const [key, child] of Object.entries(value)) {
      if (FORBIDDEN_FIELDS.has(key.toLowerCase())) return true;
      pending.push(child);
    }
  }
  return false;
}

function validSnapshot(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value) || value.schema_version !== 1) return false;
  if (!["demo", "practice", "scoring", "final"].includes(String(value.mode))) return false;
  if (typeof value.generated_at !== "string" || !Number.isFinite(Date.parse(value.generated_at))) {
    return false;
  }

  const requiredNumbers = [
    "equity",
    "start_equity",
    "total_return_pct",
    "day_pnl_pct",
    "drawdown_pct",
    "open_risk",
  ];
  if (requiredNumbers.some((key) => typeof value[key] !== "number" || !Number.isFinite(value[key]))) {
    return false;
  }

  const boundedArrays: Array<[string, number]> = [
    ["positions", 250],
    ["open_trades", 100],
    ["attribution", 100],
    ["equity_curve", 2_000],
    ["recent_decisions", 250],
    ["roster", 100],
  ];
  if (boundedArrays.some(([key, limit]) => !Array.isArray(value[key]) || value[key].length > limit)) {
    return false;
  }
  return isRecord(value.counts) && !containsPrivateData(value);
}

async function getSnapshot(env: Bindings, head = false): Promise<Response> {
  const stored = await env.SNAPSHOTS.get(SNAPSHOT_KEY, "text");
  if (!stored) {
    return json({ error: "snapshot_unavailable" }, 503, { "Retry-After": "30" });
  }
  return new Response(head ? null : stored, { status: 200, headers: API_HEADERS });
}

async function publishSnapshot(request: Request, env: Bindings): Promise<Response> {
  if (!env.PUBLISH_TOKEN) {
    console.error("snapshot publisher is not configured");
    return json({ error: "publisher_unavailable" }, 503);
  }
  if (!(await authorized(request, env))) {
    return json({ error: "unauthorized" }, 401, { "WWW-Authenticate": "Bearer" });
  }
  if (!(request.headers.get("Content-Type") ?? "").toLowerCase().startsWith("application/json")) {
    return json({ error: "content_type_must_be_json" }, 415);
  }

  let raw: string;
  try {
    raw = await readLimitedBody(request);
  } catch (error) {
    if (error instanceof PayloadTooLarge) return json({ error: "payload_too_large" }, 413);
    throw error;
  }

  let payload: unknown;
  try {
    payload = JSON.parse(raw);
  } catch {
    return json({ error: "invalid_json" }, 400);
  }
  if (!validSnapshot(payload)) return json({ error: "invalid_snapshot" }, 422);

  await env.SNAPSHOTS.put(SNAPSHOT_KEY, JSON.stringify(payload));
  console.log("snapshot accepted", {
    generated_at: payload.generated_at,
    mode: payload.mode,
  });
  return new Response(null, { status: 204, headers: API_HEADERS });
}

export default {
  async fetch(request: Request, env: Bindings): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname !== "/api/snapshot") {
      return json({ error: "not_found" }, 404);
    }

    if (request.method === "GET") return getSnapshot(env);
    if (request.method === "HEAD") return getSnapshot(env, true);
    if (request.method === "POST") return publishSnapshot(request, env);
    return json({ error: "method_not_allowed" }, 405, { Allow: "GET, HEAD, POST" });
  },
} satisfies ExportedHandler<Bindings>;
