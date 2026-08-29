const API_URL = "/api/snapshot";
const DEMO_URL = "/demo-snapshot.json";
const REFRESH_MS = 60_000;

const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const moneyExact = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

const byId = (id) => document.getElementById(id);
const safeArray = (value) => Array.isArray(value) ? value : [];
const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;

function setText(id, value) {
  const node = byId(id);
  if (node) node.textContent = String(value);
}

function tone(node, value) {
  node.classList.remove("positive", "negative", "neutral");
  node.classList.add(value > 0 ? "positive" : value < 0 ? "negative" : "neutral");
}

function signedPercent(value, digits = 2) {
  const number = finite(value);
  return `${number > 0 ? "+" : ""}${number.toFixed(digits)}%`;
}

function formatTime(value, options = {}) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
    ...options,
  }).format(date);
}

function formatDay(value) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(date);
}

async function fetchJson(url, timeoutMs = 5_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function loadSnapshot() {
  try {
    const payload = await fetchJson(API_URL);
    render(payload, "published");
  } catch {
    try {
      const demo = await fetchJson(DEMO_URL);
      render(demo, "demo");
    } catch {
      setText("desk-state", "DATA UNAVAILABLE");
      setText("freshness-label", "Snapshot unavailable");
    }
  }
}

function render(data, source) {
  const mode = source === "demo" ? "demo" : String(data.mode || "practice").toLowerCase();
  const demo = mode === "demo";
  byId("demo-banner").hidden = !demo;
  document.body.dataset.mode = mode;

  renderStatus(data, mode);
  renderMetrics(data);
  renderResearch(data);
  renderRoster(data);
  renderChart(data);
  renderStructures(data);
  renderAttribution(data);
  renderDecisions(data);
  renderLetter(data);
}

function renderStatus(data, mode) {
  const generated = new Date(data.generated_at);
  const ageMinutes = Number.isFinite(generated.getTime()) ? (Date.now() - generated.getTime()) / 60_000 : Infinity;
  const isFinal = mode === "final";
  const fresh = ageMinutes <= 15;
  const labels = {
    demo: "DEMO SCENARIO",
    practice: fresh ? "PRACTICE · LIVE" : "PRACTICE · STALE",
    scoring: fresh ? "SCORING · LIVE" : "SCORING · STALE",
    final: "FINAL SNAPSHOT",
  };

  setText("mode-pill", mode.toUpperCase());
  setText("desk-state", labels[mode] || "OBSERVING");
  setText("generated-time", formatTime(data.generated_at));
  setText("footer-updated", `Last snapshot: ${formatTime(data.generated_at)}`);
  setText(
    "freshness-label",
    mode === "demo" ? "Sample data" : isFinal ? "Desk complete" : fresh ? "Snapshot current" : "Snapshot delayed",
  );
  byId("flow-research").textContent = data.research ? "COMPLETE" : "READY";
  byId("flow-promotion").textContent = safeArray(data.research?.promoted).length ? "HIRED" : "VERIFY";
  byId("flow-execution").textContent = finite(data.counts?.pending) > 0 ? "PENDING" : "SYNCED";
}

function renderMetrics(data) {
  const equity = finite(data.equity);
  const totalReturn = finite(data.total_return_pct);
  const day = finite(data.day_pnl_pct);
  const drawdown = finite(data.drawdown_pct);

  setText("metric-equity", moneyExact.format(equity));
  setText("metric-equity-context", `From ${money.format(finite(data.start_equity, 100_000))} starting equity`);
  setText("metric-return", signedPercent(totalReturn, 3));
  setText("metric-day", signedPercent(day, 3));
  setText("metric-drawdown", `${drawdown.toFixed(3)}%`);
  setText("metric-risk", money.format(finite(data.open_risk)));
  setText("metric-structures", `${finite(data.counts?.open)} open · ${finite(data.counts?.pending)} pending`);
  setText("metric-vetoes", Math.trunc(finite(data.counts?.vetoes)));

  tone(byId("metric-return"), totalReturn);
  tone(byId("metric-day"), day);
  tone(byId("metric-drawdown"), -drawdown);
}

function windowLabel(value) {
  if (!Array.isArray(value) || value.length < 2) return "—";
  return `${String(value[0]).slice(0, 10)} → ${String(value[1]).slice(0, 10)}`;
}

function renderResearch(data) {
  const research = data.research;
  const promoted = safeArray(research?.promoted);
  setText("research-tested", research ? Math.trunc(finite(research.tested)) : "—");
  setText("research-trials", research ? Math.trunc(finite(research.cumulative_trials)) : "—");
  setText("promotion-title", promoted.length ? `${promoted.join(", ")} HIRED` : research ? "No promotion — incumbent retained" : "Awaiting research");
  setText("lab-session", research?.session ? `Session ${research.session}` : "Awaiting close");
  setText("lab-status", promoted.length ? "PROMOTED" : research ? "COMPLETE" : "IDLE");
  setText("training-window", windowLabel(research?.training_window));
  setText("validation-window", windowLabel(research?.validation_window));
  setText("lab-summary", research?.summary || "The nightly lab runs only after a completed market session.");
  setText("fast-model", research?.reasoning?.fast_model || research?.reasoning?.model || "Not configured");
  setText("reasoning-model", data.shareholder_letter?.reasoning?.reasoning_model || "Not configured");
}

function strategyRole(row) {
  if (row.origin === "research_lab") return row.mutation || "Research-lab hire";
  return {
    CARRY: "Index volatility carry",
    CRUSH: "Post-earnings volatility crush",
    DRIFT: "Post-earnings directional drift",
  }[row.strategy] || "Options strategy";
}

function renderRoster(data) {
  const roster = safeArray(data.roster);
  const mini = byId("mini-roster");
  const grid = byId("roster-grid");
  mini.replaceChildren();
  grid.replaceChildren();

  roster.forEach((row) => {
    const weight = Math.max(0, finite(row.weight));

    const miniRow = document.createElement("div");
    miniRow.className = "mini-row";
    const miniName = document.createElement("span");
    miniName.textContent = row.strategy || "—";
    const track = document.createElement("span");
    track.className = "allocation-track";
    const fill = document.createElement("i");
    fill.style.width = `${Math.min(weight * 100, 100)}%`;
    track.append(fill);
    const miniValue = document.createElement("span");
    miniValue.textContent = `${(weight * 100).toFixed(0)}%`;
    miniRow.append(miniName, track, miniValue);
    mini.append(miniRow);

    const card = document.createElement("article");
    card.className = "strategy-card";
    const top = document.createElement("div");
    top.className = "strategy-top";
    const origin = document.createElement("span");
    origin.className = "strategy-origin";
    origin.textContent = row.origin === "research_lab" ? "RESEARCH LAB HIRE" : "FOUNDING STRATEGY";
    const status = document.createElement("span");
    status.className = `strategy-status ${row.status === "fired" ? "fired" : ""}`;
    status.textContent = String(row.status || "funded").toUpperCase();
    top.append(origin, status);

    const title = document.createElement("h3");
    title.textContent = row.strategy || "UNNAMED";
    const role = document.createElement("span");
    role.className = "strategy-role";
    role.textContent = strategyRole(row);

    const weightBlock = document.createElement("div");
    weightBlock.className = "strategy-weight";
    const weightLine = document.createElement("div");
    const weightLabel = document.createElement("span");
    weightLabel.textContent = "SHARE OF RISK BUDGET";
    const weightValue = document.createElement("b");
    weightValue.textContent = `${(weight * 100).toFixed(1)}%`;
    weightLine.append(weightLabel, weightValue);
    const weightTrack = document.createElement("span");
    weightTrack.className = "allocation-track";
    const weightFill = document.createElement("i");
    weightFill.style.width = `${Math.min(weight * 100, 100)}%`;
    weightTrack.append(weightFill);
    weightBlock.append(weightLine, weightTrack);

    const note = document.createElement("p");
    note.className = "strategy-note";
    note.textContent = row.reason || (row.origin === "research_lab" ? "Funded on probation until live evidence accumulates." : "Allocation moves only when fill-derived evidence accumulates.");

    const evidence = document.createElement("div");
    evidence.className = "strategy-evidence";
    if (row.evidence) {
      evidence.textContent = `${finite(row.evidence.trades)} holdout trades · ${finite(row.evidence.wins)} wins · ${(finite(row.evidence.edge) * 100).toFixed(1)}% edge / risk · t=${finite(row.evidence.t_stat).toFixed(2)}`;
    } else {
      evidence.textContent = "Designed prior · live record updates from confirmed closes";
    }

    card.append(top, title, role, weightBlock, note, evidence);
    grid.append(card);
  });

  if (roster.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Roster not yet published.";
    grid.append(empty);
  }
}

function svgNode(name, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function renderChart(data) {
  const rows = safeArray(data.equity_curve)
    .map((row) => ({ time: new Date(row.t).getTime(), equity: finite(row.equity, NaN) }))
    .filter((row) => Number.isFinite(row.time) && Number.isFinite(row.equity) && row.equity > 0)
    .sort((a, b) => a.time - b.time);
  const group = byId("chart-dynamic");
  group.replaceChildren();

  if (rows.length === 0) {
    setText("chart-change", "—");
    setText("chart-range", "No observations");
    return;
  }

  const left = 52;
  const right = 818;
  const top = 36;
  const bottom = 243;
  const values = rows.map((row) => row.equity);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const pad = Math.max((rawMax - rawMin) * 0.18, rawMax * 0.0025, 1);
  const min = rawMin - pad;
  const max = rawMax + pad;
  const span = max - min || 1;
  const timeSpan = rows.at(-1).time - rows[0].time || 1;
  const point = (row) => ({
    x: left + ((row.time - rows[0].time) / timeSpan) * (right - left),
    y: bottom - ((row.equity - min) / span) * (bottom - top),
  });
  const points = rows.map(point);
  const line = points.map((p, index) => `${index ? "L" : "M"}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
  const area = `${line} L${points.at(-1).x.toFixed(2)},${bottom} L${points[0].x.toFixed(2)},${bottom} Z`;
  group.append(svgNode("path", { d: area, class: "equity-area" }));
  group.append(svgNode("path", { d: line, class: "equity-line" }));
  group.append(svgNode("circle", { cx: points.at(-1).x, cy: points.at(-1).y, r: 5, class: "equity-dot" }));

  [max - pad, (max + min) / 2, min + pad].forEach((value, index) => {
    const label = svgNode("text", { x: 4, y: [40, 143, 246][index], class: "chart-label" });
    label.textContent = compact.format(value);
    group.append(label);
  });
  const firstLabel = svgNode("text", { x: left, y: 278, class: "chart-label" });
  firstLabel.textContent = formatDay(rows[0].time);
  const lastLabel = svgNode("text", { x: right, y: 278, class: "chart-label", "text-anchor": "end" });
  lastLabel.textContent = formatDay(rows.at(-1).time);
  group.append(firstLabel, lastLabel);

  const change = ((rows.at(-1).equity / rows[0].equity) - 1) * 100;
  setText("chart-change", signedPercent(change, 2));
  tone(byId("chart-change"), change);
  setText("chart-range", `${rows.length} hourly marks`);
  setText("chart-desc", `Account equity moved from ${moneyExact.format(rows[0].equity)} to ${moneyExact.format(rows.at(-1).equity)}.`);
}

function structureName(trade) {
  const count = safeArray(trade.legs).length;
  if (count === 4) return "4-leg condor";
  if (count === 2) return "2-leg spread";
  return `${count}-leg structure`;
}

function addCell(row, text, className) {
  const cell = document.createElement("td");
  cell.textContent = String(text);
  if (className) cell.className = className;
  row.append(cell);
}

function renderStructures(data) {
  const trades = safeArray(data.open_trades);
  const body = byId("structures-body");
  body.replaceChildren();
  trades.forEach((trade) => {
    const row = document.createElement("tr");
    addCell(row, trade.strategy || "—");
    addCell(row, trade.underlying || "—");
    addCell(row, `${structureName(trade)} × ${finite(trade.qty, 1)}`);
    addCell(row, money.format(finite(trade.max_loss)));
    addCell(row, String(trade.status || "open").replaceAll("_", " ").toUpperCase(), "table-status");
    body.append(row);
  });
  setText("open-count", trades.length);
  byId("structures-empty").hidden = trades.length > 0;
}

function renderAttribution(data) {
  const rows = safeArray(data.attribution);
  const body = byId("attribution-body");
  body.replaceChildren();
  rows.forEach((item) => {
    const row = document.createElement("tr");
    addCell(row, item.strategy || "—");
    addCell(row, Math.trunc(finite(item.closed)));
    addCell(row, item.win_rate === null || item.win_rate === undefined ? "—" : `${finite(item.win_rate).toFixed(1)}%`);
    const pnl = finite(item.realized_pnl);
    addCell(row, `${pnl > 0 ? "+" : ""}${money.format(pnl)}`, pnl > 0 ? "positive" : pnl < 0 ? "negative" : "neutral");
    addCell(row, money.format(finite(item.risk_at_work)));
    body.append(row);
  });
  byId("attribution-empty").hidden = rows.length > 0;
}

function eventKind(event) {
  const value = String(event || "event").toLowerCase();
  if (value.includes("veto") || value.includes("error") || value.includes("kill")) return "veto";
  if (value.includes("hire") || value.includes("research") || value.includes("alloc")) return "hire";
  if (value.includes("fill") || value.includes("open") || value.includes("close")) return "fill";
  return "event";
}

function renderDecisions(data) {
  const decisions = safeArray(data.recent_decisions).slice(-9).reverse();
  const tape = byId("decision-tape");
  tape.replaceChildren();
  decisions.forEach((decision) => {
    const item = document.createElement("li");
    item.className = "decision-item";
    const time = document.createElement("time");
    time.dateTime = decision.ts || "";
    time.textContent = decision.ts ? formatTime(decision.ts, { month: undefined, day: undefined, second: "2-digit" }) : "—";
    const tag = document.createElement("span");
    const kind = eventKind(decision.event);
    tag.className = `decision-tag ${kind}`;
    tag.textContent = String(decision.event || "EVENT").replaceAll("_", " ").toUpperCase();
    const copy = document.createElement("div");
    copy.className = "decision-copy";
    const title = document.createElement("b");
    title.textContent = [decision.strategy, decision.underlying].filter(Boolean).join(" · ") || "DESK";
    const summary = document.createElement("span");
    summary.textContent = decision.summary || decision.rationale || decision.reason || "Decision recorded.";
    copy.append(title, summary);
    item.append(time, tag, copy);
    tape.append(item);
  });
  if (decisions.length === 0) {
    const item = document.createElement("li");
    item.className = "empty-state";
    item.textContent = "The public audit tape is empty.";
    tape.append(item);
  }
}

function renderLetter(data) {
  const letter = data.shareholder_letter;
  setText("letter-date", letter?.as_of ? `As of ${letter.as_of}` : "Awaiting first close");
  setText(
    "shareholder-letter",
    letter?.text || "After each session, the reasoning layer explains a deterministic fact block. The publisher rejects invented or rounded numbers.",
  );
}

loadSnapshot();
setInterval(loadSnapshot, REFRESH_MS);
