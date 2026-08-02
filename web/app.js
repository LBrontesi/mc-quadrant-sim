"use strict";

/* ---------- Constants ---------- */

const REGIME_ORDER = [
  "high_growth_low_inflation",
  "high_growth_high_inflation",
  "low_growth_high_inflation",
  "low_growth_low_inflation",
];

const REGIME_NAMES = {
  high_growth_low_inflation: "High growth / low inflation",
  high_growth_high_inflation: "High growth / high inflation",
  low_growth_high_inflation: "Low growth / high inflation",
  low_growth_low_inflation: "Low growth / low inflation",
};

const REGIME_COLORS = {
  high_growth_low_inflation: "#2f855a",
  high_growth_high_inflation: "#d97706",
  low_growth_high_inflation: "#c2410c",
  low_growth_low_inflation: "#2563eb",
};

const DEFAULT_TICKERS = ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"];
const SYNTHETIC_OPTIONS = [...DEFAULT_TICKERS, "DBMF", "KMLM", "TLT", "QQQ"];

const DEFAULT_WEIGHTS = {
  SPY: 40, IEF: 20, GLD: 10, DBC: 10, EFA: 10, VNQ: 5, TIP: 3, SHY: 2, DBMF: 5, KMLM: 5,
};

const METRIC_FIELDS = [
  ["mean", "Mean"], ["p05", "P05"], ["p50", "Median"], ["p95", "P95"], ["std", "Volatility"],
];

const state = {
  loadPayload: null,
  loadResult: null,
  selected: [],
  weights: {},
  results: null,
  diagnostics: null,
};

/* ---------- Helpers ---------- */

const $ = (id) => document.getElementById(id);

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function pct(value, digits = 1) {
  return fmt(value * 100, digits) + "%";
}

async function postJSON(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Request failed with status ${response.status}`);
  }
  return data;
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    if (!file) return resolve(null);
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read uploaded file."));
    reader.readAsText(file);
  });
}

function setStatus(el, message, isError = false) {
  el.textContent = message || "";
  el.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function defaultWeight(ticker) {
  const base = ticker.endsWith("_SIM") ? ticker.slice(0, -4) : ticker.endsWith("SIM") ? ticker.slice(0, -3) : ticker;
  return DEFAULT_WEIGHTS[base] ?? 0;
}

function downloadCSV(filename, text) {
  const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function toCSV(columns, rows) {
  const escape = (value) => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [columns.join(","), ...rows.map((row) => row.map(escape).join(","))].join("\n");
}

/* ---------- SVG charts ---------- */

const SVG_NS = "http://www.w3.org/2000/svg";
const MARGIN = { top: 12, right: 12, bottom: 26, left: 44 };

function createSvg(width, height) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "auto");
  return svg;
}

function svgText(parent, x, y, text, cls = "", anchor = "middle") {
  const el = document.createElementNS(SVG_NS, "text");
  el.setAttribute("x", x);
  el.setAttribute("y", y);
  el.setAttribute("text-anchor", anchor);
  el.setAttribute("class", cls);
  el.textContent = text;
  parent.appendChild(el);
  return el;
}

function drawAxes(svg, width, height, xTicks, yTicks, xLabel, yLabel) {
  const grid = document.createElementNS(SVG_NS, "g");
  xTicks.forEach(([x, label]) => {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", x); line.setAttribute("y1", 0);
    line.setAttribute("x2", x); line.setAttribute("y2", height);
    line.setAttribute("stroke", "#334155"); line.setAttribute("stroke-width", "1");
    grid.appendChild(line);
    svgText(grid, x, height + 14, label, "chart-title");
  });
  yTicks.forEach(([y, label]) => {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", 0); line.setAttribute("y1", y);
    line.setAttribute("x2", width); line.setAttribute("y2", y);
    line.setAttribute("stroke", "#334155"); line.setAttribute("stroke-width", "1");
    grid.appendChild(line);
    svgText(grid, -6, y + 4, label, "chart-title", "end");
  });
  if (xLabel) svgText(grid, width / 2, height + 22, xLabel, "chart-title");
  if (yLabel) {
    const el = svgText(grid, -30, height / 2, yLabel, "chart-title");
    el.setAttribute("transform", "rotate(-90 -30 " + height / 2 + ")");
  }
  svg.appendChild(grid);
}

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const ticks = [];
  for (let value = Math.ceil(min / step) * step; value <= max + 1e-9; value += step) ticks.push(value);
  if (ticks.length > count * 2) return niceTicks(min, max, count - 1);
  return ticks.length ? ticks : [min];
}

function lineChart(container, labels, series) {
  container.innerHTML = "";
  const width = 560, height = 300;
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  let min = Infinity, max = -Infinity;
  series.forEach((s) => s.values.forEach((v) => { if (v < min) min = v; if (v > max) max = v; }));
  const pad = (max - min) * 0.08 || 1;
  min -= pad; max += pad;
  const xScale = (i) => MARGIN.left + (i / Math.max(labels.length - 1, 1)) * plotW;
  const yScale = (v) => MARGIN.top + (1 - (v - min) / (max - min)) * plotH;
  const svg = createSvg(width, height);
  const xTicks = niceTicks(0, labels.length - 1, 5).map((t) => [xScale(t), labels[Math.round(t)] ?? ""]);
  const yTicks = niceTicks(min, max, 5).map((v) => [yScale(v), v.toFixed(0)]);
  drawAxes(svg, plotW, plotH, xTicks, yTicks, "Period", "Wealth");
  series.forEach((s) => {
    const points = s.values.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
    const path = document.createElementNS(SVG_NS, "polyline");
    path.setAttribute("points", points);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", s.color);
    path.setAttribute("stroke-width", "2");
    svg.appendChild(path);
  });
  container.appendChild(svg);
  const legend = document.createElement("div");
  series.forEach((s) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-dot" style="background:${s.color}"></span>${s.name}`;
    legend.appendChild(item);
  });
  container.appendChild(legend);
}

function barChart(container, items, options = {}) {
  container.innerHTML = "";
  const width = 560, height = 280;
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const values = items.map((item) => item.value);
  const max = Math.max(...values, options.min || 0) * 1.1 || 1;
  const svg = createSvg(width, height);
  const yTicks = niceTicks(0, max, 4).map((v) => [MARGIN.top + (1 - v / max) * plotH, v.toFixed(2)]);
  drawAxes(svg, plotW, plotH, [], yTicks, "", options.yLabel || "");
  const slot = plotW / items.length;
  items.forEach((item, index) => {
    const barW = slot * 0.62;
    const x = MARGIN.left + index * slot + (slot - barW) / 2;
    const y = MARGIN.top + (1 - item.value / max) * plotH;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y);
    rect.setAttribute("width", barW); rect.setAttribute("height", Math.max(plotH - (y - MARGIN.top), 0));
    rect.setAttribute("fill", item.color || "#2563eb");
    rect.setAttribute("rx", "2");
    svg.appendChild(rect);
    const label = item.label;
    const labelEl = svgText(svg, x + barW / 2, height - 8, label.length > 16 ? label.slice(0, 15) + "..." : label, "chart-title");
    labelEl.setAttribute("transform", `rotate(-18 ${x + barW / 2} ${height - 8})`);
    svgText(svg, x + barW / 2, Math.max(y - 4, 10), item.value.toFixed(options.digits ?? 2), "chart-title");
  });
  container.appendChild(svg);
}

function heatmap(container, labels, values, domain, title) {
  container.innerHTML = "";
  const n = labels.length;
  const width = 460, height = 460;
  const size = Math.min((width - 60) / n, (height - 60) / n);
  const svg = createSvg(width, height);
  const x0 = 56, y0 = 10;
  const colorFor = (value) => {
    const t = (value - domain[0]) / (domain[1] - domain[0]);
    if (t <= 0.5) {
      const k = t / 0.5;
      const r = Math.round(37 + k * (59 - 37));
      const g = Math.round(99 + k * (130 - 99));
      const b = Math.round(235 + k * (219 - 235));
      return `rgb(${r},${g},${b})`;
    }
    const k = (t - 0.5) / 0.5;
    const r = Math.round(59 + k * (194 - 59));
    const g = Math.round(130 + k * (65 - 130));
    const b = Math.round(219 + k * (12 - 219));
    return `rgb(${r},${g},${b})`;
  };
  labels.forEach((rowLabel, row) => {
    svgText(svg, x0 - 8, y0 + row * size + size / 2 + 4, rowLabel.length > 16 ? rowLabel.slice(0, 15) + "..." : rowLabel, "chart-title", "end");
    labels.forEach((colLabel, col) => {
      const value = values[row][col];
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", x0 + col * size);
      rect.setAttribute("y", y0 + row * size);
      rect.setAttribute("width", size - 2);
      rect.setAttribute("height", size - 2);
      rect.setAttribute("fill", colorFor(value));
      svg.appendChild(rect);
      svgText(svg, x0 + col * size + (size - 2) / 2, y0 + row * size + (size - 2) / 2 + 4, value.toFixed(2), "chart-title");
    });
    svgText(svg, x0 + row * size + (size - 2) / 2, height - 4, labels[row].length > 16 ? labels[row].slice(0, 15) + "..." : labels[row], "chart-title");
  });
  if (title) svgText(svg, width / 2, 6, title, "chart-title");
  container.appendChild(svg);
}

function scatterChart(container, points, xLabel, yLabel) {
  container.innerHTML = "";
  const width = 560, height = 300;
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xPad = (xMax - xMin) * 0.1 || 1;
  const yPad = (yMax - yMin) * 0.1 || 1;
  xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;
  const xScale = (v) => MARGIN.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const yScale = (v) => MARGIN.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;
  const svg = createSvg(width, height);
  const xTicks = niceTicks(xMin, xMax, 5).map((v) => [xScale(v), v.toFixed(1)]);
  const yTicks = niceTicks(yMin, yMax, 5).map((v) => [yScale(v), v.toFixed(1)]);
  drawAxes(svg, plotW, plotH, xTicks, yTicks, xLabel, yLabel);
  const seen = new Set();
  points.forEach((p) => {
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", xScale(p.x));
    circle.setAttribute("cy", yScale(p.y));
    circle.setAttribute("r", 3.4);
    circle.setAttribute("fill", p.color);
    circle.setAttribute("opacity", "0.75");
    svg.appendChild(circle);
  });
  container.appendChild(svg);
}

function histChart(container, values, bins = 45) {
  container.innerHTML = "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = 560, height = 280;
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const span = max - min || 1;
  const counts = new Array(bins).fill(0);
  values.forEach((v) => {
    const index = Math.min(bins - 1, Math.floor(((v - min) / span) * bins));
    counts[index] += 1;
  });
  const peak = Math.max(...counts) || 1;
  const svg = createSvg(width, height);
  const yTicks = niceTicks(0, peak, 4).map((v) => [MARGIN.top + (1 - v / peak) * plotH, v.toFixed(0)]);
  drawAxes(svg, plotW, plotH, [], yTicks, "Terminal wealth", "Paths");
  const slot = plotW / bins;
  counts.forEach((count, index) => {
    const barW = slot * 0.9;
    const x = MARGIN.left + index * slot + (slot - barW) / 2;
    const y = MARGIN.top + (1 - count / peak) * plotH;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y);
    rect.setAttribute("width", barW); rect.setAttribute("height", plotH - (y - MARGIN.top));
    rect.setAttribute("fill", "#2563eb");
    rect.setAttribute("opacity", "0.82");
    svg.appendChild(rect);
  });
  container.appendChild(svg);
}

/* ---------- Inputs ---------- */

function sourceValue() {
  return document.querySelector('input[name="source"]:checked').value;
}

function toggleSourceGroups() {
  const source = sourceValue();
  $("demo-group").classList.toggle("hidden", source !== "demo");
  $("yahoo-group").classList.toggle("hidden", source !== "yahoo");
  $("csv-group").classList.toggle("hidden", source !== "csv");
}

function thresholdPayload(selectId, fixedId) {
  const mode = $(selectId).value;
  return mode === "fixed" ? `fixed:${$(fixedId).value}` : mode;
}

function gatherLoadPayload() {
  const source = sourceValue();
  const payload = { source };
  if (source === "demo") {
    payload.seed = Number($("demo-seed").value);
  } else if (source === "yahoo") {
    payload.tickers = $("yahoo-tickers").value;
    payload.start = $("yahoo-start").value;
    payload.end = $("yahoo-end").value;
    payload.proxies = $("yahoo-proxies").value;
    payload.synthetic = Array.from(document.querySelectorAll('#synthetic-options input[type="checkbox"]:checked')).map((el) => el.value);
    payload.synthetic_seed = Number($("synthetic-seed").value);
  } else {
    payload.csv_prices = null;
    payload.csv_macro = null;
    payload.asset_input = document.querySelector('input[name="asset-input"]:checked').value;
    payload.monthly = $("csv-monthly").checked;
    payload.growth_col = $("csv-growth").value || "growth";
    payload.inflation_col = $("csv-inflation").value || "inflation";
  }
  return payload;
}

async function fillCsvPayload(payload) {
  if (payload.source === "csv") {
    payload.csv_prices = await readFile($("csv-prices").files[0]);
    payload.csv_macro = await readFile($("csv-macro").files[0]);
  }
}

function gatherScenario() {
  return {
    growth_threshold: thresholdPayload("growth-threshold", "growth-fixed"),
    inflation_threshold: thresholdPayload("inflation-threshold", "inflation-fixed"),
    macro_lag: Number($("macro-lag").value),
    transition_uncertainty: Number($("transition-uncertainty").value),
    periods: Number($("periods").value),
    paths: Number($("paths").value),
    seed: Number($("seed").value),
    start_state: $("start-state").value,
    distribution: $("distribution").value,
    degrees_of_freedom: Number($("degrees-of-freedom").value),
    block_size: Number($("block-size").value),
    rebalance: $("rebalance").value,
    cost_bps: Number($("cost-bps").value),
    base_currency: $("base-currency").value,
    currency_map: $("currency-map").value,
  };
}

/* ---------- Portfolio editor ---------- */

function renderTickerChecklist(tickers, defaults) {
  const container = $("ticker-list");
  container.innerHTML = "";
  tickers.forEach((ticker) => {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = ticker;
    checkbox.checked = defaults.includes(ticker);
    checkbox.addEventListener("change", renderWeightEditor);
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(ticker));
    container.appendChild(label);
  });
}

function selectedTickers() {
  return Array.from(document.querySelectorAll('#ticker-list input[type="checkbox"]:checked')).map((el) => el.value);
}

function renderWeightEditor() {
  const selected = selectedTickers();
  const container = $("weight-editor");
  container.innerHTML = "";
  selected.forEach((ticker) => {
    const row = document.createElement("div");
    row.className = "weight-row";
    const label = document.createElement("span");
    label.textContent = ticker;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = "100";
    input.step = "1";
    input.value = state.weights[ticker] ?? defaultWeight(ticker);
    input.addEventListener("input", () => { state.weights[ticker] = Number(input.value); updateWeightTotal(); });
    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);
  });
  state.selected = selected;
  updateWeightTotal();
}

function updateWeightTotal() {
  const total = selectedTickers().reduce((sum, ticker) => sum + (Number(state.weights[ticker]) || 0), 0);
  const el = $("weight-total");
  el.textContent = `Total weight: ${total.toFixed(1)}%. The simulator normalizes this to 100%.`;
  el.style.color = total <= 0 ? "var(--danger)" : "var(--muted)";
}

function gatherWeights() {
  const weights = {};
  selectedTickers().forEach((ticker) => { weights[ticker] = Number(state.weights[ticker]) || 0; });
  return weights;
}

/* ---------- Results rendering ---------- */

function renderTables(preview) {
  const render = (elementId, data) => {
    const el = $(elementId);
    if (!data || !data.rows.length) { el.innerHTML = "<p class='status'>No data.</p>"; return; }
    el.innerHTML = "<table><thead><tr>" + data.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead><tbody>" +
      data.rows.map((row) => "<tr>" + row.map((value) => `<td>${escapeHtml(value === null || value === undefined ? "" : fmtNumber(value))}</td>`).join("") + "</tr>").join("") +
      "</tbody></table>";
  };
  render("macro-preview", preview.macro);
  render("returns-preview", preview.returns);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function fmtNumber(value) {
  return typeof value === "number" ? (Math.abs(value) >= 1000 ? value.toFixed(0) : value.toFixed(4)) : value;
}

function regimeNameToState(name) {
  return REGIME_ORDER.find((state) => REGIME_NAMES[state] === name) || null;
}

function renderResults(data) {
  $("intro").classList.add("hidden");
  $("results-content").classList.remove("hidden");

  const grid = $("metric-grid");
  grid.innerHTML = "";
  METRIC_FIELDS.forEach(([key, label]) => {
    const card = document.createElement("div");
    card.className = "metric";
    card.innerHTML = `<div class="label">${label}</div><div class="value">${fmt(data.summary[key])}</div>`;
    grid.appendChild(card);
  });
  $("risk-caption").textContent =
    `Probability of loss: ${pct(data.summary.probability_of_loss)} | ` +
    `VaR (95%): ${fmt(data.summary.var_95)} | ` +
    `Expected shortfall (95%): ${fmt(data.summary.expected_shortfall_95)} | ` +
    `Worst max drawdown: ${pct(data.summary.max_drawdown_worst)}`;
  $("performance-caption").textContent =
    `Annualized return: ${pct(data.summary.annualized_return)} | ` +
    `Annualized volatility: ${pct(data.summary.annualized_volatility)} | ` +
    `Sharpe ratio (0% risk-free): ${fmt(data.summary.sharpe_ratio, 2)}`;

  lineChart($("chart-wealth"), data.wealth.periods, [
    { name: "P05", color: "#c2410c", values: data.wealth.p05 },
    { name: "Median", color: "#2563eb", values: data.wealth.median },
    { name: "P95", color: "#2f855a", values: data.wealth.p95 },
  ]);
  histChart($("chart-terminal"), data.terminal);
  barChart($("chart-regime-mix"), data.regime_mix.map((item) => ({
    label: item.label,
    value: item.share,
    color: REGIME_COLORS[regimeNameToState(item.label)] || "#2563eb",
  })), { digits: 3 });
  scatterChart(
    $("chart-macro"),
    data.macro_scatter.map((p) => ({ x: p.growth, y: p.inflation, color: REGIME_COLORS[regimeNameToState(p.regime)] || "#94a3b8" })),
    "Growth",
    "Inflation"
  );
  heatmap($("chart-transition"), data.transition.labels, data.transition.values, [0, 1], "From / To");
  barChart($("chart-observations"), Object.entries(data.observations).map(([label, value]) => ({
    label,
    value,
    color: REGIME_COLORS[regimeNameToState(label)] || "#2563eb",
  })), { digits: 0 });

  const regimeSelect = $("correlation-regime");
  regimeSelect.innerHTML = "";
  Object.keys(data.correlations).forEach((label) => {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label;
    regimeSelect.appendChild(option);
  });
  const drawCorrelation = () => {
    const label = regimeSelect.value;
    const corr = data.correlations[label];
    heatmap($("chart-correlation"), corr.labels, corr.values, [-1, 1], label);
  };
  regimeSelect.addEventListener("change", drawCorrelation);
  drawCorrelation();

  const diagnostics = data.diagnostics;
  $("diagnostics-table").innerHTML = "<table><thead><tr>" + diagnostics.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead><tbody>" +
    diagnostics.rows.map((row) => "<tr>" + row.map((value) => `<td>${escapeHtml(fmtNumber(value))}</td>`).join("") + "</tr>").join("") + "</tbody></table>";

  $("warnings-box").textContent = data.warnings.length ? data.warnings.join("\n") : "";
  state.diagnostics = diagnostics;
}

/* ---------- Handlers ---------- */

async function onLoad() {
  const button = $("load-btn");
  const message = $("load-message");
  button.disabled = true;
  try {
    const payload = await gatherLoadPayload();
    await fillCsvPayload(payload);
    state.loadPayload = payload;
    const data = await postJSON("/api/load", payload);
    state.loadResult = data;
    setStatus(message, data.message);
    renderTickerChecklist(data.tickers, data.default_tickers);
    state.weights = {};
    renderWeightEditor();
    renderTables(data);
    $("portfolio-status").textContent = `${data.tickers.length} tickers available. Select tickers and set weights.`;
    $("run-btn").disabled = false;
    $("compare-btn").disabled = false;
  } catch (error) {
    setStatus(message, error.message, true);
  } finally {
    button.disabled = false;
  }
}

function gatherSimPayload() {
  return {
    ...state.loadPayload,
    ...gatherScenario(),
    selected_tickers: selectedTickers(),
    weights: gatherWeights(),
  };
}

async function onRun() {
  const message = $("run-message");
  const button = $("run-btn");
  button.disabled = true;
  try {
    const payload = gatherSimPayload();
    const data = await postJSON("/api/simulate", payload);
    state.results = data;
    setStatus(message, data.message);
    renderResults(data);
  } catch (error) {
    setStatus(message, error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function onCompare() {
  const message = $("run-message");
  const button = $("compare-btn");
  button.disabled = true;
  try {
    const payload = gatherSimPayload();
    const data = await postJSON("/api/compare", payload);
    $("comparison-table").innerHTML = "<table><thead><tr>" + data.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead><tbody>" +
      data.rows.map((row) => "<tr>" + row.map((value) => `<td>${escapeHtml(fmtNumber(value))}</td>`).join("") + "</tr>").join("") + "</tbody></table>";
    setStatus(message, "Scenario comparison complete.");
  } catch (error) {
    setStatus(message, error.message, true);
  } finally {
    button.disabled = false;
  }
}

function onDownloadSummary() {
  if (!state.results) return;
  const summary = state.results.summary;
  const rows = Object.entries(summary).map(([key, value]) => [key, value]);
  downloadCSV("risk_summary.csv", toCSV(["metric", "value"], rows));
}

function onDownloadDiagnostics() {
  if (!state.diagnostics) return;
  downloadCSV("calibration_diagnostics.csv", toCSV(state.diagnostics.columns, state.diagnostics.rows));
}

/* ---------- Init ---------- */

function init() {
  const today = new Date().toISOString().slice(0, 10);
  $("yahoo-end").value = today;

  SYNTHETIC_OPTIONS.forEach((ticker) => {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = ticker;
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(ticker));
    $("synthetic-options").appendChild(label);
  });

  const startSelect = $("start-state");
  REGIME_ORDER.forEach((state) => {
    const option = document.createElement("option");
    option.value = REGIME_NAMES[state];
    option.textContent = REGIME_NAMES[state];
    startSelect.appendChild(option);
  });

  const transition = $("transition-uncertainty");
  transition.addEventListener("input", () => { $("transition-uncertainty-output").textContent = Number(transition.value).toFixed(2); });

  document.querySelectorAll('input[name="source"]').forEach((radio) => radio.addEventListener("change", toggleSourceGroups));
  toggleSourceGroups();

  $("load-btn").addEventListener("click", onLoad);
  $("run-btn").addEventListener("click", onRun);
  $("compare-btn").addEventListener("click", onCompare);
  $("download-summary").addEventListener("click", onDownloadSummary);
  $("download-diagnostics").addEventListener("click", onDownloadDiagnostics);

  fetch("/api/health")
    .then((response) => response.json())
    .then((data) => {
      const badge = $("connection");
      badge.textContent = "connected";
      badge.className = "badge badge-ok";
    })
    .catch(() => {
      const badge = $("connection");
      badge.textContent = "backend offline";
      badge.className = "badge badge-error";
    });
}

document.addEventListener("DOMContentLoaded", init);
