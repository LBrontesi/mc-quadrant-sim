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
  low_growth_low_inflation: "#3b82f6",
};

const DEFAULT_TICKERS = ["SPY", "IEF", "GLD", "DBC", "EFA", "VNQ", "TIP", "SHY"];
const SYNTHETIC_OPTIONS = [...DEFAULT_TICKERS, "DBMF", "KMLM", "TLT", "QQQ"];

const DEFAULT_WEIGHTS = {
  SPY: 40, IEF: 20, GLD: 10, DBC: 10, EFA: 10, VNQ: 5, TIP: 3, SHY: 2, DBMF: 5, KMLM: 5,
};

const DEFAULT_CORRELATIONS = {
  high_growth_low_inflation: -0.10,
  high_growth_high_inflation: 0.35,
  low_growth_high_inflation: 0.25,
  low_growth_low_inflation: -0.40,
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

function showOverlay(message) {
  $("overlay-text").textContent = message || "Working...";
  $("overlay").classList.remove("hidden");
}

function hideOverlay() {
  $("overlay").classList.add("hidden");
}

function notify(message, type = "info") {
  const box = $("toast-container");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  box.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 350);
  }, 4200);
}

function animateNumber(el, target, digits) {
  const start = performance.now();
  const duration = 750;
  function frame(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (target * eased).toFixed(digits);
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
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

function attachTooltip(container) {
  let tip = container.querySelector(".tooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip hidden";
    container.appendChild(tip);
  }
  const show = (x, y, html) => {
    tip.innerHTML = html;
    const maxX = container.clientWidth - 40;
    const left = Math.min(Math.max(x, 60), Math.max(maxX, 60));
    tip.style.left = left + "px";
    tip.style.top = Math.max(y, 8) + "px";
    tip.classList.remove("hidden");
  };
  const hide = () => tip.classList.add("hidden");
  return { show, hide };
}

/* ---------- SVG charts ---------- */

const SVG_NS = "http://www.w3.org/2000/svg";
const MARGIN = { top: 12, right: 12, bottom: 26, left: 46 };

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
    line.setAttribute("stroke", "#22345a"); line.setAttribute("stroke-width", "1");
    grid.appendChild(line);
    svgText(grid, x, height + 14, label, "chart-title");
  });
  yTicks.forEach(([y, label]) => {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", 0); line.setAttribute("y1", y);
    line.setAttribute("x2", width); line.setAttribute("y2", y);
    line.setAttribute("stroke", "#22345a"); line.setAttribute("stroke-width", "1");
    grid.appendChild(line);
    svgText(grid, -6, y + 4, label, "chart-title", "end");
  });
  if (xLabel) svgText(grid, width / 2, height + 22, xLabel, "chart-title");
  if (yLabel) {
    const el = svgText(grid, -32, height / 2, yLabel, "chart-title");
    el.setAttribute("transform", "rotate(-90 -32 " + height / 2 + ")");
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
  const crosshair = document.createElementNS(SVG_NS, "line");
  crosshair.setAttribute("y1", 0);
  crosshair.setAttribute("y2", plotH);
  crosshair.setAttribute("stroke", "#5b7bb0");
  crosshair.setAttribute("stroke-dasharray", "4 4");
  crosshair.setAttribute("opacity", "0.5");
  crosshair.setAttribute("x1", 0);
  crosshair.setAttribute("x2", 0);
  svg.appendChild(crosshair);
  series.forEach((s) => {
    const points = s.values.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
    const path = document.createElementNS(SVG_NS, "polyline");
    path.setAttribute("points", points);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", s.color);
    path.setAttribute("stroke-width", "2.2");
    path.setAttribute("stroke-linejoin", "round");
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
  const tooltip = attachTooltip(container);
  svg.addEventListener("mousemove", (e) => {
    const ratio = (e.offsetX - MARGIN.left) / plotW;
    const index = Math.max(0, Math.min(labels.length - 1, Math.round(ratio * (labels.length - 1))));
    crosshair.setAttribute("x1", xScale(index));
    crosshair.setAttribute("x2", xScale(index));
    crosshair.setAttribute("opacity", "1");
    const lines = [`<b>Period ${labels[index]}</b>`];
    series.forEach((s) => lines.push(`<span style="color:${s.color}">${s.name}:</span> ${fmt(s.values[index])}`));
    tooltip.show(e.offsetX, e.offsetY, lines.join("<br>"));
  });
  svg.addEventListener("mouseleave", () => { crosshair.setAttribute("opacity", "0"); tooltip.hide(); });
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
  const tooltip = attachTooltip(container);
  items.forEach((item, index) => {
    const barW = slot * 0.62;
    const x = MARGIN.left + index * slot + (slot - barW) / 2;
    const y = MARGIN.top + (1 - item.value / max) * plotH;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y);
    rect.setAttribute("width", barW); rect.setAttribute("height", Math.max(plotH - (y - MARGIN.top), 0));
    rect.setAttribute("fill", item.color || "#3b82f6");
    rect.setAttribute("rx", "3");
    rect.style.transition = "opacity 0.15s";
    rect.addEventListener("mouseenter", () => {
      rect.setAttribute("opacity", "0.8");
      tooltip.show(x + barW / 2, y, `<b>${escapeHtml(item.label)}</b><br>${item.value.toFixed(options.digits ?? 2)}`);
    });
    rect.addEventListener("mouseleave", () => { rect.setAttribute("opacity", "1"); tooltip.hide(); });
    svg.appendChild(rect);
    svgText(svg, x + barW / 2, Math.max(y - 4, 10), item.value.toFixed(options.digits ?? 2), "chart-title");
    const label = item.label;
    const labelEl = svgText(svg, x + barW / 2, height - 8, label.length > 16 ? label.slice(0, 15) + "..." : label, "chart-title");
    labelEl.setAttribute("transform", `rotate(-18 ${x + barW / 2} ${height - 8})`);
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
  const tooltip = attachTooltip(container);
  labels.forEach((rowLabel, row) => {
    svgText(svg, x0 - 8, y0 + row * size + size / 2 + 4, rowLabel.length > 16 ? rowLabel.slice(0, 15) + "..." : rowLabel, "chart-title", "end");
    labels.forEach((colLabel, col) => {
      const value = values[row][col];
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", x0 + col * size);
      rect.setAttribute("y", y0 + row * size);
      rect.setAttribute("width", size - 2);
      rect.setAttribute("height", size - 2);
      rect.setAttribute("rx", "2");
      rect.setAttribute("fill", colorFor(value));
      rect.style.cursor = "default";
      rect.addEventListener("mouseenter", () => {
        rect.setAttribute("stroke", "#fff");
        rect.setAttribute("stroke-width", "1.5");
        tooltip.show(x0 + col * size + size / 2, y0 + row * size + 8, `<b>${escapeHtml(rowLabel)} → ${escapeHtml(colLabel)}</b><br>${value.toFixed(3)}`);
      });
      rect.addEventListener("mouseleave", () => { rect.removeAttribute("stroke"); tooltip.hide(); });
      svg.appendChild(rect);
      svgText(svg, x0 + col * size + (size - 2) / 2, y0 + row * size + (size - 2) / 2 + 4, value.toFixed(2), "chart-title");
    });
    svgText(svg, x0 + row * size + (size - 2) / 2, height - 4, labels[row].length > 16 ? labels[row].slice(0, 15) + "..." : labels[row], "chart-title");
  });
  if (title) svgText(svg, width / 2, 6, title, "chart-title");
  container.appendChild(svg);
}

function scatterChart(container, points, xLabel, yLabel, legend = null) {
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
  const tooltip = attachTooltip(container);
  points.forEach((p) => {
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", xScale(p.x));
    circle.setAttribute("cy", yScale(p.y));
    circle.setAttribute("r", 3.4);
    circle.setAttribute("fill", p.color);
    circle.setAttribute("opacity", "0.75");
    circle.addEventListener("mouseenter", () => {
      circle.setAttribute("r", "5");
      circle.setAttribute("stroke", "#fff");
      circle.setAttribute("stroke-width", "1");
      const lines = p.label ? [`<b>${escapeHtml(p.label)}</b>`] : [];
      lines.push(`${xLabel}: ${fmt(p.x, 2)}`);
      lines.push(`${yLabel}: ${fmt(p.y, 2)}`);
      if (p.regime) lines.push(`<span style="color:${p.color}">${escapeHtml(p.regime)}</span>`);
      tooltip.show(xScale(p.x), yScale(p.y), lines.join("<br>"));
    });
    circle.addEventListener("mouseleave", () => {
      circle.setAttribute("r", "3.4");
      circle.removeAttribute("stroke");
      tooltip.hide();
    });
    svg.appendChild(circle);
  });
  container.appendChild(svg);
  if (legend) {
    const box = document.createElement("div");
    box.className = "legend";
    legend.forEach((entry) => {
      const item = document.createElement("span");
      item.innerHTML = `<span class="legend-dot" style="background:${entry.color}"></span>${entry.label}`;
      box.appendChild(item);
    });
    container.appendChild(box);
  }
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
  const tooltip = attachTooltip(container);
  counts.forEach((count, index) => {
    const barW = slot * 0.9;
    const x = MARGIN.left + index * slot + (slot - barW) / 2;
    const y = MARGIN.top + (1 - count / peak) * plotH;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y);
    rect.setAttribute("width", barW); rect.setAttribute("height", plotH - (y - MARGIN.top));
    rect.setAttribute("fill", "#3b82f6");
    rect.setAttribute("opacity", "0.82");
    const low = min + (index / bins) * span;
    const high = min + ((index + 1) / bins) * span;
    rect.addEventListener("mouseenter", () => {
      rect.setAttribute("opacity", "1");
      tooltip.show(x + barW / 2, y, `<b>${fmt(low)} – ${fmt(high)}</b><br>${count} paths`);
    });
    rect.addEventListener("mouseleave", () => { rect.setAttribute("opacity", "0.82"); tooltip.hide(); });
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
    risk_free_rate: Number($("risk-free").value) / 100,
    annual_inflation: Number($("annual-inflation").value) / 100,
    base_currency: $("base-currency").value,
    currency_map: $("currency-map").value,
    use_correlation_override: $("use-corr-override").checked,
    correlation_blend: Number($("corr-blend").value),
    correlation_override_targets: gatherCorrelationTargets(),
  };
}

function gatherCorrelationTargets() {
  const targets = {};
  document.querySelectorAll("#corr-sliders input[type='range']").forEach((slider) => {
    targets[slider.dataset.state] = Number(slider.value);
  });
  return targets;
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
  const selected = selectedTickers();
  const total = selected.reduce((sum, ticker) => sum + (Number(state.weights[ticker]) || 0), 0);
  const el = $("weight-total");
  el.textContent = `Total weight: ${total.toFixed(1)}%. The simulator normalizes this to 100%.`;
  el.style.color = total <= 0 ? "var(--danger)" : "var(--muted)";
  $("run-btn").disabled = !state.loadResult || selected.length === 0 || total <= 0;
  $("compare-btn").disabled = !state.loadResult || selected.length === 0 || total <= 0;
}

function gatherWeights() {
  const weights = {};
  selectedTickers().forEach((ticker) => { weights[ticker] = Number(state.weights[ticker]) || 0; });
  return weights;
}

function applyPreset() {
  const name = $("preset-select").value;
  if (!name || !state.loadResult) return;
  const preset = state.loadResult.presets.find((p) => p.name === name);
  if (!preset) return;
  const selected = selectedTickers();
  const matched = {};
  let total = 0;
  selected.forEach((ticker) => {
    const base = ticker.endsWith("_SIM") ? ticker.slice(0, -4) : ticker.endsWith("SIM") ? ticker.slice(0, -3) : ticker;
    if (preset.weights[base] !== undefined) {
      matched[ticker] = preset.weights[base];
      total += preset.weights[base];
    }
  });
  if (total <= 0) {
    notify("Preset assets are not selected. Select matching tickers first.", "error");
    return;
  }
  const factor = 100 / total;
  selected.forEach((ticker) => { state.weights[ticker] = matched[ticker] !== undefined ? matched[ticker] * factor : 0; });
  renderWeightEditor();
  notify(`Applied ${name}`, "success");
}

function populatePresets(presets) {
  const select = $("preset-select");
  const current = select.value;
  select.innerHTML = '<option value="">— choose —</option>';
  (presets || []).forEach((preset) => {
    const option = document.createElement("option");
    option.value = preset.name;
    option.textContent = preset.name;
    select.appendChild(option);
  });
  select.value = current && presets.some((p) => p.name === current) ? current : "";
}

/* ---------- Tabs ---------- */

function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
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
    const valueEl = document.createElement("div");
    valueEl.className = "value";
    card.appendChild(Object.assign(document.createElement("div"), { className: "label", textContent: label }));
    card.appendChild(valueEl);
    grid.appendChild(card);
    animateNumber(valueEl, data.summary[key], 2);
  });
  $("risk-caption").textContent =
    `${data.terms === "real" ? "Real (inflation-adjusted) | " : "Nominal | "}` +
    `Currency: ${data.currency} | ` +
    `Probability of loss: ${pct(data.summary.probability_of_loss)} | ` +
    `VaR (95%): ${fmt(data.summary.var_95)} | ` +
    `Expected shortfall (95%): ${fmt(data.summary.expected_shortfall_95)} | ` +
    `Worst max drawdown: ${pct(data.summary.max_drawdown_worst)}`;
  $("performance-caption").textContent =
    `Annualized return: ${pct(data.summary.annualized_return)} | ` +
    `Annualized volatility: ${pct(data.summary.annualized_volatility)} | ` +
    `Sharpe ratio (0% risk-free): ${fmt(data.summary.sharpe_ratio, 2)}`;

  lineChart($("chart-wealth"), data.wealth.periods, [
    { name: "P05", color: "#f97316", values: data.wealth.p05 },
    { name: "Median", color: "#3b82f6", values: data.wealth.median },
    { name: "P95", color: "#10b981", values: data.wealth.p95 },
  ]);
  histChart($("chart-terminal"), data.terminal);
  barChart($("chart-regime-mix"), data.regime_mix.map((item) => ({
    label: item.label,
    value: item.share,
    color: REGIME_COLORS[regimeNameToState(item.label)] || "#3b82f6",
  })), { digits: 3 });
  scatterChart(
    $("chart-macro"),
    data.macro_scatter.map((p) => ({
      x: p.growth,
      y: p.inflation,
      color: REGIME_COLORS[regimeNameToState(p.regime)] || "#94a3b8",
      label: p.date,
      regime: p.regime,
    })),
    "Growth",
    "Inflation",
    REGIME_ORDER.map((state) => ({ label: REGIME_NAMES[state], color: REGIME_COLORS[state] }))
  );
  heatmap($("chart-transition"), data.transition.labels, data.transition.values, [0, 1], "From / To");
  barChart($("chart-observations"), Object.entries(data.observations).map(([label, value]) => ({
    label,
    value,
    color: REGIME_COLORS[regimeNameToState(label)] || "#3b82f6",
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
  $("diagnostics-empty").classList.add("hidden");
  $("diagnostics-content").classList.remove("hidden");
}

/* ---------- Handlers ---------- */

async function onLoad() {
  const button = $("load-btn");
  const message = $("load-message");
  button.disabled = true;
  showOverlay("Loading data...");
  try {
    const payload = await gatherLoadPayload();
    await fillCsvPayload(payload);
    state.loadPayload = payload;
    const data = await postJSON("/api/load", payload);
    state.loadResult = data;
    setStatus(message, data.message);
    notify(data.message, "success");
    renderTickerChecklist(data.tickers, data.default_tickers);
    state.weights = {};
    renderWeightEditor();
    renderTables(data);
    populatePresets(data.presets);
    $("preset-apply").disabled = false;
    $("portfolio-status").textContent = `${data.tickers.length} tickers available. Select tickers and set weights.`;
    $("data-empty").classList.add("hidden");
    $("data-content").classList.remove("hidden");
    switchTab("tab-data");
  } catch (error) {
    setStatus(message, error.message, true);
    notify(error.message, "error");
  } finally {
    button.disabled = false;
    hideOverlay();
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
  showOverlay("Running simulation...");
  try {
    const payload = gatherSimPayload();
    const data = await postJSON("/api/simulate", payload);
    state.results = data;
    setStatus(message, data.message);
    notify("Simulation complete", "success");
    renderResults(data);
    switchTab("tab-results");
  } catch (error) {
    setStatus(message, error.message, true);
    notify(error.message, "error");
  } finally {
    button.disabled = false;
    hideOverlay();
  }
}

async function onCompare() {
  const message = $("run-message");
  const button = $("compare-btn");
  button.disabled = true;
  showOverlay("Comparing distributions...");
  try {
    const payload = gatherSimPayload();
    const data = await postJSON("/api/compare", payload);
    $("comparison-table").innerHTML = "<table><thead><tr>" + data.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead><tbody>" +
      data.rows.map((row) => "<tr>" + row.map((value) => `<td>${escapeHtml(fmtNumber(value))}</td>`).join("") + "</tr>").join("") + "</tbody></table>";
    setStatus(message, "Scenario comparison complete.");
    notify("Scenario comparison complete", "success");
    switchTab("tab-diagnostics");
  } catch (error) {
    setStatus(message, error.message, true);
    notify(error.message, "error");
  } finally {
    button.disabled = false;
    hideOverlay();
  }
}

function onDownloadSummary() {
  if (!state.results) return;
  const summary = state.results.summary;
  const rows = Object.entries(summary).map(([key, value]) => [key, value]);
  downloadCSV("risk_summary.csv", toCSV(["metric", "value"], rows));
  notify("Risk summary downloaded", "success");
}

function onDownloadDiagnostics() {
  if (!state.diagnostics) return;
  downloadCSV("calibration_diagnostics.csv", toCSV(state.diagnostics.columns, state.diagnostics.rows));
  notify("Diagnostics downloaded", "success");
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

  const sliderBox = $("corr-sliders");
  REGIME_ORDER.forEach((state) => {
    const label = document.createElement("label");
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "-1";
    slider.max = "1";
    slider.step = "0.05";
    slider.value = DEFAULT_CORRELATIONS[state];
    slider.dataset.state = state;
    const output = document.createElement("output");
    const update = () => { output.textContent = Number(slider.value).toFixed(2); };
    slider.addEventListener("input", update);
    update();
    label.appendChild(document.createTextNode(REGIME_NAMES[state]));
    label.appendChild(slider);
    label.appendChild(output);
    sliderBox.appendChild(label);
  });

  document.querySelectorAll('input[name="source"]').forEach((radio) => radio.addEventListener("change", toggleSourceGroups));
  toggleSourceGroups();

  document.querySelectorAll(".tab-btn").forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

  $("load-btn").addEventListener("click", onLoad);
  $("run-btn").addEventListener("click", onRun);
  $("compare-btn").addEventListener("click", onCompare);
  $("download-summary").addEventListener("click", onDownloadSummary);
  $("download-diagnostics").addEventListener("click", onDownloadDiagnostics);
  $("preset-apply").addEventListener("click", applyPreset);

  fetch("/api/health")
    .then((response) => response.json())
    .then(() => {
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
