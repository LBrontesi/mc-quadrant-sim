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
  ["mean", "Mean terminal wealth"], ["p05", "P05"], ["p50", "Median"], ["p95", "P95"],
  ["annualized_return", "Annualized return (wealth)"],
  ["annualized_volatility", "Annualized volatility (wealth)"],
  ["sharpe_ratio", "Sharpe ratio"], ["sortino_ratio", "Sortino"], ["calmar_ratio", "Calmar"],
  ["geometric_annualized_return", "CAGR"], ["probability_of_loss", "Probability of loss"],
  ["var_95", "VaR (95%)"], ["expected_shortfall_95", "Expected shortfall (95%)"],
  ["max_drawdown_worst", "Worst max drawdown"],
];

const FLOW_METRIC_FIELDS = [
  ["cash_flow_adjusted_annualized_return", "Time-weighted return"],
  ["cash_flow_adjusted_volatility", "Time-weighted volatility"],
  ["cash_flow_adjusted_sharpe_ratio", "Time-weighted Sharpe"],
  ["total_contributed", "Total contributions"],
  ["total_withdrawn", "Total withdrawals"],
];
const COST_METRIC_FIELDS = [
  ["leverage_multiple", "Leverage"], ["weighted_expense_ratio", "Weighted ETF fee"],
  ["annual_fee_drag", "Annual fee drag"], ["annual_financing_cost", "Annual financing cost"],
  ["margin_calls", "Margin calls"],
];

const PERCENT_METRICS = new Set([
  "annualized_return", "annualized_volatility", "cash_flow_adjusted_annualized_return",
  "cash_flow_adjusted_volatility", "geometric_annualized_return", "probability_of_loss",
  "max_drawdown_mean", "max_drawdown_p95", "max_drawdown_worst", "ulcer_index_mean", "ulcer_index_p95",
]);
const CURRENCY_METRICS = new Set([
  "mean", "std", "p05", "p50", "p95", "var_95", "expected_shortfall_95", "periodic_contribution",
  "periodic_withdrawal", "total_contributed", "total_withdrawn", "net_external_cash_flow",
]);

const state = {
  loadPayload: null,
  loadResult: null,
  selected: [],
  weights: {},
  results: null,
  diagnostics: null,
  lastSimPayload: null,
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

function formatMetricValue(key, value, currency = "USD") {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
  const numeric = Number(value);
  if (key === "leverage_multiple") return `${numeric.toFixed(1)}x`;
  if (key === "margin_calls") return Math.round(numeric).toLocaleString();
  if (PERCENT_METRICS.has(key)) return `${(numeric * 100).toFixed(2)}%`;
  if (CURRENCY_METRICS.has(key)) return `${currency} ${numeric.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  return numeric.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function animateMetric(el, target, key, currency) {
  if (!Number.isFinite(Number(target))) {
    el.textContent = "-";
    return;
  }
  const start = performance.now();
  const duration = 750;
  function frame(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = formatMetricValue(key, target * eased, currency);
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function defaultWeight(ticker) {
  const base = ticker.endsWith("_SIM") ? ticker.slice(0, -4) : ticker.endsWith("SIM") ? ticker.slice(0, -3) : ticker;
  return DEFAULT_WEIGHTS[base] ?? 0;
}

function downloadCSV(filename, text) {
  downloadFile(filename, text, "text/csv;charset=utf-8");
}

function downloadFile(filename, text, mime) {
  const blob = new Blob([text], { type: mime });
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
    line.setAttribute("stroke", "var(--grid)"); line.setAttribute("stroke-width", "1");
    grid.appendChild(line);
    svgText(grid, x, height + 14, label, "chart-title");
  });
  yTicks.forEach(([y, label]) => {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", 0); line.setAttribute("y1", y);
    line.setAttribute("x2", width); line.setAttribute("y2", y);
    line.setAttribute("stroke", "var(--grid)"); line.setAttribute("stroke-width", "1");
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
  crosshair.setAttribute("stroke", "var(--crosshair)");
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
  const pointerPosition = (e) => {
    const rect = svg.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (width / rect.width),
      y: (e.clientY - rect.top) * (height / rect.height),
      px: e.clientX - rect.left,
      py: e.clientY - rect.top,
    };
  };
  svg.addEventListener("mousemove", (e) => {
    const { x, px, py } = pointerPosition(e);
    const ratio = (x - MARGIN.left) / plotW;
    const index = Math.max(0, Math.min(labels.length - 1, Math.round(ratio * (labels.length - 1))));
    crosshair.setAttribute("x1", xScale(index));
    crosshair.setAttribute("x2", xScale(index));
    crosshair.setAttribute("opacity", "1");
    const lines = [`<b>Period ${labels[index]}</b>`];
    series.forEach((s) => lines.push(`<span style="color:${s.color}">${s.name}:</span> ${fmt(s.values[index])}`));
    tooltip.show(px, py, lines.join("<br>"));
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

function histChart(container, values, bins = 45, label = "Terminal wealth", color = "#3b82f6") {
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
  drawAxes(svg, plotW, plotH, [], yTicks, label, "Paths");
  const slot = plotW / bins;
  const tooltip = attachTooltip(container);
  counts.forEach((count, index) => {
    const barW = slot * 0.9;
    const x = MARGIN.left + index * slot + (slot - barW) / 2;
    const y = MARGIN.top + (1 - count / peak) * plotH;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x); rect.setAttribute("y", y);
    rect.setAttribute("width", barW); rect.setAttribute("height", plotH - (y - MARGIN.top));
    rect.setAttribute("fill", color);
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

function timelineChart(container, states) {
  container.innerHTML = "";
  const width = 560, height = 66;
  const svg = createSvg(width, height);
  const slot = width / states.length;
  const tooltip = attachTooltip(container);
  const uniqueStates = [...new Set(states)];
  states.forEach((state, index) => {
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", index * slot);
    rect.setAttribute("y", 6);
    rect.setAttribute("width", Math.max(slot - 1, 1));
    rect.setAttribute("height", 34);
    rect.setAttribute("fill", colorForState(state, uniqueStates.indexOf(state)));
    rect.addEventListener("mouseenter", () => {
      rect.setAttribute("opacity", "0.85");
      tooltip.show(index * slot + slot / 2, 30, `Period ${index + 1}<br>${escapeHtml(labelForState(state))}`);
    });
    rect.addEventListener("mouseleave", () => { rect.setAttribute("opacity", "1"); tooltip.hide(); });
    svg.appendChild(rect);
  });
  container.appendChild(svg);
  const legend = document.createElement("div");
  legend.className = "legend";
  uniqueStates.forEach((state, index) => {
    const item = document.createElement("span");
    item.innerHTML = `<span class="legend-dot" style="background:${colorForState(state, index)}"></span>${escapeHtml(labelForState(state))}`;
    legend.appendChild(item);
  });
  container.appendChild(legend);
}

function donutChart(container, items) {
  container.innerHTML = "";
  if (!items.length) return;
  const size = 110, cx = size / 2, cy = size / 2, r = 42, stroke = 15;
  const total = items.reduce((sum, item) => sum + item.value, 0) || 1;
  const svg = createSvg(size, size);
  const palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#14b8a6", "#a78bfa", "#22c55e"];
  const tooltip = attachTooltip(container);
  let angle = -Math.PI / 2;
  items.forEach((item, index) => {
    const frac = item.value / total;
    const start = angle;
    const end = angle + frac * 2 * Math.PI;
    angle = end;
    const large = frac > 0.5 ? 1 : 0;
    const x1 = cx + r * Math.cos(start);
    const y1 = cy + r * Math.sin(start);
    const x2 = cx + r * Math.cos(end);
    const y2 = cy + r * Math.sin(end);
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", palette[index % palette.length]);
    path.setAttribute("stroke-width", stroke);
    path.addEventListener("mouseenter", () => {
      tooltip.show(cx, cy, `<b>${escapeHtml(item.label)}</b><br>${item.value.toFixed(1)}%`);
    });
    path.addEventListener("mouseleave", () => tooltip.hide());
    svg.appendChild(path);
  });
  svgText(svg, cx, cy + 4, total.toFixed(0) + "%", "chart-title");
  container.appendChild(svg);
}

/* ---------- Inputs ---------- */

function activeSource() {
  return $("csv-enabled").checked ? "csv" : "yahoo";
}

function toggleSourceGroups() {
  if (state.loadResult) {
    state.loadResult = null;
    state.results = null;
    state.lastSimPayload = null;
    state.diagnostics = null;
    $("ticker-list").innerHTML = "";
    renderWeightEditor();
    $("preset-apply").disabled = true;
    $("portfolio-status").textContent = "Data source changed — click Load Data to refresh tickers.";
    $("results-content").classList.add("hidden");
    $("intro").classList.remove("hidden");
    $("data-content").classList.add("hidden");
    $("data-empty").classList.remove("hidden");
  }
  updateRunAvailability();
}

function thresholdPayload(selectId, fixedId) {
  const mode = $(selectId).value;
  return mode === "fixed" ? `fixed:${$(fixedId).value}` : mode;
}

function gatherLoadPayload() {
  const source = activeSource();
  const payload = { source };
  if (source === "yahoo") {
    payload.tickers = $("yahoo-tickers").value;
    payload.start = $("yahoo-start").value;
    payload.end = $("yahoo-end").value;
    payload.proxies = $("yahoo-proxies").value;
    payload.synthetic = Array.from(document.querySelectorAll('#synthetic-options input[type="checkbox"]:checked')).map((el) => el.value);
    payload.synthetic_seed = Number($("synthetic-seed").value);
    payload.synthetic_method = $("synthetic-method").value;
    payload.synthetic_categories = $("synthetic-categories").value;
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
    cost_bps: $("rebalance").value === "legacy" ? 0 : Number($("cost-bps").value),
    contribution: Number($("contribution").value),
    withdrawal: Number($("withdrawal").value),
    expense_ratios: $("expense-ratios").value,
    leverage_multiple: Number($("leverage-multiple").value),
    financing_rate: Number($("financing-rate").value),
    maintenance_margin: Number($("maintenance-margin").value),
    risk_free_rate: Number($("risk-free").value),
    annual_inflation: Number($("annual-inflation").value),
    base_currency: $("base-currency").value,
    currency_map: $("currency-map").value,
    use_correlation_override: $("use-corr-override").checked,
    correlation_blend: Number($("corr-blend").value),
    correlation_override_targets: gatherCorrelationTargets(),
    model: $("model-kind").value,
    hmm_states: Number($("hmm-states").value),
    threshold_window: Number($("threshold-window").value),
    duration_model: $("duration-model").value,
    garch: $("garch").checked,
    walk_forward: $("walk-forward").checked,
  };
}

function validateScenario() {
  const errors = [];
  if ($("garch").checked && $("distribution").value !== "normal") {
    errors.push("GARCH volatility clustering requires the Normal return distribution.");
  }
  if ($("base-currency").value.trim().length !== 3) {
    errors.push("Portfolio currency must be a three-letter ISO code.");
  }
  const leverage = Number($("leverage-multiple").value);
  const margin = Number($("maintenance-margin").value) / 100;
  if (leverage > 1 && $("rebalance").value === "legacy") {
    errors.push("Leverage requires monthly, quarterly, or annual rebalancing.");
  }
  if (leverage === 1 && margin > 0) {
    errors.push("Maintenance margin only applies when leverage is greater than 1.0x.");
  }
  if (leverage > 1 && margin >= 1 / leverage) {
    errors.push("Maintenance margin must be below the initial equity margin for the selected leverage.");
  }
  return errors;
}

function updateMethodologyControls() {
  const isHMM = $("model-kind").value === "hmm";
  const distribution = $("distribution").value;
  const legacy = $("rebalance").value === "legacy";
  $("quadrant-calibration").classList.toggle("hidden", isHMM);
  $("hmm-states-group").classList.toggle("hidden", !isHMM);
  $("threshold-window-group").classList.toggle("hidden", isHMM);
  $("walk-forward-group").classList.toggle("hidden", isHMM);
  $("correlation-override-controls").classList.toggle("hidden", isHMM);
  $("corr-blend-group").classList.toggle("hidden", isHMM);
  $("start-state-group").classList.toggle("hidden", isHMM);
  $("student-t-group").classList.toggle("hidden", distribution !== "student_t");
  $("block-size-group").classList.toggle("hidden", distribution !== "block_bootstrap");
  $("cost-bps").disabled = legacy;
  $("cost-bps-group").classList.toggle("methodology-muted", legacy);
  $("garch").disabled = distribution !== "normal";
  $("garch-hint").textContent = distribution === "normal"
    ? "GARCH requires the Normal return distribution."
    : "GARCH is disabled because the selected return distribution is not Normal.";
  updateRunAvailability();
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
    const weight = state.weights[ticker] ?? defaultWeight(ticker);
    state.weights[ticker] = weight;
    input.value = weight;
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
  donutChart($("donut"), selected.map((ticker) => ({ label: ticker, value: Number(state.weights[ticker]) || 0 })));
  updateRunAvailability();
}

function updateRunAvailability() {
  const selected = selectedTickers();
  const total = selected.reduce((sum, ticker) => sum + (Number(state.weights[ticker]) || 0), 0);
  const errors = validateScenario();
  const status = $("scenario-status");
  if (errors.length) {
    status.textContent = errors.join(" ");
    status.style.color = "var(--danger)";
  } else {
    status.textContent = "";
  }
  const disabled = !state.loadResult || selected.length === 0 || total <= 0 || errors.length > 0;
  $("run-btn").disabled = disabled;
  $("compare-btn").disabled = disabled;
  if (state.results && state.lastSimPayload) {
    const stale = JSON.stringify(gatherSimPayload()) !== JSON.stringify(state.lastSimPayload);
    $("stale-results").classList.toggle("hidden", !stale);
  } else {
    $("stale-results").classList.add("hidden");
  }
  updateGuide();
}

function updateGuide() {
  const guideStatus = $("guide-status");
  if (!guideStatus) return;
  const selected = selectedTickers();
  const total = selected.reduce((sum, ticker) => sum + (Number(state.weights[ticker]) || 0), 0);
  const loaded = Boolean(state.loadResult);
  const portfolioReady = loaded && selected.length > 0 && total > 0;
  const methodologyReady = portfolioReady && validateScenario().length === 0;
  const completed = [loaded, portfolioReady, methodologyReady, Boolean(state.results), Boolean(state.results)];
  const activeIndex = completed.findIndex((step) => !step);
  ["data", "portfolio", "methodology", "run", "read"].forEach((name, index) => {
    const step = $(`guide-${name}`);
    step.classList.toggle("complete", completed[index]);
    step.classList.toggle("active", index === activeIndex);
  });
  if (!loaded) guideStatus.textContent = "Next: load the market data.";
  else if (!portfolioReady) guideStatus.textContent = "Next: select at least one ticker and set a positive weight.";
  else if (!methodologyReady) guideStatus.textContent = "Next: resolve the highlighted methodology setting.";
  else if (!state.results) guideStatus.textContent = "Next: click Run Simulation.";
  else if (state.lastSimPayload && JSON.stringify(gatherSimPayload()) !== JSON.stringify(state.lastSimPayload)) guideStatus.textContent = "Inputs changed: run the simulation again to refresh the results.";
  else guideStatus.textContent = "Scenario ready: inspect Results, Diagnostics, or Data.";
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
  renderTable("macro-preview", preview.macro);
  renderTable("returns-preview", preview.returns);
}

function renderTable(elementId, data) {
  const el = $(elementId);
  if (!data || !data.rows.length) { el.innerHTML = "<p class='status'>No data.</p>"; return; }
  el.innerHTML = "<table><thead><tr>" + data.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead><tbody>" +
    data.rows.map((row) => "<tr>" + row.map((value) => `<td>${escapeHtml(value === null || value === undefined ? "" : fmtNumber(value))}</td>`).join("") + "</tr>").join("") +
    "</tbody></table>";
}

function renderCoverage(coverage) {
  const rows = Object.entries(coverage || {}).map(([ticker, range]) => [ticker, range.first, range.last]);
  $("coverage-table").innerHTML = rows.length
    ? "<table><thead><tr><th>Ticker</th><th>First</th><th>Last</th></tr></thead><tbody>" +
      rows.map((row) => `<tr><td>${escapeHtml(row[0])}</td><td>${row[1]}</td><td>${row[2]}</td></tr>`).join("") +
      "</tbody></table>"
    : "<p class='status'>No coverage data.</p>";
}

const GRADE_LABELS = { A: "High confidence", B: "Moderate confidence", C: "Proxy / weak", X: "Not feasible" };
const GRADE_COLORS = { A: "#10b981", B: "#f59e0b", C: "#f97316", X: "#ef4444" };

function renderSyntheticReport(report) {
  const container = $("synthetic-report");
  const entries = Object.entries(report || {});
  $("synthetic-card").classList.toggle("hidden", entries.length === 0);
  if (!entries.length) {
    container.innerHTML = "<p class='status'>No synthetic backfill assets selected.</p>";
    return;
  }
  const rows = entries.map(([asset, info]) => {
    const counts = Object.entries(info.observations_by_regime || {})
      .map(([state, count]) => `${labelForState(state)}: ${count}`)
      .join(" · ");
    return "<tr>" +
      `<td><strong>${escapeHtml(asset)}</strong></td>` +
      `<td><span class="grade-badge" style="color:${GRADE_COLORS[info.grade] || "var(--muted)"};border-color:${GRADE_COLORS[info.grade] || "var(--border)"}">${escapeHtml(info.grade)} · ${escapeHtml(GRADE_LABELS[info.grade] || info.grade)}</span></td>` +
      `<td>${escapeHtml(info.category || "-")}</td>` +
      `<td>${Number(info.history_months || 0)}</td>` +
      `<td>${info.factor_r2 === null || info.factor_r2 === undefined ? "-" : fmt(info.factor_r2, 2)}</td>` +
      `<td>${escapeHtml(counts || "-")}</td>` +
      `<td><span class="hint">${escapeHtml((info.warnings || []).join(" "))}</span></td>` +
      "</tr>";
  });
  container.innerHTML = "<table><thead><tr>" +
    "<th>Asset</th><th>Feasibility</th><th>Category</th><th>Observed months</th><th>Factor R²</th><th>Observations by regime</th><th>Warnings</th>" +
    "</tr></thead><tbody>" + rows.join("") + "</tbody></table>";
}

const DISTRIBUTION_LABELS = {
  normal: "Normal",
  student_t: "Student-t",
  bootstrap: "Historical bootstrap",
  block_bootstrap: "Block bootstrap",
};

function renderScenarioChips(payload, data) {
  const el = $("scenario-chips");
  if (!payload) { el.innerHTML = ""; return; }
  const chips = [];
  Object.entries(payload.weights || {}).forEach(([ticker, weight]) => {
    if (Number(weight) > 0) chips.push(`${ticker} ${Number(weight).toFixed(0)}%`);
  });
  chips.push(`${payload.periods} periods x ${payload.paths} paths`);
  if (Number(payload.contribution) > 0) chips.push(`+${fmt(payload.contribution, 0)}/period`);
  if (Number(payload.withdrawal) > 0) chips.push(`−${fmt(payload.withdrawal, 0)}/period`);
  chips.push(data.terms === "real" ? "Real terms" : "Nominal");
  chips.push(data.currency);
  chips.push(DISTRIBUTION_LABELS[payload.distribution] || payload.distribution);
  chips.push(payload.model === "hmm" ? `HMM · ${payload.hmm_states} states` : "Quadrant model");
  chips.push(payload.duration_model === "semi_markov" ? "Semi-Markov durations" : "Markov durations");
  if (Number(payload.leverage_multiple || 1) > 1) chips.push(`${Number(payload.leverage_multiple).toFixed(1)}x leverage`);
  el.innerHTML = chips.map((text) => `<span class="chip">${escapeHtml(text)}</span>`).join("");
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function fmtNumber(value) {
  return typeof value === "number" ? (Math.abs(value) >= 1000 ? value.toFixed(0) : value.toFixed(4)) : value;
}

const STATE_PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"];

function labelForState(state) {
  if (REGIME_NAMES[state]) return REGIME_NAMES[state];
  if (String(state).startsWith("state_")) return `Regime ${String(state).slice(6)}`;
  return String(state);
}

function regimeNameToState(name) {
  return REGIME_ORDER.find((state) => REGIME_NAMES[state] === name) || null;
}

function colorForState(state, index = 0) {
  return REGIME_COLORS[state] || STATE_PALETTE[index % STATE_PALETTE.length];
}

function colorForLabel(label, index = 0) {
  return colorForState(regimeNameToState(label), index);
}

function renderMetricGrid(containerId, fields, data) {
  const grid = $(containerId);
  grid.innerHTML = "";
  fields.forEach(([key, label]) => {
    const card = document.createElement("div");
    card.className = "metric";
    const valueEl = document.createElement("div");
    valueEl.className = "value";
    card.appendChild(Object.assign(document.createElement("div"), { className: "label", textContent: label }));
    card.appendChild(valueEl);
    grid.appendChild(card);
    animateMetric(valueEl, data.summary[key], key, data.currency);
  });
}

function renderResults(data) {
  $("intro").classList.add("hidden");
  $("results-content").classList.remove("hidden");
  renderScenarioChips(state.lastSimPayload, data);
  $("macro-chart-title").textContent = data.model_kind === "hmm" ? "HMM states / macro history" : "Macro quadrants";
  renderMetricGrid("metric-grid", METRIC_FIELDS, data);
  const hasCashFlows = Number(data.summary.periodic_contribution || 0) > 0 || Number(data.summary.periodic_withdrawal || 0) > 0;
  $("cash-flow-performance").classList.toggle("hidden", !hasCashFlows);
  if (hasCashFlows) renderMetricGrid("flow-metric-grid", FLOW_METRIC_FIELDS, data);
  const costs = data.costs || {};
  const hasCosts = Number(costs.leverage_multiple || 1) > 1 || Number(costs.weighted_expense_ratio || 0) > 0;
  $("cost-assumptions").classList.toggle("hidden", !hasCosts);
  if (hasCosts) renderMetricGrid("cost-metric-grid", COST_METRIC_FIELDS, { summary: costs, currency: data.currency });
  $("stale-results").classList.add("hidden");
  $("risk-caption").textContent =
    `${data.terms === "real" ? "Real (inflation-adjusted) | " : "Nominal | "}` +
    `Currency: ${data.currency} | ` +
    `Probability of loss: ${pct(data.summary.probability_of_loss)} | ` +
    `VaR (95%): ${formatMetricValue("var_95", data.summary.var_95, data.currency)} | ` +
    `Expected shortfall (95%): ${formatMetricValue("expected_shortfall_95", data.summary.expected_shortfall_95, data.currency)} | ` +
    `Worst max drawdown: ${pct(data.summary.max_drawdown_worst)}`;
  const riskFree = Number(state.lastSimPayload?.risk_free_rate || 0);
  $("performance-caption").textContent =
    `Annualized return: ${pct(data.summary.annualized_return)} | ` +
    `Annualized volatility: ${pct(data.summary.annualized_volatility)} | ` +
    `Sharpe ratio (${fmt(riskFree, 2)}% risk-free): ${fmt(data.summary.sharpe_ratio, 2)} | ` +
    `Ulcer index: ${fmt(data.summary.ulcer_index_mean, 2)} (p95 ${fmt(data.summary.ulcer_index_p95, 2)}) | ` +
    `Terminal skew: ${fmt(data.summary.terminal_skewness, 2)} | ` +
    `Excess kurtosis: ${fmt(data.summary.terminal_kurtosis, 2)}`;

  lineChart($("chart-wealth"), data.wealth.periods, [
    { name: "P05", color: "#f97316", values: data.wealth.p05 },
    { name: "Median", color: "#3b82f6", values: data.wealth.median },
    { name: "P95", color: "#10b981", values: data.wealth.p95 },
  ]);
  histChart($("chart-terminal"), data.terminal);
  histChart($("chart-drawdowns"), data.drawdowns, 45, "Maximum drawdown", "#f97316");
  timelineChart($("chart-timeline"), data.regime_timeline);
  barChart($("chart-regime-mix"), data.regime_mix.map((item) => ({
    label: item.label,
    value: item.share,
    color: colorForLabel(item.label, data.regime_mix.indexOf(item)),
  })), { digits: 3 });
  const macroLabels = [...new Set(data.macro_scatter.map((point) => point.regime))];
  scatterChart(
    $("chart-macro"),
    data.macro_scatter.map((p) => ({
      x: p.growth,
      y: p.inflation,
      color: colorForLabel(p.regime, macroLabels.indexOf(p.regime)),
      label: p.date,
      regime: p.regime,
    })),
    "Growth",
    "Inflation",
    macroLabels.map((label, index) => ({ label, color: colorForLabel(label, index) }))
  );
  heatmap($("chart-transition"), data.transition.labels, data.transition.values, [0, 1], "From / To");
  barChart($("chart-observations"), Object.entries(data.observations).map(([label, value]) => ({
    label,
    value,
    color: colorForLabel(label, Object.keys(data.observations).indexOf(label)),
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
  const validation = data.validation;
  $("validation-panel").classList.toggle("hidden", !validation);
  if (validation) {
    const summary = validation.summary;
    $("validation-summary").textContent =
      `Out-of-sample advantage: ${summary.advantage_mean > 0 ? "+" : ""}${fmt(summary.advantage_mean, 4)} log-likelihood units/period · ` +
      `positive split share: ${pct(summary.advantage_positive_share)} · ` +
      `one-step regime hit rate: ${pct(summary.regime_hit_rate, 0)} · ${summary.splits} splits.`;
    renderTable("validation-table", { columns: validation.columns, rows: validation.rows });
  }
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
    state.results = null;
    state.lastSimPayload = null;
    setStatus(message, data.message);
    notify(data.message, "success");
    renderTickerChecklist(data.tickers, data.default_tickers);
    state.weights = {};
    renderWeightEditor();
    renderTables(data);
    renderCoverage(data.coverage);
    renderSyntheticReport(data.synthetic);
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
    state.lastSimPayload = payload;
    state.results = data;
    updateGuide();
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
  if (!state.results) {
    notify("Run a simulation first.", "error");
    return;
  }
  if (state.lastSimPayload && JSON.stringify(gatherSimPayload()) !== JSON.stringify(state.lastSimPayload)) {
    notify("Run the current inputs before exporting results.", "error");
    return;
  }
  const summary = state.results.summary;
  const rows = Object.entries(summary).map(([key, value]) => [key, value]);
  downloadCSV("risk_summary.csv", toCSV(["metric", "value"], rows));
  notify("Risk summary downloaded", "success");
}

function onDownloadDiagnostics() {
  if (!state.diagnostics) {
    notify("Run a simulation first.", "error");
    return;
  }
  if (state.lastSimPayload && JSON.stringify(gatherSimPayload()) !== JSON.stringify(state.lastSimPayload)) {
    notify("Run the current inputs before exporting results.", "error");
    return;
  }
  downloadCSV("calibration_diagnostics.csv", toCSV(state.diagnostics.columns, state.diagnostics.rows));
  notify("Diagnostics downloaded", "success");
}

async function onDownloadWealth() {
  if (!state.lastSimPayload) {
    notify("Run a simulation first.", "error");
    return;
  }
  if (JSON.stringify(gatherSimPayload()) !== JSON.stringify(state.lastSimPayload)) {
    notify("Run the current inputs before exporting results.", "error");
    return;
  }
  showOverlay("Exporting wealth paths...");
  try {
    const data = await postJSON("/api/wealth", state.lastSimPayload);
    downloadCSV("wealth_paths.csv", data.csv);
    notify("Wealth paths downloaded", "success");
  } catch (error) {
    notify(error.message, "error");
  } finally {
    hideOverlay();
  }
}

/* ---------- Settings persistence ---------- */

const CONTROL_IDS = [
  "yahoo-tickers", "yahoo-start", "yahoo-end", "yahoo-proxies", "synthetic-seed",
  "synthetic-method", "synthetic-categories",
  "csv-growth", "csv-inflation", "base-currency", "currency-map", "corr-blend",
  "growth-threshold", "growth-fixed", "inflation-threshold", "inflation-fixed",
  "macro-lag", "transition-uncertainty", "periods", "paths", "seed", "distribution",
  "degrees-of-freedom", "block-size", "rebalance", "cost-bps", "contribution", "withdrawal",
  "risk-free", "annual-inflation", "expense-ratios", "leverage-multiple", "financing-rate", "maintenance-margin",
  "model-kind", "hmm-states", "threshold-window", "duration-model",
];

function saveControls() {
  const data = {};
  CONTROL_IDS.forEach((id) => {
    const el = $(id);
    if (el) data[id] = el.value;
  });
  data.synthetic = Array.from(document.querySelectorAll('#synthetic-options input[type="checkbox"]:checked')).map((el) => el.value);
  data.csvEnabled = $("csv-enabled").checked;
  data.csvMonthly = $("csv-monthly").checked;
  data.useCorr = $("use-corr-override").checked;
  data.garch = $("garch").checked;
  data.walkForward = $("walk-forward").checked;
  data.corrTargets = gatherCorrelationTargets();
  localStorage.setItem("mcq-controls", JSON.stringify(data));
}

function restoreControls() {
  try {
    const raw = localStorage.getItem("mcq-controls");
    if (!raw) return;
    const data = JSON.parse(raw);
    CONTROL_IDS.forEach((id) => {
      const el = $(id);
      if (el && data[id] !== undefined) el.value = data[id];
    });
    if (data.csvEnabled !== undefined) $("csv-enabled").checked = data.csvEnabled;
    document.querySelectorAll("#synthetic-options input[type='checkbox']").forEach((checkbox) => {
      checkbox.checked = (data.synthetic || []).includes(checkbox.value);
    });
    if (data.csvMonthly !== undefined) $("csv-monthly").checked = data.csvMonthly;
    if (data.useCorr !== undefined) $("use-corr-override").checked = data.useCorr;
    if (data.garch !== undefined) $("garch").checked = data.garch;
    if (data.walkForward !== undefined) $("walk-forward").checked = data.walkForward;
    document.querySelectorAll("#corr-sliders input[type='range']").forEach((slider) => {
      if (data.corrTargets && data.corrTargets[slider.dataset.state] !== undefined) slider.value = data.corrTargets[slider.dataset.state];
    });
  } catch (error) {
    // ignore corrupted saved settings
  }
}

/* ---------- Theme ---------- */

function applyTheme(theme) {
  document.body.classList.toggle("light", theme === "light");
  $("theme-toggle").textContent = theme === "light" ? "Dark theme" : "Light theme";
  localStorage.setItem("mcq-theme", theme);
}

function toggleTheme() {
  const next = document.body.classList.contains("light") ? "dark" : "light";
  applyTheme(next);
}

/* ---------- Convenience actions ---------- */

function equalizeWeights() {
  const selected = selectedTickers();
  if (!selected.length) {
    notify("Select at least one ticker first.", "error");
    return;
  }
  const weight = 100 / selected.length;
  selected.forEach((ticker) => { state.weights[ticker] = weight; });
  renderWeightEditor();
  notify("Weights equalized", "success");
}

function resetControls() {
  localStorage.removeItem("mcq-controls");
  location.reload();
}

function setSectionsOpen(open) {
  document.querySelectorAll("#controls details").forEach((details) => { details.open = open; });
}

function onDownloadJson() {
  if (!state.results) {
    notify("Run a simulation first.", "error");
    return;
  }
  if (state.lastSimPayload && JSON.stringify(gatherSimPayload()) !== JSON.stringify(state.lastSimPayload)) {
    notify("Run the current inputs before exporting results.", "error");
    return;
  }
  downloadFile("results.json", JSON.stringify(state.results, null, 2), "application/json;charset=utf-8");
  notify("Results downloaded", "success");
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

  function syncCsvEnabled() {
    const bothFiles = $("csv-prices").files.length > 0 && $("csv-macro").files.length > 0;
    $("csv-enabled").checked = bothFiles;
    toggleSourceGroups();
  }

  $("csv-enabled").addEventListener("change", toggleSourceGroups);
  $("csv-prices").addEventListener("change", syncCsvEnabled);
  $("csv-macro").addEventListener("change", syncCsvEnabled);

  document.querySelectorAll(".tab-btn").forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

  $("load-btn").addEventListener("click", onLoad);
  $("run-btn").addEventListener("click", onRun);
  $("compare-btn").addEventListener("click", onCompare);
  $("download-summary").addEventListener("click", onDownloadSummary);
  $("download-diagnostics").addEventListener("click", onDownloadDiagnostics);
  $("download-wealth").addEventListener("click", onDownloadWealth);
  $("download-json").addEventListener("click", onDownloadJson);
  $("preset-apply").addEventListener("click", applyPreset);
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("equalize-btn").addEventListener("click", equalizeWeights);
  $("reset-btn").addEventListener("click", resetControls);
  $("expand-all").addEventListener("click", () => setSectionsOpen(true));
  $("collapse-all").addEventListener("click", () => setSectionsOpen(false));

  document.addEventListener("input", () => { saveControls(); updateMethodologyControls(); });
  document.addEventListener("change", () => { saveControls(); updateMethodologyControls(); });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !$("run-btn").disabled) {
      event.preventDefault();
      onRun();
    }
  });

  restoreControls();
  toggleSourceGroups();
  updateMethodologyControls();
  applyTheme(localStorage.getItem("mcq-theme") || "dark");

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
