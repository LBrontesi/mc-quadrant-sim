"use strict";

import { postJSON } from "./api-client.js";

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
  ["annualized_return", "Annualized return"],
  ["annualized_volatility", "Annualized volatility"],
  ["effective_risk_free_rate", "Effective risk-free rate"],
  ["sharpe_ratio", "Sharpe ratio"], ["sortino_ratio", "Sortino"], ["calmar_ratio", "Calmar"],
  ["geometric_annualized_return", "CAGR"], ["probability_of_loss", "Probability of loss"],
  ["var_95", "VaR (95%)"], ["expected_shortfall_95", "Expected shortfall (95%)"],
  ["max_drawdown_worst", "Worst max drawdown"],
];

const FLOW_METRIC_FIELDS = [
  ["total_contributed", "Total contributions"],
  ["total_withdrawn", "Total withdrawals"],
  ["net_external_cash_flow", "Net external cash flow"],
];
const GOAL_METRIC_FIELDS = [
  ["target_wealth", "Target wealth"],
  ["goal_success_probability", "Target success"],
  ["expected_goal_shortfall", "Expected shortfall to target"],
  ["risk_of_ruin", "Risk of ruin"],
  ["omega_ratio", "Omega ratio"],
  ["worst_rolling_return_p05", "P05 worst rolling return"],
  ["max_underwater_months_p95", "P95 time underwater"],
  ["recovery_months_p95", "P95 recovery time"],
];
const COST_METRIC_FIELDS = [
  ["leverage_multiple", "Leverage"], ["weighted_expense_ratio", "Weighted ETF fee"],
  ["annual_fee_drag", "Annual fee drag"], ["annual_financing_cost", "Annual financing cost"],
  ["effective_financing_rate", "Effective financing rate"], ["margin_calls", "Margin calls"],
];

const PERCENT_METRICS = new Set([
  "annualized_return", "annualized_volatility", "cash_flow_adjusted_annualized_return",
  "cash_flow_adjusted_volatility", "geometric_annualized_return", "probability_of_loss",
  "effective_risk_free_rate", "effective_financing_rate",
  "max_drawdown_mean", "max_drawdown_p95", "max_drawdown_worst", "ulcer_index_mean", "ulcer_index_p95",
  "goal_success_probability", "risk_of_ruin", "unrecovered_at_horizon", "worst_rolling_return",
  "worst_rolling_return_p05", "median_worst_rolling_return",
]);
const CURRENCY_METRICS = new Set([
  "mean", "std", "p05", "p50", "p95", "var_95", "expected_shortfall_95", "periodic_contribution",
  "periodic_withdrawal", "total_contributed", "total_withdrawn", "net_external_cash_flow",
  "target_wealth", "expected_goal_shortfall",
]);

const state = {
  loadPayload: null,
  loadResult: null,
  selected: [],
  weights: {},
  results: null,
  diagnostics: null,
  lastSimPayload: null,
  labWeights: {},
  pairedResults: null,
  pendingPortfolio: null,
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

function quantileFromSorted(sorted, probability) {
  if (!sorted.length) return 0;
  const position = Math.max(0, Math.min(sorted.length - 1, probability * (sorted.length - 1)));
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function sampleQuantile(values, probability) {
  return quantileFromSorted([...values].sort((a, b) => a - b), probability);
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

let overlayStartedAt = 0;
let overlayTimer = null;

function showOverlay(message, stage = "Preparing data and model inputs") {
  $("overlay-text").textContent = message || "Working...";
  $("overlay-stage").textContent = stage;
  overlayStartedAt = performance.now();
  const updateElapsed = () => {
    const seconds = Math.max(0, Math.round((performance.now() - overlayStartedAt) / 1000));
    $("overlay-elapsed").textContent = `${seconds}s elapsed`;
  };
  updateElapsed();
  clearInterval(overlayTimer);
  overlayTimer = setInterval(updateElapsed, 1000);
  $("overlay").classList.remove("hidden");
}

function hideOverlay() {
  clearInterval(overlayTimer);
  overlayTimer = null;
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
  if (key.includes("_months")) return `${numeric.toFixed(1)} mo`;
  if (PERCENT_METRICS.has(key)) return `${(numeric * 100).toFixed(2)}%`;
  if (CURRENCY_METRICS.has(key)) return `${currency} ${numeric.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  return numeric.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function compactCurrency(value, currency = "USD") {
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(value);
  } catch {
    return `${currency} ${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  }
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

let revealObserver = null;

function registerReveals(root = document) {
  if (!revealObserver) return;
  const selector = ".workspace-heading, .guide-mini, .settings-panel, .allocation-bar, .card, .metric";
  const elements = [...(root.querySelectorAll?.(selector) || [])];
  if (root.matches?.(selector)) elements.unshift(root);
  elements.forEach((element) => {
    if (element.dataset.revealReady) return;
    element.dataset.revealReady = "true";
    element.classList.add("reveal");
    revealObserver.observe(element);
  });
}

function setupExperience() {
  const progress = $("scroll-progress-bar");
  let scrollFrame = 0;
  const updateScrollProgress = () => {
    scrollFrame = 0;
    const available = Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
    progress.style.transform = `scaleX(${Math.min(Math.max(window.scrollY / available, 0), 1)})`;
  };
  window.addEventListener("scroll", () => {
    if (!scrollFrame) scrollFrame = requestAnimationFrame(updateScrollProgress);
  }, { passive: true });
  updateScrollProgress();

  if (!(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches)) {
    revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -6%" });
    registerReveals();
    new MutationObserver((mutations) => {
      mutations.forEach((mutation) => mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) registerReveals(node);
      }));
    }).observe($("workspace"), { childList: true, subtree: true });
  }

  const stage = $("quadrant-stage");
  stage.addEventListener("pointermove", (event) => {
    const bounds = stage.getBoundingClientRect();
    stage.style.setProperty("--stage-x", `${((event.clientX - bounds.left) / bounds.width - 0.5) * 16}px`);
    stage.style.setProperty("--stage-y", `${((event.clientY - bounds.top) / bounds.height - 0.5) * 16}px`);
  });
  stage.addEventListener("pointerleave", () => {
    stage.style.setProperty("--stage-x", "0px");
    stage.style.setProperty("--stage-y", "0px");
  });

  document.querySelectorAll("details").forEach((details) => details.addEventListener("toggle", () => {
    if (!details.open) return;
    details.classList.remove("details-flash");
    requestAnimationFrame(() => details.classList.add("details-flash"));
  }));

  document.querySelectorAll(".site-nav [data-tab-target]").forEach((link) => link.addEventListener("click", () => {
    switchTab(link.dataset.tabTarget);
  }));

  const updateHeroPaths = () => {
    $("hero-path-count").textContent = Math.max(0, Number($("paths").value) || 0).toLocaleString();
  };
  $("paths").addEventListener("input", updateHeroPaths);
  updateHeroPaths();
}

/* ---------- SVG charts ---------- */

const SVG_NS = "http://www.w3.org/2000/svg";
const MARGIN = { top: 12, right: 12, bottom: 26, left: 46 };

function createSvg(width, height) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(height));
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
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
    line.setAttribute("x1", x); line.setAttribute("y1", MARGIN.top);
    line.setAttribute("x2", x); line.setAttribute("y2", MARGIN.top + height);
    line.setAttribute("stroke", "var(--grid)"); line.setAttribute("stroke-width", "1");
    grid.appendChild(line);
    svgText(grid, x, MARGIN.top + height + 14, label, "chart-title");
  });
  yTicks.forEach(([y, label]) => {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", MARGIN.left); line.setAttribute("y1", y);
    line.setAttribute("x2", MARGIN.left + width); line.setAttribute("y2", y);
    line.setAttribute("stroke", "var(--grid)"); line.setAttribute("stroke-width", "1");
    grid.appendChild(line);
    svgText(grid, MARGIN.left - 6, y + 4, label, "chart-title", "end");
  });
  if (xLabel) svgText(grid, MARGIN.left + width / 2, MARGIN.top + height + 22, xLabel, "chart-title");
  if (yLabel) {
    const labelX = 10;
    const labelY = MARGIN.top + height / 2;
    const el = svgText(grid, labelX, labelY, yLabel, "chart-title");
    el.setAttribute("transform", `rotate(-90 ${labelX} ${labelY})`);
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

function lineChart(container, labels, series, options = {}) {
  container.innerHTML = "";
  if (!labels.length || !series.length) {
    container.innerHTML = "<p class='status'>No values to chart.</p>";
    return;
  }
  const width = 560, height = 300;
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  let min = Infinity, max = -Infinity;
  series.forEach((s) => s.values.forEach((v) => { if (v < min) min = v; if (v > max) max = v; }));
  const pad = (max - min) * 0.08 || 1;
  min -= pad; max += pad;
  const numericX = Boolean(options.numericX) && labels.every((value) => Number.isFinite(Number(value)));
  const xMin = numericX ? Number(labels[0]) : 0;
  const xMax = numericX ? Number(labels.at(-1)) : labels.length - 1;
  const xScale = (i) => {
    const value = numericX ? Number(labels[i]) : i;
    return MARGIN.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotW;
  };
  const yScale = (v) => MARGIN.top + (1 - (v - min) / (max - min)) * plotH;
  const svg = createSvg(width, height);
  const xTicks = numericX
    ? niceTicks(xMin, xMax, 5).map((value) => [MARGIN.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotW, value])
    : niceTicks(0, labels.length - 1, 5).map((t) => [xScale(Math.round(t)), labels[Math.round(t)] ?? ""]);
  const yFormatter = options.yFormatter || ((value) => value.toFixed(0));
  const xFormatter = options.xFormatter || ((value) => String(value));
  const formattedXTicks = xTicks.map(([x, label]) => [x, xFormatter(label)]);
  const yTicks = niceTicks(min, max, 5).map((v) => [yScale(v), yFormatter(v)]);
  drawAxes(
    svg,
    plotW,
    plotH,
    formattedXTicks,
    yTicks,
    options.xLabel || "Period",
    options.yLabel || "Wealth",
  );
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
    const index = numericX
      ? labels.reduce((best, value, candidate) => (
        Math.abs(Number(value) - (xMin + ratio * (xMax - xMin))) < Math.abs(Number(labels[best]) - (xMin + ratio * (xMax - xMin))) ? candidate : best
      ), 0)
      : Math.max(0, Math.min(labels.length - 1, Math.round(ratio * (labels.length - 1))));
    crosshair.setAttribute("x1", xScale(index));
    crosshair.setAttribute("x2", xScale(index));
    crosshair.setAttribute("opacity", "1");
    const title = options.tooltipTitle
      ? options.tooltipTitle(labels[index])
      : `${options.xLabel || "Period"} ${xFormatter(labels[index])}`;
    const valueFormatter = options.valueFormatter || ((value) => fmt(value));
    const lines = [`<b>${escapeHtml(title)}</b>`];
    series.forEach((s) => lines.push(`<span style="color:${s.color}">${s.name}:</span> ${escapeHtml(valueFormatter(s.values[index]))}`));
    tooltip.show(px, py, lines.join("<br>"));
  });
  svg.addEventListener("mouseleave", () => { crosshair.setAttribute("opacity", "0"); tooltip.hide(); });
}

function bandChart(container, labels, band, options = {}) {
  container.innerHTML = "";
  if (!labels.length || !band.low?.length || !band.median?.length || !band.high?.length) {
    container.innerHTML = "<p class='status'>No values to chart.</p>";
    return;
  }
  const width = 560, height = 300;
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const allValues = [...band.low, ...band.median, ...band.high].filter(Number.isFinite);
  let min = Math.min(...allValues), max = Math.max(...allValues);
  if (options.includeZero !== false) {
    min = Math.min(min, 0);
    max = Math.max(max, 0);
  }
  const pad = (max - min) * 0.08 || 0.01;
  min -= pad;
  max += pad;
  const numericX = Boolean(options.numericX) && labels.every((value) => Number.isFinite(Number(value)));
  const xMin = numericX ? Number(labels[0]) : 0;
  const xMax = numericX ? Number(labels.at(-1)) : labels.length - 1;
  const xScale = (index) => {
    const value = numericX ? Number(labels[index]) : index;
    return MARGIN.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotW;
  };
  const yScale = (value) => MARGIN.top + (1 - (value - min) / (max - min)) * plotH;
  const yFormatter = options.yFormatter || ((value) => value.toFixed(2));
  const xFormatter = options.xFormatter || ((value) => String(value));
  const svg = createSvg(width, height);
  const xTicks = numericX
    ? niceTicks(xMin, xMax, 5).map((value) => [MARGIN.left + ((value - xMin) / Math.max(xMax - xMin, 1)) * plotW, xFormatter(value)])
    : [...new Set(niceTicks(0, labels.length - 1, 5).map(Math.round))]
        .map((index) => [xScale(index), xFormatter(labels[index])]);
  const yTicks = niceTicks(min, max, 5).map((value) => [yScale(value), yFormatter(value)]);
  drawAxes(svg, plotW, plotH, xTicks, yTicks, options.xLabel || "Period", options.yLabel || "Value");

  const area = document.createElementNS(SVG_NS, "polygon");
  const upper = band.high.map((value, index) => `${xScale(index)},${yScale(value)}`);
  const lower = band.low.map((value, index) => `${xScale(index)},${yScale(value)}`).reverse();
  area.setAttribute("points", [...upper, ...lower].join(" "));
  area.setAttribute("fill", band.color || "#b8d0d6");
  area.setAttribute("opacity", "0.16");
  svg.appendChild(area);
  [
    { values: band.low, opacity: "0.48", width: "1.2" },
    { values: band.high, opacity: "0.48", width: "1.2" },
    { values: band.median, opacity: "1", width: "2.4" },
  ].forEach((series) => {
    const path = document.createElementNS(SVG_NS, "polyline");
    path.setAttribute("points", series.values.map((value, index) => `${xScale(index)},${yScale(value)}`).join(" "));
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", band.color || "#b8d0d6");
    path.setAttribute("stroke-width", series.width);
    path.setAttribute("opacity", series.opacity);
    path.setAttribute("stroke-linejoin", "round");
    svg.appendChild(path);
  });
  const crosshair = document.createElementNS(SVG_NS, "line");
  crosshair.setAttribute("y1", 0);
  crosshair.setAttribute("y2", plotH);
  crosshair.setAttribute("stroke", "var(--crosshair)");
  crosshair.setAttribute("stroke-dasharray", "4 4");
  crosshair.setAttribute("opacity", "0");
  svg.appendChild(crosshair);
  container.appendChild(svg);

  const legend = document.createElement("div");
  legend.className = "legend";
  const labelsForBand = options.bandLabels || ["P05", "Median", "P95"];
  legend.innerHTML = `<span class="legend-item"><span class="legend-band" style="background:${band.color || "#b8d0d6"}"></span>${escapeHtml(labelsForBand[0])}–${escapeHtml(labelsForBand[2])}</span><span class="legend-item"><span class="legend-line" style="background:${band.color || "#b8d0d6"}"></span>${escapeHtml(labelsForBand[1])}</span>`;
  container.appendChild(legend);

  const tooltip = attachTooltip(container);
  const valueFormatter = options.valueFormatter || yFormatter;
  svg.addEventListener("mousemove", (event) => {
    const rect = svg.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (width / rect.width);
    const ratio = (x - MARGIN.left) / plotW;
    const index = numericX
      ? labels.reduce((best, value, candidate) => (
        Math.abs(Number(value) - (xMin + ratio * (xMax - xMin))) < Math.abs(Number(labels[best]) - (xMin + ratio * (xMax - xMin))) ? candidate : best
      ), 0)
      : Math.max(0, Math.min(labels.length - 1, Math.round(ratio * (labels.length - 1))));
    crosshair.setAttribute("x1", xScale(index));
    crosshair.setAttribute("x2", xScale(index));
    crosshair.setAttribute("opacity", "1");
    const title = options.tooltipTitle
      ? options.tooltipTitle(labels[index])
      : `${options.xLabel || "Period"} ${xFormatter(labels[index])}`;
    tooltip.show(
      event.clientX - rect.left,
      event.clientY - rect.top,
      `<b>${escapeHtml(title)}</b><br>${escapeHtml(labelsForBand[0])}: ${escapeHtml(valueFormatter(band.low[index]))}<br>${escapeHtml(labelsForBand[1])}: ${escapeHtml(valueFormatter(band.median[index]))}<br>${escapeHtml(labelsForBand[2])}: ${escapeHtml(valueFormatter(band.high[index]))}`,
    );
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

function scatterChart(container, points, xLabel, yLabel, legend = null, options = {}) {
  container.innerHTML = "";
  if (!points.length) {
    container.innerHTML = "<p class='status'>No drawdown episodes to chart.</p>";
    return;
  }
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
  const xFormatter = options.xFormatter || ((value) => value.toFixed(1));
  const yFormatter = options.yFormatter || ((value) => value.toFixed(1));
  const xTicks = niceTicks(xMin, xMax, 5).map((v) => [xScale(v), xFormatter(v)]);
  const yTicks = niceTicks(yMin, yMax, 5).map((v) => [yScale(v), yFormatter(v)]);
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
      lines.push(`${xLabel}: ${escapeHtml(options.xValueFormatter ? options.xValueFormatter(p.x) : fmt(p.x, 2))}`);
      lines.push(`${yLabel}: ${escapeHtml(options.yValueFormatter ? options.yValueFormatter(p.y) : fmt(p.y, 2))}`);
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
  let min = Infinity;
  let max = -Infinity;
  values.forEach((value) => {
    if (!Number.isFinite(value)) return;
    min = Math.min(min, value);
    max = Math.max(max, value);
  });
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    container.innerHTML = "<p class='status'>No finite values to chart.</p>";
    return;
  }
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

function timelineChart(container, timelines) {
  container.innerHTML = "";
  const width = 560;
  const rowHeight = 26;
  const labelWidth = 88;
  const chipSize = 8;
  const height = timelines.length * rowHeight + 8;
  const svg = createSvg(width, height);
  const slot = (width - labelWidth) / timelines[0].states.length;
  const tooltip = attachTooltip(container);
  const allStates = [...new Set(timelines.flatMap((t) => t.states))];
  timelines.forEach((timeline, rowIndex) => {
    const rowTop = 4 + rowIndex * rowHeight;
    const chip = document.createElementNS(SVG_NS, "rect");
    chip.setAttribute("x", 2);
    chip.setAttribute("y", rowTop + (rowHeight - 4 - chipSize) / 2);
    chip.setAttribute("width", chipSize);
    chip.setAttribute("height", chipSize);
    chip.setAttribute("rx", 2);
    chip.setAttribute("fill", timeline.color);
    svg.appendChild(chip);
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", 14);
    label.setAttribute("y", rowTop + 17);
    label.setAttribute("font-size", "11");
    label.setAttribute("font-weight", "700");
    label.setAttribute("fill", "#e6ecf7");
    label.textContent = timeline.label;
    svg.appendChild(label);
    timeline.states.forEach((state, index) => {
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", labelWidth + index * slot);
      rect.setAttribute("y", rowTop);
      rect.setAttribute("width", Math.max(slot - 1, 1));
      rect.setAttribute("height", rowHeight - 4);
      rect.setAttribute("fill", colorForState(state, allStates.indexOf(state)));
      rect.addEventListener("mouseenter", () => {
        rect.setAttribute("opacity", "0.85");
        tooltip.show(labelWidth + index * slot + slot / 2, rowTop, `Period ${index + 1}<br>${escapeHtml(labelForState(state))}`);
      });
      rect.addEventListener("mouseleave", () => { rect.setAttribute("opacity", "1"); tooltip.hide(); });
      svg.appendChild(rect);
    });
  });
  container.appendChild(svg);
  const legend = document.createElement("div");
  legend.className = "legend";
  allStates.forEach((state, index) => {
    const item = document.createElement("span");
    item.innerHTML = `<span class="legend-dot" style="background:${colorForState(state, index)}"></span>${escapeHtml(labelForState(state))}`;
    legend.appendChild(item);
  });
  container.appendChild(legend);
}

/* ---------- Inputs ---------- */

function activeSource() {
  return $("csv-enabled").checked ? "csv" : "yahoo";
}

function parseMarketTickers() {
  return [...new Set(String($("yahoo-tickers").value || "")
    .split(/[,;\s]+/)
    .map((ticker) => ticker.trim().toUpperCase())
    .filter(Boolean))];
}

function setMarketTickers(tickers) {
  $("yahoo-tickers").value = [...new Set(tickers.map((ticker) => String(ticker).trim().toUpperCase()).filter(Boolean))].join(", ");
  $("yahoo-tickers").dispatchEvent(new Event("input", { bubbles: true }));
}

function syncUniversePreset() {
  const current = parseMarketTickers().join(",");
  document.querySelectorAll(".universe-preset").forEach((button) => {
    const selected = String(button.dataset.tickers || "").replaceAll(" ", "") === current;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function renderTickerComposer() {
  const tickers = parseMarketTickers();
  const container = $("market-ticker-chips");
  container.innerHTML = "";
  tickers.forEach((ticker) => {
    const token = document.createElement("span");
    token.className = "ticker-token";
    const symbol = document.createElement("strong");
    symbol.textContent = ticker;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${ticker}`);
    remove.addEventListener("click", () => setMarketTickers(tickers.filter((item) => item !== ticker)));
    token.append(symbol, remove);
    container.appendChild(token);
  });
  $("ticker-count").textContent = `${tickers.length} selected`;
  $("market-universe-summary").textContent = `${tickers.length} ${tickers.length === 1 ? "asset" : "assets"}`;
  syncUniversePreset();
}

function syncHistoryRange() {
  const start = $("yahoo-start").value;
  const end = $("yahoo-end").value;
  const endDate = end ? new Date(`${end}T12:00:00`) : new Date();
  document.querySelectorAll("#history-ranges button").forEach((button) => {
    let expected = button.dataset.start;
    if (button.dataset.years) {
      const date = new Date(endDate);
      date.setFullYear(date.getFullYear() - Number(button.dataset.years));
      expected = date.toISOString().slice(0, 10);
    }
    const selected = start === expected;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  if (!start) $("market-window-summary").textContent = "Choose dates";
  else {
    const startYear = start.slice(0, 4);
    const endYear = end ? end.slice(0, 4) : "today";
    $("market-window-summary").textContent = `${startYear}–${endYear}`;
  }
}

function syncMacroVintageExplainer() {
  const pointInTime = $("macro-vintage").value === "initial_release";
  $("macro-vintage-explainer").innerHTML = pointInTime
    ? "Most realistic research mode. Uses historical releases to remove revision look-ahead and requires <code>FRED_API_KEY</code> on the server."
    : "Fastest setup. Release lags reduce timing bias, but revised observations still contain information unavailable at the time.";
}

function syncDataSourceUI() {
  const custom = activeSource() === "csv";
  const liveButton = $("data-source-live");
  const csvButton = $("data-source-csv");
  liveButton.classList.toggle("active", !custom);
  csvButton.classList.toggle("active", custom);
  liveButton.setAttribute("aria-pressed", String(!custom));
  csvButton.setAttribute("aria-pressed", String(custom));
  liveButton.querySelector(".source-status").textContent = custom ? "Select" : "Active";
  csvButton.querySelector(".source-status").textContent = custom ? "Active" : "Select";
  $("yahoo-group").classList.toggle("hidden", custom);
  if (custom) $("custom-data-toggle").open = true;
  $("market-source-summary").textContent = custom ? "Custom CSV" : "Live feeds";
  $("market-universe-summary").textContent = custom ? "From file" : `${parseMarketTickers().length} assets`;
  if (custom) $("market-window-summary").textContent = "File history";
  else syncHistoryRange();
}

function enhanceSelects(root = document) {
  root.querySelectorAll("select:not([data-enhanced])").forEach((select) => {
    const wrapper = document.createElement("span");
    wrapper.className = "select-control";
    select.parentNode.insertBefore(wrapper, select);
    wrapper.appendChild(select);
    const arrow = document.createElement("span");
    arrow.className = "select-arrow";
    arrow.setAttribute("aria-hidden", "true");
    wrapper.appendChild(arrow);
    select.dataset.enhanced = "true";
  });
}

function setupMarketDataExperience() {
  enhanceSelects();
  $("yahoo-tickers").addEventListener("input", renderTickerComposer);
  [$("yahoo-start"), $("yahoo-end")].forEach((input) => input.addEventListener("input", syncHistoryRange));
  $("macro-vintage").addEventListener("change", syncMacroVintageExplainer);
  document.querySelectorAll(".universe-preset").forEach((button) => button.addEventListener("click", () => {
    setMarketTickers(String(button.dataset.tickers || "").split(","));
  }));
  const addTicker = () => {
    const input = $("ticker-add");
    const additions = input.value.split(/[,;\s]+/).filter(Boolean);
    if (!additions.length) return;
    setMarketTickers([...parseMarketTickers(), ...additions]);
    input.value = "";
    input.focus();
  };
  $("ticker-add-btn").addEventListener("click", addTicker);
  $("ticker-add").addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    addTicker();
  });
  document.querySelectorAll("#history-ranges button").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.start) $("yahoo-start").value = button.dataset.start;
    else {
      const end = $("yahoo-end").value ? new Date(`${$("yahoo-end").value}T12:00:00`) : new Date();
      end.setFullYear(end.getFullYear() - Number(button.dataset.years));
      $("yahoo-start").value = end.toISOString().slice(0, 10);
    }
    $("yahoo-start").dispatchEvent(new Event("input", { bubbles: true }));
  }));
  const chooseSource = (custom) => {
    $("csv-enabled").checked = custom;
    $("custom-data-toggle").open = custom;
    syncDataSourceUI();
    toggleSourceGroups();
    saveControls();
    if (custom) $("custom-data-toggle").scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  $("data-source-live").addEventListener("click", () => chooseSource(false));
  $("data-source-csv").addEventListener("click", () => chooseSource(true));
  renderTickerComposer();
  syncHistoryRange();
  syncMacroVintageExplainer();
  syncDataSourceUI();
}

function resetResultsView() {
  ["growth", "returns", "drawdowns", "correlations", "monthly", "lab", "compare", "diagnostics", "data"]
    .forEach((name) => {
      const content = $(`${name}-content`);
      if (content) content.classList.add("hidden");
    });
  document.querySelectorAll(".results-empty").forEach((el) => el.classList.remove("hidden"));
  $("diagnostics-empty").classList.remove("hidden");
  $("data-empty").classList.remove("hidden");
  $("intro").classList.remove("hidden");
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
    $("portfolio-status").textContent = "Data source changed — run the simulation again to refresh.";
    resetResultsView();
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
    payload.macro_vintage = $("macro-vintage").value;
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
    payload.rate_col = $("csv-rate").value.trim();
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
  const quadrantModel = $("model-kind").value === "quadrant";
  const parametricReturns = ["normal", "student_t"].includes($("distribution").value);
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
    cost_bps: ["legacy", "buy_hold"].includes($("rebalance").value) ? 0 : Number($("cost-bps").value),
    contribution: Number($("contribution").value),
    withdrawal: Number($("withdrawal").value),
    initial_value: Number($("initial-value").value),
    target_wealth: Number($("target-wealth").value),
    expense_ratios: $("expense-ratios").value,
    leverage_multiple: Number($("leverage-multiple").value),
    financing_rate: Number($("financing-rate").value),
    financing_inflation_sensitivity: Number($("financing-inflation-sensitivity").value),
    maintenance_margin: Number($("maintenance-margin").value),
    workers: Number($("workers").value || 1),
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
    min_regime_duration: Number($("min-regime-duration").value),
    garch: $("garch").checked,
    walk_forward: $("walk-forward").checked,
    probabilistic_regimes: quadrantModel && $("probabilistic-regimes").checked,
    regime_temperature: Number($("regime-temperature").value),
    regime_smoothing_window: Number($("regime-smoothing-window").value),
    regime_hysteresis: Number($("regime-hysteresis").value),
    regime_confirmation_periods: Number($("regime-confirmation-periods").value),
    duration_prior_strength: Number($("duration-prior-strength").value),
    mean_prior_strength: Number($("mean-prior-strength").value),
    parameter_draws: quadrantModel ? Number($("parameter-draws").value) : 0,
    parameter_block_size: Number($("parameter-block-size").value),
    joint_macro: quadrantModel && parametricReturns && $("joint-macro").checked,
    macro_transition_weight: Number($("macro-transition-weight").value),
    dynamic_correlation: parametricReturns && $("dynamic-correlation").checked,
    dcc_alpha: Number($("dcc-alpha").value),
    dcc_beta: Number($("dcc-beta").value),
    dcc_asymmetry: Number($("dcc-asymmetry").value),
  };
}

function updateResourceEstimate() {
  $("run-btn").textContent = "Run analysis";
}

function validateScenario() {
  const errors = [];
  if (activeSource() === "csv" && (!$('csv-prices').files.length || !$('csv-macro').files.length)) {
    errors.push("Attach both asset and macro CSV files to use custom data.");
  }
  if ($("garch").checked && $("distribution").value !== "normal") {
    errors.push("GARCH volatility clustering requires the Normal return distribution.");
  }
  const dccTotal = Number($("dcc-alpha").value) + Number($("dcc-beta").value) + Number($("dcc-asymmetry").value);
  if ($("dynamic-correlation").checked && dccTotal >= 1) {
    errors.push("Dynamic-correlation α + β + γ must be below 1.");
  }
  if ($("base-currency").value.trim().length !== 3) {
    errors.push("Portfolio currency must be a three-letter ISO code.");
  }
  const leverage = Number($("leverage-multiple").value);
  const margin = Number($("maintenance-margin").value) / 100;
  if (leverage > 1 && ["legacy", "buy_hold"].includes($("rebalance").value)) {
    errors.push("Leverage requires monthly, quarterly, or annual rebalancing.");
  }
  if (leverage === 1 && margin > 0) {
    errors.push("Maintenance margin only applies when leverage is greater than 1.0x.");
  }
  if (leverage > 1 && margin >= 1 / leverage) {
    errors.push("Maintenance margin must be below the initial equity margin for the selected leverage.");
  }
  if (!(Number($("initial-value").value) > 0)) {
    errors.push("Initial portfolio value must be positive.");
  }
  if (!(Number($("target-wealth").value) > 0)) {
    errors.push("Target wealth must be positive.");
  }
  const periods = Number($("periods").value);
  const paths = Number($("paths").value);
  const workers = Number($("workers").value);
  if (!Number.isInteger(periods) || periods < 1 || periods > 360) {
    errors.push("Periods must be between 1 and 360.");
  }
  if (!Number.isInteger(paths) || paths < 1 || paths > 120000) {
    errors.push("Paths must be between 1 and 120,000.");
  }
  if (!Number.isInteger(workers) || workers < 1 || workers > 16) {
    errors.push("Workers must be between 1 and 16.");
  }
  return errors;
}

function updateMethodologyControls() {
  const isHMM = $("model-kind").value === "hmm";
  const distribution = $("distribution").value;
  const legacy = ["legacy", "buy_hold"].includes($("rebalance").value);
  $("quadrant-calibration").classList.toggle("hidden", isHMM);
  $("hmm-states-group").classList.toggle("hidden", !isHMM);
  $("threshold-window-group").classList.toggle("hidden", isHMM);
  $("advanced-regime-controls").classList.toggle("hidden", isHMM);
  $("walk-forward-group").classList.toggle("hidden", isHMM);
  $("correlation-override-controls").classList.toggle("hidden", isHMM);
  $("corr-blend-group").classList.toggle("hidden", isHMM);
  $("start-state-group").classList.toggle("hidden", isHMM);
  $("min-duration-group").classList.toggle("hidden", $("duration-model").value !== "semi_markov");
  $("student-t-group").classList.toggle("hidden", distribution !== "student_t");
  $("block-size-group").classList.toggle("hidden", distribution !== "block_bootstrap");
  $("cost-bps").disabled = legacy;
  $("cost-bps-group").classList.toggle("methodology-muted", legacy);
  $("garch").disabled = distribution !== "normal";
  const parametric = distribution === "normal" || distribution === "student_t";
  $("joint-macro").disabled = !parametric || isHMM;
  $("dynamic-correlation").disabled = !parametric;
  $("parameter-draws").disabled = isHMM;
  $("transition-uncertainty").disabled = Number($("parameter-draws").value) > 0;
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

let assetColors = {};

function assetColor(ticker) {
  if (!assetColors[ticker] && state.loadResult) {
    const index = state.loadResult.tickers.indexOf(ticker);
    assetColors[ticker] = STATE_PALETTE[Math.max(index, 0) % STATE_PALETTE.length];
  }
  return assetColors[ticker] || "#3b82f6";
}

function renderWeightEditor() {
  const selected = selectedTickers();
  const container = $("allocation-list");
  container.innerHTML = "";
  if (!selected.length) {
    container.innerHTML = "<p class='status'>No assets selected.</p>";
    state.selected = selected;
    updateWeightTotal();
    return;
  }
  selected.forEach((ticker) => {
    const pill = document.createElement("div");
    pill.className = "asset-pill";
    const swatch = document.createElement("span");
    swatch.className = "asset-swatch";
    swatch.style.background = assetColor(ticker);
    const name = document.createElement("span");
    name.className = "asset-ticker";
    name.textContent = ticker;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = "100";
    input.step = "1";
    input.className = "asset-weight";
    const weight = state.weights[ticker] ?? defaultWeight(ticker);
    state.weights[ticker] = weight;
    input.value = weight;
    input.addEventListener("input", () => { state.weights[ticker] = Number(input.value); updateWeightTotal(); });
    const unit = document.createElement("span");
    unit.className = "asset-pct";
    unit.textContent = "%";
    const remove = document.createElement("button");
    remove.className = "asset-remove";
    remove.type = "button";
    remove.title = "Remove asset";
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      const checkbox = document.querySelector(`#ticker-list input[value="${CSS.escape(ticker)}"]`);
      if (checkbox) checkbox.checked = false;
      delete state.weights[ticker];
      renderWeightEditor();
    });
    pill.append(swatch, name, input, unit, remove);
    container.appendChild(pill);
  });
  state.selected = selected;
  updateWeightTotal();
}

function updateWeightTotal() {
  const selected = selectedTickers();
  const total = selected.reduce((sum, ticker) => sum + (Number(state.weights[ticker]) || 0), 0);
  const el = $("weight-total");
  el.textContent = `Total: ${total.toFixed(1)}% — the simulation normalizes this to 100%.`;
  el.style.color = total <= 0 ? "var(--danger)" : "var(--muted)";
  updateRunAvailability();
}

function updateRunAvailability() {
  updateResourceEstimate();
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
  const hasData = Boolean(state.loadResult);
  $("run-btn").disabled = errors.length > 0 || (hasData && (selected.length === 0 || total <= 0));
  $("compare-btn").disabled = !state.results;
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
  if (!loaded && activeSource() === "csv") guideStatus.textContent = "Next: attach both asset and macro CSV files.";
  else if (!loaded) guideStatus.textContent = "Loading market data — this happens automatically.";
  else if (!portfolioReady) guideStatus.textContent = "Next: select at least one ticker and set a positive weight.";
  else if (!methodologyReady) guideStatus.textContent = "Next: resolve the highlighted methodology setting.";
  else if (!state.results) guideStatus.textContent = "Ready: click Run analysis.";
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
  $("preset-apply").disabled = !select.value;
}

/* ---------- Tabs ---------- */

function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
}

function focusResults() {
  document.querySelectorAll(".settings-panel > details").forEach((details) => { details.open = false; });
  const tabs = $("result-tabs");
  tabs.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  document.querySelector(`#result-tabs .tab-btn.active`)?.focus({ preventScroll: true });
}

function editScenario() {
  const settings = $("simulation-settings");
  settings.open = true;
  settings.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  $("periods").focus({ preventScroll: true });
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
  chips.push(`Target ${formatMetricValue("target_wealth", payload.target_wealth, data.currency)}`);
  if (Number(payload.contribution) > 0) chips.push(`+${fmt(payload.contribution, 0)}/period`);
  if (Number(payload.withdrawal) > 0) chips.push(`−${fmt(payload.withdrawal, 0)}/period`);
  chips.push(data.terms === "real" ? "Real terms" : "Nominal");
  chips.push(data.currency);
  chips.push(DISTRIBUTION_LABELS[payload.distribution] || payload.distribution);
  chips.push(payload.model === "hmm" ? `HMM · ${payload.hmm_states} states` : "Quadrant model");
  chips.push(payload.duration_model === "semi_markov" ? "Semi-Markov durations" : "Markov durations");
  if (payload.probabilistic_regimes) chips.push("Soft moment weights");
  if (Number(payload.parameter_draws) > 0) chips.push(`${payload.parameter_draws} parameter draws`);
  if (payload.joint_macro) chips.push("Joint macro paths");
  if (payload.dynamic_correlation) chips.push("Dynamic dependence");
  if (Number(payload.leverage_multiple || 1) > 1) chips.push(`${Number(payload.leverage_multiple).toFixed(1)}x leverage`);
  el.innerHTML = chips.map((text) => `<span class="chip">${escapeHtml(text)}</span>`).join("");
}

function renderMethodologyReport(data) {
  const methodology = data.methodology || {};
  const validation = data.validation?.summary || {};
  const validationAvailable = Boolean(data.validation?.summary);
  const safeguards = [
    [methodology.point_in_time, methodology.point_in_time ? "Point-in-time macro vintages" : "Latest-revised macro history"],
    [methodology.availability_aligned, methodology.availability_aligned ? "Release-calendar aligned" : "Period-lag approximation"],
    [methodology.regime_assignment === "probabilistic", "Soft regime weights for return moments"],
    [methodology.transition_estimator === "persistence_filtered_hard_labels", "Persistence-filtered transitions"],
    [Number(methodology.parameter_draws) > 0, `${Number(methodology.parameter_draws) || 0} parameter recalibrations`],
    [Boolean(methodology.joint_macro), "Joint macro/market paths"],
    [methodology.rate_model === "joint_macro_path", "Stochastic policy-rate paths"],
    [Boolean(methodology.dynamic_correlation), "Asymmetric dynamic dependence"],
    [validationAvailable && validation.advantage_vs_student_t_mean > 0, validationAvailable
      ? (validation.advantage_vs_student_t_mean > 0 ? "Beats Student-t benchmark" : "Student-t benchmark not beaten")
      : "Validation unavailable"],
  ];
  const score = safeguards.filter(([passed]) => passed).length;
  const scoreEl = $("methodology-score");
  scoreEl.textContent = `${score}/${safeguards.length} safeguards`;
  scoreEl.dataset.level = score >= 6 ? "strong" : score >= 4 ? "mixed" : "weak";
  $("methodology-badges").innerHTML = safeguards.map(([passed, label]) =>
    `<span class="methodology-badge ${passed ? "integrity-good" : "integrity-warn"}">${passed ? "✓" : "!"} ${escapeHtml(label)}</span>`
  ).join("");
}

function renderPersistence(data) {
  const persistence = data.persistence;
  $("persistence-panel").classList.toggle("hidden", !persistence);
  if (!persistence) return;
  const validation = data.validation?.summary || {};
  const low = Boolean(
    persistence.low_persistence_warning || Number(validation.actual_switches_per_decade || 0) > 24
  );
  const status = $("persistence-status");
  status.textContent = low ? "Review persistence" : "Persistence calibrated";
  status.classList.toggle("integrity-warn", low);
  const metrics = [
    ["Expected switches / decade", fmt(persistence.expected_switches_per_decade, 1)],
    ["Simulated switches / decade", fmt(persistence.simulated_switches_per_decade, 1)],
    ["OOS observed switches / decade", fmt(validation.actual_switches_per_decade, 1)],
    ["Months between switches", fmt(persistence.expected_months_between_switches, 1)],
  ];
  $("persistence-summary").innerHTML = metrics.map(([label, value]) =>
    `<div class="metric"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div></div>`
  ).join("");
  renderTable("persistence-table", {
    columns: ["State", "Expected months", "Historical median", "Historical mean", "Episodes"],
    rows: (persistence.states || []).map((state) => [
      state.label,
      fmt(state.expected_months, 1),
      state.historical_median_months == null ? "—" : fmt(state.historical_median_months, 1),
      state.historical_mean_months == null ? "—" : fmt(state.historical_mean_months, 1),
      state.historical_episodes,
    ]),
  });
  const warning = $("persistence-warning");
  warning.classList.toggle("hidden", !low);
  warning.textContent = low
    ? "Persistence is unusually low. Review the macro smoothing, hysteresis, confirmation, or minimum-duration settings."
    : "";
}

function renderParameterUncertainty(data) {
  const uncertainty = data.parameter_uncertainty;
  $("parameter-uncertainty-card").classList.toggle("hidden", !uncertainty);
  if (!uncertainty) return;
  const definitions = [
    ["annualized_return", "Annualized return", "percent"],
    ["annualized_volatility", "Annualized volatility", "percent"],
    ["average_persistence", "Average persistence", "percent"],
    ["terminal_median", "Terminal median", "currency"],
  ];
  $("parameter-bands").innerHTML = definitions.map(([key, label, kind]) => {
    const band = uncertainty.bands[key];
    if (!band) return "";
    const formatter = kind === "percent"
      ? (value) => pct(value, 1)
      : (value) => formatMetricValue("p50", value, data.currency);
    return `<div class="uncertainty-item"><span>${escapeHtml(label)}</span><strong>${formatter(band.median)}</strong>` +
      `<small>P05 ${formatter(band.p05)} · P95 ${formatter(band.p95)}</small></div>`;
  }).join("");
}

function renderMacroPaths(data) {
  const macro = data.macro_paths;
  $("macro-path-card").classList.toggle("hidden", !macro);
  const grid = $("macro-path-grid");
  grid.innerHTML = "";
  if (!macro) return;
  Object.entries(macro.series || {}).forEach(([name, series]) => {
    const card = document.createElement("div");
    card.className = "card";
    const heading = document.createElement("h4");
    heading.textContent = name === "interest_rate" ? "Short rate (Fed funds)" : name;
    const chart = document.createElement("div");
    chart.className = "chart";
    card.append(heading, chart);
    grid.appendChild(card);
    lineChart(chart, macro.periods, [
      { name: "P05", color: "#f97316", values: series.p05 },
      { name: "Median", color: "#3b82f6", values: series.median },
      { name: "P95", color: "#10b981", values: series.p95 },
    ]);
  });
}

function renderRegimeProbabilities(data) {
  const probabilities = (data.regime_probabilities || []).filter((item) => item.probability > 0);
  $("probability-panel").classList.toggle("hidden", probabilities.length === 0);
  if (!probabilities.length) return;
  barChart($("chart-regime-probabilities"), probabilities.map((item, index) => ({
    label: item.label,
    value: item.probability,
    color: colorForState(item.state, index),
  })), { digits: 3 });
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

const MONTH_ABBREV = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function monthlyReturnColor(returnValue, maxAbs) {
  const intensity = Math.min(Math.abs(returnValue) / maxAbs, 1);
  const alpha = 0.10 + intensity * 0.72;
  if (returnValue >= 0) return `rgba(16, 185, 129, ${alpha.toFixed(3)})`;
  return `rgba(239, 68, 68, ${alpha.toFixed(3)})`;
}

function renderMonthlyCalendar(data) {
  const container = $("chart-monthly");
  container.innerHTML = "";
  const returns = data.monthly_returns || [];
  const startDate = data.start_date ? new Date(`${data.start_date}T00:00:00`) : null;
  if (!startDate || returns.length === 0) {
    container.innerHTML = "<p class='status'>No monthly data to display.</p>";
    return;
  }
  const byYear = {};
  returns.forEach((rate, index) => {
    const date = new Date(startDate.getFullYear(), startDate.getMonth() + index, 1);
    const year = date.getFullYear();
    if (!byYear[year]) byYear[year] = {};
    byYear[year][date.getMonth()] = rate;
  });
  const maxAbs = Math.max(...returns.map((rate) => Math.abs(rate)), 0.001);
  const years = Object.keys(byYear).sort();

  const table = document.createElement("table");
  table.className = "monthly-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th"));
  MONTH_ABBREV.forEach((month) => {
    const th = document.createElement("th");
    th.textContent = month;
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);

  const body = document.createElement("tbody");
  years.forEach((year) => {
    const row = document.createElement("tr");
    const yearCell = document.createElement("td");
    yearCell.className = "monthly-year";
    yearCell.textContent = year;
    row.appendChild(yearCell);
    MONTH_ABBREV.forEach((_, monthIndex) => {
      const cell = document.createElement("td");
      const rate = byYear[year][monthIndex];
      if (rate !== undefined) {
        cell.style.background = monthlyReturnColor(rate, maxAbs);
        cell.textContent = (rate * 100).toFixed(1) + "%";
        cell.title = `${year}-${String(monthIndex + 1).padStart(2, "0")}: ${(rate * 100).toFixed(2)}%`;
      }
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  table.appendChild(body);
  container.appendChild(table);
}

const METRIC_EXPLORER_META = {
  terminal_wealth: { label: "Terminal wealth", kind: "currency", color: "#b8d0d6" },
  max_drawdown: { label: "Maximum drawdown", kind: "percent", color: "#d97706" },
  annualized_return: { label: "Annualized return", kind: "percent", color: "#a8b79b" },
  geometric_annualized_return: { label: "CAGR", kind: "percent", color: "#8ca4aa" },
  annualized_volatility: { label: "Annualized volatility", kind: "percent", color: "#8ca4aa" },
  sharpe_ratio: { label: "Sharpe ratio", kind: "number", color: "#b8d0d6" },
};

function formatExplorerValue(value, kind, currency) {
  if (kind === "currency") return formatMetricValue("p50", value, currency);
  if (kind === "percent") return pct(value, 1);
  return fmt(value, 2);
}

function renderMetricExplorer(data) {
  const select = $("metric-explorer-select");
  const render = () => {
    const key = select.value;
    const metric = data.metric_distributions?.[key];
    if (!metric) return;
    const meta = METRIC_EXPLORER_META[key];
    const summary = metric.summary;
    $("metric-distribution-stats").innerHTML = [
      ["P05", summary.p05], ["Median", summary.median], ["Mean", summary.mean], ["P95", summary.p95],
    ].map(([label, value]) => `<div><small>${label}</small><strong>${escapeHtml(formatExplorerValue(value, meta.kind, data.currency))}</strong></div>`).join("");
    histChart($("chart-metric-explorer"), metric.sample, 45, meta.label, meta.color);
  };
  select.onchange = render;
  render();
}

function renderRepresentativeScenarios(data) {
  const scenarios = data.representative_scenarios || [];
  const selector = $("path-selector");
  selector.innerHTML = "";
  const render = (scenario) => {
    selector.querySelectorAll("button").forEach((button) => {
      const selected = button.dataset.scenario === scenario.label;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    $("path-caption").textContent = `${scenario.label.toUpperCase()} path · terminal ${formatMetricValue("p50", scenario.terminal, data.currency)}`;
    lineChart($("chart-scenario-path"), data.wealth.periods, [
      { name: scenario.label.toUpperCase(), color: "#b8d0d6", values: scenario.wealth },
    ]);
    timelineChart($("chart-scenario-regimes"), [
      { label: scenario.label.toUpperCase(), color: "#b8d0d6", states: scenario.regimes },
    ]);
  };
  scenarios.forEach((scenario) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.scenario = scenario.label;
    button.textContent = scenario.label === "median" ? "Median" : scenario.label.toUpperCase();
    button.addEventListener("click", () => render(scenario));
    selector.appendChild(button);
  });
  const initial = scenarios.find((scenario) => scenario.label === "median") || scenarios[0];
  if (initial) render(initial);
}

function renderPlanningCharts(data) {
  const goal = data.goal_curve;
  if (goal?.targets?.length) {
    lineChart(
      $("chart-goal-probability"),
      goal.targets,
      [{ name: "Success probability", color: "#a78bfa", values: goal.success_probability }],
      {
        numericX: true,
        xLabel: "Target wealth",
        yLabel: "Probability",
        xFormatter: (value) => compactCurrency(value, data.currency),
        yFormatter: (value) => pct(value, 0),
        valueFormatter: (value) => pct(value, 1),
        tooltipTitle: (value) => `Target ${formatMetricValue("target_wealth", value, data.currency)}`,
      },
    );
  }

  const horizons = data.rolling_horizons;
  if (horizons?.months?.length) {
    bandChart(
      $("chart-rolling-horizon"),
      horizons.months,
      {
        low: horizons.p05,
        median: horizons.median,
        high: horizons.p95,
        color: "#6fa58c",
      },
      {
        numericX: true,
        xLabel: "Holding period",
        yLabel: "Annualized return",
        xFormatter: (months) => `${Number(months) / 12}y`,
        yFormatter: (value) => pct(value, 0),
        valueFormatter: (value) => pct(value, 1),
        tooltipTitle: (months) => `${Number(months) / 12}-year rolling windows`,
      },
    );
    const sampled = Number(horizons.sample_paths || 0) < Number(horizons.total_paths || 0);
    $("rolling-horizon-caption").textContent =
      `Annualized returns across every available rolling window${sampled ? ` using ${Number(horizons.sample_paths).toLocaleString()} representative paths` : ""}.`;
  }

  const drawdown = data.drawdown_fan;
  if (drawdown?.periods?.length) {
    bandChart(
      $("chart-drawdown-fan"),
      drawdown.periods,
      { low: drawdown.p05, median: drawdown.median, high: drawdown.p95, color: "#d77b72" },
      {
        xLabel: "Simulation month",
        yLabel: "Below peak",
        yFormatter: (value) => pct(value, 0),
        valueFormatter: (value) => pct(value, 1),
        bandLabels: ["P05 deeper", "Median", "P95 shallower"],
      },
    );
  }

  const episodes = data.drawdown_episodes;
  if (episodes) {
    scatterChart(
      $("chart-drawdown-episodes"),
      (episodes.points || []).map((point) => ({
        x: point.duration_months,
        y: point.depth,
        color: point.recovered ? "#6fa58c" : "#d97706",
        label: `Path ${Number(point.path) + 1}`,
        regime: point.recovered ? "Recovered" : "Still underwater",
      })),
      "Duration",
      "Maximum depth",
      [
        { label: "Recovered", color: "#6fa58c" },
        { label: "Still underwater", color: "#d97706" },
      ],
      {
        xFormatter: (value) => `${value.toFixed(0)}m`,
        yFormatter: (value) => pct(value, 0),
        xValueFormatter: (value) => `${value.toFixed(0)} months`,
        yValueFormatter: (value) => pct(value, 1),
      },
    );
    $("drawdown-episode-caption").textContent =
      `Each point is one drawdown episode from ${Number(episodes.source_paths || 0).toLocaleString()} representative paths${episodes.sampled ? "; the display is deterministically bounded" : ""}.`;
  }

  const recovery = data.recovery_required;
  if (recovery?.periods?.length) {
    bandChart(
      $("chart-recovery-required"),
      recovery.periods,
      { low: recovery.p05, median: recovery.median, high: recovery.p95, color: "#d7a86e" },
      {
        xLabel: "Simulation month",
        yLabel: "Required gain",
        yFormatter: (value) => pct(value, 0),
        valueFormatter: (value) => pct(value, 1),
      },
    );
    $("recovery-required-caption").textContent =
      `Gain required to regain the previous peak from the current drawdown${recovery.capped ? `; extreme values are capped at ${pct(recovery.cap, 0)} for readability` : ""}.`;
  }
}

function renderDecisionAnalytics(data) {
  if (data.success) {
    lineChart($("chart-success"), data.success.periods, [
      { name: "Survival", color: "#a8b79b", values: data.success.survival },
      { name: "Preservation", color: "#b8d0d6", values: data.success.preservation },
      { name: "Profit", color: "#d7a86e", values: data.success.profit },
      { name: "Target", color: "#a78bfa", values: data.success.target },
    ]);
  }
  renderRepresentativeScenarios(data);
  renderMetricExplorer(data);
  renderPlanningCharts(data);
  const sequence = data.sequence_risk;
  $("sequence-risk-card").classList.toggle("hidden", !sequence);
  if (sequence) {
    $("sequence-risk-badge").textContent = `${pct(sequence.probability_negative_drag, 0)} negative drag`;
    $("sequence-risk-caption").textContent =
      `Median money-weighted return minus CAGR: ${sequence.median_drag >= 0 ? "+" : ""}${pct(sequence.median_drag, 2)}. ` +
      "Points below the equality relationship indicate that contribution timing reduced the investor's realized return.";
    scatterChart(
      $("chart-sequence-risk"),
      sequence.points.map((point) => ({ x: point.cagr, y: point.mwrr, color: point.drag < 0 ? "#d77b72" : "#6fa58c", label: `Drag ${pct(point.drag, 2)}`, regime: point.drag < 0 ? "Negative sequence drag" : "Positive sequence effect" })),
      "CAGR",
      "Money-weighted return",
      [
        { label: "Negative sequence drag", color: "#d77b72" },
        { label: "Positive sequence effect", color: "#6fa58c" },
      ],
    );
  }
}

function renderResults(data) {
  $("intro").classList.add("hidden");
  ["growth", "returns", "drawdowns", "correlations", "monthly"].forEach((name) => {
    const content = $(`${name}-content`);
    if (content) content.classList.remove("hidden");
  });
  document.querySelectorAll(".results-empty").forEach((el) => el.classList.add("hidden"));
  renderScenarioChips(state.lastSimPayload, data);
  renderMethodologyReport(data);
  renderParameterUncertainty(data);
  renderPersistence(data);
  $("macro-chart-title").textContent = data.model_kind === "hmm" ? "HMM states / macro history" : "Macro quadrants";
  renderMetricGrid("metric-grid", METRIC_FIELDS, data);
  renderMetricGrid("goal-metric-grid", GOAL_METRIC_FIELDS, data);
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
    `Time-weighted annualized return: ${pct(data.summary.annualized_return)} | ` +
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
  timelineChart($("chart-timeline"), [
    { label: "P05", color: "#f97316", states: data.regime_timelines.p05 },
    { label: "Median", color: "#3b82f6", states: data.regime_timelines.median },
    { label: "P95", color: "#10b981", states: data.regime_timelines.p95 },
  ]);
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

  renderMonthlyCalendar(data);
  renderMacroPaths(data);
  renderRegimeProbabilities(data);
  renderDecisionAnalytics(data);
  initializeResearchLab();
  $("lab-empty").classList.add("hidden");
  $("lab-content").classList.remove("hidden");

  const diagnostics = data.diagnostics;
  $("diagnostics-table").innerHTML = "<table><thead><tr>" + diagnostics.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead><tbody>" +
    diagnostics.rows.map((row) => "<tr>" + row.map((value) => `<td>${escapeHtml(fmtNumber(value))}</td>`).join("") + "</tr>").join("") + "</tbody></table>";
  $("warnings-box").textContent = data.warnings.length ? data.warnings.join("\n") : "";
  const validation = data.validation;
  $("validation-panel").classList.toggle("hidden", !validation);
  if (validation) {
    const summary = validation.summary;
    $("validation-summary").textContent =
      `Advantage vs Student-t: ${summary.advantage_vs_student_t_mean > 0 ? "+" : ""}${fmt(summary.advantage_vs_student_t_mean, 4)} log-score/period · ` +
      `HAC t-stat: ${fmt(summary.dm_t_statistic_vs_student_t, 2)} · ` +
      `Brier: ${fmt(summary.regime_brier_score, 3)} vs ${fmt(summary.benchmark_brier_score, 3)} benchmark · ` +
      `Switches/decade: ${fmt(summary.predicted_switches_per_decade, 1)} predicted vs ${fmt(summary.actual_switches_per_decade, 1)} observed · ` +
      `Duration log score: ${fmt(summary.duration_log_score_mean, 3)} · ` +
      `VaR breaches: ${pct(summary.var_95_breach_rate)} · ${summary.splits} splits.`;
    const preferredColumns = [
      "date", "actual_state", "predicted_state", "actual_switch",
      "predicted_switch_probability", "transition_brier_score",
      "transition_log_score", "current_regime_age", "completed_duration",
      "duration_log_score", "vintage_expected_duration",
    ];
    const indexes = preferredColumns
      .map((column) => validation.columns.indexOf(column))
      .filter((index) => index >= 0);
    renderTable("validation-table", {
      columns: indexes.map((index) => validation.columns[index]),
      rows: validation.rows.map((row) => indexes.map((index) => row[index])),
    });
  }
  state.diagnostics = diagnostics;
  $("diagnostics-empty").classList.add("hidden");
  $("diagnostics-content").classList.remove("hidden");
}

/* ---------- Handlers ---------- */

async function onLoad() {
  const message = $("load-message");
  showOverlay("Loading market data...");
  try {
    const payload = await gatherLoadPayload();
    await fillCsvPayload(payload);
    state.loadPayload = payload;
    state.results = null;
    state.lastSimPayload = null;
    const data = await postJSON("/api/load", payload);
    state.loadResult = data;
    setStatus(message, data.message);
    renderTickerChecklist(data.tickers, data.default_tickers);
    state.weights = {};
    if (state.pendingPortfolio) {
      const selected = new Set(state.pendingPortfolio.selected || []);
      document.querySelectorAll("#ticker-list input[type='checkbox']").forEach((checkbox) => {
        checkbox.checked = selected.has(checkbox.value);
      });
      state.weights = { ...state.pendingPortfolio.weights };
      state.pendingPortfolio = null;
    }
    assetColors = {};
    renderWeightEditor();
    renderTables(data);
    renderCoverage(data.coverage);
    const timing = data.data_timing || {};
    const timingStatus = $("data-timing-status");
    timingStatus.className = `data-timing-status ${timing.point_in_time ? "timing-good" : "timing-warning"}`;
    timingStatus.textContent = timing.point_in_time
      ? "✓ Point-in-time values aligned by historical availability date."
      : "! Latest-revised values: the configured release lag reduces timing bias but cannot remove revision look-ahead.";
    const macroLag = $("macro-lag");
    if (timing.availability_aligned) {
      if (!macroLag.disabled) macroLag.dataset.previousValue = macroLag.value;
      macroLag.value = "0";
      macroLag.disabled = true;
      $("macro-lag-hint").textContent = "Release dates are explicit, so no additional period lag is applied.";
    } else {
      macroLag.disabled = false;
      if (macroLag.dataset.previousValue) macroLag.value = macroLag.dataset.previousValue;
      $("macro-lag-hint").textContent = "Use one period with latest-revised data; availability-dated data uses its actual release calendar.";
    }
    renderSyntheticReport(data.synthetic);
    populatePresets(data.presets);
    $("portfolio-status").textContent = `${data.tickers.length} tickers available. Toggle assets and set weights.`;
    $("data-empty").classList.add("hidden");
    $("data-content").classList.remove("hidden");
    switchTab("tab-data");
    updateGuide();
  } catch (error) {
    setStatus(message, error.message, true);
  } finally {
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
  showOverlay(
    "Running analysis...",
    `${Number($("paths").value).toLocaleString()} paths · ${Number($("periods").value)} months`,
  );
  try {
    const currentLoadPayload = gatherLoadPayload();
    await fillCsvPayload(currentLoadPayload);
    const dataInputsChanged = JSON.stringify(currentLoadPayload) !== JSON.stringify(state.loadPayload);
    if (!state.loadResult || dataInputsChanged) {
      await onLoad();
      if (!state.loadResult) return;
      showOverlay(
        "Running analysis...",
        `${Number($("paths").value).toLocaleString()} paths · ${Number($("periods").value)} months`,
      );
    }
    const payload = gatherSimPayload();
    const data = await postJSON("/api/simulate", payload);
    state.lastSimPayload = payload;
    state.results = data;
    updateGuide();
    setStatus(message, data.message);
    notify("Analysis complete", "success");
    renderResults(data);
    switchTab("tab-growth");
    requestAnimationFrame(focusResults);
  } catch (error) {
    setStatus(message, error.message, true);
    notify(error.message, "error");
  } finally {
    hideOverlay();
    updateRunAvailability();
  }
}

async function onCompare() {
  const message = $("run-message");
  const button = $("compare-btn");
  button.disabled = true;
  showOverlay("Comparing distributions...", "Running Normal and Student-t scenarios with identical assumptions");
  try {
    const payload = gatherSimPayload();
    const data = await postJSON("/api/compare", payload);
    $("comparison-table").innerHTML = "<table><thead><tr>" + data.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead><tbody>" +
      data.rows.map((row) => "<tr>" + row.map((value) => `<td>${escapeHtml(fmtNumber(value))}</td>`).join("") + "</tr>").join("") + "</tbody></table>";
    setStatus(message, "Scenario comparison complete.");
    notify("Scenario comparison complete", "success");
    $("compare-content").classList.remove("hidden");
    document.querySelectorAll("#tab-compare .results-empty").forEach((el) => el.classList.add("hidden"));
    switchTab("tab-compare");
    requestAnimationFrame(focusResults);
  } catch (error) {
    setStatus(message, error.message, true);
    notify(error.message, "error");
  } finally {
    hideOverlay();
    updateRunAvailability();
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
  showOverlay("Exporting sampled paths...", "Replaying the original seeded path chunk and retaining up to 1,000 paths");
  try {
    const data = await postJSON("/api/wealth", { ...state.lastSimPayload, export_paths: 1000 });
    downloadCSV("wealth_paths_sample.csv", data.csv);
    const qualifier = data.sampled ? ` sampled from ${data.requested_paths.toLocaleString()}` : "";
    notify(`${data.exported_paths.toLocaleString()} wealth paths downloaded${qualifier}`, "success");
  } catch (error) {
    notify(error.message, "error");
  } finally {
    hideOverlay();
  }
}

/* ---------- Settings persistence ---------- */

const CONTROL_IDS = [
  "yahoo-tickers", "yahoo-start", "yahoo-end", "yahoo-proxies", "synthetic-seed",
  "synthetic-method", "synthetic-categories", "macro-vintage",
  "csv-growth", "csv-inflation", "csv-rate", "base-currency", "currency-map", "corr-blend",
  "growth-threshold", "growth-fixed", "inflation-threshold", "inflation-fixed",
  "macro-lag", "transition-uncertainty", "periods", "paths", "workers", "seed", "distribution",
  "degrees-of-freedom", "block-size", "rebalance", "cost-bps", "contribution", "withdrawal",
  "initial-value", "target-wealth",
  "risk-free", "annual-inflation", "expense-ratios", "leverage-multiple", "financing-rate",
  "financing-inflation-sensitivity", "maintenance-margin",
  "model-kind", "hmm-states", "threshold-window", "duration-model", "min-regime-duration",
  "regime-temperature", "regime-smoothing-window", "regime-hysteresis",
  "regime-confirmation-periods", "duration-prior-strength", "mean-prior-strength",
  "parameter-draws", "parameter-block-size",
  "macro-transition-weight", "dcc-alpha", "dcc-beta", "dcc-asymmetry",
];

function saveControls() {
  const data = { schemaVersion: 4 };
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
  data.probabilisticRegimes = $("probabilistic-regimes").checked;
  data.jointMacro = $("joint-macro").checked;
  data.dynamicCorrelation = $("dynamic-correlation").checked;
  data.corrTargets = gatherCorrelationTargets();
  localStorage.setItem("mcq-controls", JSON.stringify(data));
}

function applyControlSnapshot(data) {
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
  if (data.probabilisticRegimes !== undefined) $("probabilistic-regimes").checked = data.probabilisticRegimes;
  if (data.jointMacro !== undefined) $("joint-macro").checked = data.jointMacro;
  if (data.dynamicCorrelation !== undefined) $("dynamic-correlation").checked = data.dynamicCorrelation;
  document.querySelectorAll("#corr-sliders input[type='range']").forEach((slider) => {
    if (data.corrTargets && data.corrTargets[slider.dataset.state] !== undefined) slider.value = data.corrTargets[slider.dataset.state];
  });
  if (data.selected && data.weights) {
    state.pendingPortfolio = { selected: data.selected, weights: data.weights };
  }
}

function restoreControls() {
  try {
    const raw = localStorage.getItem("mcq-controls");
    if (raw) {
      const data = JSON.parse(raw);
      if (!data.schemaVersion && String(data.paths) === "10000" && String(data.periods) === "120") {
        data.paths = "100000";
      }
      if (Number(data.schemaVersion || 0) < 3 && String(data.paths) === "100000" && String(data.periods) === "60") {
        data.periods = "120";
      }
      if (Number(data.schemaVersion || 0) < 4 && String(data["min-regime-duration"]) === "3") {
        data["min-regime-duration"] = "5";
      }
      applyControlSnapshot(data);
    }
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

/* ---------- Research lab ---------- */

const LAB_PRESETS = {
  balanced: { SPY: 60, IEF: 30, GLD: 10 },
  defensive: { SPY: 30, IEF: 50, GLD: 20 },
  growth: { SPY: 80, IEF: 10, GLD: 10 },
};

function tickerBase(ticker) {
  return String(ticker).replace(/_SIM$/, "").replace(/SIM$/, "");
}

function setLabPreset(name) {
  const tickers = selectedTickers();
  const weights = {};
  if (name === "equal") {
    const equal = tickers.length ? 100 / tickers.length : 0;
    tickers.forEach((ticker) => { weights[ticker] = equal; });
  } else {
    const preset = LAB_PRESETS[name] || LAB_PRESETS.balanced;
    tickers.forEach((ticker) => { weights[ticker] = preset[tickerBase(ticker)] || 0; });
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0);
    if (!total && tickers.length) {
      tickers.forEach((ticker) => { weights[ticker] = 100 / tickers.length; });
    }
  }
  state.labWeights = weights;
  document.querySelectorAll(".lab-preset").forEach((button) => button.classList.toggle("active", button.dataset.preset === name));
  renderLabWeights();
}

function renderLabWeights() {
  const container = $("lab-weights");
  if (!container) return;
  const tickers = selectedTickers();
  container.innerHTML = "";
  tickers.forEach((ticker) => {
    const label = document.createElement("label");
    label.innerHTML = `<span>${escapeHtml(ticker)}</span>`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.max = "100";
    input.step = "1";
    input.value = Number(state.labWeights[ticker] || 0).toFixed(1);
    input.addEventListener("input", () => {
      state.labWeights[ticker] = Math.max(0, Number(input.value) || 0);
      updateLabWeightTotal();
    });
    const unit = document.createElement("small");
    unit.textContent = "%";
    label.append(input, unit);
    container.appendChild(label);
  });
  updateLabWeightTotal();
}

function updateLabWeightTotal() {
  const total = Object.values(state.labWeights).reduce((sum, value) => sum + (Number(value) || 0), 0);
  $("lab-weight-total").textContent = `${total.toFixed(1)}%`;
  $("lab-weight-total").style.color = total > 0 ? "var(--accent)" : "var(--danger)";
  $("portfolio-compare-btn").disabled = total <= 0;
}

function initializeResearchLab() {
  if (!$("lab-weights")) return;
  const tickers = selectedTickers();
  const sameUniverse = tickers.length && tickers.every((ticker) => Object.hasOwn(state.labWeights, ticker));
  if (!sameUniverse) setLabPreset("balanced");
  else renderLabWeights();
}

function pairedSummaryRows(portfolioA, portfolioB) {
  return [
    ["Median terminal wealth", "p50", true],
    ["Annualized return", "annualized_return", true],
    ["Annualized volatility", "annualized_volatility", false],
    ["Sharpe ratio", "sharpe_ratio", true],
    ["Probability of loss", "probability_of_loss", false],
    ["Target success", "goal_success_probability", true],
    ["Expected target shortfall", "expected_goal_shortfall", false],
    ["Risk of ruin", "risk_of_ruin", false],
    ["Omega ratio", "omega_ratio", true],
    ["P95 time underwater", "max_underwater_months_p95", false],
    ["Worst max drawdown", "max_drawdown_worst", false],
  ].map(([label, key, higherIsBetter]) => [label, portfolioA.summary[key], portfolioB.summary[key], portfolioB.summary[key] - portfolioA.summary[key], key, higherIsBetter]);
}

function renderPairedComparison(portfolioB) {
  const portfolioA = state.results;
  const count = Math.min(portfolioA.terminal.length, portfolioB.terminal.length);
  const differences = portfolioB.terminal.slice(0, count).map((value, index) => value - portfolioA.terminal[index]);
  const winRate = differences.filter((value) => value > 0).length / Math.max(count, 1);
  const totalPaths = Math.min(
    Number(portfolioA.reporting_sample?.total_paths || count),
    Number(portfolioB.reporting_sample?.total_paths || count),
  );
  const sampleNote = count < totalPaths
    ? ` · deterministic sample ${count.toLocaleString()} of ${totalPaths.toLocaleString()}`
    : "";
  $("paired-win-rate").textContent = `Portfolio B wins ${pct(winRate, 1)} of paired paths${sampleNote}`;
  const meanDifference = differences.reduce((sum, value) => sum + value, 0) / Math.max(count, 1);
  const variance = differences.reduce((sum, value) => sum + (value - meanDifference) ** 2, 0) / Math.max(count - 1, 1);
  const margin = 1.96 * Math.sqrt(variance / Math.max(count, 1));
  const conditionalLosses = differences.filter((value) => value < 0).map((value) => -value);
  const conditionalRegret = conditionalLosses.reduce((sum, value) => sum + value, 0) / Math.max(conditionalLosses.length, 1);
  const sortedA = [...portfolioA.terminal].sort((a, b) => a - b);
  const sortedB = [...portfolioB.terminal].sort((a, b) => a - b);
  const quantileChecks = Array.from({ length: 101 }, (_, index) => {
    const probability = index / 100;
    return quantileFromSorted(sortedB, probability) >= quantileFromSorted(sortedA, probability);
  });
  const dominanceShare = quantileChecks.filter(Boolean).length / quantileChecks.length;
  const evidence = [
    ["Mean paired difference", formatMetricValue("p50", meanDifference, portfolioA.currency)],
    ["95% Monte Carlo CI", `${formatMetricValue("p50", meanDifference - margin, portfolioA.currency)} to ${formatMetricValue("p50", meanDifference + margin, portfolioA.currency)}`],
    ["Median paired difference", formatMetricValue("p50", sampleQuantile(differences, 0.5), portfolioA.currency)],
    ["B higher by quantile", pct(dominanceShare, 0)],
    ["Conditional regret", formatMetricValue("p50", conditionalRegret, portfolioA.currency)],
  ];
  $("paired-evidence").innerHTML = evidence.map(([label, value]) =>
    `<div class="uncertainty-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`
  ).join("");
  const rows = pairedSummaryRows(portfolioA, portfolioB).map(([label, a, b, difference, key, higherIsBetter]) => {
    const percentKey = PERCENT_METRICS.has(key);
    const formatter = percentKey ? (value) => pct(value, 2) : (value) => formatMetricValue(key, value, portfolioA.currency);
    const differenceText = `${difference >= 0 ? "+" : ""}${formatter(difference)}`;
    const favorable = higherIsBetter ? difference >= 0 : difference <= 0;
    return `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(formatter(a))}</td><td>${escapeHtml(formatter(b))}</td><td class="${favorable ? "difference-positive" : "difference-negative"}">${escapeHtml(differenceText)}</td></tr>`;
  });
  $("paired-summary").innerHTML = `<table><thead><tr><th>Metric</th><th>Portfolio A</th><th>Portfolio B</th><th>B − A</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
  histChart($("chart-paired-difference"), differences, 45, "Terminal wealth difference", "#a8b79b");
  lineChart($("chart-paired-wealth"), portfolioA.wealth.periods, [
    { name: "A P05", color: "#78909c", values: portfolioA.wealth.p05 },
    { name: "A median", color: "#b8d0d6", values: portfolioA.wealth.median },
    { name: "A P95", color: "#d7e4e7", values: portfolioA.wealth.p95 },
    { name: "B P05", color: "#a56f3f", values: portfolioB.wealth.p05 },
    { name: "B median", color: "#d7a86e", values: portfolioB.wealth.median },
    { name: "B P95", color: "#f0cf9f", values: portfolioB.wealth.p95 },
  ]);
  const percentileLabels = Array.from({ length: 51 }, (_, index) => index * 2);
  lineChart(
    $("chart-paired-quantiles"),
    percentileLabels,
    [
      { name: "Portfolio A", color: "#b8d0d6", values: percentileLabels.map((value) => quantileFromSorted(sortedA, value / 100)) },
      { name: "Portfolio B", color: "#d7a86e", values: percentileLabels.map((value) => quantileFromSorted(sortedB, value / 100)) },
    ],
    {
      numericX: true,
      xLabel: "Terminal percentile",
      yLabel: "Terminal wealth",
      xFormatter: (value) => `${value}%`,
      yFormatter: (value) => compactCurrency(value, portfolioA.currency),
      valueFormatter: (value) => formatMetricValue("p50", value, portfolioA.currency),
      tooltipTitle: (value) => `${value}th percentile`,
    },
  );
  $("paired-results").classList.remove("hidden");
  state.pairedResults = portfolioB;
}

async function onPortfolioCompare() {
  if (!state.results || !state.lastSimPayload) return;
  const button = $("portfolio-compare-btn");
  const status = $("portfolio-compare-status");
  button.disabled = true;
  showOverlay("Running paired portfolio...", "Reusing the same seed and market-path assumptions");
  try {
    const payload = { ...state.lastSimPayload, weights: { ...state.labWeights }, walk_forward: false };
    const data = await postJSON("/api/simulate", payload);
    renderPairedComparison(data);
    setStatus(status, "Paired comparison complete. Both portfolios used identical seeded market paths.");
    notify("Paired comparison complete", "success");
  } catch (error) {
    setStatus(status, error.message, true);
    notify(error.message, "error");
  } finally {
    hideOverlay();
    updateLabWeightTotal();
  }
}

async function onRebalancingSensitivity() {
  if (!state.lastSimPayload) return;
  const button = $("rebalance-sensitivity-btn");
  const status = $("rebalance-status");
  button.disabled = true;
  const schedules = [["Monthly", "monthly"], ["Quarterly", "quarterly"], ["Annual", "annual"], ["Buy and hold", "buy_hold"]];
  const results = [];
  showOverlay("Analyzing rebalancing...", "Running paired schedule 1 of 4");
  try {
    for (let index = 0; index < schedules.length; index += 1) {
      const [label, rebalance] = schedules[index];
      $("overlay-stage").textContent = `Running ${label.toLowerCase()} schedule · ${index + 1} of ${schedules.length}`;
      const payload = {
        ...state.lastSimPayload,
        rebalance,
        cost_bps: rebalance === "buy_hold" ? 0 : state.lastSimPayload.cost_bps,
        paths: Math.min(Number(state.lastSimPayload.paths), 1_000),
        parameter_draws: 0,
        walk_forward: false,
        workers: 1,
      };
      const data = await postJSON("/api/simulate", payload);
      results.push([label, data.summary.p50, data.summary.annualized_return, data.summary.annualized_volatility, data.summary.max_drawdown_mean, data.summary.sharpe_ratio]);
    }
    $("rebalance-sensitivity-table").innerHTML = "<table><thead><tr><th>Schedule</th><th>Median wealth</th><th>Return</th><th>Volatility</th><th>Mean drawdown</th><th>Sharpe</th></tr></thead><tbody>" +
      results.map((row) => `<tr><td><strong>${row[0]}</strong></td><td>${formatMetricValue("p50", row[1], state.results.currency)}</td><td>${pct(row[2])}</td><td>${pct(row[3])}</td><td>${pct(row[4])}</td><td>${fmt(row[5], 2)}</td></tr>`).join("") + "</tbody></table>";
    setStatus(status, "Sensitivity sweep complete using identical seeds and up to 1,000 paths per schedule.");
  } catch (error) {
    setStatus(status, error.message, true);
  } finally {
    hideOverlay();
    button.disabled = false;
  }
}

function scenarioLibrary() {
  try { return JSON.parse(localStorage.getItem("mcq-scenario-library") || "[]"); }
  catch (error) { return []; }
}

function captureScenarioSnapshot() {
  saveControls();
  const controls = JSON.parse(localStorage.getItem("mcq-controls") || "{}");
  controls.csvEnabled = false;
  controls.selected = selectedTickers();
  controls.weights = gatherWeights();
  controls.savedAt = new Date().toISOString();
  return controls;
}

function refreshScenarioLibrary() {
  const select = $("saved-scenario-select");
  const selected = select.value;
  select.innerHTML = '<option value="">Choose a saved scenario</option>';
  scenarioLibrary().forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.name;
    select.appendChild(option);
  });
  select.value = selected;
}

function saveScenarioToLibrary() {
  const name = $("scenario-name").value.trim() || `Scenario ${new Date().toLocaleDateString()}`;
  const library = scenarioLibrary();
  library.unshift({ id: String(Date.now()), name, data: captureScenarioSnapshot() });
  localStorage.setItem("mcq-scenario-library", JSON.stringify(library.slice(0, 20)));
  refreshScenarioLibrary();
  $("saved-scenario-select").value = library[0].id;
  notify(`Saved “${name}” locally`, "success");
}

function encodeScenario(data) {
  const bytes = new TextEncoder().encode(JSON.stringify(data));
  let binary = "";
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function decodeScenario(encoded) {
  const normalized = encoded.replaceAll("-", "+").replaceAll("_", "/");
  const binary = atob(normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "="));
  return JSON.parse(new TextDecoder().decode(Uint8Array.from(binary, (character) => character.charCodeAt(0))));
}

function applyScenarioSnapshot(data, reload = true) {
  applyControlSnapshot(data);
  $("csv-enabled").checked = false;
  renderTickerComposer();
  syncDataSourceUI();
  updateMethodologyControls();
  saveControls();
  if (reload) onLoad();
}

async function shareScenario() {
  const encoded = encodeScenario(captureScenarioSnapshot());
  const url = `${location.href.split("?")[0]}?scenario=${encoded}`;
  try {
    await navigator.clipboard.writeText(url);
    notify("Reproducible scenario link copied", "success");
  } catch (error) {
    window.prompt("Copy this reproducible scenario link:", url);
  }
}

function loadSavedScenario() {
  const id = $("saved-scenario-select").value;
  const entry = scenarioLibrary().find((item) => item.id === id);
  if (!entry) return notify("Choose a saved scenario first.", "error");
  applyScenarioSnapshot(entry.data);
  notify(`Loaded “${entry.name}”`, "success");
}

function deleteSavedScenario() {
  const id = $("saved-scenario-select").value;
  if (!id) return;
  localStorage.setItem("mcq-scenario-library", JSON.stringify(scenarioLibrary().filter((item) => item.id !== id)));
  refreshScenarioLibrary();
  notify("Saved scenario deleted", "success");
}

function restoreScenarioFromUrl() {
  const encoded = new URLSearchParams(location.search).get("scenario");
  if (!encoded) return;
  try {
    applyScenarioSnapshot(decodeScenario(encoded), false);
    notify("Shared scenario loaded", "success");
  } catch (error) {
    notify("The shared scenario link is invalid.", "error");
  }
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
  const macroTransition = $("macro-transition-weight");
  macroTransition.addEventListener("input", () => { $("macro-transition-weight-output").textContent = Number(macroTransition.value).toFixed(2); });

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
    const hasFiles = $("csv-prices").files.length > 0 || $("csv-macro").files.length > 0;
    if (hasFiles) $("csv-enabled").checked = true;
    syncDataSourceUI();
    toggleSourceGroups();
  }

  $("csv-enabled").addEventListener("change", () => {
    syncDataSourceUI();
    toggleSourceGroups();
  });
  $("csv-prices").addEventListener("change", syncCsvEnabled);
  $("csv-macro").addEventListener("change", syncCsvEnabled);

  document.querySelectorAll(".tab-btn").forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

  $("run-btn").addEventListener("click", onRun);
  $("compare-btn").addEventListener("click", onCompare);
  $("download-summary").addEventListener("click", onDownloadSummary);
  $("download-diagnostics").addEventListener("click", onDownloadDiagnostics);
  $("download-wealth").addEventListener("click", onDownloadWealth);
  $("download-json").addEventListener("click", onDownloadJson);
  $("preset-apply").addEventListener("click", applyPreset);
  $("preset-select").addEventListener("change", () => { $("preset-apply").disabled = !$("preset-select").value; });
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("equalize-btn").addEventListener("click", equalizeWeights);
  $("reset-btn").addEventListener("click", resetControls);
  $("edit-scenario").addEventListener("click", editScenario);
  $("portfolio-compare-btn").addEventListener("click", onPortfolioCompare);
  $("rebalance-sensitivity-btn").addEventListener("click", onRebalancingSensitivity);
  document.querySelectorAll(".lab-preset").forEach((button) => button.addEventListener("click", () => setLabPreset(button.dataset.preset)));
  $("save-scenario-btn").addEventListener("click", saveScenarioToLibrary);
  $("share-scenario-btn").addEventListener("click", shareScenario);
  $("load-scenario-btn").addEventListener("click", loadSavedScenario);
  $("delete-scenario-btn").addEventListener("click", deleteSavedScenario);

  document.addEventListener("input", () => { saveControls(); updateMethodologyControls(); });
  document.addEventListener("change", () => { saveControls(); updateMethodologyControls(); });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !$("run-btn").disabled) {
      event.preventDefault();
      onRun();
    }
  });

  restoreControls();
  setupMarketDataExperience();
  restoreScenarioFromUrl();
  refreshScenarioLibrary();
  setupExperience();
  $("transition-uncertainty-output").textContent = Number($("transition-uncertainty").value).toFixed(2);
  $("macro-transition-weight-output").textContent = Number($("macro-transition-weight").value).toFixed(2);
  toggleSourceGroups();
  updateMethodologyControls();
  applyTheme(localStorage.getItem("mcq-theme") || "light");

  fetch("/api/health")
    .then((response) => response.json())
    .then(() => {
      const badge = $("connection");
      badge.textContent = "connected";
      badge.className = "badge badge-ok";
      if (new URLSearchParams(window.location.search).has("skipAutoLoad")) {
        setStatus($("load-message"), "Connected. Choose custom files or run to load market data.");
        updateRunAvailability();
      } else {
        onLoad();
      }
    })
    .catch(() => {
      const badge = $("connection");
      badge.textContent = "backend offline";
      badge.className = "badge badge-error";
    });
}

document.addEventListener("DOMContentLoaded", init);
