import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/app.js", import.meta.url), "utf8");
const functions = [
  source.slice(source.indexOf("function resolvePreset("), source.indexOf("function populatePresets(")),
  source.slice(source.indexOf("async function onLoad("), source.indexOf("function gatherSimPayload(")),
  source.slice(source.indexOf("function tickerBase("), source.indexOf("function setLabPreset(")),
].join("\n");

function setup(tickers = ["SPY", "IEF", "GLD"], sourceKind = "yahoo") {
  const preset = { name: "Classic 60/40", weights: { SPY: 60, IEF: 40 } };
  const controls = new Map();
  const element = (id) => {
    if (!controls.has(id)) controls.set(id, {
      value: "", disabled: false, dataset: {}, classList: { add() {}, remove() {} }, dispatchEvent() {},
    });
    return controls.get(id);
  };
  element("preset-select").value = preset.name;
  element("yahoo-tickers").value = tickers.join(", ");
  let boxes = tickers.map((value) => ({ value, checked: value === "GLD" }));
  const state = { loadResult: { tickers, presets: [preset] }, weights: { GLD: 100 }, pendingPortfolio: null };
  const messages = [];
  const context = vm.createContext({
    state, $: element, Event: class {},
    document: { querySelectorAll: () => boxes },
    activeSource: () => sourceKind,
    parseMarketTickers: () => element("yahoo-tickers").value.split(/,\s*/),
    setMarketTickers: (values) => { element("yahoo-tickers").value = [...new Set(values)].join(", "); },
    selectedTickers: () => boxes.filter((box) => box.checked).map((box) => box.value),
    gatherWeights: () => ({ ...state.weights }),
    gatherLoadPayload: () => ({ tickers: element("yahoo-tickers").value }),
    fillCsvPayload() {}, showOverlay() {}, hideOverlay() {}, setStatus() {},
    renderTickerChecklist: (values, defaults) => { boxes = values.map((value) => ({ value, checked: defaults.includes(value) })); },
    renderWeightEditor() {}, renderTables() {}, renderCoverage() {}, renderSyntheticReport() {},
    populatePresets() {}, switchTab() {}, updateGuide() {}, saveControls() {},
    notify: (...args) => messages.push(args),
    postJSON: async () => ({ tickers: ["SPY", "IEF", "GLD"], default_tickers: ["SPY", "IEF", "GLD"] }),
    assetColors: {},
  });
  vm.runInContext(functions, context);
  return { context, state, element, messages, selected: () => boxes.filter((box) => box.checked).map((box) => box.value) };
}

test("choosing a preset replaces selection and uses its exact weights", async () => {
  const app = setup();
  await app.context.applyPreset();
  assert.deepEqual(app.selected(), ["SPY", "IEF"]);
  assert.equal(JSON.stringify(app.state.weights), '{"SPY":60,"IEF":40}');
});

test("synthetic aliases are supported without duplicating real tickers", async () => {
  const app = setup(["SPY_SIM", "SPY", "IEFSIM", "GLD"]);
  await app.context.applyPreset();
  assert.deepEqual(app.selected(), ["SPY", "IEFSIM"]);
  assert.equal(JSON.stringify(app.state.weights), '{"SPY":60,"IEFSIM":40}');
});

test("missing Yahoo assets are fetched and the full preset is applied", async () => {
  const app = setup(["GLD"]);
  await app.context.applyPreset();
  assert.equal(app.element("yahoo-tickers").value, "GLD, SPY, IEF");
  assert.deepEqual(app.selected(), ["SPY", "IEF"]);
  assert.equal(JSON.stringify(app.state.weights), '{"SPY":60,"IEF":40}');
});

test("missing CSV assets preserve the allocation instead of applying a partial preset", async () => {
  const app = setup(["SPY", "GLD"], "csv");
  await app.context.applyPreset();
  assert.deepEqual(app.selected(), ["GLD"]);
  assert.deepEqual(app.state.weights, { GLD: 100 });
  assert.match(app.messages[0][0], /needs IEF/);
});

for (const failure of ["network", "missing ticker"]) {
  test(`${failure} during automatic loading preserves allocation and universe`, async () => {
    const app = setup(["GLD"]);
    const oldResult = app.state.loadResult;
    app.context.postJSON = async () => {
      if (failure === "network") throw new Error("Offline");
      return { tickers: ["SPY", "GLD"], default_tickers: ["SPY"] };
    };
    await app.context.applyPreset();
    assert.equal(app.state.loadResult, oldResult);
    assert.deepEqual(app.selected(), ["GLD"]);
    assert.deepEqual(app.state.weights, { GLD: 100 });
    assert.equal(app.element("yahoo-tickers").value, "GLD");
    assert.equal(app.element("preset-select").value, "");
    assert.equal(app.element("preset-select").disabled, false);
  });
}

test("data refresh retains the chosen portfolio", async () => {
  const app = setup();
  await app.context.applyPreset();
  await app.context.onLoad();
  assert.deepEqual(app.selected(), ["SPY", "IEF"]);
  assert.equal(JSON.stringify(app.state.weights), '{"SPY":60,"IEF":40}');
});
