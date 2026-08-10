import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import web_app
from web_app import WEB_DIR, WebHandler


def test_web_assets_are_served_from_expected_directory():
    web_dir = Path(WEB_DIR)
    html = (web_dir / "index.html").read_text(encoding="utf-8")

    assert (web_dir / "index.html").is_file()
    assert (web_dir / "app.js").is_file()
    assert (web_dir / "api-client.js").is_file()
    assert (web_dir / "resource-planner.js").is_file()
    assert (web_dir / "style.css").is_file()
    assert 'src="logo.jpg"' in html
    assert 'src="/logo.jpg"' not in html


def test_web_ui_exposes_shared_methodology_controls():
    html = (Path(WEB_DIR) / "index.html").read_text(encoding="utf-8")
    javascript = (Path(WEB_DIR) / "app.js").read_text(encoding="utf-8")

    for control in ("model-kind", "hmm-states", "duration-model", "threshold-window", "garch", "walk-forward"):
        assert f'id="{control}"' in html
    for control in (
        "macro-vintage",
        "probabilistic-regimes",
        "parameter-draws",
        "joint-macro",
        "dynamic-correlation",
        "methodology-badges",
        "parameter-bands",
    ):
        assert f'id="{control}"' in html
    for control in ("synthetic-method", "synthetic-categories", "synthetic-report"):
        assert f'id="{control}"' in html
    assert "cash_flow_adjusted_annualized_return" in javascript
    assert "validation-panel" in html
    assert "guide-status" in html
    assert "updateGuide" in javascript
    assert "renderSyntheticReport" in javascript
    assert "renderMethodologyReport" in javascript
    assert "renderParameterUncertainty" in javascript
    assert "renderMacroPaths" in javascript
    assert 'risk_free_rate: Number($("risk-free").value)' in javascript
    assert 'type="module" src="app.js?' in html
    assert 'id="resource-card"' not in html
    assert 'id="hero-title"' in html
    assert 'id="quadrant-stage"' in html
    assert 'id="scroll-progress-bar"' in html
    for control in (
        "data-source-live",
        "data-source-csv",
        "market-ticker-chips",
        "ticker-add",
        "history-ranges",
        "market-universe-summary",
        "macro-vintage-explainer",
        "chart-success",
        "path-selector",
        "metric-explorer-select",
        "sequence-risk-card",
        "tab-lab",
        "portfolio-compare-btn",
        "rebalance-sensitivity-btn",
        "save-scenario-btn",
        "share-scenario-btn",
    ):
        assert f'id="{control}"' in html
    assert "setupExperience" in javascript
    assert "setupMarketDataExperience" in javascript
    assert "renderTickerComposer" in javascript
    assert "enhanceSelects" in javascript
    assert "renderDecisionAnalytics" in javascript
    assert "onPortfolioCompare" in javascript
    assert "onRebalancingSensitivity" in javascript
    assert "captureScenarioSnapshot" in javascript
    assert 'id="paths" value="100000"' in html
    assert 'id="periods" value="120"' in html
    assert '<button id="run-btn" class="btn btn-primary btn-lg" disabled>Run analysis</button>' in html
    assert '$("run-btn").textContent = "Run analysis"' in javascript
    assert "Run ${paths.toLocaleString()} paths" not in javascript
    assert 'String(data.paths) === "10000"' in javascript
    assert "Estimated memory" not in javascript
    assert "MEMORY_LIMIT_MB" not in javascript
    assert 'setAttribute("height", "auto")' not in javascript
    assert 'setAttribute("height", String(height))' in javascript


def test_web_backend_serves_health_and_load_endpoints():
    server = ThreadingHTTPServer(("127.0.0.1", 0), WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base_url}/api/health") as response:
            health = json.loads(response.read())
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert response.headers["Cache-Control"] == "no-store"
        assert health["ok"] is True

        with urlopen(f"{base_url}/app.js") as response:
            assert response.headers["Cache-Control"] == "no-cache"

        request = Request(
            f"{base_url}/api/load",
            data=json.dumps(
                {
                    "source": "csv",
                    "csv_prices": (
                        "Date,SPY,IEF\n2020-01-31,100,50\n2020-02-29,110,51\n"
                        "2020-03-31,120,52\n2020-04-30,115,53\n"
                    ),
                    "csv_macro": (
                        "Date,growth,inflation\n2020-01-31,2.0,1.0\n2020-02-29,2.5,4.0\n"
                        "2020-03-31,-1.0,4.5\n2020-04-30,-1.5,1.2\n"
                    ),
                    "asset_input": "Price levels",
                    "monthly": True,
                    "growth_col": "growth",
                    "inflation_col": "inflation",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            load = json.loads(response.read())
        assert load["ok"] is True
        assert load["tickers"]
        assert load["coverage"]

        try:
            urlopen(f"{base_url}/%2e%2e/pyproject.toml")
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("Static server allowed a path outside WEB_DIR")
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_web_backend_rejects_oversized_or_non_object_json(monkeypatch):
    monkeypatch.setattr(web_app, "MAX_REQUEST_BYTES", 1)
    server = ThreadingHTTPServer(("127.0.0.1", 0), WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(
            f"{base_url}/api/load",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request)
        assert exc_info.value.code == 413

        monkeypatch.setattr(web_app, "MAX_REQUEST_BYTES", 1024)
        request = Request(
            f"{base_url}/api/load",
            data=b"[]",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request)
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        thread.join(timeout=5)
