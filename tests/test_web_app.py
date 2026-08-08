import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from web_app import WEB_DIR, WebHandler


def test_web_assets_are_served_from_expected_directory():
    web_dir = Path(WEB_DIR)

    assert (web_dir / "index.html").is_file()
    assert (web_dir / "app.js").is_file()
    assert (web_dir / "style.css").is_file()


def test_web_ui_exposes_shared_methodology_controls():
    html = (Path(WEB_DIR) / "index.html").read_text(encoding="utf-8")
    javascript = (Path(WEB_DIR) / "app.js").read_text(encoding="utf-8")

    for control in ("model-kind", "hmm-states", "duration-model", "threshold-window", "garch", "walk-forward"):
        assert f'id="{control}"' in html
    for control in ("synthetic-method", "synthetic-categories", "synthetic-report"):
        assert f'id="{control}"' in html
    assert "cash_flow_adjusted_annualized_return" in javascript
    assert "validation-panel" in html
    assert "guide-status" in html
    assert "updateGuide" in javascript
    assert "renderSyntheticReport" in javascript
    assert 'risk_free_rate: Number($("risk-free").value)' in javascript


def test_web_backend_serves_health_and_load_endpoints():
    server = ThreadingHTTPServer(("127.0.0.1", 0), WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base_url}/api/health") as response:
            health = json.loads(response.read())
        assert health["ok"] is True

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
    finally:
        server.shutdown()
        thread.join(timeout=5)
