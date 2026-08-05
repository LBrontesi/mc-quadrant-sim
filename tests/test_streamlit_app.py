import pytest
from pathlib import Path

streamlit = pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_streamlit_app_loads_demo_data():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception
    assert at.subheader[0].value == "Portfolio"
    assert any(button.label == "Run Simulation" for button in at.button)


def test_streamlit_app_full_simulation_flow():
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    at.multiselect[0].set_value(["SPY", "IEF", "GLD", "DBC"])
    at.run()
    at.button[0].click()
    at.run()
    assert not at.exception
    assert len(at.metric) >= 10
    assert any("Simulation complete" in message.value for message in at.success)
