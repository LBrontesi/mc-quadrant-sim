from pathlib import Path

import pytest

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
    assert any("Out-of-sample validation" in caption.value for caption in at.caption)


def test_streamlit_app_hmm_model_flow():
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    at.multiselect[0].set_value(["SPY", "IEF", "GLD", "DBC"])
    at.run()
    at.sidebar.selectbox(key="model_kind").set_value("hmm")
    at.sidebar.selectbox(key="duration_model").set_value("semi_markov")
    at.sidebar.number_input(key="hmm_states").set_value(3)
    at.run()
    at.button[0].click()
    at.run()
    assert not at.exception
    assert any("Simulation complete" in message.value for message in at.success)
    assert any("HMM fitted" in warning.value for warning in at.warning)
