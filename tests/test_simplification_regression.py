import hashlib
import json
from pathlib import Path

import numpy as np

import mc_quadrants.simulation as simulation_module
from mc_quadrants.simulation import simulate_returns
from tests.test_backfill import _dataset, _run
from tests.test_simulation import _calibrated_model

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "simplification_baseline.json"


def _sha256(values: np.ndarray) -> str:  # Hash numeric array bytes so exact deterministic output can be checked.
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def test_simplification_matches_pre_change_snapshot(monkeypatch):  # Compare representative outputs with the pre-change baseline.
    expected = json.loads(SNAPSHOT_PATH.read_text())["outputs"]

    def disable_native_regime(*args, **kwargs):  # Force the portable Python regime generator for stable comparison.
        return None

    def disable_native_returns(**kwargs):  # Force the portable Python return generator for stable comparison.
        return None

    monkeypatch.setattr(simulation_module, "simulate_regime_macro_native", disable_native_regime)
    monkeypatch.setattr(simulation_module, "simulate_parametric_native", disable_native_returns)
    result = simulate_returns(_calibrated_model(), periods=6, paths=10, random_seed=1, return_regime_codes=True)
    stitched, report = _run(_dataset())
    actual = {
        "returns_shape": list(result.returns.shape),
        "returns_sha256": _sha256(result.returns),
        "regimes_shape": list(result.regimes.shape),
        "regimes_sha256": _sha256(result.regimes),
        "backfill_shape": list(stitched.shape),
        "backfill_values_sha256": _sha256(stitched.to_numpy()),
        "backfill_report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest(),
    }
    assert actual == expected
