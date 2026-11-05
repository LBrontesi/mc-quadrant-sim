"""Background lifecycle, restart, cancellation, and real seeded API checks."""

import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import web_app
from mc_quadrants.jobs import JobsBusy, JobStore
from mc_quadrants.progress import JobCancelled, check_cancelled, progress_context, report_progress


def finished(store, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job["status"] in {"completed", "cancelled", "failed", "interrupted"}:
            return job
        time.sleep(0.01)
    pytest.fail("Job did not finish")


def test_cancel_stops_at_checkpoint_and_releases_capacity(tmp_path):
    entered, release = threading.Event(), threading.Event()
    continued = []

    def work(payload):
        report_progress("Calibrating", 1, 8)
        entered.set()
        assert release.wait(5)
        check_cancelled()
        continued.append(True)
        return {"ok": True}

    semaphore = threading.BoundedSemaphore(1)
    store = JobStore(tmp_path, {"simulate": work}, semaphore)
    job = store.submit("simulate", {})
    assert entered.wait(5)
    assert store.get(job["id"])["progress"]["completed"] == 1
    with pytest.raises(JobsBusy):
        store.submit("simulate", {})
    assert store.cancel(job["id"])["status"] == "cancelling"
    release.set()
    assert finished(store, job["id"])["status"] == "cancelled"
    assert not continued
    assert semaphore.acquire(timeout=2)
    semaphore.release()
    with pytest.raises(ValueError, match="no completed result"):
        store.result(job["id"])
    assert store.cancel(job["id"])["status"] == "cancelled"


def test_idempotent_submit_persists_result_and_retry_recovers_restart(tmp_path):
    calls = []

    def work(payload):
        calls.append(payload)
        return {"ok": True, "answer": payload["value"]}

    store = JobStore(tmp_path, {"simulate": work}, threading.BoundedSemaphore(1))
    job = store.submit("simulate", {"value": 42}, job_id="a" * 32)
    assert finished(store, job["id"])["status"] == "completed"
    assert store.submit("simulate", {"value": 42}, job_id=job["id"])["id"] == job["id"]
    assert len(calls) == 1
    with pytest.raises(ValueError, match="different request"):
        store.submit("simulate", {"value": 43}, job_id=job["id"])
    recovered = JobStore(tmp_path, {"simulate": work}, threading.BoundedSemaphore(1))
    assert recovered.result(job["id"])["result"]["answer"] == 42
    assert recovered.result(job["id"])["payload"] == {"value": 42}
    assert "payload" not in recovered.get(job["id"])
    unfinished = {**store.jobs[job["id"]], "id": "b" * 32, "status": "running"}
    (tmp_path / f"{unfinished['id']}.json").write_text(json.dumps(unfinished))
    restarted = JobStore(tmp_path, {"simulate": work}, threading.BoundedSemaphore(1))
    assert restarted.get(unfinished["id"])["status"] == "interrupted"
    retry = restarted.retry(unfinished["id"], retry_id="c" * 32)
    assert finished(restarted, retry["id"])["status"] == "completed"
    assert restarted.retry(unfinished["id"], retry_id="c" * 32)["id"] == retry["id"]
    assert len(calls) == 2


def test_failed_job_is_recoverable_and_unexpected_errors_are_private(tmp_path):
    def work(payload):
        raise RuntimeError("private server details")

    store = JobStore(tmp_path, {"simulate": work}, threading.BoundedSemaphore(1))
    job = store.submit("simulate", {})
    result = finished(store, job["id"])
    assert result["status"] == "failed"
    assert "private" not in result["error"]
    with pytest.raises(ValueError):
        store.submit("unknown", {})
    with pytest.raises(ValueError):
        store.submit("simulate", {}, job_id="../../secrets")
    with pytest.raises(KeyError):
        store.result("../../secrets")


def test_retention_removes_both_request_and_result(tmp_path):
    store = JobStore(tmp_path, {"simulate": lambda p: p}, threading.BoundedSemaphore(1), max_jobs=1)
    first = store.submit("simulate", {"ok": True})
    finished(store, first["id"])
    # Waiting on the semaphore also ensures the finishing worker released it.
    assert store.semaphore.acquire(timeout=2)
    store.semaphore.release()
    second = store.submit("simulate", {"ok": True})
    finished(store, second["id"])
    assert not (tmp_path / f"{first['id']}.json").exists()
    assert not (tmp_path / f"{first['id']}.result").exists()
    assert not (tmp_path / f"{first['id']}.request").exists()
    with pytest.raises(KeyError):
        store.get(first["id"])


def test_damaged_saved_result_becomes_retryable(tmp_path):
    handlers = {"simulate": lambda p: {"ok": True}}
    store = JobStore(tmp_path, handlers, threading.BoundedSemaphore(1))
    job = store.submit("simulate", {"large_csv": "data"})
    finished(store, job["id"])
    metadata = json.loads((tmp_path / f"{job['id']}.json").read_text())
    assert "payload" not in metadata
    result = tmp_path / f"{job['id']}.result"
    result.write_text("broken")
    with pytest.raises(ValueError, match="could not be read"):
        store.result(job["id"])
    assert store.get(job["id"])["status"] == "failed"
    result.unlink()
    recovered = JobStore(tmp_path, handlers, threading.BoundedSemaphore(1))
    retry = recovered.retry(job["id"])
    assert finished(recovered, retry["id"])["status"] == "completed"


def test_progress_context_is_isolated_and_restored():
    messages = []
    with progress_context(messages.append, lambda: False):
        report_progress("outer")
        with pytest.raises(JobCancelled), progress_context(messages.append, lambda: True):
            report_progress("cancelled")
        thread = threading.Thread(target=report_progress, args=("unrelated",))
        thread.start()
        thread.join()
        report_progress("outer again")
    report_progress("no context")
    assert [m["stage"] for m in messages] == ["outer", "outer again"]


def test_parallel_cancellation_does_not_schedule_remaining_chunks(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import mc_quadrants.pipeline as pipeline

    started = []
    both_started, release, cancel = threading.Event(), threading.Event(), threading.Event()
    stopped = []

    def chunk(spec):
        started.append(spec)
        if len(started) == 2:
            both_started.set()
        assert release.wait(5)
        return spec

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", lambda **kw: ThreadPoolExecutor(max_workers=kw["max_workers"]))
    monkeypatch.setattr(pipeline, "_run_chunk", chunk)

    def consume():
        try:
            with progress_context(lambda p: None, cancel.is_set):
                list(pipeline._parallel_chunks({}, list(range(20)), 2))
        except JobCancelled:
            stopped.append(True)

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        assert both_started.wait(5)
        cancel.set()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert stopped and sorted(started) == [0, 1]


def test_rebalancing_job_preserves_schedule_settings_and_pairing(monkeypatch):
    import mc_quadrants.api as api

    calls = []

    def simulate(payload):
        calls.append(payload)
        return {"currency": "EUR", "summary": {key: 1 for key in (
            "p50", "annualized_return", "annualized_volatility", "max_drawdown_mean", "sharpe_ratio")}}

    monkeypatch.setattr(api, "build_simulate_response", simulate)
    result = api.build_rebalance_response({"paths": 20000, "cost_bps": 12, "random_seed": 42})
    assert [p["rebalance"] for p in calls] == ["monthly", "quarterly", "annual", "buy_hold"]
    assert [p["cost_bps"] for p in calls] == [12, 12, 12, 0]
    assert all(p["random_seed"] == 42 and p["paths"] == 1000 and p["parameter_draws"] == 0 for p in calls)
    assert result["currency"] == "EUR" and len(result["rows"]) == 4


def _assert_parameter_summary_close(actual, expected):
    """Allow only floating-point roundoff, preserving diagnostic schema and labels."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_parameter_summary_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_parameter_summary_close(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-14)
    else:
        assert actual == expected


def _assert_job_response_equal(actual, expected):
    actual, expected = dict(actual), dict(expected)
    # Cached model copies can change the last bits of BLAS reductions in the
    # parameter diagnostics on Linux. Paths, risk results and other metadata
    # still have to match exactly; progress must not alter simulation output.
    _assert_parameter_summary_close(actual.pop("parameter_uncertainty"), expected.pop("parameter_uncertainty"))
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_job_response_comparison_only_tolerates_diagnostic_roundoff():
    expected = {"parameter_uncertainty": {"rows": [[0, 0.01]], "columns": ["draw", "return"]}, "wealth": [100.0]}
    rounded = {**expected, "parameter_uncertainty": {**expected["parameter_uncertainty"], "rows": [[0, 0.01 + 1e-17]]}}
    _assert_job_response_equal(rounded, expected)
    with pytest.raises(AssertionError):
        _assert_job_response_equal({**rounded, "wealth": [100.0 + 1e-10]}, expected)
    with pytest.raises(AssertionError):
        _assert_job_response_equal({**expected, "parameter_uncertainty": {**expected["parameter_uncertainty"], "rows": [[0, 0.010001]]}}, expected)
    with pytest.raises(AssertionError):
        _assert_job_response_equal({**expected, "parameter_uncertainty": {}}, expected)


def test_real_api_job_matches_synchronous_seeded_result(tmp_path):
    from test_api import _csv_payload

    from mc_quadrants.api import build_simulate_response

    payload = _csv_payload(paths=32, periods=12, workers=1, parameter_draws=2, walk_forward=False)
    expected = build_simulate_response(payload)
    stages = []
    with progress_context(stages.append, lambda: False):
        actual = build_simulate_response(payload)
    _assert_job_response_equal(actual, expected)
    assert any("Simulating" in p["stage"] for p in stages)
    assert stages[-1]["stage"] == "Preparing charts and results"
    store = JobStore(tmp_path, {"simulate": build_simulate_response}, threading.BoundedSemaphore(1))
    job = store.submit("simulate", payload)
    assert finished(store, job["id"])["status"] == "completed"
    _assert_job_response_equal(store.result(job["id"])["result"], json.loads(json.dumps(expected)))


def test_http_job_submission_disconnect_poll_result_and_cancel(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def work(payload):
        report_progress("Calibrating", 2, 8)
        entered.set()
        assert release.wait(5)
        check_cancelled()
        return {"ok": True, "value": payload["value"]}

    store = JobStore(tmp_path, {"simulate": work}, threading.BoundedSemaphore(1))
    monkeypatch.setattr(web_app, "JOB_STORE", store)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_app.WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, payload=None):
        req = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read())

    try:
        status, response = request("/api/jobs", {"operation": "simulate", "payload": {"value": 17}})
        assert status == 202
        job_id = response["job"]["id"]
        assert entered.wait(5)
        assert request(f"/api/jobs/{job_id}")[1]["job"]["progress"]["total"] == 8
        with pytest.raises(HTTPError) as error:
            request("/api/jobs", {"operation": "simulate", "payload": {}})
        assert error.value.code == 429
        assert request(f"/api/jobs/{job_id}/cancel", {})[1]["job"]["status"] == "cancelling"
        release.set()
        assert finished(store, job_id)["status"] == "cancelled"
        with pytest.raises(HTTPError) as error:
            request(f"/api/jobs/{job_id}/result")
        assert error.value.code == 409
        retry = request(f"/api/jobs/{job_id}/retry", {})[1]["job"]
        assert finished(store, retry["id"])["status"] == "completed"
        assert request(f"/api/jobs/{retry['id']}/result")[1]["result"]["value"] == 17
        with pytest.raises(HTTPError) as error:
            request("/api/jobs/missing")
        assert error.value.code == 404
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
