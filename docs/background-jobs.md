# Background analysis jobs

The browser submits simulations, safe-rate analyses, wealth exports and the
four-schedule rebalancing sweep as background jobs. A lost HTTP connection does
not end the calculation. Progress comes from actual server checkpoints; any
counts describe the current stage, not a predicted overall completion percentage.

## User flow

- **Cancel analysis** requests cooperative cancellation. The UI displays
  “Stopping…” until the current computation reaches a safe checkpoint. This
  includes bootstrap fits, simulation models and path chunks, walk-forward
  origins, and safe-rate evaluations. A single fit, native call, data fetch or
  already-running worker chunk is not forcibly interrupted.
- **Leave running** stops browser polling, keeping the server job and its ID.
  **Reconnect** resumes polling that ID. Network failures also retry polling
  automatically without resubmitting the simulation.
- Refreshing an unfinished job reconnects automatically. Completed jobs can be
  restored later using **Restore results**. Original settings and uploaded CSV
  inputs are restored for simulation and safe-rate results. Browser storage must
  be available for automatic recovery.
- A recovered portfolio comparison is displayed as the saved portfolio's
  standalone analysis: the earlier in-memory Portfolio A result is not restored.
  The entire rebalancing sweep is one job and is restored as a complete table.
- A server restart marks unfinished jobs **interrupted**. **Retry analysis**
  starts a new computation from the saved request, with the same input settings
  and seeds. It does not resume a partial numerical state or guarantee identical
  live market data if upstream data changed. Successful results survive a restart
  when the same job directory remains available.

## HTTP API

Create a job with `POST /api/jobs`:

```json
{
  "id": "a8ff59a2e4a64bba955a23ecc015f53c",
  "operation": "simulate",
  "payload": {"source": "csv", "paths": 1000}
}
```

The payload uses the corresponding existing endpoint's schema; the example
above omits required data and scenario fields for brevity. Supported operations
are `simulate`, `safe-rate`, `wealth`, and `rebalance`. The optional ID must be
32 lowercase hexadecimal characters. Browser IDs use cryptographic randomness.
Submitting the same ID and payload again retrieves the existing job, making an
uncertain submission safe to retry. Reusing the ID with different inputs fails.

| Request | Response |
|---|---|
| `POST /api/jobs` | `202`, job ID, status and current progress |
| `GET /api/jobs/{id}` | Status, creation/update timestamps, stage and counts |
| `GET /api/jobs/{id}/result` | Completed result plus original payload; `409` before completion |
| `POST /api/jobs/{id}/cancel` | Cancellation request; terminal states remain unchanged |
| `POST /api/jobs/{id}/retry` | New job for stopped work; optional body `{"id": "<new-random-id>"}` supports idempotent retry |

Missing or expired IDs return `404`; capacity exhaustion returns `429`. Jobs
share the existing heavy-operation semaphore with synchronous endpoints. There
is no unbounded queue. Computation failures are stored as failed job states;
validation messages are exposed, while unexpected exception details remain in
server logs. Damaged results become retryable failures.

## Storage and deployment

`JOB_DIRECTORY` defaults to `output/jobs`. Metadata, original requests and
completed results are written separately with atomic replacement and private
file permissions. Progress updates do not rewrite uploaded CSV data. Completed
results are loaded on demand. The browser stores a job capability and small
control snapshot, not the result payload or uploaded CSV contents.

`MAX_STORED_JOBS` defaults to 32 and `JOB_RETENTION_SECONDS` to 86400. Access and
submission prune expired terminal jobs; capacity pressure removes the oldest
terminal jobs first. Active jobs are never evicted. Dismissing a browser entry
forgets its local pointer; server data remain until retention pruning.

Use one application process per job directory. Multi-instance coordination and
distributed workers are not implemented. Mount persistent storage to retain
jobs across container replacement; local files on ephemeral hosting cannot
provide that guarantee. Job IDs are bearer capabilities, not an account system.
Application data, including uploaded histories, are retained on the server for
the configured retention window.

## Verification

`tests/test_jobs.py` exercises cancellation, capacity release, idempotent
submission/retry, restart recovery, retention, failed-result recovery, request
isolation, HTTP lifecycle, rebalancing settings, and equality with synchronous
seeded simulation output. `tests/jobs.test.mjs` covers network reconnection,
uncertain submission, saved-result retrieval, cancellation and leaving a job
running. The browser smoke test reloads an active job, checks that no duplicate
job was submitted, cancels another run, retries from saved CSV inputs, verifies
charts and checks mobile overflow.

```sh
uv run --extra dev pytest -q
node --test tests/*.test.mjs
uv run --extra dev ruff check .
BASE_URL=http://127.0.0.1:7860 uv run --with 'playwright<2' python tests/browser_smoke.py
```

For local browser checks, `PLAYWRIGHT_CHROMIUM_EXECUTABLE` can select an installed
Chromium executable and `BROWSER_SCREENSHOT_DIR` can retain progress/result PNGs.
