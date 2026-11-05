"""Bounded, persistent background jobs for the single-process web server.

Random capability IDs are required to read or cancel a job. No listing endpoint
exposes other clients' jobs. A restart retains results and marks unfinished
work interrupted; retrying starts a new computation rather than guessing at a
partially computed state.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

from mc_quadrants.progress import JobCancelled, check_cancelled, progress_context

LOGGER = logging.getLogger(__name__)
TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


class JobsBusy(Exception):
    pass


class JobStore:
    def __init__(self, directory, handlers, semaphore, *, retention_seconds=86400, max_jobs=32):
        self.directory = Path(directory)
        self.handlers = handlers
        self.semaphore = semaphore
        self.retention_seconds = retention_seconds
        self.max_jobs = max_jobs
        self.lock = threading.RLock()
        self.jobs = {}
        self.cancellations = {}
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path in self.directory.glob("*.json"):
            if not re.fullmatch(r"[0-9a-f]{32}", path.stem):
                continue
            try:
                job = json.loads(path.read_text())
                if job["id"] != path.stem or job["operation"] not in handlers:
                    continue
                if "payload" not in job:
                    job["payload"] = json.loads((self.directory / f"{path.stem}.request").read_text())
                elif not (self.directory / f"{path.stem}.request").exists():
                    self._write_json(self.directory / f"{path.stem}.request", job["payload"])
                if job["status"] == "completed" and not (self.directory / f"{path.stem}.result").exists():
                    job.update(status="failed", error="The saved result is missing. Retry to run the analysis again.", updated_at=time.time())
                    self._write(job)
                if job["status"] not in TERMINAL:
                    job.update(status="interrupted", error="The server restarted before this job finished. Retry to start again.", updated_at=time.time())
                    self._write(job)
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                LOGGER.exception("Could not recover job %s", path.stem)
        self._prune()

    def _write_json(self, path, value):
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                temporary.chmod(0o600)
                json.dump(value, handle)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write(self, job):
        # Large uploaded CSVs are stored once, not rewritten at every checkpoint.
        self._write_json(self.directory / f"{job['id']}.json", self._public(job))

    def _prune(self, reserve=0):
        terminal = sorted((j for j in self.jobs.values() if j["status"] in TERMINAL), key=lambda j: j["updated_at"])
        for job in terminal:
            if time.time() - job["updated_at"] <= self.retention_seconds and len(self.jobs) + reserve <= self.max_jobs:
                continue
            for suffix in (".json", ".result", ".request"):
                (self.directory / f"{job['id']}{suffix}").unlink(missing_ok=True)
            self.jobs.pop(job["id"])

    @staticmethod
    def _public(job):
        return {key: value for key, value in job.items() if key not in {"payload"}}

    def get(self, job_id):
        with self.lock:
            self._prune()
            return self._public(self.jobs[job_id])

    def result(self, job_id):
        with self.lock:
            self._prune()
            job = self.jobs[job_id]
            if job["status"] != "completed":
                raise ValueError("This job has no completed result.")
            try:
                result = json.loads((self.directory / f"{job_id}.result").read_text())
            except (OSError, ValueError) as exc:
                job.update(status="failed", error="The saved result could not be read. Retry to run the analysis again.", updated_at=time.time())
                self._write(job)
                raise ValueError(job["error"]) from exc
            return {"job": self._public(job), "payload": job["payload"], "result": result}

    def submit(self, operation, payload, *, job_id=None):
        if operation not in self.handlers or not isinstance(payload, dict):
            raise ValueError("Choose a supported job operation and an object payload.")
        job_id = job_id or uuid.uuid4().hex
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Invalid job ID.")
        with self.lock:
            # The browser saves its random ID before submission. Retrying an
            # uncertain POST can therefore never launch a duplicate job.
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job["operation"] != operation or job["payload"] != payload:
                    raise ValueError("This job ID already belongs to a different request.")
                return self._public(job)
            self._prune(reserve=1)
            if len(self.jobs) >= self.max_jobs or not self.semaphore.acquire(blocking=False):
                raise JobsBusy("Another analysis is running. Try again when it finishes.")
            event = threading.Event()
            now = time.time()
            job = dict(id=job_id, operation=operation, payload=payload, status="running",
                       created_at=now, updated_at=now, progress={"stage": "Preparing analysis", "completed": None, "total": None})
            try:
                self._write_json(self.directory / f"{job_id}.request", payload)
                self._write(job)
                self.jobs[job_id] = job
                self.cancellations[job_id] = event
                threading.Thread(target=self._run, args=(job_id, event), daemon=True, name=f"analysis-{job_id[:8]}").start()
            except Exception:
                self.jobs.pop(job_id, None)
                self.cancellations.pop(job_id, None)
                (self.directory / f"{job_id}.json").unlink(missing_ok=True)
                (self.directory / f"{job_id}.request").unlink(missing_ok=True)
                self.semaphore.release()
                raise
            return self._public(job)

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] not in TERMINAL:
                self.cancellations[job_id].set()
                job.update(status="cancelling", updated_at=time.time())
                self._write(job)
            return self._public(job)

    def retry(self, job_id, *, retry_id=None):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] not in {"failed", "interrupted", "cancelled"}:
                raise ValueError("Only a stopped job can be retried.")
            return self.submit(job["operation"], job["payload"], job_id=retry_id)

    def _run(self, job_id, event):
        def progress(value):
            with self.lock:
                job = self.jobs[job_id]
                job.update(progress=value, updated_at=time.time())
                self._write(job)

        try:
            with progress_context(progress, event.is_set):
                check_cancelled()
                job = self.jobs[job_id]
                result = self.handlers[job["operation"]](job["payload"])
                check_cancelled()
                with self.lock:
                    check_cancelled()
                    self._write_json(self.directory / f"{job_id}.result", result)
                    job.update(status="completed", updated_at=time.time(), progress={"stage": "Complete", "completed": 1, "total": 1})
                    self._write(job)
        except Exception as exc:
            cancelled = isinstance(exc, JobCancelled) or event.is_set()
            if not cancelled and not isinstance(exc, ValueError):
                LOGGER.exception("Background analysis failed: %s", job_id)
            with self.lock:
                job = self.jobs[job_id]
                job.update(status="cancelled" if cancelled else "failed", updated_at=time.time(),
                           error="Analysis cancelled." if cancelled else str(exc) if isinstance(exc, ValueError) else "The server could not complete this analysis.")
                try:
                    self._write(job)
                except OSError:
                    LOGGER.exception("Could not persist stopped job %s", job_id)
        finally:
            with self.lock:
                self.cancellations.pop(job_id, None)
            self.semaphore.release()
