"use strict";

const JOB_KEY = "mcq-background-job";
const HEAVY = new Set(["/api/simulate", "/api/safe-rate", "/api/wealth", "/api/rebalance"]);
const TERMINAL = new Set(["completed", "cancelled", "failed", "interrupted"]);
let activeJob = null;
let watchGeneration = 0;

function announce(detail) {
  window.dispatchEvent(new CustomEvent("analysis-job", { detail }));
}

export function savedJob() {
  try { return JSON.parse(localStorage.getItem(JOB_KEY) || "null"); }
  catch { return null; }
}

function remember(job) {
  try { localStorage.setItem(JOB_KEY, JSON.stringify(job)); }
  catch { announce({ storageUnavailable: true }); }
}

export function forgetJob() {
  localStorage.removeItem(JOB_KEY);
}

export function detachJob() {
  watchGeneration += 1;
}

export async function requestJSON(path, payload) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, {
      method: payload === undefined ? "GET" : "POST",
      headers: payload === undefined ? {} : { "Content-Type": "application/json" },
      body: payload === undefined ? undefined : JSON.stringify(payload),
      signal: controller.signal,
    });
    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
      ? await response.json() : { error: await response.text() };
    if (!response.ok || data.ok === false) {
      const error = new Error(data.error || `Request failed with status ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return data;
  } finally {
    clearTimeout(timeout);
  }
}

export async function cancelJob() {
  const job = activeJob || savedJob();
  if (!job) return;
  const data = await requestJSON(`/api/jobs/${job.id}/cancel`, {});
  announce({ job: data.job });
  return data.job;
}

export async function resumeJob(job = savedJob()) {
  if (!job) return null;
  if (activeJob) throw new Error("An analysis is already connected in this tab.");
  activeJob = job;
  const generation = ++watchGeneration;
  let failures = 0;
  try {
    for (;;) {
      if (generation !== watchGeneration) throw new Error("Analysis left running. Use Reconnect to retrieve it.");
      try {
        const response = await requestJSON(`/api/jobs/${job.id}`);
        if (generation !== watchGeneration) continue;
        const current = response.job;
        job = { ...job, status: current.status };
        activeJob = job;
        remember(job);
        announce({ job: current });
        failures = 0;
        if (current.status === "completed") {
          let envelope;
          try { envelope = await requestJSON(`/api/jobs/${job.id}/result`); }
          catch (error) {
            if (error.status === 409) continue; // Server marked a damaged saved result failed.
            throw error;
          }
          if (generation !== watchGeneration) continue;
          remember({ ...job, delivered: true });
          return envelope;
        }
        if (TERMINAL.has(current.status)) {
          const error = new Error(current.error || "Analysis stopped.");
          error.job = current;
          throw error;
        }
      } catch (error) {
        if (error.job || (error.status && error.status < 500)) throw error;
        failures += 1;
        announce({ disconnected: true, message: "Connection lost. Reconnecting; your analysis continues on the server." });
      }
      await new Promise((resolve) => setTimeout(resolve, Math.min(1000 * Math.max(1, failures), 5000)));
    }
  } finally {
    activeJob = null;
    announce({ settled: true });
  }
}

export async function retryJob() {
  const previous = savedJob();
  if (!previous) return null;
  const id = crypto.randomUUID().replaceAll("-", "");
  const next = { ...previous, id, delivered: false, status: "submitting" };
  remember(next);
  try {
    const data = await requestJSON(`/api/jobs/${previous.id}/retry`, { id });
    return await resumeJob({ ...next, status: data.job.status });
  } catch (error) {
    if (error.status) remember(previous);
    throw error;
  }
}

export async function postJSON(path, payload, context = null) {
  if (!HEAVY.has(path)) return requestJSON(path, payload);
  const previous = savedJob();
  if (activeJob || (previous && !previous.delivered && !TERMINAL.has(previous.status))) {
    throw new Error("Reconnect to your existing analysis before starting another one.");
  }
  const operation = path.split("/").at(-1);
  const job = { id: crypto.randomUUID().replaceAll("-", ""), operation, context, status: "submitting", delivered: false };
  remember(job); // Save the capability before any network request can succeed.
  announce({ job: { ...job, progress: { stage: "Submitting analysis" } } });
  try {
    const request = { id: job.id, operation, payload };
    try {
      await requestJSON("/api/jobs", request);
    } catch (error) {
      if (error.status) throw error;
      await requestJSON("/api/jobs", request); // Same ID: safe after an uncertain response.
    }
    return (await resumeJob(job)).result;
  } catch (error) {
    if (error.status && error.status < 500 && error.status !== 404) {
      if (previous) remember(previous); else forgetJob();
    }
    throw error;
  }
}
