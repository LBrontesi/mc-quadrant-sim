import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../web/api-client.js", import.meta.url), "utf8");
let moduleId = 0;
async function setup(handler) {
  const values = new Map();
  globalThis.localStorage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
  globalThis.window = new EventTarget();
  globalThis.fetch = async (path, options) => {
    const answer = await handler(path, options.body ? JSON.parse(options.body) : undefined);
    return new Response(JSON.stringify(answer.body), { status: answer.status || 200, headers: { "content-type": "application/json" } });
  };
  return import(`data:text/javascript;base64,${Buffer.from(source + `\n// module ${moduleId++}`).toString("base64")}`);
}

const completed = (id) => ({ id, operation: "simulate", status: "completed", progress: { stage: "Complete" } });

test("job survives polling connection loss without resubmitting computation", async () => {
  let polls = 0;
  let submits = 0;
  let id;
  const client = await setup((path, payload) => {
    if (path === "/api/jobs") {
      submits += 1; id = payload.id;
      return { status: 202, body: { job: { id, status: "running" } } };
    }
    if (path.endsWith("/result")) return { body: { job: completed(id), payload: { paths: 32 }, result: { ok: true, value: 9 } } };
    if (++polls === 1) throw new TypeError("offline");
    return { body: { job: completed(id) } };
  });
  const updates = [];
  window.addEventListener("analysis-job", (event) => updates.push(event.detail));
  assert.deepEqual(await client.postJSON("/api/simulate", { paths: 32 }), { ok: true, value: 9 });
  assert.equal(submits, 1);
  assert.equal(client.savedJob().delivered, true);
  assert.ok(updates.some((event) => event.disconnected));
});

test("uncertain submission repeats the same capability ID", async () => {
  const ids = [];
  const client = await setup((path, payload) => {
    if (path === "/api/jobs") {
      ids.push(payload.id);
      if (ids.length === 1) throw new TypeError("response lost");
      return { body: { job: completed(payload.id) } };
    }
    if (path.endsWith("/result")) return { body: { result: { ok: true }, payload: {}, job: completed(ids[0]) } };
    return { body: { job: completed(ids[0]) } };
  });
  await client.postJSON("/api/simulate", {});
  assert.equal(ids.length, 2);
  assert.equal(ids[0], ids[1]);
});

test("reload restores saved result without a new POST", async () => {
  const client = await setup((path, payload) => {
    assert.equal(payload, undefined);
    if (path.endsWith("/result")) return { body: { result: { value: 7 }, payload: { paths: 48 }, job: completed("a".repeat(32)) } };
    return { body: { job: completed("a".repeat(32)) } };
  });
  localStorage.setItem("mcq-background-job", JSON.stringify({ id: "a".repeat(32), operation: "simulate", status: "running" }));
  const restored = await client.resumeJob();
  assert.equal(restored.result.value, 7);
  assert.equal(restored.payload.paths, 48);
});

test("cancellation reaches server and stopped jobs retain retry information", async () => {
  let cancelled = false;
  const id = "a".repeat(32);
  const client = await setup((path) => {
    if (path.endsWith("/cancel")) cancelled = true;
    return { body: { job: { id, status: cancelled ? "cancelled" : "running", error: "Analysis cancelled." } } };
  });
  localStorage.setItem("mcq-background-job", JSON.stringify({ id, operation: "simulate", status: "running" }));
  await client.cancelJob();
  await assert.rejects(client.resumeJob(), /cancelled/);
  assert.equal(client.savedJob().status, "cancelled");
});

test("busy submission preserves the previous recoverable result", async () => {
  const client = await setup(() => ({ status: 429, body: { error: "busy" } }));
  const previous = { id: "a".repeat(32), delivered: true, status: "completed" };
  localStorage.setItem("mcq-background-job", JSON.stringify(previous));
  await assert.rejects(client.postJSON("/api/simulate", {}), /busy/);
  assert.deepEqual(client.savedJob(), previous);
});

test("leaving running preserves ID and reconnect fetches the same job", async () => {
  const id = "a".repeat(32);
  let done = false;
  const client = await setup((path) => {
    if (path.endsWith("/result")) return { body: { job: completed(id), payload: {}, result: { ok: true } } };
    return { body: { job: done ? completed(id) : { id, status: "running" } } };
  });
  localStorage.setItem("mcq-background-job", JSON.stringify({ id, status: "running" }));
  window.addEventListener("analysis-job", (event) => {
    if (event.detail.job?.status === "running") client.detachJob();
  });
  await assert.rejects(client.resumeJob(), /left running/);
  assert.equal(client.savedJob().id, id);
  done = true;
  assert.equal((await client.resumeJob()).result.ok, true);
});
