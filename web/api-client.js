"use strict";

export async function postJSON(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };
  if (!response.ok || data.ok === false) {
    const retry = response.status === 429 ? " The server is busy; retry in a moment." : "";
    throw new Error((data.error || `Request failed with status ${response.status}`) + retry);
  }
  return data;
}
