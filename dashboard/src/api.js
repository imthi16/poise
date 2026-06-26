// Thin API client for the POISE FastAPI backend (§7).

async function jget(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

async function jpost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

export const api = {
  health: () => jget("/health"),
  telemetry: () => jget("/v1/telemetry"),
  state: () => jget("/v1/state"),
  generate: (payload) => jpost("/v1/generate", payload),
  setMode: (mode) => jpost("/v1/config/mode", { mode }),
  run: (id) => jget(`/v1/runs/${id}`),
};
