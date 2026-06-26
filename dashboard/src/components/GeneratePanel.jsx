import React, { useState } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { api } from "../api.js";

// Trigger a generation and visualize the per-token budget trace alongside temp.
export default function GeneratePanel({ mode }) {
  const [prompt, setPrompt] = useState("Explain thermal throttling in one paragraph.");
  const [maxTokens, setMaxTokens] = useState(64);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function onGenerate() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.generate({
        prompt, max_new_tokens: Number(maxTokens), mode, return_trace: true,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const traceData = (result?.trace ?? []).map((t) => ({
    i: t.i, budget: t.budget, temp_c: t.temp_c, power_w: t.power_w,
  }));

  return (
    <div className="panel">
      <h2>Generate</h2>
      <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} />
      <div className="row">
        <label>max tokens
          <input type="number" value={maxTokens} min={1} max={1024}
            onChange={(e) => setMaxTokens(e.target.value)} />
        </label>
        <button onClick={onGenerate} disabled={busy}>
          {busy ? "generating…" : `generate (${mode})`}
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {result && (
        <>
          <div className="metrics-row">
            <span>{result.tokens} tok</span>
            <span>{result.metrics.tok_per_s.toFixed(1)} tok/s</span>
            <span>{result.metrics.energy_per_token_j.toFixed(3)} J/tok</span>
            <span>peak {result.metrics.peak_temp_c.toFixed(1)}°C</span>
            <span>mean depth {result.metrics.mean_budget.toFixed(1)}</span>
            <span>throttle {result.metrics.throttle_events}</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={traceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="i" label={{ value: "token", position: "insideBottom", offset: -3 }} />
              <YAxis yAxisId="b" domain={[0, 32]} label={{ value: "budget", angle: -90 }} />
              <YAxis yAxisId="t" orientation="right" />
              <Tooltip />
              <Legend />
              <Area yAxisId="b" type="stepAfter" dataKey="budget" stroke="#31a354"
                fill="#bae4b3" name="layer budget" isAnimationActive={false} />
              <Line yAxisId="t" type="monotone" dataKey="temp_c" stroke="#d6452c"
                dot={false} name="temp °C" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
          <details>
            <summary>output text</summary>
            <pre>{result.text}</pre>
          </details>
        </>
      )}
    </div>
  );
}
