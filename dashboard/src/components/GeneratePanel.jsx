import React, { useState } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { api } from "../api.js";
import { tokens } from "../theme.js";

// The workload console: run a generation and read back the per-token budget trace —
// the record of how depth was governed while the tokens streamed.
export default function WorkloadPanel({ mode, therm, ceiling = 87, setpoint = 80 }) {
  const [prompt, setPrompt] = useState("Explain thermal throttling in one paragraph.");
  const [maxTokens, setMaxTokens] = useState(64);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function onGenerate() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.generate({ prompt, max_new_tokens: Number(maxTokens), mode, return_trace: true });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const trace = (result?.trace ?? []).map((t) => ({ i: t.i, budget: t.budget, temp_c: t.temp_c }));
  const m = result?.metrics;

  return (
    <section className="gov-panel gov-workload">
      <p className="gov-eyebrow">Workload</p>

      <label className="gov-field">
        <span className="gov-field-cap">Prompt</span>
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} spellCheck={false} />
      </label>

      <div className="gov-run">
        <label className="gov-field gov-field-inline">
          <span className="gov-field-cap">Max tokens</span>
          <input type="number" value={maxTokens} min={1} max={1024}
            onChange={(e) => setMaxTokens(e.target.value)} />
        </label>
        <button className="gov-run-btn" onClick={onGenerate} disabled={busy}
          style={busy ? undefined : { borderColor: therm, color: therm }}>
          {busy ? "running…" : `Run · ${mode}`}
        </button>
      </div>

      {error && <div className="gov-alert" role="alert">{error}</div>}

      {m && (
        <>
          <div className="gov-readouts gov-readouts-3">
            <Metric label="Throughput" value={m.tok_per_s.toFixed(1)} unit="tok/s" />
            <Metric label="Energy" value={m.energy_per_token_j.toFixed(3)} unit="J/tok" />
            <Metric label="Peak temp" value={m.peak_temp_c.toFixed(1)} unit="°C"
              warn={m.peak_temp_c >= ceiling} />
            <Metric label="Mean depth" value={m.mean_budget.toFixed(1)} unit="layers" />
            <Metric label="Tokens" value={result.tokens} />
            <Metric label="Throttles" value={m.throttle_events} warn={m.throttle_events > 0} />
          </div>

          <div className="gov-trace">
            <div className="gov-trace-head">
              <span className="gov-trace-title">PER-TOKEN BUDGET</span>
              <span className="gov-trace-sub">depth chosen as each token was emitted</span>
            </div>
            <ResponsiveContainer width="100%" height={190}>
              <ComposedChart data={trace} margin={{ top: 6, right: 6, bottom: 2, left: -16 }}>
                <defs>
                  <linearGradient id="budgetFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={tokens.cool} stopOpacity={0.28} />
                    <stop offset="100%" stopColor={tokens.cool} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={tokens.lineSoft} vertical={false} />
                <XAxis dataKey="i" tick={{ fill: tokens.mutedDim, fontSize: 10, fontFamily: "monospace" }}
                  axisLine={{ stroke: tokens.line }} tickLine={false} />
                <YAxis yAxisId="b" domain={[0, 32]} width={30}
                  tick={{ fill: tokens.mutedDim, fontSize: 10, fontFamily: "monospace" }}
                  axisLine={false} tickLine={false} />
                <YAxis yAxisId="t" orientation="right" domain={["auto", "auto"]} hide />
                <Tooltip contentStyle={{
                  background: tokens.panel2, border: `1px solid ${tokens.line}`, borderRadius: 3,
                  fontFamily: "monospace", fontSize: 12, color: tokens.ink,
                }} labelStyle={{ color: tokens.mutedDim }} />
                <ReferenceLine yAxisId="t" y={setpoint} stroke={tokens.warm} strokeDasharray="5 4"
                  strokeOpacity={0.6} />
                <Area yAxisId="b" type="stepAfter" dataKey="budget" stroke={tokens.cool} strokeWidth={2}
                  fill="url(#budgetFill)" dot={false} isAnimationActive={false} name="layer budget" />
                <Line yAxisId="t" type="monotone" dataKey="temp_c" stroke={therm} strokeWidth={1.5}
                  strokeOpacity={0.85} dot={false} isAnimationActive={false} name="temp °C" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <details className="gov-output">
            <summary>Model output</summary>
            <pre>{result.text}</pre>
          </details>
        </>
      )}
    </section>
  );
}

function Metric({ label, value, unit, warn }) {
  return (
    <div className={"gov-ro" + (warn ? " warn" : "")}>
      <div className="gov-ro-label">{label}</div>
      <div className="gov-ro-value">{value}{unit && <span className="u">{unit}</span>}</div>
    </div>
  );
}
