import React, { useEffect, useState, useRef } from "react";
import { api } from "./api.js";
import TelemetryPanel from "./components/TelemetryPanel.jsx";
import GeneratePanel from "./components/GeneratePanel.jsx";

const MODES = ["static", "pid", "ppo"];
const HISTORY_LEN = 120;

export default function App() {
  const [state, setState] = useState(null);
  const [history, setHistory] = useState([]);
  const [mode, setMode] = useState("pid");
  const [err, setErr] = useState(null);
  const tick = useRef(0);

  // Poll telemetry + controller state ~2 Hz for the live strip charts.
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const [s, t] = await Promise.all([api.state(), api.telemetry()]);
        if (!alive) return;
        setState(s);
        setMode(s.mode);
        setErr(null);
        setHistory((h) => {
          const next = [...h, {
            t: tick.current++,
            temp_c: t.temp_c,
            power_w: t.power_w,
            budget: s.current_budget,
          }];
          return next.slice(-HISTORY_LEN);
        });
      } catch (e) {
        if (alive) setErr(String(e));
      }
    };
    const id = setInterval(poll, 500);
    poll();
    return () => { alive = false; clearInterval(id); };
  }, []);

  async function changeMode(m) {
    try {
      await api.setMode(m);
      setMode(m);
    } catch (e) {
      setErr(`mode switch failed: ${e}`);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>POISE</h1>
        <p className="tagline">
          Hardware-state-conditioned variable-depth execution — depth follows device physics.
        </p>
        <div className="modes">
          {MODES.map((m) => (
            <button key={m} className={m === mode ? "active" : ""} onClick={() => changeMode(m)}>
              {m}
            </button>
          ))}
        </div>
      </header>
      {err && <div className="error">{err} — is the API running on :8000?</div>}
      <main>
        <TelemetryPanel history={history} state={state} />
        <GeneratePanel mode={mode} />
      </main>
      <footer>
        <small>
          Live numbers are demo telemetry. Headline energy/quality/throughput come only
          from the eval harness (eval/report.py), reported with variance.
        </small>
      </footer>
      <style>{css}</style>
    </div>
  );
}

const css = `
  :root { font-family: system-ui, sans-serif; }
  .app { max-width: 980px; margin: 0 auto; padding: 16px; color: #1c2733; }
  header h1 { margin: 0; letter-spacing: 2px; }
  .tagline { color: #5a6b7b; margin: 4px 0 12px; }
  .modes button, .row button { margin-right: 8px; padding: 6px 14px; border-radius: 6px;
    border: 1px solid #b9c4cf; background: #f3f6f9; cursor: pointer; }
  .modes button.active { background: #2c7fb8; color: white; border-color: #2c7fb8; }
  main { display: grid; grid-template-columns: 1fr; gap: 16px; }
  @media (min-width: 880px) { main { grid-template-columns: 1fr 1fr; } }
  .panel { border: 1px solid #e1e7ee; border-radius: 10px; padding: 14px; background: white; }
  .stats { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
  .stat { background: #f3f6f9; border-radius: 8px; padding: 8px 12px; min-width: 84px; }
  .stat.warn { background: #fdecea; color: #b3261e; }
  .stat-label { font-size: 11px; color: #5a6b7b; }
  .stat-value { font-size: 20px; font-weight: 600; }
  textarea, input { width: 100%; box-sizing: border-box; font-family: inherit; }
  .row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
  .metrics-row { display: flex; flex-wrap: wrap; gap: 12px; font-size: 13px; color: #2c3e50;
    margin: 8px 0; }
  .error { background: #fdecea; color: #b3261e; padding: 8px 12px; border-radius: 6px; }
  pre { white-space: pre-wrap; background: #f3f6f9; padding: 8px; border-radius: 6px; }
  footer { margin-top: 18px; color: #5a6b7b; }
`;
