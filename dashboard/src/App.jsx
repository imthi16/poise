import React, { useEffect, useState, useRef } from "react";
import { api } from "./api.js";
import { tokens, thermColor, thermZone } from "./theme.js";
import InstrumentPanel from "./components/TelemetryPanel.jsx";
import WorkloadPanel from "./components/GeneratePanel.jsx";

const MODES = [
  { id: "static", desc: "fixed depth" },
  { id: "pid", desc: "reactive control" },
  { id: "ppo", desc: "anticipatory policy" },
];
const HISTORY_LEN = 160;
const POLL_MS = 500;

export default function App() {
  const [state, setState] = useState(null);
  const [tele, setTele] = useState(null);
  const [history, setHistory] = useState([]);
  const [mode, setMode] = useState("pid");
  const [err, setErr] = useState(null);
  const tick = useRef(0);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const [s, t] = await Promise.all([api.state(), api.telemetry()]);
        if (!alive) return;
        setState(s);
        setTele(t);
        setMode(s.mode);
        setErr(null);
        setHistory((h) =>
          [...h, { t: tick.current++, temp_c: t.temp_c, power_w: t.power_w, budget: s.current_budget }].slice(
            -HISTORY_LEN
          )
        );
      } catch (e) {
        if (alive) setErr(String(e));
      }
    };
    const id = setInterval(poll, POLL_MS);
    poll();
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  async function changeMode(m) {
    try {
      await api.setMode(m);
      setMode(m);
    } catch (e) {
      setErr(`Could not switch to ${m} mode — ${e}`);
    }
  }

  const ambient = state?.ambient_c ?? 25;
  const setpoint = state?.setpoint_c ?? 80;
  const ceiling = state?.ceiling_c ?? 87;
  const temp = state?.temp_c;
  const therm = thermColor(temp, ambient, setpoint, ceiling);
  const zone = thermZone(temp, setpoint, ceiling);
  const nearCeiling = temp != null && temp >= ceiling - 3;
  const live = state != null && err == null;

  return (
    <div className="gov" style={{ "--therm": therm }}>
      <div className="gov-rail">
        <div className="gov-brand">
          <span className="gov-wordmark">POISE</span>
          <span className="gov-instr">Thermal Governor</span>
        </div>
        <div className="gov-rail-right">
          <span className={"gov-pulse" + (nearCeiling ? " alarm" : "") + (live ? "" : " off")} />
          <span className="gov-zone">{live ? zone.label : "NO SIGNAL"}</span>
          <span className="gov-sep" />
          <span className="gov-dev">jetson · agx orin</span>
        </div>
      </div>

      <header className="gov-head">
        <h1>
          Depth follows <span className="gov-em">physics</span>.
        </h1>
        <p>
          Per-token transformer depth is governed by the chip's live thermal state — not by input
          difficulty. Watch the junction heat and the layer budget shed to stay under the throttle
          ceiling.
        </p>
        <div className="gov-modes" role="group" aria-label="Control mode">
          {MODES.map((m) => {
            const locked = m.id === "ppo" && state && !state.policy_loaded;
            const active = m.id === mode;
            return (
              <button
                key={m.id}
                className={"gov-mode" + (active ? " active" : "") + (locked ? " locked" : "")}
                onClick={() => !locked && changeMode(m.id)}
                disabled={locked}
                aria-pressed={active}
                title={
                  locked
                    ? "Train a PPO policy and set POISE_POLICY_PATH to enable this mode."
                    : `Switch to ${m.id} — ${m.desc}`
                }
              >
                <span className="gov-mode-id">{m.id}</span>
                <span className="gov-mode-desc">{locked ? "policy not loaded" : m.desc}</span>
              </button>
            );
          })}
        </div>
      </header>

      {err && (
        <div className="gov-alert" role="alert">
          <strong>Instrument offline.</strong> {err} — start the API with{" "}
          <code>bash scripts/serve.sh</code> on port 8000.
        </div>
      )}

      <main className="gov-main">
        <InstrumentPanel state={state} tele={tele} history={history} therm={therm} zone={zone} />
        <WorkloadPanel mode={mode} therm={therm} ceiling={ceiling} setpoint={setpoint} />
      </main>

      <footer className="gov-foot">
        <span>
          Live readings are demo telemetry. Headline energy · quality · throughput come only from the
          eval harness (<code>eval/report.py</code>), reported with variance.
        </span>
      </footer>

      <style>{styles}</style>
    </div>
  );
}

const T = tokens;
const styles = `
  .gov {
    --bg:${T.bg}; --panel:${T.panel}; --panel2:${T.panel2}; --line:${T.line};
    --line-soft:${T.lineSoft}; --ink:${T.ink}; --muted:${T.muted}; --muted-dim:${T.mutedDim};
    --mono: ui-monospace, "JetBrains Mono", "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
    --sans: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background:
      radial-gradient(1100px 620px at 82% -8%, color-mix(in srgb, var(--therm) 12%, transparent), transparent 60%),
      var(--bg);
    color: var(--ink); font-family: var(--sans);
    min-height: 100vh; padding: 0 clamp(16px, 4vw, 48px) 48px;
    max-width: 1240px; margin: 0 auto;
    -webkit-font-smoothing: antialiased;
  }

  /* ---- status rail ---- */
  .gov-rail {
    display:flex; align-items:center; justify-content:space-between;
    padding: 16px 2px; border-bottom: 1px solid var(--line);
    font-family: var(--mono); font-size: 12px;
  }
  .gov-brand { display:flex; align-items:baseline; gap:12px; }
  .gov-wordmark { font-weight:700; letter-spacing:.42em; font-size:15px; color:var(--ink); }
  .gov-instr {
    text-transform:uppercase; letter-spacing:.28em; font-size:10px; color:var(--muted-dim);
    padding-left:12px; border-left:1px solid var(--line);
  }
  .gov-rail-right { display:flex; align-items:center; gap:12px; color:var(--muted); }
  .gov-zone { color: var(--therm); letter-spacing:.16em; font-size:11px; transition: color .5s ease; }
  .gov-dev { text-transform:uppercase; letter-spacing:.16em; color:var(--muted-dim); font-size:10px; }
  .gov-sep { width:1px; height:12px; background:var(--line); }
  .gov-pulse {
    width:9px; height:9px; border-radius:50%; background:var(--therm);
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--therm) 70%, transparent);
    transition: background .5s ease;
  }
  .gov-pulse.off { background: var(--muted-dim); }
  .gov-pulse.alarm { animation: govpulse 1.05s ease-out infinite; }
  @keyframes govpulse {
    0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--therm) 75%, transparent); }
    100% { box-shadow: 0 0 0 11px transparent; }
  }

  /* ---- header ---- */
  .gov-head { padding: 40px 2px 26px; max-width: 760px; }
  .gov-head h1 {
    margin:0; font-size: clamp(34px, 5.4vw, 58px); line-height:1.02; font-weight:680;
    letter-spacing:-0.02em;
  }
  .gov-em { color: var(--therm); transition: color .6s ease; }
  .gov-head p { margin:18px 0 0; color:var(--muted); font-size:15px; line-height:1.6; max-width:620px; }

  /* ---- mode switch ---- */
  .gov-modes { display:flex; gap:10px; margin-top:26px; flex-wrap:wrap; }
  .gov-mode {
    display:flex; flex-direction:column; gap:3px; align-items:flex-start;
    padding:10px 16px; min-width:132px; cursor:pointer; text-align:left;
    background:var(--panel); border:1px solid var(--line); border-radius:2px;
    color:var(--muted); transition: border-color .18s ease, background .18s ease, color .18s ease;
  }
  .gov-mode:hover:not(:disabled) { border-color: color-mix(in srgb, var(--therm) 55%, var(--line)); }
  .gov-mode-id { font-family:var(--mono); font-size:14px; letter-spacing:.14em; text-transform:uppercase; color:var(--ink); }
  .gov-mode-desc { font-size:11px; letter-spacing:.04em; }
  .gov-mode.active {
    border-color: var(--therm); background: color-mix(in srgb, var(--therm) 10%, var(--panel));
  }
  .gov-mode.active .gov-mode-id { color: var(--therm); }
  .gov-mode.locked { opacity:.42; cursor:not-allowed; }
  .gov-mode:focus-visible { outline:2px solid var(--therm); outline-offset:2px; }

  /* ---- layout ---- */
  .gov-main { display:grid; grid-template-columns:1fr; gap:18px; margin-top:8px; }
  @media (min-width: 940px) { .gov-main { grid-template-columns: 1.15fr 1fr; align-items:start; } }

  .gov-panel {
    background:
      linear-gradient(180deg, color-mix(in srgb, var(--panel2) 60%, var(--panel)), var(--panel));
    border:1px solid var(--line); border-radius:4px; padding:20px;
  }
  .gov-eyebrow {
    font-family:var(--mono); text-transform:uppercase; letter-spacing:.26em; font-size:10px;
    color:var(--muted-dim); margin:0 0 16px; display:flex; align-items:center; gap:10px;
  }
  .gov-eyebrow::after { content:""; flex:1; height:1px; background:var(--line-soft); }

  .gov-alert {
    margin-top:16px; padding:12px 16px; border-radius:3px; font-size:13px;
    background: color-mix(in srgb, ${T.crit} 12%, var(--panel));
    border:1px solid color-mix(in srgb, ${T.crit} 45%, var(--line)); color:#ffd6dc;
  }
  .gov-alert code, .gov-foot code { font-family:var(--mono); color:var(--ink); font-size:.92em; }

  .gov-foot {
    margin-top:28px; padding-top:18px; border-top:1px solid var(--line);
    color:var(--muted-dim); font-size:12px; line-height:1.6; font-family:var(--mono); letter-spacing:.02em;
  }

  /* shared readout primitives used by the panels */
  .gov-readouts { display:grid; grid-template-columns:repeat(2,1fr); gap:1px; background:var(--line-soft);
    border:1px solid var(--line-soft); border-radius:3px; overflow:hidden; }
  .gov-ro { background:var(--panel); padding:11px 13px; }
  .gov-ro-label { font-family:var(--mono); font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted-dim); }
  .gov-ro-value { font-family:var(--mono); font-size:19px; color:var(--ink); margin-top:3px; font-variant-numeric:tabular-nums; }
  .gov-ro-value .u { font-size:11px; color:var(--muted); margin-left:3px; }
  .gov-ro.warn .gov-ro-value { color:${T.crit}; }

  .gov-readouts-4 { grid-template-columns:repeat(4,1fr); }
  .gov-readouts-3 { grid-template-columns:repeat(3,1fr); margin-top:16px; }
  @media (max-width:560px){ .gov-readouts-4,.gov-readouts-3 { grid-template-columns:repeat(2,1fr); } }

  /* ---- instrument: gauge + ladder ---- */
  .gov-hero { display:grid; grid-template-columns:auto 1fr; gap:20px; margin-bottom:18px; }
  @media (max-width:420px){ .gov-hero { grid-template-columns:1fr; } }

  .gov-gauge { display:flex; flex-direction:column; align-items:center; width:150px; }
  .gov-gauge svg { overflow:visible; }
  .gov-gauge-readout { text-align:center; margin-top:6px; }
  .gov-gauge-value { font-family:var(--mono); font-size:34px; font-weight:600; line-height:1;
    font-variant-numeric:tabular-nums; transition:color .5s ease; }
  .gov-gauge-unit { font-size:15px; color:var(--muted); margin-left:2px; }
  .gov-gauge-label { font-family:var(--mono); font-size:9.5px; letter-spacing:.16em; color:var(--muted-dim);
    margin-top:6px; text-transform:uppercase; }

  .gov-ladder { display:flex; flex-direction:column; justify-content:center; padding-left:4px; }
  .gov-ladder-cap, .gov-ladder-foot { font-family:var(--mono); font-size:10px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--muted-dim); }
  .gov-ladder-foot { margin-top:10px; }
  .gov-ladder-rungs { display:flex; flex-direction:column; gap:7px; margin:12px 0; }
  .gov-rung { display:flex; align-items:center; gap:10px; height:22px; }
  .gov-rung-bar { display:block; height:9px; border-radius:2px; background:var(--line);
    transition: width .5s ease, background .5s ease, box-shadow .5s ease; min-width:30px; }
  .gov-rung-num { font-family:var(--mono); font-size:13px; color:var(--muted-dim);
    width:22px; font-variant-numeric:tabular-nums; transition:color .4s ease; }
  .gov-rung.on .gov-rung-num { font-weight:600; }
  .gov-rung-mark { font-family:var(--mono); font-size:10px; letter-spacing:.1em; }

  /* ---- traces ---- */
  .gov-trace { margin-top:18px; }
  .gov-trace-head { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:8px; }
  .gov-trace-title { font-family:var(--mono); font-size:10px; letter-spacing:.2em; color:var(--muted); }
  .gov-trace-sub { font-family:var(--mono); font-size:10px; color:var(--muted-dim); letter-spacing:.04em; }

  /* ---- workload console ---- */
  .gov-field { display:block; margin-top:6px; }
  .gov-field-cap { display:block; font-family:var(--mono); font-size:10px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--muted-dim); margin-bottom:6px; }
  .gov-workload textarea, .gov-workload input {
    width:100%; box-sizing:border-box; background:var(--bg); color:var(--ink);
    border:1px solid var(--line); border-radius:3px; padding:10px 12px;
    font-family:var(--mono); font-size:13px; resize:vertical;
  }
  .gov-workload textarea:focus, .gov-workload input:focus {
    outline:none; border-color:var(--therm); box-shadow:0 0 0 1px color-mix(in srgb,var(--therm) 40%,transparent);
  }
  .gov-run { display:flex; align-items:flex-end; gap:12px; margin-top:14px; }
  .gov-field-inline { margin-top:0; width:130px; flex:none; }
  .gov-run-btn {
    flex:1; padding:11px 18px; cursor:pointer; background:var(--panel2);
    border:1px solid var(--line); border-radius:3px; color:var(--ink);
    font-family:var(--mono); font-size:13px; letter-spacing:.1em; text-transform:uppercase;
    transition: border-color .18s ease, color .18s ease, background .18s ease;
  }
  .gov-run-btn:hover:not(:disabled) { background: color-mix(in srgb, var(--therm) 12%, var(--panel2)); }
  .gov-run-btn:disabled { opacity:.5; cursor:progress; }
  .gov-run-btn:focus-visible { outline:2px solid var(--therm); outline-offset:2px; }

  .gov-output { margin-top:16px; }
  .gov-output summary { font-family:var(--mono); font-size:11px; letter-spacing:.12em; text-transform:uppercase;
    color:var(--muted); cursor:pointer; }
  .gov-output pre { white-space:pre-wrap; background:var(--bg); border:1px solid var(--line);
    border-radius:3px; padding:12px; margin-top:10px; font-family:var(--mono); font-size:12.5px;
    line-height:1.6; color:var(--ink); max-height:280px; overflow:auto; }

  @media (prefers-reduced-motion: reduce) {
    .gov-pulse.alarm { animation:none; }
    .gov *, .gov *::before, .gov *::after { transition:none !important; }
  }
`;
