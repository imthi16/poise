import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// Live temperature / power / current-budget strip charts + headline stats.
export default function TelemetryPanel({ history, state }) {
  const setpoint = state?.setpoint_c ?? 80;
  return (
    <div className="panel">
      <h2>Live hardware state</h2>
      <div className="stats">
        <Stat label="Temp (°C)" value={fmt(state?.temp_c)} warn={state?.temp_c > setpoint} />
        <Stat label="Budget (layers)" value={state?.current_budget ?? "—"} />
        <Stat label="Mode" value={state?.mode ?? "—"} />
        <Stat label="Throttled" value={state?.throttled ? "YES" : "no"} warn={state?.throttled} />
        <Stat label="Policy" value={state?.policy_loaded ? "loaded" : "PID only"} />
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={history}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="t" tick={false} />
          <YAxis yAxisId="temp" domain={["auto", "auto"]} />
          <Tooltip />
          <ReferenceLine yAxisId="temp" y={setpoint} stroke="#e0a000"
            label="setpoint" strokeDasharray="4 4" />
          <Line yAxisId="temp" type="monotone" dataKey="temp_c" stroke="#d6452c"
            dot={false} name="temp °C" isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={history}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="t" tick={false} />
          <YAxis yAxisId="pw" />
          <YAxis yAxisId="b" orientation="right" domain={[0, 32]} />
          <Tooltip />
          <Line yAxisId="pw" type="monotone" dataKey="power_w" stroke="#2c7fb8"
            dot={false} name="power W" isAnimationActive={false} />
          <Line yAxisId="b" type="stepAfter" dataKey="budget" stroke="#31a354"
            dot={false} name="budget" isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Stat({ label, value, warn }) {
  return (
    <div className={"stat" + (warn ? " warn" : "")}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

const fmt = (x) => (typeof x === "number" ? x.toFixed(1) : "—");
