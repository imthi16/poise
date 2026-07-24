import React from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Line,
} from "recharts";
import { tokens } from "../theme.js";

// The instrument face: a junction-temperature gauge and a depth ladder that move in
// opposition — heat rises on the left, the layer budget sheds on the right — plus the
// secondary readouts and a live thermochromic trace.
export default function InstrumentPanel({ state, tele, history, therm, zone }) {
  const ambient = state?.ambient_c ?? 25;
  const setpoint = state?.setpoint_c ?? 80;
  const ceiling = state?.ceiling_c ?? 87;
  const temp = state?.temp_c;
  const budget = state?.current_budget;
  const total = state?.layer_total ?? 32;
  const budgetSet = state?.budget_set ?? [16, 20, 24, 28, 32];
  const power = tele?.power_w ?? state?.power_w;
  const clock = tele?.gpu_clock_mhz ?? state?.gpu_clock_mhz;
  const util = tele?.gpu_util ?? state?.gpu_util;
  const throttled = state?.throttled;
  const err = temp != null ? temp - setpoint : null; // the actual PID error signal

  return (
    <section className="gov-panel gov-instrument">
      <p className="gov-eyebrow">Junction · Depth</p>

      <div className="gov-hero">
        <Gauge temp={temp} ambient={ambient} setpoint={setpoint} ceiling={ceiling} therm={therm} zone={zone} />
        <Ladder budget={budget} total={total} budgetSet={budgetSet} therm={therm} />
      </div>

      <div className="gov-readouts gov-readouts-4">
        <Readout label="Δ error" value={err == null ? "—" : signed(err)} unit="°C"
          warn={err != null && err > 0} />
        <Readout label="Power" value={fmt(power, 1)} unit="W" />
        <Readout label="GPU clock" value={fmt(clock, 0)} unit="MHz" />
        <Readout label="Throttle" value={throttled ? "ACTIVE" : "clear"} warn={throttled} />
      </div>

      <div className="gov-trace">
        <div className="gov-trace-head">
          <span className="gov-trace-title">TEMPERATURE ▸ BUDGET</span>
          <span className="gov-trace-sub">{history.length} samples · 2 Hz</span>
        </div>
        <ResponsiveContainer width="100%" height={170}>
          <AreaChart data={history} margin={{ top: 6, right: 6, bottom: 0, left: -14 }}>
            <defs>
              <linearGradient id="tempFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={therm} stopOpacity={0.32} />
                <stop offset="100%" stopColor={therm} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={tokens.lineSoft} vertical={false} />
            <XAxis dataKey="t" tick={false} axisLine={{ stroke: tokens.line }} height={4} />
            <YAxis yAxisId="temp" domain={[ambient, ceiling + 2]} width={34}
              tick={{ fill: tokens.mutedDim, fontSize: 10, fontFamily: "monospace" }}
              axisLine={false} tickLine={false} />
            <YAxis yAxisId="b" orientation="right" domain={[0, total]} hide />
            <Tooltip
              contentStyle={{
                background: tokens.panel2, border: `1px solid ${tokens.line}`, borderRadius: 3,
                fontFamily: "monospace", fontSize: 12, color: tokens.ink,
              }}
              labelStyle={{ color: tokens.mutedDim }} />
            <ReferenceLine yAxisId="temp" y={ceiling} stroke={tokens.crit} strokeWidth={1}
              strokeDasharray="2 2" label={{ value: `ceiling ${ceiling}°`, fill: tokens.crit, fontSize: 10, position: "insideTopRight" }} />
            <ReferenceLine yAxisId="temp" y={setpoint} stroke={tokens.warm} strokeWidth={1}
              strokeDasharray="5 4" label={{ value: `setpoint ${setpoint}°`, fill: tokens.warm, fontSize: 10, position: "insideBottomRight" }} />
            <Area yAxisId="temp" type="monotone" dataKey="temp_c" stroke={therm} strokeWidth={2}
              fill="url(#tempFill)" dot={false} isAnimationActive={false} name="temp °C" />
            <Line yAxisId="b" type="stepAfter" dataKey="budget" stroke={tokens.cool} strokeWidth={1.5}
              strokeOpacity={0.7} dot={false} isAnimationActive={false} name="budget" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function Gauge({ temp, ambient, setpoint, ceiling, therm, zone }) {
  const W = 132, H = 300, pad = 22;
  const top = pad, bottom = H - pad;
  const scaleMin = ambient, scaleMax = ceiling + 3;
  const y = (v) => bottom - ((clampN(v, scaleMin, scaleMax) - scaleMin) / (scaleMax - scaleMin)) * (bottom - top);
  const barX = W / 2 - 13, barW = 26;
  const fillY = temp != null ? y(temp) : bottom;
  const known = temp != null;

  return (
    <div className="gov-gauge">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label={`Junction temperature ${known ? temp.toFixed(1) : "unknown"} degrees`}>
        {/* track */}
        <rect x={barX} y={top} width={barW} height={bottom - top} rx={4}
          fill={tokens.bg} stroke={tokens.line} />
        {/* fill */}
        <rect x={barX} y={fillY} width={barW} height={bottom - fillY} rx={4}
          fill={therm} style={{ transition: "y .6s cubic-bezier(.4,0,.2,1), height .6s cubic-bezier(.4,0,.2,1), fill .5s ease" }} />
        {/* ceiling marker */}
        <Marker y={y(ceiling)} W={W} barX={barX} barW={barW} color={tokens.crit} label={`${ceiling}°`} solid />
        {/* setpoint marker */}
        <Marker y={y(setpoint)} W={W} barX={barX} barW={barW} color={tokens.warm} label={`${setpoint}°`} />
        {/* ambient tick */}
        <text x={barX - 8} y={bottom + 4} textAnchor="end" fill={tokens.mutedDim}
          fontSize="9" fontFamily="monospace">{Math.round(ambient)}°</text>
      </svg>
      <div className="gov-gauge-readout">
        <div className="gov-gauge-value" style={{ color: therm }}>
          {known ? temp.toFixed(1) : "––.–"}<span className="gov-gauge-unit">°C</span>
        </div>
        <div className="gov-gauge-label">JUNCTION · {zone?.label ?? "—"}</div>
      </div>
    </div>
  );
}

function Marker({ y, W, barX, barW, color, label, solid }) {
  return (
    <g style={{ transition: "transform .5s ease" }}>
      <line x1={barX - 5} y1={y} x2={barX + barW + 5} y2={y} stroke={color}
        strokeWidth={solid ? 1.5 : 1} strokeDasharray={solid ? "" : "3 3"} />
      <text x={barX + barW + 9} y={y + 3} fill={color} fontSize="10" fontFamily="monospace">{label}</text>
    </g>
  );
}

function Ladder({ budget, total, budgetSet, therm }) {
  const rungs = [...budgetSet].sort((a, b) => b - a); // full depth at top
  const maxD = Math.max(...budgetSet, total);
  return (
    <div className="gov-ladder">
      <div className="gov-ladder-cap">DEPTH BUDGET</div>
      <div className="gov-ladder-rungs">
        {rungs.map((d) => {
          const active = d === budget;
          const w = 38 + (d / maxD) * 62; // wider rung = deeper compute
          return (
            <div key={d} className={"gov-rung" + (active ? " on" : "")}>
              <span className="gov-rung-bar" style={{ width: `${w}%`, ...(active ? { background: therm, boxShadow: `0 0 14px -2px ${therm}` } : {}) }} />
              <span className="gov-rung-num" style={active ? { color: therm } : undefined}>{d}</span>
              {active && <span className="gov-rung-mark" style={{ color: therm }}>◄ active</span>}
            </div>
          );
        })}
      </div>
      <div className="gov-ladder-foot">{total} layers total</div>
    </div>
  );
}

function Readout({ label, value, unit, warn }) {
  return (
    <div className={"gov-ro" + (warn ? " warn" : "")}>
      <div className="gov-ro-label">{label}</div>
      <div className="gov-ro-value">{value}{unit && value !== "—" && <span className="u">{unit}</span>}</div>
    </div>
  );
}

const clampN = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const fmt = (x, d = 1) => (typeof x === "number" ? x.toFixed(d) : "—");
const signed = (x) => (x >= 0 ? "+" : "") + x.toFixed(1);
