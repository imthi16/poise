// POISE — "Thermal Governor" design system.
//
// The signature of this UI is thermochromic: a single accent color, `thermColor`,
// is derived from live junction temperature and drives the hero gauge, the depth
// ladder, and the page accent. The color scale is not decorative — it encodes the
// two values POISE fights to respect: the setpoint and the throttle ceiling.

export const tokens = {
  bg: "#0A0D11", // anodized instrument casing (neutral-cool so warmth reads as heat)
  panel: "#10151C",
  panel2: "#161D26",
  line: "#22303F", // brushed-steel hairline
  lineSoft: "#1A2531",
  ink: "#DCE6F0",
  muted: "#7C8C9E",
  mutedDim: "#566472",
  // Thermal ramp — cold → cool → setpoint → hot → throttle.
  cold: "#35B7E8",
  cool: "#2FD6B0",
  warm: "#F3C33F",
  hot: "#F0803A",
  crit: "#F0455B",
};

// The color ramp, positioned along a control-relevant parameter u ∈ [0,1].
const RAMP = [
  [0.0, [0x35, 0xb7, 0xe8]],
  [0.3, [0x2f, 0xd6, 0xb0]],
  [0.55, [0xf3, 0xc3, 0x3f]], // == setpoint
  [0.8, [0xf0, 0x80, 0x3a]],
  [1.0, [0xf0, 0x45, 0x5b]],
];

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const lerp = (a, b, t) => a + (b - a) * t;
const hex = (r, g, b) =>
  "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");

// Map temperature to the ramp parameter. Below the setpoint, temperature spans the
// lower 55% of the ramp (gradual warming during normal load); the setpoint→ceiling
// band is deliberately expanded to the top 45% — that narrow band is where POISE's
// control actually fights, so it earns the visual emphasis.
export function thermU(temp, ambient = 25, setpoint = 80, ceiling = 87) {
  if (temp == null || Number.isNaN(temp)) return 0;
  if (temp <= setpoint) {
    return clamp((0.55 * (temp - ambient)) / Math.max(1, setpoint - ambient), 0, 0.55);
  }
  return clamp(0.55 + (0.45 * (temp - setpoint)) / Math.max(1, ceiling - setpoint), 0, 1);
}

export function thermColor(temp, ambient = 25, setpoint = 80, ceiling = 87) {
  const u = thermU(temp, ambient, setpoint, ceiling);
  for (let i = 1; i < RAMP.length; i++) {
    const [p1, c1] = RAMP[i - 1];
    const [p2, c2] = RAMP[i];
    if (u <= p2) {
      const t = (u - p1) / (p2 - p1);
      return hex(lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t));
    }
  }
  return hex(...RAMP[RAMP.length - 1][1]);
}

// Discrete thermal zone → instrument-panel status label.
export function thermZone(temp, setpoint = 80, ceiling = 87) {
  if (temp == null) return { label: "—", key: "idle" };
  if (temp >= ceiling) return { label: "THROTTLING", key: "crit" };
  if (temp >= ceiling - 3) return { label: "THROTTLE RISK", key: "hot" };
  if (temp >= setpoint) return { label: "OVER SETPOINT", key: "warm" };
  if (temp >= setpoint - 8) return { label: "APPROACHING", key: "cool" };
  return { label: "NOMINAL", key: "cold" };
}
