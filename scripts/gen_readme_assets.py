#!/usr/bin/env python3
"""Generate the README vector assets for POISE — the 'Thermal Governor' visual system.

Static SVGs (GitHub renders these inline). Derived from the same thermochromic logic as
the live dashboard (poise/dashboard/src/theme.js) so they are renderings, not fabrications.
"""
from pathlib import Path
import math

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# ---- design tokens (mirror dashboard/src/theme.js) ----
BG, PANEL, PANEL2 = "#0A0D11", "#10151C", "#161D26"
LINE, LINE_SOFT = "#22303F", "#1A2531"
INK, MUTED, MUTED_DIM = "#DCE6F0", "#7C8C9E", "#566472"
WARM, CRIT, COOL = "#F3C33F", "#F0455B", "#2FD6B0"
MONO = "ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

AMBIENT, SETPOINT, CEILING, TOTAL = 25, 80, 87, 32
BUDGETS = [16, 20, 24, 28, 32]
RAMP = [(0.0, (0x35, 0xB7, 0xE8)), (0.3, (0x2F, 0xD6, 0xB0)), (0.55, (0xF3, 0xC3, 0x3F)),
        (0.8, (0xF0, 0x80, 0x3A)), (1.0, (0xF0, 0x45, 0x5B))]


def clamp(x, a, b): return max(a, min(b, x))
def lerp(a, b, t): return a + (b - a) * t


def therm_u(t):
    if t <= SETPOINT:
        return clamp(0.55 * (t - AMBIENT) / (SETPOINT - AMBIENT), 0, 0.55)
    return clamp(0.55 + 0.45 * (t - SETPOINT) / (CEILING - SETPOINT), 0, 1)


def therm(t):
    u = therm_u(t)
    for i in range(1, len(RAMP)):
        p1, c1 = RAMP[i - 1]
        p2, c2 = RAMP[i]
        if u <= p2:
            k = (u - p1) / (p2 - p1)
            return "#%02x%02x%02x" % tuple(round(lerp(c1[j], c2[j], k)) for j in range(3))
    return "#%02x%02x%02x" % RAMP[-1][1]


def control(t):  # PID intent: shed depth as temp climbs
    if t < SETPOINT - 6: return 32
    if t < SETPOINT - 2: return 28
    if t < SETPOINT + 1: return 24
    if t < CEILING - 2: return 20
    return 16


def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill=INK, family=MONO, weight=400, anchor="start", spacing=None, opacity=None):
    a = f' letter-spacing="{spacing}"' if spacing else ""
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{a}{o}>{esc(s)}</text>')


# simulate a coherent trajectory: load ramps, governor sheds depth, temp holds over setpoint
def sim(n=90):
    temp, load, pts = 68.0, 40.0, []
    for i in range(n):
        load = clamp(40 + 55 * (i / n), 0, 100)
        b = control(temp)
        heat = (load / 100) * (0.42 + 0.58 * b / TOTAL)
        temp += ((AMBIENT + heat * 68) - temp) * 0.10
        pts.append((temp, b))
    return pts


# ======================================================================== HERO
def hero():
    W, H = 1240, 300
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{SANS}" role="img" aria-label="POISE — depth follows physics">']
    s.append('<defs>')
    s.append(f'<radialGradient id="glow" cx="82%" cy="0%" r="70%">'
             f'<stop offset="0%" stop-color="{WARM}" stop-opacity="0.16"/>'
             f'<stop offset="55%" stop-color="{CRIT}" stop-opacity="0.05"/>'
             f'<stop offset="100%" stop-color="{BG}" stop-opacity="0"/></radialGradient>')
    s.append(f'<linearGradient id="tfill" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0%" stop-color="{WARM}" stop-opacity="0.28"/>'
             f'<stop offset="100%" stop-color="{WARM}" stop-opacity="0"/></linearGradient>')
    s.append('</defs>')
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(f'<rect width="{W}" height="{H}" fill="url(#glow)"/>')
    # hairline frame
    s.append(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" fill="none" stroke="{LINE}"/>')

    # left: wordmark + copy
    s.append(text(56, 96, "POISE", 66, INK, SANS, 800, spacing="14"))
    s.append(text(60, 132, "POWER-OPTIMIZED INFERENCE VIA STATE-AWARE EXECUTION", 13, MUTED, MONO, 500, spacing="3"))
    s.append(f'<rect x="60" y="150" width="46" height="2" fill="{WARM}"/>')
    for i, ln in enumerate([
        "An on-device LLM engine that varies per-token transformer",
        "depth in response to live hardware physics.",
    ]):
        s.append(text(60, 186 + i * 27, ln, 18, INK, SANS, 400))
    s.append(text(60, 250, "hardware-state-conditioned adaptive computation", 13, WARM, MONO, 500, spacing="1"))

    # right: the thesis — temperature climbs toward the ceiling while depth steps down.
    # Two stacked bands so the two scales never collide.
    gx, gy, gw, gh = 720, 58, 464, 200
    s.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="4" fill="{PANEL}" stroke="{LINE}"/>')
    px0, px1 = gx + 16, gx + gw - 16
    pts = sim(80)
    n = len(pts)
    def X(i): return px0 + i * (px1 - px0) / (n - 1)

    # --- temperature band (upper) ---
    t_top, t_bot, tmin, tmax = gy + 30, gy + 122, 60, 90
    def ty(v): return t_bot - (clamp(v, tmin, tmax) - tmin) / (tmax - tmin) * (t_bot - t_top)
    s.append(text(px0, gy + 22, "JUNCTION TEMPERATURE", 9, MUTED, MONO, 500, spacing="2"))
    s.append(f'<line x1="{px0}" y1="{ty(CEILING):.1f}" x2="{px1}" y2="{ty(CEILING):.1f}" stroke="{CRIT}" stroke-width="1.2" stroke-dasharray="2 2"/>')
    s.append(text(px1, ty(CEILING) - 6, f"ceiling {CEILING}°", 10, CRIT, MONO, 400, anchor="end"))
    s.append(f'<line x1="{px0}" y1="{ty(SETPOINT):.1f}" x2="{px1}" y2="{ty(SETPOINT):.1f}" stroke="{WARM}" stroke-width="1" stroke-dasharray="5 4" opacity="0.75"/>')
    s.append(text(px1, ty(SETPOINT) + 13, f"setpoint {SETPOINT}°", 10, WARM, MONO, 400, anchor="end"))
    tline = " ".join(f"{X(i):.1f},{ty(pts[i][0]):.1f}" for i in range(n))
    s.append(f'<polygon points="{px0},{t_bot} {tline} {px1},{t_bot}" fill="url(#tfill)"/>')
    s.append(f'<polyline points="{tline}" fill="none" stroke="{therm(pts[-1][0])}" stroke-width="2.4"/>')
    s.append(f'<circle cx="{X(n-1):.1f}" cy="{ty(pts[-1][0]):.1f}" r="3.5" fill="{therm(pts[-1][0])}"/>')

    # --- depth band (lower), rescaled 16..32 so the descent is unmistakable ---
    b_top, b_bot = gy + 150, gy + 184
    def by(v): return b_bot - (v - 16) / (32 - 16) * (b_bot - b_top)
    s.append(text(px0, gy + 146, "LAYER BUDGET   32 ▸ 16", 9, COOL, MONO, 500, spacing="2"))
    dparts = [f'M {X(0):.1f} {by(pts[0][1]):.1f}']
    for i in range(1, n):
        dparts.append(f'L {X(i):.1f} {by(pts[i-1][1]):.1f} L {X(i):.1f} {by(pts[i][1]):.1f}')
    s.append(f'<path d="{" ".join(dparts)}" fill="none" stroke="{COOL}" stroke-width="2"/>')
    s.append(f'<circle cx="{X(n-1):.1f}" cy="{by(pts[-1][1]):.1f}" r="3" fill="{COOL}"/>')

    s.append('</svg>')
    (OUT / "hero.svg").write_text("\n".join(s))
    print("wrote", OUT / "hero.svg")


# =================================================================== DASHBOARD
def dashboard():
    W, H = 1240, 812
    TEMP = 82.5
    ACC = therm(TEMP)           # amber-orange: over setpoint, governor holding
    BUD = control(TEMP)         # -> 20
    ERR = TEMP - SETPOINT
    ZONE = "OVER SETPOINT"
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{SANS}" role="img" aria-label="POISE Thermal Governor dashboard (rendering)">']
    s.append('<defs>')
    s.append(f'<radialGradient id="dglow" cx="84%" cy="-4%" r="60%">'
             f'<stop offset="0%" stop-color="{ACC}" stop-opacity="0.10"/>'
             f'<stop offset="100%" stop-color="{BG}" stop-opacity="0"/></radialGradient>')
    s.append(f'<linearGradient id="dtemp" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0%" stop-color="{ACC}" stop-opacity="0.34"/>'
             f'<stop offset="100%" stop-color="{ACC}" stop-opacity="0"/></linearGradient>')
    s.append(f'<linearGradient id="dbud" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0%" stop-color="{COOL}" stop-opacity="0.26"/>'
             f'<stop offset="100%" stop-color="{COOL}" stop-opacity="0"/></linearGradient>')
    s.append('</defs>')
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(f'<rect width="{W}" height="{H}" fill="url(#dglow)"/>')

    PAD = 28
    # ---- rail ----
    s.append(text(PAD, 34, "POISE", 15, INK, MONO, 700, spacing="6"))
    s.append(f'<line x1="{PAD+78}" y1="20" x2="{PAD+78}" y2="40" stroke="{LINE}"/>')
    s.append(text(PAD + 92, 33, "THERMAL GOVERNOR", 10, MUTED_DIM, MONO, 500, spacing="4"))
    rx = W - PAD
    s.append(text(rx, 33, "jetson · agx orin", 10, MUTED_DIM, MONO, 500, anchor="end", spacing="2"))
    s.append(f'<line x1="{rx-150}" y1="20" x2="{rx-150}" y2="40" stroke="{LINE}"/>')
    s.append(text(rx - 164, 33, ZONE, 11, ACC, MONO, 500, anchor="end", spacing="2"))
    s.append(f'<circle cx="{rx-288}" cy="29" r="4.5" fill="{ACC}"/>')
    s.append(f'<circle cx="{rx-288}" cy="29" r="8.5" fill="none" stroke="{ACC}" stroke-opacity="0.4"/>')
    s.append(f'<line x1="{PAD}" y1="52" x2="{W-PAD}" y2="52" stroke="{LINE}"/>')

    # ---- header ----
    s.append(text(PAD, 118, "Depth follows ", 40, INK, SANS, 680))
    s.append(text(PAD + 340, 118, "physics.", 40, ACC, SANS, 680))
    s.append(text(PAD, 150, "Per-token transformer depth is governed by the chip's live thermal state — not by input difficulty.", 15, MUTED, SANS, 400))
    # modes
    modes = [("static", "fixed depth", False), ("pid", "reactive control", True), ("ppo", "policy not loaded", None)]
    mx = PAD
    for mid, desc, active in modes:
        w = 150
        fill = f'{ACC}1A' if active else PANEL
        stroke = ACC if active else LINE
        op = ' opacity="0.42"' if active is None else ""
        s.append(f'<g{op}><rect x="{mx}" y="176" width="{w}" height="46" rx="2" fill="{PANEL}" stroke="{stroke}"/>')
        if active:
            s.append(f'<rect x="{mx}" y="176" width="{w}" height="46" rx="2" fill="{ACC}" fill-opacity="0.10"/>')
        s.append(text(mx + 14, 199, mid.upper(), 13, ACC if active else INK, MONO, 600, spacing="2"))
        s.append(text(mx + 14, 214, desc, 10, MUTED, SANS, 400))
        lock = " 🔒" if active is None else ""
        if lock:
            s.append(text(mx + w - 14, 199, "□", 12, MUTED, MONO, 400, anchor="end"))
        s.append('</g>')
        mx += w + 10

    # ---- panels ----
    ptop = 250
    pw = (W - 2 * PAD - 16) / 2
    lx, rxp = PAD, PAD + pw + 16
    ph = H - ptop - PAD
    for px in (lx, rxp):
        s.append(f'<rect x="{px}" y="{ptop}" width="{pw}" height="{ph}" rx="4" fill="{PANEL}" stroke="{LINE}"/>')

    def eyebrow(px, label):
        y = ptop + 30
        s.append(text(px + 20, y, label, 10, MUTED_DIM, MONO, 500, spacing="3"))
        s.append(f'<line x1="{px+20+len(label)*7.4}" y1="{y-4}" x2="{px+pw-20}" y2="{y-4}" stroke="{LINE_SOFT}"/>')

    # ===== LEFT: instrument =====
    eyebrow(lx, "JUNCTION · DEPTH")
    # gauge
    gcx = lx + 70
    g_top, g_bot = ptop + 60, ptop + 250
    smin, smax = AMBIENT, CEILING + 3
    def gy(v): return g_bot - (clamp(v, smin, smax) - smin) / (smax - smin) * (g_bot - g_top)
    bar_x, bar_w = gcx - 15, 30
    s.append(f'<rect x="{bar_x}" y="{g_top}" width="{bar_w}" height="{g_bot-g_top}" rx="4" fill="{BG}" stroke="{LINE}"/>')
    fy = gy(TEMP)
    s.append(f'<rect x="{bar_x}" y="{fy:.1f}" width="{bar_w}" height="{g_bot-fy:.1f}" rx="4" fill="{ACC}"/>')
    # markers
    s.append(f'<line x1="{bar_x-6}" y1="{gy(CEILING):.1f}" x2="{bar_x+bar_w+6}" y2="{gy(CEILING):.1f}" stroke="{CRIT}" stroke-width="1.4"/>')
    s.append(text(bar_x + bar_w + 10, gy(CEILING) + 3, f"{CEILING}°", 10, CRIT, MONO))
    s.append(f'<line x1="{bar_x-6}" y1="{gy(SETPOINT):.1f}" x2="{bar_x+bar_w+6}" y2="{gy(SETPOINT):.1f}" stroke="{WARM}" stroke-width="1" stroke-dasharray="3 3"/>')
    s.append(text(bar_x + bar_w + 10, gy(SETPOINT) + 3, f"{SETPOINT}°", 10, WARM, MONO))
    s.append(text(bar_x - 8, g_bot + 3, "25°", 9, MUTED_DIM, MONO, anchor="end"))
    s.append(text(gcx, g_bot + 42, f"{TEMP:.1f}", 34, ACC, MONO, 600, anchor="middle"))
    s.append(text(gcx + 52, g_bot + 42, "°C", 15, MUTED, MONO, anchor="middle"))
    s.append(text(gcx, g_bot + 62, f"JUNCTION · {ZONE}", 9, MUTED_DIM, MONO, 500, anchor="middle", spacing="1"))

    # ladder
    ladx = lx + 190
    lady = ptop + 66
    maxd = max(BUDGETS)
    for i, d in enumerate(sorted(BUDGETS, reverse=True)):
        yy = lady + i * 34
        on = (d == BUD)
        w = 60 + (d / maxd) * 150
        col = ACC if on else LINE
        s.append(f'<rect x="{ladx+34}" y="{yy}" width="{w:.0f}" height="9" rx="2" fill="{col}"/>')
        s.append(text(ladx, yy + 9, str(d), 13, ACC if on else MUTED_DIM, MONO, 600 if on else 400))
        if on:
            s.append(text(ladx + 34 + w + 12, yy + 9, "◄ active", 10, ACC, MONO, 500, spacing="1"))
    s.append(text(ladx, lady - 14, "DEPTH BUDGET", 10, MUTED_DIM, MONO, 500, spacing="2"))
    s.append(text(ladx, lady + len(BUDGETS) * 34 + 8, "32 layers total", 10, MUTED_DIM, MONO, 400))

    # readouts (4)
    def readout(px, py, w, label, val, unit, warn=False):
        s.append(f'<rect x="{px}" y="{py}" width="{w}" height="58" fill="{PANEL2}" stroke="{LINE_SOFT}"/>')
        s.append(text(px + 13, py + 22, label.upper(), 10, MUTED_DIM, MONO, 500, spacing="1"))
        vcol = CRIT if warn else INK
        s.append(text(px + 13, py + 44, val, 18, vcol, MONO, 500))
        if unit:
            s.append(text(px + 13 + len(val) * 10.5, py + 44, unit, 11, MUTED, MONO))
    roy = ptop + 326
    row = (pw - 40) / 4
    readout(lx + 20, roy, row, "Δ error", f"+{ERR:.1f}", "°C", warn=True)
    readout(lx + 20 + row, roy, row, "power", "27.3", "W")
    readout(lx + 20 + 2 * row, roy, row, "gpu clock", "1300", "MHz")
    readout(lx + 20 + 3 * row, roy, row, "throttle", "clear", "")

    # live trace
    trcy = ptop + 410
    trh = 108
    trx0, trx1 = lx + 44, lx + pw - 20
    s.append(text(lx + 20, trcy - 8, "TEMPERATURE ▸ BUDGET", 10, MUTED, MONO, 500, spacing="2"))
    s.append(text(lx + pw - 20, trcy - 8, "live · 2 Hz", 10, MUTED_DIM, MONO, anchor="end"))
    tmin, tmax = AMBIENT, CEILING + 2
    def trty(v): return trcy + (1 - (clamp(v, tmin, tmax) - tmin) / (tmax - tmin)) * trh
    def trby(v): return trcy + (1 - v / TOTAL) * trh
    pts = sim(80); n = len(pts)
    def trx(i): return trx0 + i * (trx1 - trx0) / (n - 1)
    for gv in range(0, 5):
        gt = tmin + (tmax - tmin) * gv / 4
        yy = trty(gt)
        s.append(f'<line x1="{trx0}" y1="{yy:.1f}" x2="{trx1}" y2="{yy:.1f}" stroke="{LINE_SOFT}"/>')
        s.append(text(lx + 38, yy + 3, f"{round(gt)}°", 9, MUTED_DIM, MONO, anchor="end"))
    s.append(f'<line x1="{trx0}" y1="{trty(SETPOINT):.1f}" x2="{trx1}" y2="{trty(SETPOINT):.1f}" stroke="{WARM}" stroke-dasharray="5 4" opacity="0.6"/>')
    s.append(f'<line x1="{trx0}" y1="{trty(CEILING):.1f}" x2="{trx1}" y2="{trty(CEILING):.1f}" stroke="{CRIT}" stroke-dasharray="2 2" opacity="0.7"/>')
    # budget area
    dparts = [f'M {trx(0):.1f} {trby(pts[0][1]):.1f}']
    for i in range(1, n):
        dparts.append(f'L {trx(i):.1f} {trby(pts[i-1][1]):.1f} L {trx(i):.1f} {trby(pts[i][1]):.1f}')
    dpath = " ".join(dparts)
    s.append(f'<path d="{dpath} L {trx(n-1):.1f} {trcy+trh} L {trx(0):.1f} {trcy+trh} Z" fill="url(#dbud)"/>')
    s.append(f'<path d="{dpath}" fill="none" stroke="{COOL}" stroke-width="1.6" opacity="0.8"/>')
    # temp area+line
    tline = " ".join(f"{trx(i):.1f},{trty(pts[i][0]):.1f}" for i in range(n))
    s.append(f'<polygon points="{trx0},{trcy+trh} {tline} {trx1},{trcy+trh}" fill="url(#dtemp)"/>')
    s.append(f'<polyline points="{tline}" fill="none" stroke="{ACC}" stroke-width="2"/>')
    s.append(f'<circle cx="{trx(n-1):.1f}" cy="{trty(pts[-1][0]):.1f}" r="3" fill="{ACC}"/>')

    # ===== RIGHT: workload =====
    eyebrow(rxp, "WORKLOAD")
    # prompt box
    s.append(f'<rect x="{rxp+20}" y="{ptop+70}" width="{pw-40}" height="64" rx="3" fill="{BG}" stroke="{LINE}"/>')
    s.append(text(rxp + 20, ptop + 62, "PROMPT", 10, MUTED_DIM, MONO, 500, spacing="2"))
    s.append(text(rxp + 34, ptop + 100, "Explain thermal throttling in one", 13, INK, MONO))
    s.append(text(rxp + 34, ptop + 120, "paragraph.", 13, INK, MONO))
    # run button
    s.append(f'<rect x="{rxp+20}" y="{ptop+150}" width="{pw-40}" height="42" rx="3" fill="{PANEL2}" stroke="{ACC}"/>')
    s.append(text(rxp + pw / 2, ptop + 176, "RUN · PID", 13, ACC, MONO, 600, anchor="middle", spacing="2"))
    # metrics (6, 3col)
    metrics = [("throughput", "23.4", "tok/s", False), ("energy", "0.318", "J/tok", False),
               ("peak temp", "84.6", "°C", False), ("mean depth", "22.4", "layers", False),
               ("tokens", "64", "", False), ("throttles", "0", "", False)]
    mcol = (pw - 40) / 3
    my0 = ptop + 210
    for i, (lab, val, unit, warn) in enumerate(metrics):
        c, r = i % 3, i // 3
        px = rxp + 20 + c * mcol
        py = my0 + r * 58
        s.append(f'<rect x="{px}" y="{py}" width="{mcol}" height="58" fill="{PANEL2}" stroke="{LINE_SOFT}"/>')
        s.append(text(px + 12, py + 22, lab.upper(), 9, MUTED_DIM, MONO, 500, spacing="1"))
        s.append(text(px + 12, py + 44, val, 17, INK, MONO, 500))
        if unit:
            s.append(text(px + 12 + len(val) * 10, py + 44, unit, 10, MUTED, MONO))

    # per-token trace
    ptry = ptop + 352
    ptrh = 150
    ptrx0, ptrx1 = rxp + 44, rxp + pw - 20
    s.append(text(rxp + 20, ptry - 8, "PER-TOKEN BUDGET", 10, MUTED, MONO, 500, spacing="2"))
    s.append(text(rxp + pw - 20, ptry - 8, "depth per emitted token", 10, MUTED_DIM, MONO, anchor="end"))
    def pby(v): return ptry + (1 - v / TOTAL) * ptrh
    N = 64
    tokpts, t = [], 70.0
    for i in range(N):
        b = control(t)
        t += ((AMBIENT + 0.5 * (0.42 + 0.58 * b / TOTAL) * 70) - t) * 0.05 + 0.18
        t = clamp(t, AMBIENT, CEILING + 1)
        tokpts.append((t, b))
    def ptx(i): return ptrx0 + i * (ptrx1 - ptrx0) / (N - 1)
    for gv in range(0, 33, 8):
        yy = pby(gv)
        s.append(f'<line x1="{ptrx0}" y1="{yy:.1f}" x2="{ptrx1}" y2="{yy:.1f}" stroke="{LINE_SOFT}"/>')
        s.append(text(rxp + 38, yy + 3, str(gv), 9, MUTED_DIM, MONO, anchor="end"))
    dparts = [f'M {ptx(0):.1f} {pby(tokpts[0][1]):.1f}']
    for i in range(1, N):
        dparts.append(f'L {ptx(i):.1f} {pby(tokpts[i-1][1]):.1f} L {ptx(i):.1f} {pby(tokpts[i][1]):.1f}')
    dpath = " ".join(dparts)
    s.append(f'<path d="{dpath} L {ptx(N-1):.1f} {ptry+ptrh} L {ptx(0):.1f} {ptry+ptrh} Z" fill="url(#dbud)"/>')
    s.append(f'<path d="{dpath}" fill="none" stroke="{COOL}" stroke-width="2"/>')
    tl = " ".join(f"{ptx(i):.1f},{ (ptry + (1-(clamp(tokpts[i][0],AMBIENT,CEILING+1)-AMBIENT)/(CEILING+1-AMBIENT))*ptrh):.1f}" for i in range(N))
    s.append(f'<polyline points="{tl}" fill="none" stroke="{ACC}" stroke-width="1.4" opacity="0.85"/>')

    s.append('</svg>')
    (OUT / "dashboard.svg").write_text("\n".join(s))
    print("wrote", OUT / "dashboard.svg")


hero()
dashboard()
