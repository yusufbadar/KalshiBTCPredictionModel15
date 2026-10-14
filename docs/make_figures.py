"""Draws the pictures used in README.md.

    python docs/make_figures.py

Reads the reports written by ``python -m kalshi_bot.research.report``
(storage/reports/backtest*.json) and by ``python -m kalshi_bot.research.tape_replay
--per-market 0.10 --per-close 0.15`` (storage/reports/tape_replay.json) and
writes black-and-white SVGs to docs/img/. Plain Python - no plotting library.
The live numbers are the ones recorded in RESEARCH.md sections 7 and 8.
"""
from __future__ import annotations

import html
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "storage" / "reports"
OUT = ROOT / "docs" / "img"

# the dashboard's palette
BG, PANEL, LINE, LINE2 = "#000000", "#0c0c0c", "#1f1f1f", "#303030"
TEXT, MUTED, DIM, HI = "#f4f4f4", "#9c9c9c", "#5a5a5a", "#ffffff"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,'JetBrains Mono',Menlo,Consolas,'Liberation Mono',monospace"
W = 1200

# first live day (RESEARCH.md section 7)
LIVE = {"trades": 24, "wins": 21, "losses": 3, "net": -10.06, "avg_win": 0.57, "avg_loss": -7.36,
        "dd_before": 27.70, "dd_after": 14.40, "replay_days": 79}

# 13 Oct 2026, 1:44 PM ET, market KXBTC15M-26OCT131345-45 (RESEARCH.md section 8), around the bot's losing buy.
# From the public trade tape and Kalshi's published one-second index.
# MOVED: (seconds from the order, lowest and highest price YES traded at on Kalshi in that quarter second)
MOVED = ((-3.00, .97, .98), (-2.75, .97, .97), (-2.50, .97, .97), (-2.25, .97, .98), (-2.00, .98, .98),
         (-1.75, .98, .98), (-1.50, .98, .99), (-1.25, .98, .99), (-1.00, .98, .99), (-0.75, .98, .99),
         (-0.50, .98, .99), (-0.25, .53, .99), (0.00, .53, .81), (0.25, .56, .61), (0.50, .58, .62),
         (0.75, .57, .62), (1.00, .53, .58), (1.25, .51, .54), (1.50, .48, .52), (1.75, .48, .51), (2.00, .46, .49))
# MOVED_INDEX: (seconds from the order at which a new index value was stamped, Bitcoin against the line in dollars)
MOVED_INDEX = ((-3.00, 11.15), (-2.06, 11.45), (-1.06, 12.25), (-0.06, 8.91), (0.94, -14.36), (1.94, -13.86))
MOVED_BUY = {"price": 0.81, "model": 95.7, "sold": 27_703}


class Svg:
    def __init__(self, w: int, h: int, label: str):
        self.w, self.h, self.label = w, h, label
        self.body: list[str] = []
        self.defs: list[str] = []
        self.css: list[str] = []

    def add(self, s: str) -> "Svg":
        self.body.append(s)
        return self

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1.5, rx=0, extra=""):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        return self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st}{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=LINE2, sw=1.5, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                        f'stroke-width="{sw}"{d}{extra}/>')

    def circle(self, cx, cy, r, fill="none", stroke=None, sw=1.5, extra=""):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        return self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"{st}{extra}/>')

    def path(self, d, fill="none", stroke=HI, sw=2.0, extra=""):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        return self.add(f'<path d="{d}" fill="{fill}"{st} stroke-linejoin="round" stroke-linecap="round"{extra}/>')

    def text(self, x, y, s, size=20, fill=TEXT, weight=400, anchor="start", mono=False, ls=0.0, extra=""):
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        sp = f' letter-spacing="{ls}"' if ls else ""
        c = ' class="m"' if mono else ""
        return self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}"'
                        f'{a}{sp}{c}{extra}>{html.escape(str(s))}</text>')

    def hatch(self) -> str:
        """Diagonal stripes: the black-and-white way to mark a loss."""
        if not any('id="hatch"' in d for d in self.defs):
            self.defs.append('<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" '
                             'patternTransform="rotate(45)"><rect width="7" height="7" fill="#000"/>'
                             '<rect width="3" height="7" fill="#c8c8c8"/></pattern>')
        return "url(#hatch)"

    def save(self, name: str) -> None:
        css = f"text{{font-family:{SANS}}}.m{{font-family:{MONO}}}" + "".join(self.css)
        label = html.escape(self.label)
        out = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
               f'height="{self.h}" role="img" aria-label="{label}"><title>{label}</title><style>{css}</style>'
               + (f'<defs>{"".join(self.defs)}</defs>' if self.defs else "") + "".join(self.body) + "</svg>\n")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(out, encoding="utf-8", newline="\n")
        print(f"  {name:18s} {len(out) / 1024:6.1f} KB")


def frame(h: int, label: str, kicker: str, title: str, sub: str | None = None) -> Svg:
    s = Svg(W, h, label)
    s.rect(1, 1, W - 2, h - 2, fill=BG, stroke=LINE2, sw=2, rx=24)
    s.text(48, 60, kicker.upper(), size=16, fill=MUTED, weight=700, ls=2.6)
    s.text(48, 106, title, size=34, weight=700, fill=HI)
    if sub:
        s.text(48, 142, sub, size=21, fill=MUTED)
    return s


def pill(s: Svg, x: float, y: float, txt: str, size: int = 15, solid: bool = False, anchor: str = "start") -> float:
    """Rounded label; returns its width. ``anchor='end'`` puts its right edge at x."""
    w = len(txt) * size * 0.62 + 28
    h = size + 18
    if anchor == "end":
        x -= w
    s.rect(x, y, w, h, fill=HI if solid else "none", stroke=None if solid else LINE2, rx=h / 2)
    s.text(x + w / 2, y + h / 2 + size * 0.36, txt, size=size, weight=700, fill="#000" if solid else TEXT,
           anchor="middle", mono=True, ls=0.6)
    return w


def check(s: Svg, x: float, y: float, size: float = 9) -> None:
    s.path(f"M{x - size:.1f},{y:.1f} l{size * 0.7:.1f},{size * 0.75:.1f} l{size * 1.3:.1f},{-size * 1.5:.1f}", sw=2.6)


def smooth(a: float, b: float, u: float) -> float:
    t = min(1.0, max(0.0, (u - a) / (b - a)))
    return t * t * (3 - 2 * t)


def money(v: float, cents: bool = False) -> str:
    sign = "−" if v < 0 else "+"
    return f"{sign}${abs(v):,.2f}" if cents else f"{sign}${abs(v):,.0f}"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- hero
def hero() -> None:
    h = 440
    s = Svg(W, h, "Lock-In: a trading bot for Kalshi's 15-minute Bitcoin markets. "
                  "It waits for the final minute and buys the near-certain side.")
    s.rect(1, 1, W - 2, h - 2, fill=BG, stroke=LINE2, sw=2, rx=24)
    s.rect(56, 58, 58, 58, rx=15, stroke=HI, sw=3.5)
    s.text(85, 96, "15", size=25, weight=700, anchor="middle", mono=True)
    s.text(134, 102, "LOCK-IN", size=44, weight=800, fill=HI, ls=4)
    s.text(56, 200, "Waits for the final minute.", size=38, weight=700, fill=HI)
    s.text(56, 248, "Buys the near-certain side.", size=38, weight=700, fill=MUTED)
    s.text(56, 300, "A trading bot for Kalshi's 15-minute Bitcoin markets.", size=21, fill=MUTED)
    x = 56.0
    for chip in ("BTC 15-MIN + HOURLY", "PAPER · DEMO · LIVE", "PYTHON 3.12+"):
        x += pill(s, x, 340, chip, size=14) + 12

    # chart: the last three minutes of a market
    x0, x1, y0, y1, t0 = 716, 1136, 66, 362, -180
    X = lambda t: x0 + (t - t0) / -t0 * (x1 - x0)  # noqa: E731
    ys = 252
    for gy in (y0 + 52, y0 + 126, y0 + 260):
        s.line(x0, gy, x1, gy, stroke=LINE)
    s.rect(X(-60), y0, x1 - X(-60), y1 - y0, fill=HI, rx=6, extra=' opacity="0.075"')
    s.text((X(-60) + x1) / 2, y0 + 26, "FINAL 60 SECONDS", size=13, fill=MUTED, weight=700, anchor="middle", ls=1.8)
    s.line(x0, ys, x1, ys, stroke=MUTED, sw=1.5, dash="6 7")
    s.text(x1 - 8, ys + 25, "the line to beat", size=15, fill=MUTED, anchor="end", mono=True)
    for t, lab, anc in ((-180, "3 min left", "start"), (-60, "1 min left", "middle"), (0, "close", "end")):
        s.text(X(t), y1 + 26, lab, size=14, fill=MUTED, anchor=anc, mono=True)
    rnd, noise, pts = random.Random(15), 0.0, []
    for t in range(t0, 1):
        u = (t - t0) / -t0
        base = -0.62 + 1.75 * smooth(0.18, 0.52, u) + 0.16 * math.sin(7.5 * u) - 0.10 * smooth(0.75, 1.0, u)
        noise = 0.82 * noise + rnd.gauss(0, 0.075)
        pts.append((X(t), ys - (base + noise) * 84))
    d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    ib = -t0 - 42                                        # the bot buys with 42 s left
    fb = 50 * sum(seg[:ib]) / sum(seg)                   # % of the loop at which the line reaches that point
    bx, by = pts[ib]
    ex, ey = pts[-1]
    s.add('<g class="lp">')
    s.path(d, sw=3, extra=' pathLength="1" class="ln"')
    s.add('<g class="by">')
    s.circle(bx, by, 9, stroke=HI, sw=2, extra=' class="pu"')
    s.circle(bx, by, 6.5, fill=HI, stroke=BG, sw=2)
    pill(s, bx - 72, by - 62, "BUY YES 93¢", size=14, solid=True)
    s.add("</g>")
    s.add('<g class="en">')
    s.circle(ex, ey, 6.5, fill=HI, stroke=BG, sw=2)
    s.text(x1 - 8, ys - 16, "settles YES · pays $1", size=15, fill=TEXT, weight=700, anchor="end", mono=True)
    s.add("</g></g>")
    # default state is the finished picture; the animation only plays where it is supported
    s.css.append(
        ".ln{stroke-dasharray:1;stroke-dashoffset:0;animation:ln 10s linear infinite}"
        "@keyframes ln{0%{stroke-dashoffset:1}50%,100%{stroke-dashoffset:0}}"
        ".by{animation:by 10s linear infinite}"
        f"@keyframes by{{0%,{fb:.1f}%{{opacity:0}}{fb + 2:.1f}%,95%{{opacity:1}}100%{{opacity:0}}}}"
        ".en{animation:en 10s linear infinite}@keyframes en{0%,50%{opacity:0}52%,95%{opacity:1}100%{opacity:0}}"
        ".lp{animation:lp 10s linear infinite}@keyframes lp{0%{opacity:0}2%,95%{opacity:1}100%{opacity:0}}"
        ".pu{transform-box:fill-box;transform-origin:center;animation:pu 1.6s ease-out infinite}"
        "@keyframes pu{0%{transform:scale(.6);opacity:.9}100%{transform:scale(2.6);opacity:0}}"
        "@media (prefers-reduced-motion:reduce){.ln,.by,.en,.lp,.pu{animation:none}.pu{opacity:0}}")
    s.save("hero.svg")


# --------------------------------------------------------------------------- the market
def market() -> None:
    s = frame(420, "A Kalshi market is a yes-or-no question. Each side costs between 1 and 99 cents and the "
                   "winning side pays one dollar.",
              "The market", "Every 15 minutes, one yes-or-no question",
              "You can buy YES or NO. The side that turns out right pays $1.00. The other pays nothing.")
    tx, ty, tw, th = 48, 176, 548, 204
    s.rect(tx, ty, tw, th, fill=PANEL, stroke=LINE2, rx=18)
    s.text(tx + 26, ty + 40, "KALSHI · BITCOIN · 15 MINUTES", size=14, fill=MUTED, weight=700, mono=True, ls=1.6)
    s.text(tx + 26, ty + 82, "Will Bitcoin be above", size=27, weight=600, fill=HI)
    s.text(tx + 26, ty + 118, "$83,581.29 at 3:15 PM?", size=27, weight=600, fill=HI)
    bw = (tw - 52 - 16) / 2
    s.rect(tx + 26, ty + 142, bw, 44, fill=HI, rx=11)
    s.text(tx + 26 + bw / 2, ty + 171, "YES   94¢", size=20, weight=700, fill="#000", anchor="middle", mono=True)
    s.rect(tx + 42 + bw, ty + 142, bw, 44, stroke=MUTED, rx=11)
    s.text(tx + 42 + bw + bw / 2, ty + 171, "NO   7¢", size=20, weight=700, anchor="middle", mono=True)
    rx0, rx1 = 640, 1152
    for i, (cond, pay) in enumerate((("If it finishes above", "YES pays $1.00"), ("If it finishes below", "NO pays $1.00"))):
        y = ty + i * 76
        s.rect(rx0, y, rx1 - rx0, 62, fill=PANEL, stroke=LINE, rx=14)
        s.text(rx0 + 24, y + 39, cond, size=20, fill=TEXT)
        s.text(rx1 - 24, y + 39, pay, size=20, weight=700, fill=HI, anchor="end", mono=True)
    s.text(rx0 + 2, ty + 178, "The price is the crowd's odds:", size=19, fill=MUTED)
    s.text(rx0 + 2, ty + 204, "94¢ means about a 94% chance.", size=19, fill=TEXT, weight=600)
    s.save("market.svg")


# --------------------------------------------------------------------------- the final minute
def timeline() -> None:
    s = frame(520, "A 15-minute market: for 14 minutes the bot only watches. The result is the average of the 60 "
                   "one-second prices in the final minute. With 25 seconds left, 35 of the 60 are already known.",
              "How a market ends", "The result is the average of the last 60 seconds",
              "Kalshi takes one Bitcoin price per second during the final minute and averages all 60.")
    bw, gap, y = 66, 6, 198
    total = 15 * bw + 14 * gap
    x0 = (W - total) / 2
    s.text(x0, y - 14, "3:00 PM · opens", size=16, fill=MUTED, mono=True)
    s.text(x0 + total / 2 - 36, y - 14, "14 minutes: the bot only watches", size=17, fill=MUTED, anchor="middle")
    s.text(x0 + total, y - 14, "3:15 PM · closes", size=16, fill=MUTED, anchor="end", mono=True)
    for i in range(15):
        bx = x0 + i * (bw + gap)
        if i == 14:
            s.rect(bx, y, bw, 40, fill=HI, rx=8)
        else:
            s.rect(bx, y, bw, 40, fill=PANEL, stroke=LINE2, rx=8)
    ys, hs = 304, 84
    lx = x0 + 14 * (bw + gap)
    s.add(f'<polygon points="{lx:.1f},{y + 40} {lx + bw:.1f},{y + 40} {x0 + total:.1f},{ys} {x0:.1f},{ys}" '
          f'fill="{HI}" opacity="0.07"/>')
    sw, sg, done = (total - 59 * 4) / 60, 4, 35
    for i in range(60):
        bx = x0 + i * (sw + sg)
        if i < done:
            s.rect(bx, ys, sw, hs, fill=HI, rx=3)
        else:
            s.rect(bx, ys, sw, hs, stroke=DIM, sw=1.5, rx=3)
    nx = x0 + done * (sw + sg) - sg / 2
    s.line(nx, ys - 16, nx, ys + hs + 16, stroke=HI, sw=2.5)
    s.rect(nx - 78, ys - 46, 156, 30, fill=BG, stroke=HI, sw=1.5, rx=15)
    s.text(nx, ys - 25, "NOW · 25 s left", size=14, weight=700, anchor="middle", mono=True, ls=0.6)
    by = ys + hs + 30
    s.line(x0, by, nx - 10, by, stroke=HI, sw=2)
    s.line(nx + 10, by, x0 + total, by, stroke=DIM, sw=2, dash="5 6")
    s.text((x0 + nx - 10) / 2, by + 30, "35 prices already in. They cannot change.", size=19, fill=HI, weight=600,
           anchor="middle")
    s.text((nx + 10 + x0 + total) / 2, by + 30, "25 still to come", size=19, fill=MUTED, anchor="middle")
    s.text(48, 488, "More than half of the answer is locked in. The rest can only nudge it.", size=21, fill=TEXT)
    s.save("final-minute.svg")


def funnel() -> None:
    s = frame(540, "Chart: how far the result can still move as the close approaches. Kalshi's 60-second average "
                   "firms up much faster than a market that only looks at the final price.",
              "Why the final minute", "The answer firms up fast",
              "Room left for the result to move, next to a market that only looked at the very last price.")
    px0, px1, py0, py1 = 112, 1140, 196, 430
    X = lambda m: px0 + (60 - m) / 60 * (px1 - px0)  # noqa: E731
    Y = lambda v: py1 - v / 100 * (py1 - py0)  # noqa: E731
    spot = lambda m: 100 * math.sqrt(m / 60)  # noqa: E731
    avg = lambda m: 100 * math.sqrt(m * (m + 1) * (2 * m + 1) / 21600 / 60)  # noqa: E731  (model/fair_value.py)
    for v in (0, 50, 100):
        s.line(px0, Y(v), px1, Y(v), stroke=LINE)
        s.text(px0 - 14, Y(v) + 6, f"{v}%", size=16, fill=MUTED, anchor="end", mono=True)
    for m, lab in ((60, "60 s left"), (45, "45"), (30, "30"), (15, "15"), (0, "close")):
        s.line(X(m), py1, X(m), py1 + 8, stroke=LINE2)
        s.text(X(m), py1 + 32, lab, size=16, fill=MUTED, anchor="middle", mono=True)
    ms = [60 - i * 0.5 for i in range(121)]
    area = f"M{X(60):.1f},{Y(0):.1f} " + " ".join(f"L{X(m):.1f},{Y(avg(m)):.1f}" for m in ms) + " Z"
    s.path(area, fill=HI, stroke=None, extra=' opacity="0.13"')
    s.path("M" + " L".join(f"{X(m):.1f},{Y(spot(m)):.1f}" for m in ms), stroke=MUTED, sw=2.5, extra=' stroke-dasharray="7 7"')
    s.path("M" + " L".join(f"{X(m):.1f},{Y(avg(m)):.1f}" for m in ms), sw=3.5)
    for m in (30, 10):
        s.line(X(m), Y(spot(m)), X(m), Y(0), stroke=DIM, sw=1.5, dash="3 5")
        s.circle(X(m), Y(spot(m)), 6, fill=BG, stroke=MUTED, sw=2.5)
        s.circle(X(m), Y(avg(m)), 6.5, fill=HI, stroke=BG, sw=2)
        s.text(X(m) + 12, Y(spot(m)) - 12, f"{spot(m):.0f}%", size=19, fill=MUTED, weight=700, mono=True)
        s.text(X(m) + 12, Y(avg(m)) - 12, f"{avg(m):.0f}%", size=19, fill=HI, weight=700, mono=True)
    lx, ly = px1 - 430, py0 + 6
    s.line(lx, ly, lx + 44, ly, stroke=MUTED, sw=2.5, dash="7 7")
    s.text(lx + 56, ly + 6, "if only the final price counted", size=18, fill=MUTED)
    s.line(lx, ly + 30, lx + 44, ly + 30, stroke=HI, sw=3.5)
    s.text(lx + 56, ly + 36, "Kalshi's 60-second average", size=18, fill=HI, weight=600)
    s.text(48, 508, "With 30 seconds left the average has about a third of the wiggle room. "
                    "With 10 seconds left, a tenth.", size=21, fill=TEXT)
    s.save("certainty.svg")


# --------------------------------------------------------------------------- the six checks
def checks() -> None:
    s = frame(728, "Six checks before every order: final 55 seconds, model at least 90 percent sure, price safely "
                   "clear of the line, still at least one cent cheaper than fair value after fees, the market has not "
                   "just moved against it, inside the risk limits.",
              "The rule", "Six questions before any order",
              "It buys only when every answer is yes. Most markets end without a trade.")
    rows = (
        ("clock", "Is it the final minute?", "Only the last 55 seconds count. Earlier is guessing.", "55 s or less left"),
        ("target", "Is one side a strong favourite?", "The model must rate it at least 90% likely to win.", "model ≥ 90%"),
        ("gap", "Is the price safely clear of the line?", "Far enough that a normal wobble will not flip it.", "≥ 1.5 σ, ≥ $17 early"),
        ("tag", "Is it still on sale?", "At least 1¢ cheaper than it is worth, after Kalshi's fee.", "edge ≥ 1¢"),
        ("pulse", "Is the market calm right now?", "If its price just dropped or sellers just rushed in, it waits a second.",
         "no sudden drop"),
        ("shield", "Is the bet small enough?", "Inside every risk limit, with fresh prices on screen.", "≤ 6% of account"),
    )
    y0, rh = 176, 78
    for i, (icon, title, desc, rule) in enumerate(rows):
        y = y0 + i * rh
        s.rect(48, y, 1104, rh - 12, fill=PANEL, stroke=LINE, rx=14)
        s.text(74, y + 41, str(i + 1), size=20, fill=MUTED, weight=700, mono=True)
        cx, cy = 128, y + 33
        if icon == "clock":
            s.circle(cx, cy, 15, stroke=HI, sw=2.2)
            s.path(f"M{cx},{cy - 9} V{cy} L{cx + 7},{cy + 4}", sw=2.2)
        elif icon == "target":
            s.circle(cx, cy, 15, stroke=HI, sw=2.2)
            s.circle(cx, cy, 8, stroke=HI, sw=2.2)
            s.circle(cx, cy, 2.5, fill=HI)
        elif icon == "gap":
            s.line(cx - 15, cy - 13, cx + 15, cy - 13, stroke=HI, sw=2.2)
            s.line(cx - 15, cy + 13, cx + 15, cy + 13, stroke=HI, sw=2.2, dash="4 4")
            s.path(f"M{cx},{cy - 7} V{cy + 7} M{cx - 4},{cy - 3} L{cx},{cy - 8} L{cx + 4},{cy - 3} "
                   f"M{cx - 4},{cy + 3} L{cx},{cy + 8} L{cx + 4},{cy + 3}", sw=2)
        elif icon == "tag":
            s.path(f"M{cx - 15},{cy - 11} H{cx + 2} L{cx + 15},{cy} L{cx + 2},{cy + 11} H{cx - 15} Z", sw=2.2)
            s.circle(cx - 8, cy, 2.6, fill=HI)
        elif icon == "pulse":
            s.path(f"M{cx - 17},{cy} H{cx - 8} L{cx - 3},{cy - 13} L{cx + 4},{cy + 13} L{cx + 8},{cy} H{cx + 17}", sw=2.2)
        else:
            s.path(f"M{cx},{cy - 16} L{cx + 13},{cy - 11} V{cy - 2} C{cx + 13},{cy + 7} {cx + 7},{cy + 12} {cx},{cy + 16} "
                   f"C{cx - 7},{cy + 12} {cx - 13},{cy + 7} {cx - 13},{cy - 2} V{cy - 11} Z", sw=2.2)
        s.text(172, y + 29, title, size=22, weight=700, fill=HI)
        s.text(172, y + 53, desc, size=18, fill=MUTED)
        pill(s, 1130, y + 15, rule, size=16, anchor="end")
    fy = y0 + len(rows) * rh + 4
    s.rect(48, fy, 1104, 50, fill=HI, rx=14)
    s.text(W / 2, fy + 32, "Six yes answers → buy at once, hold under a minute, collect $1 per contract if right.",
           size=20, weight=700, fill="#000", anchor="middle")
    s.save("checks.svg")


# --------------------------------------------------------------------------- backtest
def backtest(b: dict, hr: dict) -> dict:
    sm = b["summary"]
    eq = [(t, v) for t, v in b["equity"]]
    eh = [(t, v) for t, v in hr["equity"]]
    t0, t1 = min(eq[0][0], eh[0][0]), max(eq[-1][0], eh[-1][0])
    d0, d1 = (datetime.fromtimestamp(t, timezone.utc) for t in (t0, t1))
    days = b["by_day"]
    up = sum(1 for d in days if d["pnl"] > 0)
    folds = b["folds"]
    wk_up = sum(1 for f in folds if f["pnl"] > 0)
    s = frame(770, f"Backtest equity curve, {d0:%d %b} to {d1:%d %b %Y}: {money(sm['pnl'])} over {sm['markets']:,} "
                   f"markets at test size; {up} of {len(days)} days profitable.",
              f"Backtest · {d0.day} {d0:%b} to {d1.day} {d1:%b %Y}", f"Replayed on {len(days)} days of real Kalshi trades",
              "Each week was traded with settings chosen from earlier weeks only, so none of it is hindsight.")
    tiles = ((money(sm["pnl"]), "profit at test size"), (f"{sm['markets']:,}", "markets traded"),
             (f"{100 * sm['market_win_rate']:.1f}%", "of those markets won"), (f"{wk_up} of {len(folds)}", "test weeks profitable"))
    tw = (1104 - 3 * 16) / 4
    for i, (v, lab) in enumerate(tiles):
        x = 48 + i * (tw + 16)
        s.rect(x, 170, tw, 84, fill=PANEL, stroke=LINE, rx=14)
        s.text(x + 22, 212, v, size=30, weight=700, fill=HI, mono=True)
        s.text(x + 22, 238, lab, size=16.5, fill=MUTED)
    px0, px1, py0, py1 = 104, 1140, 292, 524
    ymax = 1000 * math.ceil(max(v for _, v in eq) / 1000)
    X = lambda t: px0 + (t - t0) / (t1 - t0) * (px1 - px0)  # noqa: E731
    Y = lambda v: py1 - v / ymax * (py1 - py0)  # noqa: E731
    for v in range(0, int(ymax) + 1, 2000):
        s.line(px0, Y(v), px1, Y(v), stroke=LINE)
        s.text(px0 - 14, Y(v) + 6, f"${v // 1000}k" if v else "$0", size=16, fill=MUTED, anchor="end", mono=True)
    line15 = [(X(eq[0][0]), Y(0))] + [(X(t), Y(v)) for t, v in eq]
    s.path(f"M{line15[0][0]:.1f},{Y(0):.1f} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in line15)
           + f" L{line15[-1][0]:.1f},{Y(0):.1f} Z", fill=HI, stroke=None, extra=' opacity="0.09"')
    lineh = [(X(eh[0][0]), Y(0))] + [(X(t), Y(v)) for t, v in eh]
    s.path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in lineh), stroke=MUTED, sw=2.2)
    s.path("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in line15), sw=2.6)
    s.text(px1 - 150, Y(eq[-1][1]) + 8, f"15-minute markets  {money(eq[-1][1])}", size=17, weight=700, fill=HI,
           anchor="end", mono=True)
    s.text(px1 - 10, Y(eh[-1][1]) - 14, f"hourly strikes  {money(eh[-1][1])}", size=16, fill=MUTED, anchor="end", mono=True)
    # daily bars
    by0, by1 = 580, 676
    hi_, lo_ = max(d["pnl"] for d in days), min(d["pnl"] for d in days)
    zero = by0 + hi_ / (hi_ - lo_) * (by1 - by0)
    bw = (px1 - px0) / ((t1 - t0) / 86400) - 3.5
    s.text(px0, by0 - 18, "PROFIT OR LOSS, DAY BY DAY", size=14, fill=MUTED, weight=700, ls=2)
    s.text(px1, by0 - 18, f"{up} days up · {len(days) - up} days down", size=16, fill=MUTED, anchor="end", mono=True)
    for d in days:
        day = datetime.strptime(f"{d['label']} {d0.year}", "%a %b %d %Y").replace(tzinfo=timezone.utc)
        cx = X(min(max(day.timestamp() + 43200, t0), t1))
        hgt = abs(d["pnl"]) / (hi_ - lo_) * (by1 - by0)
        if d["pnl"] >= 0:
            s.rect(cx - bw / 2, zero - hgt, bw, max(hgt, 1), fill=HI, rx=1.5)
        else:
            s.rect(cx - bw / 2, zero, bw, max(hgt, 1), fill=s.hatch(), stroke="#c8c8c8", sw=1, rx=1.5)
    s.line(px0, zero, px1, zero, stroke=LINE2)
    for mth in range(d0.month + 1, d1.month + 1):
        tm = datetime(d0.year, mth, 1, tzinfo=timezone.utc)
        s.line(X(tm.timestamp()), by1 + 6, X(tm.timestamp()), by1 + 14, stroke=LINE2)
        s.text(X(tm.timestamp()), by1 + 36, f"1 {tm:%b}".upper(), size=15, fill=MUTED, anchor="middle", mono=True)
    s.text(48, 742, "Test size: up to 300 contracts per market. Fills, fees and delays are simulated from the public "
                    "trade tape. Not a forecast.", size=17, fill=MUTED)
    s.save("backtest.svg")
    return {"start": f"{d0.day} {d0:%b}", "end": f"{d1.day} {d1:%b %Y}", "days": len(days), "days_up": up,
            "weeks": len(folds), "weeks_up": wk_up, "best_day": hi_, "worst_day": lo_}


# --------------------------------------------------------------------------- payoff
def payoff(b: dict) -> dict:
    sm = b["summary"]
    p = sm["avg_price"]
    fee = 0.07 * p * (1 - p)
    win, loss = 1 - p - fee, p + fee
    be, bt, live = loss / (win + loss), sm["win_rate"], LIVE["wins"] / LIVE["trades"]
    s = frame(480, f"A win earns about {100 * win:.1f} cents per contract and a loss costs about {100 * loss:.1f} cents, "
                   f"so the bot must be right more than {100 * be:.1f}% of the time. Backtest: {100 * bt:.1f}%. "
                   f"First live day: {100 * live:.1f}%.",
              "The catch", "Small wins, big losses",
              "It buys near-certain things, so each win is small and each miss costs nearly the whole stake.")
    s.text(48, 200, f"One contract at the average price paid, {100 * p:.0f}¢", size=21, weight=700, fill=HI)
    sc = 400
    s.text(48, 252, "if right", size=19, fill=MUTED)
    s.rect(140, 232, win * sc, 30, fill=HI, rx=3)
    s.text(140 + win * sc + 14, 255, f"+{100 * win:.1f}¢", size=21, weight=700, fill=HI, mono=True)
    s.text(48, 312, "if wrong", size=19, fill=MUTED)
    s.rect(140, 292, loss * sc, 30, fill=s.hatch(), stroke="#c8c8c8", sw=1.5, rx=3)
    s.text(140 + loss * sc + 14, 315, f"−{100 * loss:.1f}¢", size=21, weight=700, fill=HI, mono=True)
    s.text(48, 370, f"One loss wipes out about {loss / win:.0f} wins.", size=21, fill=TEXT)
    # how often it has to be right
    ax0, ax1, ay, lo = 706, 1146, 300, 85.0
    s.text(ax0 - 6, 200, "So it has to be right this often", size=21, weight=700, fill=HI)
    XV = lambda v: ax0 + (100 * v - lo) / (100 - lo) * (ax1 - ax0)  # noqa: E731
    s.rect(ax0, ay - 13, XV(be) - ax0, 26, fill=s.hatch(), stroke="#c8c8c8", sw=1.5, rx=4)
    s.rect(XV(be), ay - 13, ax1 - XV(be), 26, fill=HI, rx=4, extra=' opacity="0.9"')
    s.text(ax0, ay - 28, "85%", size=15, fill=MUTED, mono=True)
    s.text(ax1, ay - 28, "100%", size=15, fill=MUTED, anchor="end", mono=True)
    s.line(XV(be), ay - 44, XV(be), ay + 22, stroke=HI, sw=2.5)
    s.text(XV(be), ay - 54, f"break-even {100 * be:.1f}%", size=17, weight=700, fill=HI, anchor="middle", mono=True)
    s.circle(XV(bt), ay, 8, fill="#000", stroke=HI, sw=3)
    s.line(XV(bt), ay + 12, XV(bt), ay + 34, stroke=MUTED, sw=1.5)
    s.text(XV(bt), ay + 56, f"backtest {100 * bt:.1f}%", size=17, weight=700, fill=HI, anchor="middle", mono=True)
    s.circle(XV(live), ay, 8, fill=HI, stroke="#000", sw=3)
    s.line(XV(live), ay + 12, XV(live), ay + 34, stroke=MUTED, sw=1.5)
    s.text(XV(live), ay + 56, f"first live day {100 * live:.1f}%", size=17, weight=700, fill=HI, anchor="middle", mono=True)
    s.text((ax0 + XV(be)) / 2, ay + 92, "loses money", size=17, fill=MUTED, anchor="middle")
    s.text((XV(be) + ax1) / 2, ay + 92, "makes money", size=17, fill=MUTED, anchor="middle")
    s.text(48, 440, "A few points of win rate decide everything. That is why the rules are strict and the bets are small.",
           size=21, fill=TEXT)
    s.save("payoff.svg")
    return {"avg_price_c": 100 * p, "win_c": 100 * win, "loss_c": 100 * loss, "wins_per_loss": loss / win,
            "break_even": 100 * be, "backtest_win_rate": 100 * bt, "live_win_rate": 100 * live}


# --------------------------------------------------------------------------- first live day
def liveday() -> None:
    L = LIVE
    s = frame(620, f"First live day: {L['trades']} trades, {L['wins']} won, {L['losses']} lost, net {money(L['net'], True)}. "
                   "All three losses were bought 58 to 60 seconds before the close with a lead of 10 to 13 dollars.",
              "Real money · 10 to 11 Oct 2026", f"First live day: {L['wins']} of {L['trades']} right, and still down $10",
              "A $105 account risking 10% per trade. Three losses cost more than 21 wins earned.")
    sz, gp, y = 32, 7, 180
    for i in range(L["trades"]):
        x = 48 + i * (sz + gp)
        if i < L["wins"]:
            s.rect(x, y, sz, sz, fill=HI, rx=6)
        else:
            s.rect(x, y, sz, sz, fill=s.hatch(), stroke="#c8c8c8", sw=1.5, rx=6)
    xw = 48 + L["wins"] * (sz + gp)
    s.text(48, y + 62, f"{L['wins']} wins · average {money(L['avg_win'], True)}", size=19, fill=HI, mono=True)
    s.text(xw + L["losses"] * (sz + gp) - gp, y + 62, f"{L['losses']} losses · average {money(L['avg_loss'], True)}",
           size=19, fill=HI, anchor="end", mono=True)
    s.text(1152, y + 30, money(L["net"], True), size=38, weight=700, fill=HI, anchor="end", mono=True)
    s.text(1152, y + 62, "net result", size=17, fill=MUTED, anchor="end")
    s.text(48, 300, "WHAT THE THREE LOSSES HAD IN COMMON", size=15, fill=MUTED, weight=700, ls=2.2)
    cards = (("58 to 60 s", "left on the clock when it bought"), ("$10 to $13", "ahead of the line: a thin lead"),
             ("$23 to $26", "one-second jump that flipped it"))
    cw = (1104 - 2 * 20) / 3
    for i, (v, dsc) in enumerate(cards):
        x = 48 + i * (cw + 20)
        s.rect(x, 318, cw, 96, fill=PANEL, stroke=LINE, rx=14)
        s.text(x + 22, 362, v, size=30, weight=700, fill=HI, mono=True)
        s.text(x + 22, 392, dsc, size=17.5, fill=MUTED)
    s.text(48, 460, "WHAT CHANGED THE SAME DAY", size=15, fill=MUTED, weight=700, ls=2.2)
    for i, txt in enumerate(("Wait five more seconds: act in the final 55, not 60.",
                             "Early in the minute, demand a lead of about $17.")):
        check(s, 62, 502 + i * 38)
        s.text(90, 508 + i * 38, txt, size=20, fill=TEXT)
    s.rect(672, 478, 480, 96, fill=HI, rx=14)
    s.text(696, 522, f"${L['dd_before']:.2f} → ${L['dd_after']:.2f}", size=30, weight=700, fill="#000", mono=True)
    s.text(696, 552, f"worst drop in a {L['replay_days']}-day replay · profit unchanged", size=16.5, fill="#222", weight=600)
    s.save("first-live-day.svg")


# --------------------------------------------------------------------------- the market moved first
def moved_first() -> None:
    buy = MOVED_BUY
    s = frame(700, "A real losing trade on 13 October 2026. On Kalshi the price of YES fell from 98 cents to as low "
                   "as 53 cents while the bot's Bitcoin price still showed 9 to 12 dollars above the line. The bot "
                   f"bought at {100 * buy['price']:.0f} cents. Just under a second later its price feed showed "
                   "Bitcoin 14 dollars below the line.",
              "13 Oct 2026 · 1:44 PM ET · real money", "The market knew a second before the bot did",
              "One losing trade, quarter second by quarter second. Kalshi's traders moved first; the price feed followed.")
    x0, x1 = 150, 1128
    t0, t1 = MOVED[0][0], MOVED[-1][0] + 0.25
    X = lambda t: x0 + (t - t0) / (t1 - t0) * (x1 - x0)  # noqa: E731
    cw = (x1 - x0) / len(MOVED)
    # --- what Kalshi's traders paid
    ya, yb, lo, hi = 204, 386, 0.40, 1.00
    Y = lambda p: yb - (p - lo) / (hi - lo) * (yb - ya)  # noqa: E731
    s.text(48, ya - 22, "PRICE OF YES ON KALSHI", size=14.5, fill=MUTED, weight=700, ls=2.0)
    for p in (1.0, 0.8, 0.6, 0.4):
        s.line(x0, Y(p), x1, Y(p), stroke=LINE)
        s.text(x0 - 14, Y(p) + 6, f"{100 * p:.0f}¢", size=16, fill=MUTED, anchor="end", mono=True)
    for t, plo, phi in MOVED:
        top, bot = Y(phi), Y(plo)
        s.rect(X(t) + 7, top - 3, cw - 14, max(bot - top, 0) + 6, fill=HI, rx=3)
    # --- what the bot's price feed said
    yc, yd, ilo, ihi = 478, 584, -20.0, 16.0
    YI = lambda v: yd - (v - ilo) / (ihi - ilo) * (yd - yc)  # noqa: E731
    s.text(48, yc - 26, "BITCOIN AGAINST THE LINE, AS THE BOT'S PRICE FEED SHOWED IT", size=14.5, fill=MUTED, weight=700, ls=2.0)
    s.line(x0, YI(0), x1, YI(0), stroke=MUTED, dash="5 6")
    s.text(x0 - 14, YI(0) + 6, "the line", size=16, fill=MUTED, anchor="end")
    d = ""
    for i, (t, v) in enumerate(MOVED_INDEX):
        end = MOVED_INDEX[i + 1][0] if i + 1 < len(MOVED_INDEX) else t1
        d += f"{'M' if i == 0 else 'L'}{X(t):.1f},{YI(v):.1f} L{X(end):.1f},{YI(v):.1f} "
    s.path(d, sw=3)
    s.text(X(-2.95), YI(11.2) - 13, "$11 to $12 above: YES looks safe", size=17, fill=TEXT)
    s.text(X(2.25), YI(-14.4) - 14, "$14 below", size=17, fill=TEXT, anchor="end")
    # --- the three moments
    xm, xo, xf = X(-0.25), X(0.0), X(MOVED_INDEX[4][0])
    for x, dash in ((xm, "3 5"), (xo, None), (xf, "3 5")):
        s.line(x, ya - 2, x, yd + 6, stroke=HI if dash is None else MUTED, sw=2 if dash is None else 1.5, dash=dash)
    s.circle(xo, Y(buy["price"]), 8, fill=BG, stroke=HI, sw=3)
    s.text(xm - 12, Y(0.50), f"{buy['sold']:,} contracts", size=16.5, fill=HI, weight=700, anchor="end")
    s.text(xm - 12, Y(0.50) + 21, "sold in a quarter second", size=16.5, fill=MUTED, anchor="end")
    s.text(xo + 62, Y(0.90), f"the bot buys YES at {100 * buy['price']:.0f}¢", size=17, fill=HI, weight=700)
    s.text(xo + 62, Y(0.90) + 22, f"its model still says {buy['model']:.0f}% sure", size=16.5, fill=MUTED)
    by = yb + 30
    s.path(f"M{xm:.1f},{by - 7} V{by} H{xf:.1f} V{by - 7}", stroke=HI, sw=2)
    s.text((xm + xf) / 2, by + 24, "about one second", size=17, fill=HI, weight=700, anchor="middle", mono=True)
    s.text(xf + 12, yc + 6, "the price feed catches up", size=16.5, fill=MUTED)
    fy = 626
    s.rect(48, fy, 1104, 50, fill=HI, rx=14)
    s.text(W / 2, fy + 32, "Now it waits: after a drop of more than 10¢ or a rush of selling, it looks again a second later.",
           size=20, weight=700, fill="#000", anchor="middle")
    s.save("moved-first.svg")


def two_checks(rep: dict) -> dict:
    v = rep["variants"]
    off, on = v["without the checks"], v["shipped rule (both checks)"]
    kept = 100 * on["trades"] / off["trades"]
    s = frame(470, f"The live code replayed trade by trade over {off['days']} days. Without the two checks: "
                   f"{off['trades']:,} trades, {off['lost_pct']:.1f} percent lost, profit {off['pnl']:.0f} dollars, worst dip "
                   f"{off['dip']:.0f} dollars. With them: {on['trades']:,} trades, {on['lost_pct']:.1f} percent lost, profit "
                   f"{on['pnl']:.0f} dollars, worst dip {on['dip']:.0f} dollars.",
              "What waiting is worth", f"Fewer bad buys, {kept:.0f}% of the trades kept",
              f"The live code replayed trade by trade over {off['days']} days of real markets. "
              f"${rep['equity']:.0f} account, at most {100 * rep['per_market']:.0f}% per trade.")
    cols = (("TRADES", 470), ("LOST", 620), ("PROFIT", 800), ("WORST DIP", 975), ("WORST DAY", 1128))
    for head, x in cols:
        s.text(x, 196, head, size=14.5, fill=MUTED, weight=700, anchor="end", mono=True, ls=1.6)
    for i, (name, r, solid) in enumerate((("Without the two checks", off, False), ("With the two checks", on, True))):
        y = 214 + i * 92
        s.rect(48, y, 1104, 78, fill=HI if solid else PANEL, stroke=None if solid else LINE, rx=14)
        ink = "#000" if solid else HI
        s.text(72, y + 47, name, size=22, weight=700, fill=ink)
        vals = (f"{r['trades']:,}", f"{r['lost_pct']:.1f}%", money(r["pnl"]), f"${r['dip']:.0f}", money(r["worst_day"]))
        for (_, x), val in zip(cols, vals):
            s.text(x, y + 49, val, size=27, weight=700, fill=ink, anchor="end", mono=True)
    s.text(48, 432, f"Losing days: {off['red_days']} of {off['days']} without, {on['red_days']} of {on['days']} with. "
                    "A replay is not a promise: live trading is the real test.", size=19, fill=MUTED)
    s.save("two-checks.svg")
    return {"days": off["days"], "markets": rep["markets"], "kept_pct": kept,
            "without": {k: off[k] for k in ("trades", "per_day", "lost", "lost_pct", "pnl", "dip", "worst_day", "red_days")},
            "with": {k: on[k] for k in ("trades", "per_day", "lost", "lost_pct", "pnl", "dip", "worst_day", "red_days")}}


# --------------------------------------------------------------------------- limits
def risk() -> None:
    s = frame(520, "Built-in limits for a 100 dollar account: at most 6 dollars in one contract, 10 dollars on one "
                   "closing moment, trading stops for the day after losing 15 dollars, and the bot switches itself "
                   "off after a 30 percent fall from the peak.",
              "Safety rails", "Limits it cannot argue with",
              "Shown for a $100 account. Each limit is a share of the balance, so it scales as the account changes.")
    tiles = (("$6", ("the most it puts into", "one contract"), "6% OF ACCOUNT"),
             ("$10", ("the most riding on", "one closing moment"), "10% OF ACCOUNT"),
             ("−$15", ("a day's loss that stops", "it until tomorrow"), "15% OF ACCOUNT"),
             ("−30%", ("a fall from the peak", "that switches it off"), "KILL SWITCH"))
    tw = (1104 - 3 * 16) / 4
    for i, (v, lines, foot) in enumerate(tiles):
        x = 48 + i * (tw + 16)
        s.rect(x, 174, tw, 160, fill=PANEL, stroke=LINE, rx=14)
        s.text(x + 22, 228, v, size=42, weight=700, fill=HI, mono=True)
        s.text(x + 22, 260, lines[0], size=18, fill=TEXT)
        s.text(x + 22, 283, lines[1], size=18, fill=TEXT)
        s.text(x + 22, 314, foot, size=13.5, fill=MUTED, weight=700, mono=True, ls=1.4)
    items = ("Demo by default; live stays locked until you unlock it", "Orders sized to the cash actually free, fees included",
             "No trade on a stale price feed or order book", "One bot per account; a second copy refuses to start",
             "Kill switch is remembered across restarts", "Dashboard: password, lockout, local machine only")
    for i, txt in enumerate(items):
        x, y = (48 if i % 2 == 0 else 612), 384 + (i // 2) * 40
        check(s, x + 12, y - 6)
        s.text(x + 40, y, txt, size=18.5, fill=TEXT)
    s.save("safety.svg")


# --------------------------------------------------------------------------- server
def server() -> None:
    s = frame(440, "Deployment: the bot runs as a service on a small cloud server. The dashboard listens only inside "
                   "the server and is reached from a laptop through an encrypted SSH tunnel.",
              "Always on", "Runs on a small cloud server, laptop closed",
              "The dashboard never faces the internet. You reach it through an encrypted SSH tunnel.")
    y, hgt = 180, 190

    def box(x, w, head, lines, solid=False):
        s.rect(x, y, w, hgt, fill=HI if solid else PANEL, stroke=None if solid else LINE2, rx=16)
        ink, dim = ("#000", "#333") if solid else (HI, TEXT)
        s.text(x + 22, y + 38, head, size=14.5, fill="#444" if solid else MUTED, weight=700, mono=True, ls=1.8)
        for i, ln in enumerate(lines):
            s.text(x + 22, y + 76 + i * 28, ln, size=18.5, fill=ink if i == 0 else dim, weight=700 if i == 0 else 400)

    box(48, 250, "YOUR LAPTOP", ("Browser", "localhost:8000", "open it when you like"))
    box(458, 400, "SERVER · UBUNTU", ("Lock-In, running as a service", "dashboard: inside the server only",
                                      "password, lockout after 5 tries", "firewall: only SSH is open"), solid=True)
    box(968, 184, "KALSHI", ("Exchange", "prices in", "orders out"))
    for x0, x1, top, bot in ((298, 458, "SSH tunnel", "encrypted"), (858, 968, "", "a few ms")):
        cy = y + hgt / 2
        s.line(x0 + 10, cy - 7, x1 - 10, cy - 7, stroke=HI, sw=2)
        s.line(x0 + 10, cy + 7, x1 - 10, cy + 7, stroke=HI, sw=2)
        if top:
            s.text((x0 + x1) / 2, cy - 22, top, size=16, fill=HI, weight=700, anchor="middle", mono=True)
        s.text((x0 + x1) / 2, cy + 34, bot, size=15, fill=MUTED, anchor="middle", mono=True)
    s.text(48, 410, "Reboot or crash: the service comes back and the bot resumes by itself. Stop means stop.",
           size=19, fill=MUTED)
    s.save("server.svg")


# --------------------------------------------------------------------------- calibration
def calibration(b: dict) -> None:
    rows = b["calibration"]
    mid = rows[3]
    s = frame(570, "Calibration: in every confidence band the favourite won at least as often as the model said, "
                   "except the very top band where they match.",
              "Is the model honest?",
              f"When it says {100 * mid['model']:.0f}%, it happens about {100 * mid['realised']:.0f}% of the time",
              "Every market in the backtest, final minute, grouped by how sure the model was about the favourite.")
    tx0, tx1, lo = 330, 940, 60.0
    XV = lambda v: tx0 + (100 * v - lo) / (100 - lo) * (tx1 - tx0)  # noqa: E731
    fmt = lambda v: f"{100 * v:.0f}%" if v < 0.98 else f"{100 * v:.1f}%"  # noqa: E731
    y0, rh = 200, 44
    for v in (60, 70, 80, 90, 100):
        x = XV(v / 100)
        s.line(x, y0 - 22, x, y0 + 6 * rh + 22, stroke=LINE)
        s.text(x, y0 + 6 * rh + 48, f"{v}%", size=15, fill=MUTED, anchor="middle", mono=True)
    for i, r in enumerate(rows):
        y = y0 + i * rh
        s.text(48, y + 6, f"model said {fmt(r['model'])}", size=18.5, fill=MUTED, mono=True)
        s.line(XV(r["model"]), y, XV(r["realised"]), y, stroke=HI, sw=2.5)
        s.circle(XV(r["model"]), y, 7, fill=BG, stroke=MUTED, sw=2.5)
        s.circle(XV(r["realised"]), y, 7.5, fill=HI, stroke=BG, sw=1.5)
        s.text(1152, y + 6, f"happened {fmt(r['realised'])}", size=18.5, fill=HI, weight=700, anchor="end", mono=True)
    ly = y0 + 6 * rh + 84
    s.circle(56, ly - 6, 7, fill=BG, stroke=MUTED, sw=2.5)
    s.text(74, ly, "what the model expected", size=17.5, fill=MUTED)
    s.circle(320, ly - 6, 7.5, fill=HI)
    s.text(338, ly, "what actually happened", size=17.5, fill=TEXT)
    s.text(1152, ly, "A dot to the right of its circle means the model was cautious.", size=17.5, fill=MUTED, anchor="end")
    s.save("calibration.svg")


# --------------------------------------------------------------------------- stress tests
def stress(b: dict) -> None:
    rb = {r["label"]: r for r in b["robustness"]}
    rows = (("As tested", "House parameters, whole period"),
            ("Bot reacts four times slower (1 s)", "Latency 1.0 s instead of 0.25 s"),
            ("Smaller orders (20 contracts max)", "Max 20 contracts per order"),
            ("Half as much available to buy", "Participation 25% of each confirming trade"),
            ("Every fill 2¢ worse", "Every fill 2c worse than the tape"),
            ("Model made 50% more cautious", "Volatility factor 1.5 (much more conservative)"))
    s = frame(520, "Stress tests of the backtest: profit stays positive with slower reactions, smaller orders, less "
                   "liquidity, worse fills and a more cautious model.",
              "Stress tests", "Still positive when conditions get worse",
              "The same replay with one thing made harder each time. Current rule, whole period, test size.")
    top = rb[rows[0][1]]["pnl"]
    bx0, sc, y0, rh = 470, 540 / top, 184, 46
    for i, (lab, key) in enumerate(rows):
        v, y = rb[key]["pnl"], y0 + i * rh
        s.text(48, y + 24, lab, size=19.5, fill=HI if i == 0 else TEXT, weight=700 if i == 0 else 400)
        s.rect(bx0, y + 4, v * sc, 28, fill=HI, rx=4, extra="" if i == 0 else ' opacity="0.55"')
        s.text(bx0 + v * sc + 14, y + 25, money(v), size=19.5, weight=700, fill=HI, mono=True)
    s.text(48, 488, "A replay cannot capture everything. Live trading is the real test.", size=19, fill=MUTED)
    s.save("stress-tests.svg")


# --------------------------------------------------------------------------- how it is built
def architecture() -> None:
    s = frame(480, "How the bot is built: prices come in from Kalshi, a model works out the odds, six checks decide "
                   "whether to buy, risk limits size the order, and the order goes back to Kalshi. Every step is "
                   "recorded in a trade log and shown on the dashboard.",
              "Under the hood", "From a price to an order",
              "The same rule runs in the research replay and in the live bot, and a test fails if they disagree.")
    s.rect(48, 172, 1104, 46, stroke=HI, sw=2, rx=12)
    s.text(W / 2, 202, "KALSHI EXCHANGE", size=16, weight=700, fill=HI, anchor="middle", mono=True, ls=3)
    steps = (("PRICE FEED", "Bitcoin's price,", "every Kalshi trade"), ("MODEL", "the odds that", "each side wins"),
             ("SIX CHECKS", "buy only if", "all six pass"), ("RISK LIMITS", "size the order", "to the account"),
             ("ORDER", "buy now", "or cancel"))
    gap, y, hgt = 28, 268, 96
    bw = (1104 - 4 * gap) / 5

    def arrow(x1, y1, x2, y2, dash=None):
        s.line(x1, y1, x2, y2, stroke=HI if not dash else DIM, sw=2, dash=dash)
        if dash:
            return
        dx, dy = (x2 > x1) - (x2 < x1), (y2 > y1) - (y2 < y1)
        s.path(f"M{x2},{y2} L{x2 - 9 * dx - 6 * dy},{y2 - 9 * dy - 6 * dx} L{x2 - 9 * dx + 6 * dy},{y2 - 9 * dy + 6 * dx} Z",
               fill=HI, stroke=HI, sw=1)

    for i, (head, l1, l2) in enumerate(steps):
        x = 48 + i * (bw + gap)
        s.rect(x, y, bw, hgt, fill=PANEL, stroke=LINE2, rx=14)
        s.text(x + 18, y + 32, head, size=14.5, fill=MUTED, weight=700, mono=True, ls=1.8)
        s.text(x + 18, y + 60, l1, size=18, fill=HI, weight=600)
        s.text(x + 18, y + 82, l2, size=18, fill=HI, weight=600)
        if i < 4:
            arrow(x + bw + 4, y + hgt / 2, x + bw + gap - 3, y + hgt / 2)
        arrow(x + bw / 2, y + hgt, x + bw / 2, 410, dash="4 6")
    cx1, cx5 = 48 + bw / 2, 48 + 4 * (bw + gap) + bw / 2
    arrow(cx1, 218, cx1, y - 3)
    s.text(cx1 + 14, 249, "prices", size=15.5, fill=MUTED, mono=True)
    arrow(cx5, y, cx5, 221)
    s.text(cx5 - 14, 249, "orders", size=15.5, fill=MUTED, anchor="end", mono=True)
    s.rect(48, 410, 1104, 46, fill=HI, rx=12)
    s.text(W / 2, 440, "TRADE LOG + DASHBOARD: every step is recorded and shown live", size=16, weight=700, fill="#000",
           anchor="middle", mono=True, ls=0.8)
    s.save("architecture.svg")


# --------------------------------------------------------------------------- story
def story() -> None:
    s = frame(400, "Timeline: March 2026 candle-pattern bot scrapped; 27 September lock-in rules frozen; "
                   "28 September to 10 October tested on unseen data; 10 October live with 100 dollars; "
                   "11 October first live day lost 10 dollars and the rules were tightened; 13 October two more "
                   "losses traced to the market moving before the price feed, and the bot now waits when that happens.",
              "The story so far", "Built, tested, shipped, corrected twice")
    nodes = (("MAR 2026", "Candle-pattern bot", ("replayed on real fills:", "it loses. Scrapped.")),
             ("27 SEP", "Lock-in rules, v1", ("written down", "and frozen")),
             ("28 SEP TO 10 OCT", "Unseen-data test", ("two fresh weeks,", "still profitable")),
             ("10 OCT", "Live with $100", ("v2, on a small", "cloud server")),
             ("11 OCT", "First day: −$10", ("three early buys lost,", "v3 waits longer")),
             ("13 OCT", "Market moved first", ("bought on stale news,", "v4 waits a second")))
    x0, x1, y = 124, 1076, 216
    s.line(x0, y, x1, y, stroke=LINE2, sw=3)
    for i, (when, head, lines) in enumerate(nodes):
        x = x0 + i * (x1 - x0) / (len(nodes) - 1)
        last = i == len(nodes) - 1
        s.circle(x, y, 11, fill=HI if not last else BG, stroke=HI, sw=3)
        if last:
            s.circle(x, y, 4.5, fill=HI)
        s.text(x, y - 34, when, size=14, fill=MUTED, weight=700, anchor="middle", mono=True, ls=1.0)
        s.text(x, y + 50, head, size=18.5, fill=HI, weight=700, anchor="middle")
        for j, ln in enumerate(lines):
            s.text(x, y + 78 + j * 23, ln, size=15.5, fill=MUTED, anchor="middle")
    s.save("story.svg")


def main() -> None:
    missing = [n for n in ("backtest.json", "backtest_kxbtcd.json") if not (REPORTS / n).exists()]
    if missing:
        raise SystemExit(f"{', '.join(missing)} not found in {REPORTS}. Build the reports first:\n"
                         "  python -m kalshi_bot.research.report --series KXBTC15M\n"
                         "  python -m kalshi_bot.research.report --series KXBTCD")
    b, hr = load("backtest.json"), load("backtest_kxbtcd.json")
    print(f"writing {OUT}")
    hero()
    market()
    timeline()
    funnel()
    checks()
    facts = {"backtest": backtest(b, hr), "payoff": payoff(b)}
    liveday()
    moved_first()
    if (REPORTS / "tape_replay.json").exists():
        facts["two_checks"] = two_checks(load("tape_replay.json"))
    else:
        print("  (two-checks.svg skipped: run  python -m kalshi_bot.research.tape_replay --per-market 0.10 --per-close 0.15)")
    risk()
    server()
    calibration(b)
    stress(b)
    architecture()
    story()
    sm, sh = b["summary"], hr["summary"]
    facts["15m"] = {k: sm[k] for k in ("markets", "pnl", "market_win_rate", "win_rate", "pnl_per_contract_c", "fees",
                                       "max_drawdown", "worst_market", "best_market", "avg_price")}
    facts["hourly"] = {k: sh[k] for k in ("markets", "pnl", "market_win_rate", "pnl_per_contract_c", "max_drawdown")}
    facts["hourly"]["weeks_up"] = f"{sum(1 for f in hr['folds'] if f['pnl'] > 0)} of {len(hr['folds'])}"
    facts["hourly"]["days_up"] = f"{sum(1 for d in hr['by_day'] if d['pnl'] > 0)} of {len(hr['by_day'])}"
    print("\nnumbers used in the README:")
    print(json.dumps(facts, indent=1, default=lambda v: round(v, 4)))


if __name__ == "__main__":
    main()
