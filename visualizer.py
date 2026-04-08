"""
Professional ATC Visualization
Real Physics, Dynamic Radar, Conflict Lines, ICAO Logs
Run: python visualizer.py [task_level]
"""

import sys
import math
import argparse

import matplotlib
matplotlib.use("TkAgg")   # Use TkAgg; falls back automatically on other platforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import numpy as np

from runway_algorithm import run_simulation, Status


# ─────────────────────────────────────────────
# Colour Palette
# ─────────────────────────────────────────────
class C:
    BG        = "#050A14"
    PANEL     = "#0F1724"
    TEXT      = "#E0E6ED"
    MUTED     = "#5C6B7F"
    BORDER    = "#1E2D40"

    EMERGENCY = "#FF2A2A"
    CRITICAL  = "#FF8C00"
    WARNING   = "#FFD700"
    NORMAL    = "#00FF9D"
    INFO      = "#00BFFF"
    LANDED    = "#4CAF50"

    RADAR_RING  = "#0D1F33"
    RUNWAY_OPEN = "#1A2E44"
    RUNWAY_SHUT = "#3B1010"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def ac_color(ac: dict) -> str:
    if ac.get("emergency"):  return C.EMERGENCY
    if ac.get("critical"):   return C.CRITICAL
    if ac.get("low_fuel"):   return C.WARNING
    st = ac.get("status", "")
    if st in ("LANDED", "TAXIING", "EXITED"): return C.LANDED
    return C.NORMAL


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"T+{m:02d}:{s:02d}"


# ─────────────────────────────────────────────
# Main Visualizer
# ─────────────────────────────────────────────
class ATCVisualizer:
    RADAR_RANGE = 60   # NM radius displayed

    def __init__(self, task_level: int):
        self.task_level = task_level
        print(f"[ATC] Running Task {task_level} simulation — please wait...")
        result = run_simulation(task_level)
        self.snapshots   = result["snapshots"]
        self.weather     = result["weather"]
        self.scores      = result["scores"]
        self.total_frames = len(self.snapshots)
        self.frame_idx    = 0
        self.sweep_angle  = 0.0

        print(f"[ATC] Simulation complete — {self.total_frames} frames captured.")
        print(f"[ATC] Final score: {self.scores['score']:.4f}")

        # ── Figure Layout ──────────────────────────────────
        self.fig = plt.figure(figsize=(22, 12), facecolor=C.BG)
        self.fig.canvas.manager.set_window_title(f"AI ATC System — Task {task_level}")

        gs = gridspec.GridSpec(
            5, 12, figure=self.fig,
            height_ratios=[0.7, 3, 3, 2.5, 0.5],
            hspace=0.18, wspace=0.08,
        )

        self.ax_hdr      = self.fig.add_subplot(gs[0, :8])
        self.ax_wth      = self.fig.add_subplot(gs[0, 8:])
        self.ax_radar    = self.fig.add_subplot(gs[1:3, :6])
        self.ax_rwy      = self.fig.add_subplot(gs[1:3, 6:9])
        self.ax_tbl      = self.fig.add_subplot(gs[1:3, 9:])
        self.ax_timeline = self.fig.add_subplot(gs[3, :9])
        self.ax_logs     = self.fig.add_subplot(gs[3, 9:])
        self.ax_stats    = self.fig.add_subplot(gs[4, :])
        self._style_all()

    def _style_all(self):
        for ax in (self.ax_hdr, self.ax_wth, self.ax_rwy, self.ax_tbl,
                   self.ax_logs, self.ax_stats):
            ax.set_facecolor(C.PANEL)
            ax.axis("off")
        self.ax_radar.set_facecolor(C.BG)
        self.ax_radar.axis("off")
        self.ax_timeline.set_facecolor(C.PANEL)
        for sp in self.ax_timeline.spines.values():
            sp.set_edgecolor(C.BORDER)

    # ── Draw helpers ──────────────────────────────────────

    def _draw_header(self, snap: dict):
        ax = self.ax_hdr
        ax.clear(); ax.set_facecolor(C.BG); ax.axis("off")
        t  = fmt_time(snap["sim_time"])
        pct = (self.frame_idx + 1) / max(1, self.total_frames) * 100
        ax.text(0.02, 0.5, f"AI ATC CONTROL SYSTEM", color=C.INFO,
                fontsize=15, fontweight="bold", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.5, f"TASK {self.task_level}  |  {t}",
                color=C.TEXT, fontsize=13, ha="center", va="center", transform=ax.transAxes)
        ax.text(0.98, 0.5, f"{pct:.0f}% complete",
                color=C.MUTED, fontsize=10, ha="right", va="center", transform=ax.transAxes)
        # Progress bar
        bar_ax = self.fig.add_axes([0.015, 0.94, 0.65, 0.006])
        bar_ax.set_facecolor(C.BORDER)
        bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1); bar_ax.axis("off")
        bar_ax.barh(0.5, pct / 100, height=1.0, color=C.INFO, align="center")

    def _draw_weather(self, snap: dict):
        ax = self.ax_wth
        ax.clear(); ax.set_facecolor(C.PANEL); ax.axis("off")
        w = self.weather
        icon = {"CLEAR": "CLEAR SKY", "RAIN": "RAIN", "STORM": "STORM"}.get(w.condition, w.condition)
        col  = {"CLEAR": C.NORMAL,    "RAIN": C.WARNING, "STORM": C.EMERGENCY}.get(w.condition, C.TEXT)
        ax.text(0.5, 0.80, "WEATHER", color=C.MUTED, ha="center", fontsize=9, transform=ax.transAxes)
        ax.text(0.5, 0.55, icon,      color=col,    ha="center", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.28, f"Wind {w.wind_speed:.0f}kt / {w.wind_dir:.0f}deg", color=C.TEXT, ha="center", fontsize=9, transform=ax.transAxes)
        ax.text(0.5, 0.08, f"Vis {w.visibility:.0f}km",  color=C.TEXT, ha="center", fontsize=9, transform=ax.transAxes)

    def _draw_radar(self, snap: dict):
        ax = self.ax_radar
        ax.clear()
        ax.set_facecolor(C.BG)
        R = self.RADAR_RANGE
        ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        ax.set_aspect("equal"); ax.axis("off")

        # Grid rings
        for r in (10, 20, 30, 40, 50):
            circle = Circle((0, 0), r, fill=False, edgecolor=C.RADAR_RING, linewidth=0.8, linestyle="--")
            ax.add_patch(circle)
            ax.text(0, r + 1.5, f"{r}nm", color=C.MUTED, ha="center", fontsize=6)

        # Cardinal lines
        for deg in (0, 90, 180, 270):
            rad = math.radians(deg)
            ax.plot([0, R * math.sin(rad)], [0, R * math.cos(rad)],
                    color=C.RADAR_RING, linewidth=0.5, linestyle=":")

        # Rotating sweep
        self.sweep_angle = (self.sweep_angle + 4.0) % 360
        rad = math.radians(self.sweep_angle)
        sx, sy = (R - 4) * math.sin(rad), (R - 4) * math.cos(rad)
        ax.plot([0, sx], [0, sy], color=C.NORMAL, linewidth=1.2, alpha=0.75)

        # Airport centre marker
        ax.plot(0, 0, "+", color=C.INFO, markersize=14, markeredgewidth=2)
        ax.text(0, -R + 3, "RADAR DISPLAY", color=C.TEXT, ha="center", fontsize=9, fontweight="bold")

        # Conflict pairs
        aircraft = snap["aircraft"]
        conflict_segs = []
        active = [a for a in aircraft if a["status"] not in ("EXITED", "CRASHED")]
        for i, a1 in enumerate(active):
            for a2 in active[i + 1:]:
                dist = math.hypot(a1["x"] - a2["x"], a1["y"] - a2["y"])
                if dist < 3.0:
                    conflict_segs.append(((a1["x"], a1["y"]), (a2["x"], a2["y"])))

        if conflict_segs:
            lc = LineCollection(conflict_segs, colors=C.EMERGENCY, linewidths=1.5, linestyles="dotted")
            ax.add_collection(lc)
            ax.text(R - 2, R - 5, "SEPARATION\nLOSS", color=C.EMERGENCY,
                    ha="right", fontsize=8, fontweight="bold")

        # Aircraft blips
        for ac in aircraft:
            st = ac["status"]
            if st == "EXITED":
                continue

            x, y = ac["x"], ac["y"]
            col  = ac_color(ac)
            size = 6 if ac["wake"] == "HEAVY" else (4 if ac["wake"] == "MEDIUM" else 3)

            # Faded pulse based on sweep proximity
            ac_ang   = math.degrees(math.atan2(ac["x"], ac["y"])) % 360
            ang_diff = abs(ac_ang - self.sweep_angle)
            if ang_diff > 180:
                ang_diff = 360 - ang_diff
            alpha = 1.0 if ang_diff < 20 else max(0.2, 1.0 - ang_diff / 100)

            dot = Circle((x, y), size / 2.0, facecolor=col, edgecolor="white",
                         linewidth=0.5, alpha=alpha, zorder=3)
            ax.add_patch(dot)

            # Data tag
            if alpha > 0.3:
                tag = f"{ac['id']}\n{int(ac['altitude'])}ft {int(ac['speed'])}kt"
                ax.text(x + size / 2 + 0.8, y + size / 2, tag,
                        color=col, fontsize=5.5, alpha=alpha, zorder=4)
                # Velocity vector
                hdg = math.radians(ac["heading"])
                vx  = x + 6 * math.sin(hdg)
                vy  = y + 6 * math.cos(hdg)
                ax.plot([x, vx], [y, vy], color=col, linewidth=0.6, alpha=alpha * 0.6)

    def _draw_runways(self, snap: dict):
        ax = self.ax_rwy
        ax.clear(); ax.set_facecolor(C.PANEL)
        ax.set_xlim(0, 14); ax.set_ylim(0, 10)
        ax.axis("off")
        ax.text(7, 9.5, "RUNWAY STATUS", color=C.TEXT, ha="center", fontweight="bold", fontsize=11)

        y_positions = [7.0, 3.5]
        for i, rwy_snap in enumerate(snap["runways"]):
            y     = y_positions[i]
            open_ = rwy_snap["is_active"]
            col   = C.RUNWAY_OPEN if open_ else C.RUNWAY_SHUT
            edge  = "#ffffff" if open_ else C.EMERGENCY

            # Runway surface
            rect = Rectangle((0.5, y - 0.45), 12, 0.9, facecolor=col, edgecolor=edge, linewidth=2)
            ax.add_patch(rect)

            # Centre-line dashes
            if open_:
                for xd in np.arange(1.5, 12, 1.5):
                    ax.plot([xd, xd + 0.7], [y, y], color=C.TEXT, linewidth=1.5, alpha=0.6)

            # Threshold lights
            if open_:
                for xl in [0.7, 12.1]:
                    ax.plot(xl, y, "o", color=C.WARNING, markersize=6)

            # Label
            status_lbl = "OPEN" if open_ else "CLOSED"
            lbl_col    = C.NORMAL if open_ else C.EMERGENCY
            ax.text(6.5, y + 1.1, f"{rwy_snap['name']}  [{status_lbl}]",
                    color=lbl_col, ha="center", fontsize=10, fontweight="bold")

            # Landing aircraft icon
            prog = rwy_snap.get("progress")
            occ  = rwy_snap.get("occupant")
            if open_ and occ and prog is not None:
                rx = 0.5 + prog * 12.0
                ax.text(rx, y, u"\u2708", fontsize=18, ha="center", va="center",
                        color=C.INFO, zorder=5)
                ax.text(6.5, y - 0.9, f"ON RUNWAY: {occ}",
                        color=C.INFO, ha="center", fontsize=8, fontweight="bold")
            elif open_:
                ax.text(6.5, y - 0.9, "CLEAR", color=C.NORMAL, ha="center", fontsize=8)

    def _draw_table(self, snap: dict):
        ax = self.ax_tbl
        ax.clear(); ax.set_facecolor(C.PANEL)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        ax.text(0.5, 0.96, "TRAFFIC LIST", color=C.TEXT, ha="center",
                fontweight="bold", fontsize=10, transform=ax.transAxes)

        hdrs = ["CALL", "FUEL", "ALT", "SPD", "STATE"]
        xs   = [0.10,   0.30,   0.50,  0.70,  0.90]
        for h, x in zip(hdrs, xs):
            ax.text(x, 0.90, h, color=C.MUTED, ha="center", fontsize=8, transform=ax.transAxes)
        ax.axhline(y=0.89, color=C.BORDER, linewidth=0.8)

        sorted_ac = sorted(snap["aircraft"], key=lambda a: a["priority_score"], reverse=True)
        y = 0.83
        for ac in sorted_ac[:9]:
            if y < 0.05:
                break
            col  = ac_color(ac)
            fcol = C.CRITICAL if ac["fuel"] < 15 else (C.WARNING if ac["fuel"] < 25 else C.TEXT)
            st   = ac["status"]
            scol = (C.EMERGENCY if st == "CRASHED" else
                    C.LANDED    if st in ("LANDED", "EXITED") else C.INFO)

            ax.text(xs[0], y, ac["id"],            color=col,  ha="center", fontsize=8, fontweight="bold", transform=ax.transAxes)
            ax.text(xs[1], y, f"{ac['fuel']:.0f}m", color=fcol, ha="center", fontsize=8, transform=ax.transAxes)
            ax.text(xs[2], y, f"{int(ac['altitude'])}", color=C.MUTED, ha="center", fontsize=7, transform=ax.transAxes)
            ax.text(xs[3], y, f"{int(ac['speed'])}",    color=C.MUTED, ha="center", fontsize=7, transform=ax.transAxes)
            ax.text(xs[4], y, st[:8],              color=scol, ha="center", fontsize=7, transform=ax.transAxes)
            y -= 0.088

    def _draw_timeline(self, snap: dict):
        ax = self.ax_timeline
        ax.clear(); ax.set_facecolor(C.PANEL)
        max_t = max(3600.0, snap["sim_time"] + 600.0)
        ax.set_xlim(0, max_t); ax.set_ylim(-0.5, len(snap["runways"]))
        ax.set_xlabel("Simulation time (s)", color=C.MUTED, fontsize=8)
        ax.tick_params(colors=C.MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C.BORDER)
        ax.set_facecolor(C.PANEL)
        ax.set_title("RUNWAY OCCUPANCY TIMELINE", color=C.TEXT, fontsize=9, pad=4)
        ax.xaxis.label.set_color(C.MUTED)
        ax.tick_params(axis="both", colors=C.MUTED)

        # Current time cursor
        ax.axvline(snap["sim_time"], color=C.EMERGENCY, linewidth=1.2, linestyle="--", alpha=0.8)

        for i, rwy_snap in enumerate(snap["runways"]):
            col = C.INFO if rwy_snap["is_active"] else C.EMERGENCY
            ax.text(-30, i, rwy_snap["name"], color=col, va="center", ha="right", fontsize=7)
            for slot in rwy_snap["slots"]:
                w = slot["end"] - slot["start"]
                ax.barh(i, w, left=slot["start"], height=0.55, color=C.INFO,
                        alpha=0.35, edgecolor=C.INFO, linewidth=0.5)
                mid = slot["start"] + w / 2
                ax.text(mid, i, slot["id"][:5], color=C.TEXT, ha="center",
                        va="center", fontsize=6, zorder=5)

    def _draw_logs(self, snap: dict):
        ax = self.ax_logs
        ax.clear(); ax.set_facecolor(C.PANEL)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 0.97, "ATC LOG", color=C.TEXT, ha="center",
                fontweight="bold", fontsize=9, transform=ax.transAxes)

        y = 0.88
        for ev in reversed(snap["events"][-7:]):
            col = C.EMERGENCY if "CONFLICT" in ev or "CRASH" in ev else (
                  C.WARNING   if "FUEL"     in ev else C.MUTED)
            ax.text(0.05, y, ev, color=col, fontsize=7, va="top",
                    transform=ax.transAxes, wrap=True)
            y -= 0.125
            if y < 0.0:
                break

    def _draw_stats(self, snap: dict):
        ax = self.ax_stats
        ax.clear(); ax.set_facecolor(C.BG); ax.axis("off")
        m = snap["metrics"]
        sc = self.scores

        parts = [
            f"SCORE {sc['score']:.4f}",
            f"REWARD {sc['reward']:.2f}",
            f"CRASHES {m['crashes']}",
            f"DELAY {m['total_delay']:.0f}s",
            f"EFF {m['efficiency']:.0f}%",
        ]
        cols = [C.NORMAL, C.INFO, C.EMERGENCY if m["crashes"] > 0 else C.TEXT, C.WARNING, C.LANDED]
        x = 0.05
        for part, col in zip(parts, cols):
            ax.text(x, 0.5, part, color=col, fontsize=10, fontweight="bold",
                    va="center", transform=ax.transAxes, family="monospace")
            x += 0.19

        ax.text(0.99, 0.5, f"TASK {self.task_level} / {self.weather.condition}",
                color=C.MUTED, fontsize=9, ha="right", va="center", transform=ax.transAxes)

    # ── Animation ─────────────────────────────────────────

    def update(self, frame: int):
        """Called each animation frame — advances through pre-computed snapshots."""
        self.frame_idx = min(frame, self.total_frames - 1)
        snap = self.snapshots[self.frame_idx]
        self.sweep_angle = (self.sweep_angle + 3.0) % 360.0

        self._draw_header(snap)
        self._draw_weather(snap)
        self._draw_radar(snap)
        self._draw_runways(snap)
        self._draw_table(snap)
        self._draw_timeline(snap)
        self._draw_logs(snap)
        self._draw_stats(snap)

        return []

    def animate(self, interval_ms: int = 80):
        anim = animation.FuncAnimation(
            self.fig, self.update,
            frames=self.total_frames,
            interval=interval_ms,
            blit=False,
            repeat=True,
        )
        return anim


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ATC Visualizer")
    parser.add_argument("task", nargs="?", type=int, default=None,
                        help="Task level 1-3 (prompt if omitted)")
    args = parser.parse_args()

    if args.task in (1, 2, 3):
        task = args.task
    else:
        print("Select task level:")
        print("  1 — Clear Skies  (5 aircraft, good conditions)")
        print("  2 — Fuel Pressure (10 aircraft, rain, fuel-critical aircraft)")
        print("  3 — Full Emergency (15 aircraft, storm, runway closure, emergency)")
        try:
            task = int(input("Task [1-3]: ").strip() or "1")
        except (ValueError, EOFError):
            task = 1
        task = max(1, min(3, task))

    viz  = ATCVisualizer(task)
    anim = viz.animate()
    # plt.tight_layout(pad=0.5)  # Can cause warnings with custom gridspecs
    plt.show()

    print(f"\nFinal Score:  {viz.scores['score']:.4f}")
    print(f"Reward:       {viz.scores['reward']:.2f}")


if __name__ == "__main__":
    main()
