"""
figs/eta_vs_rtt.py
Genera figs/eta_vs_rtt.pdf — figura S5: eficiencia η vs RTT artificial.

Datos fuente: sensitivity_rtt_step.csv (experimento con ESP32 real, WiFi LAN).
Estilo: idéntico a rtt_comparison.py (88×58 mm, 600 dpi, palette #1976D2/#388E3C).
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── rutas ─────────────────────────────────────────────────────────────────────
# Delay sweep on the real ESP32 (Condition C, profile P2, closed loop). delta = 0 is the
# factorial C/P2 set; delta > 0 from scripts/run_factorial.py ... --delay-ms d.
FIGS_DIR = Path(__file__).parent
DATA = Path(__file__).parents[1] / "data"
RUN_GLOBS = [DATA / "factorial" / "runs", DATA / "sensitivity_rtt" / "runs"]

# ── estilo (idéntico a rtt_comparison.py) ─────────────────────────────────────
FIG_W_IN  = 88 / 25.4
FIG_H_IN  = 58 / 25.4
C_PO      = "#1976D2"
C_PSO     = "#388E3C"
C_ACCENT  = "#C62828"
ETA_THR   = 97.0        # umbral η [%]


def style_ax(ax, lang="es"):
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    if lang == "es":
        ax.set_xlabel("Latencia efectiva: RTT medido + retardo (ms)", fontsize=7)
    else:
        ax.set_xlabel("Effective latency: measured RTT + injected delay (ms)", fontsize=7)
    ax.set_ylabel(r"$\eta_\mathrm{track}$ (%)", fontsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.25)


def load_runs() -> dict:
    """Mean and std of eta per (algorithm, injected delay); x = measured RTT + delay [ms]."""
    import json
    groups = {}
    for d in RUN_GLOBS:
        for f in d.glob("C_*_P2_r*.json"):
            r = json.loads(f.read_text())
            if r["cond"] != "C" or r["profile"] != "P2":
                continue
            groups.setdefault((r["algo"], r["delay_ms"]), []).append(r)
    data = {}
    for (algo, delay), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lat = np.array([r["rtt_mean_ms"] + delay for r in rs])
        eta = np.array([r["eta_pct"] for r in rs])
        d = data.setdefault(algo, {"delay": [], "rtt": [], "eta": [], "eta_std": [], "n": []})
        d["delay"].append(delay); d["rtt"].append(lat.mean()); d["eta"].append(eta.mean())
        d["eta_std"].append(eta.std(ddof=1) if len(eta) > 1 else 0.0); d["n"].append(len(eta))
    return data


def find_breakpoint(rtts, etas, threshold):
    for i in range(len(etas) - 1):
        if etas[i] >= threshold >= etas[i + 1]:
            s = (etas[i + 1] - etas[i]) / (rtts[i + 1] - rtts[i] + 1e-12)
            return rtts[i] + (threshold - etas[i]) / s
    return None


def main():
    data = load_runs()
    for a, d in data.items():
        print(a, [(dl, round(x, 1), round(e, 2), round(sd, 2), n) for dl, x, e, sd, n in zip(d['delay'], d['rtt'], d['eta'], d['eta_std'], d['n'])])

    for lang in ["es", "en"]:
        fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
        style_ax(ax, lang)

        cfg = {
            "po":  dict(color=C_PO,  marker="o", label="P&O"),
            "pso": dict(color=C_PSO, marker="s", label="PSO SCAN+TRACK"),
        }

        for algo, d in data.items():
            c = cfg[algo]
            ax.errorbar(
                d["rtt"], d["eta"], yerr=d["eta_std"],
                color=c["color"], marker=c["marker"], capsize=2,
                linestyle="-", linewidth=1.2, markersize=4,
                label=c["label"],
            )
        # ponytail: no eta threshold line; the data do not support a single RTT threshold
        ax.set_xlim(left=0)
        ax.legend(fontsize=6, loc="lower left", framealpha=0.85)

        fig.tight_layout()
        if lang == "es":
            out = FIGS_DIR / "eta_vs_rtt.pdf"
            out_png = FIGS_DIR / "eta_vs_rtt.png"
        else:
            out = FIGS_DIR / "eta_vs_rtt_en.pdf"
            out_png = FIGS_DIR / "eta_vs_rtt_en.png"

        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Guardado: {out}")

        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Guardado: {out_png}")
        plt.close(fig)


if __name__ == "__main__":
    main()
