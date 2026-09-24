#!/usr/bin/env python3
"""
scripts/monte_carlo_packet_loss.py

Monte Carlo SIL simulation to quantify MPPT tracking efficiency (eta_track)
under stochastic WiFi network latency, jitter, and packet loss rates (0% to 20%).

Responds directly to Reviewer #2 (Comments 2.2, 2.4) and Reviewer #4 (Comment 4.2).
Outputs:
  - figs/packet_loss_sensitivity.pdf
  - figs/packet_loss_sensitivity.png
  - figs/packet_loss_summary.csv
  - figs/packet_loss_stats.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTT_DIR = os.path.join(ROOT, "sections", "sec3_platform_architecture", "experimental_data_rtt")
FIGS_DIR = os.path.join(ROOT, "figs")

# ── Shared with the HIL campaign (scripts/run_factorial.py) ───────────────────
# Same firmware port, same PV module, same profiles, same client timeout, and the
# RTT pool is the closed-loop RTT actually measured on the real ESP32 (Condition C).
import glob
import sys
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from run_factorial import PORTS, HTTP_TIMEOUT_S, POWERSIM, PROFILES  # noqa: E402
from core.models.pv_cell import PVCell, PVModuleParams  # noqa: E402

FACT_DIR = os.path.join(ROOT, "data", "factorial", "runs")


def _rtt_pool(algo):
    files = glob.glob(os.path.join(FACT_DIR, f"C_{algo}_*_requests.csv"))
    files = [f for f in files if "_d" not in os.path.basename(f)]
    if not files:
        raise SystemExit("Run the Condition C campaign first (scripts/run_factorial.py)")
    df = pd.concat(pd.read_csv(f) for f in files)
    return df.loc[df.ok == 1, "rtt_ms"].to_numpy(float)


PV = PVCell(PVModuleParams.from_json(os.path.join(POWERSIM, "data", "panels", "generic_85w.json")))
G_GRID = np.linspace(50.0, 1100.0, 106)
P_MPP_GRID = np.array([PV.mpp(G=g, T_celsius=25.0)[2] for g in G_GRID])
PROFILE_DATA = {k: np.loadtxt(os.path.join(ROOT, "data", f"{v}.csv"), delimiter=",", skiprows=1)
                for k, v in PROFILES.items()}


def get_irradiance(profile_name, t):
    d = PROFILE_DATA[profile_name]
    return float(np.interp(t, d[:, 0], d[:, 1]))


def run_simulation(algo_name, profile_name, p_loss, rtt_pool, t_duration=10.0, seed=None):
    """Quasi-static event-driven loop: the inner converter loop holds v_pv = vref
    (tau_conv = 0.29 ms << RTT). A lost exchange costs the client timeout with the old vref."""
    rng = np.random.default_rng(seed)
    ctrl = PORTS[algo_name.lower()](seed)
    t, v_ref = 0.0, None
    e_act = e_mpp = 0.0
    while t < t_duration:
        g = get_irradiance(profile_name, t)
        v = 15.1 if v_ref is None else v_ref
        i = PV.current(v, G=g, T_celsius=25.0)
        p_act = v * i
        p_mpp = float(np.interp(g, G_GRID, P_MPP_GRID))
        if rng.random() < p_loss:
            dt = HTTP_TIMEOUT_S
        else:
            dt = rng.choice(rtt_pool) / 1000.0
            v_ref = ctrl.step(v, i)
        dt = min(dt, t_duration - t)
        e_act += p_act * dt
        e_mpp += p_mpp * dt
        t += dt
    return 100.0 * e_act / e_mpp


# ── Monte Carlo Experiment Matrix ─────────────────────────────────────────────
def main():
    print("Running Monte Carlo Packet Loss and Jitter Evaluation...")
    p_loss_rates = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20] # 0% to 20%
    algorithms = ["PO", "PSO"]
    profiles = ["P1", "P2", "P3"]
    N_RUNS = 50  # independent runs per cell

    results = []

    for algo in algorithms:
        rtt_pool = _rtt_pool(algo.lower())
        for prof in profiles:
            dur = 10.0
            print(f"  Evaluating {algo} on {prof}...")
            
            # Baseline (0% loss); loss cells use independent seeds -> Mann-Whitney U
            baseline_etas = [run_simulation(algo, prof, 0.0, rtt_pool, dur, seed=1000+r) for r in range(N_RUNS)]
            
            for p_loss in p_loss_rates:
                if p_loss == 0.0:
                    etas = baseline_etas
                else:
                    etas = [run_simulation(algo, prof, p_loss, rtt_pool, dur, seed=2000 + int(p_loss*1000) + r) for r in range(N_RUNS)]
                
                mean_eta = float(np.mean(etas))
                std_eta = float(np.std(etas, ddof=1))
                ci = stats.bootstrap((np.asarray(etas),), np.mean, confidence_level=0.95,
                                     n_resamples=5000, random_state=0).confidence_interval
                ci_lo, ci_hi = float(ci.low), float(ci.high)
                
                # Mann-Whitney U vs baseline (0% loss), independent samples
                if p_loss > 0.0:
                    stat, p_val = stats.mannwhitneyu(baseline_etas, etas, alternative="two-sided")
                else:
                    stat, p_val = 0.0, 1.0

                results.append({
                    "algorithm": algo,
                    "profile": prof,
                    "packet_loss_pct": round(p_loss * 100, 1),
                    "mean_eta": round(mean_eta, 3),
                    "std_eta": round(std_eta, 3),
                    "ci95_lo": round(ci_lo, 3),
                    "ci95_hi": round(ci_hi, 3),
                    "mannwhitney_p": float(p_val)
                })

    df_res = pd.DataFrame(results)
    csv_out = os.path.join(FIGS_DIR, "packet_loss_summary.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"Saved summary table to: {csv_out}")

    json_out = os.path.join(FIGS_DIR, "packet_loss_stats.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    # ── Publication Plot (figs/packet_loss_sensitivity.pdf) ───────────────────
    print("Generating publication figure...")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    colors = {"PO": "#1976D2", "PSO": "#388E3C"}
    markers = {"PO": "o", "PSO": "s"}

    for idx, prof in enumerate(profiles):
        ax = axes[idx]
        prof_title = {"P1": "P1 (Slow Ramp)", "P2": "P2 (Fast Step)", "P3": "P3 (Composite)"}[prof]
        ax.set_title(prof_title, fontsize=8, fontweight="bold")
        
        for algo in algorithms:
            subset = df_res[(df_res["algorithm"] == algo) & (df_res["profile"] == prof)]
            x = subset["packet_loss_pct"]
            y = subset["mean_eta"]
            y_err_lo = y - subset["ci95_lo"]
            y_err_hi = subset["ci95_hi"] - y
            
            label = ("P&O" if algo == "PO" else algo) if idx == 0 else None
            ax.errorbar(x, y, yerr=[y_err_lo, y_err_hi], fmt=markers[algo]+"-", 
                        color=colors[algo], label=label, capsize=2.5, markersize=4,
                        lw=1.2, alpha=0.85)

        ax.axhline(97.0, color="#C62828", linestyle="--", lw=0.8, alpha=0.7)
        if idx == 0:
            ax.text(1.0, 97.2, "97% reference", color="#C62828", fontsize=6.5)
        ax.set_xlabel("Packet Loss Rate (%)", fontsize=7.5)
        ax.set_xticks(p_loss_rates_pct := [0, 2, 5, 10, 15, 20])
        ax.set_xticklabels([f"{v}%" for v in p_loss_rates_pct], fontsize=6.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(r"Tracking Efficiency $\eta_{\mathrm{track}}$ (%)", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower left", framealpha=0.85)

    plt.tight_layout()
    pdf_path = os.path.join(FIGS_DIR, "packet_loss_sensitivity.pdf")
    png_path = os.path.join(FIGS_DIR, "packet_loss_sensitivity.png")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {pdf_path} and {png_path} successfully.")

if __name__ == "__main__":
    main()
