"""Generate RTT figures and robust summary statistics for the paper.

Outputs:
  - figs/rtt_po_wokwi.pdf
  - figs/rtt_pso_wokwi.pdf
  - figs/rtt_po_esp32.pdf
  - figs/rtt_pso_esp32.pdf
  - figs/rtt_violin_comparison.pdf
  - figs/rtt_qq_esp32.pdf
  - figs/rtt_stats_summary.csv
  - figs/rtt_stat_tests.json
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Campaign of 2026-09-23: open-loop probe, same fixed (V, I) sequence on both targets,
# n = 1000 per algorithm (scripts/run_factorial.py probe). Wokwi = Condition B, ESP32 = C.
RTT_DIR = os.path.join(ROOT, "data", "rtt_probe")
FIGS_DIR = os.path.dirname(os.path.abspath(__file__))

FIG_W_IN = 88 / 25.4
FIG_H_IN = 58 / 25.4
C_PO = "#1976D2"
C_PSO = "#388E3C"
C_ACCENT = "#C62828"
ALPHA = 0.72

FILE_CANDIDATES = {
    "wokwi_po": [os.path.join(RTT_DIR, "wokwi_po_probe.csv")],
    "wokwi_pso": [os.path.join(RTT_DIR, "wokwi_pso_probe.csv")],
    "esp_po": [os.path.join(RTT_DIR, "esp32_po_probe.csv")],
    "esp_pso": [os.path.join(RTT_DIR, "esp32_pso_probe.csv")],
}

# Use up to 1000 RTT samples per dataset when available.
N_TARGET_SAMPLES = 1000


def load_rtt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing RTT file: {path}")
    df = pd.read_csv(path, comment="#")
    if "ok" in df.columns:  # probe files: lost requests carry the timeout, not an RTT
        df = df[df["ok"] == 1]
    if "rtt_ms" in df.columns:
        values = pd.to_numeric(df["rtt_ms"], errors="coerce").dropna().to_numpy(float)
    else:
        values = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        raise ValueError(f"No valid RTT samples in: {path}")
    return values


def choose_dataset(candidates, min_samples=None):
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        values = load_rtt(candidate)
        if min_samples is not None and values.size < min_samples:
            continue
        return values, candidate
    # If all candidates exist but fail min_samples, use first existing file explicitly.
    for candidate in candidates:
        if os.path.exists(candidate):
            return load_rtt(candidate), candidate
    raise FileNotFoundError(f"No valid dataset found in candidates: {candidates}")


def robust_stats(x):
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    low_thr = q1 - 1.5 * iqr
    high_thr = q3 + 1.5 * iqr
    out_rate = np.mean((x < low_thr) | (x > high_thr))
    return {
        "n": int(x.size),
        "mean_ms": float(np.mean(x)),
        "std_ms": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
        "cv": float(np.std(x, ddof=1) / np.mean(x)) if x.size > 1 else float("nan"),
        "min_ms": float(np.min(x)),
        "p25_ms": float(q1),
        "p50_ms": float(q2),
        "p75_ms": float(q3),
        "iqr_ms": float(iqr),
        "p90_ms": float(np.percentile(x, 90)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "max_ms": float(np.max(x)),
        "outlier_rate_iqr": float(out_rate),
    }


def cliffs_delta(a, b):
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


# Bootstrap settings: the RTT distribution is non-Gaussian (Shapiro-Wilk
# p < 1e-18), so closed-form t-intervals are not valid. We use the percentile
# bootstrap to obtain 95% CIs on the mean and p99 of each condition.
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20240517  # fixed for reproducibility


def bootstrap_ci(x, stat_fn, n_boot=N_BOOTSTRAP, alpha=0.05, seed=BOOTSTRAP_SEED):
    """Percentile bootstrap CI for an arbitrary statistic of a 1-D array."""
    rng = np.random.default_rng(seed)
    n = x.size
    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = x[rng.integers(0, n, n)]
        estimates[i] = stat_fn(sample)
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_summary(x):
    """95% percentile-bootstrap CIs for the mean and p99 of an RTT sample."""
    mean_lo, mean_hi = bootstrap_ci(x, np.mean)
    p99_lo, p99_hi = bootstrap_ci(x, lambda s: np.percentile(s, 99))
    return {
        "mean_ci95_lo_ms": mean_lo,
        "mean_ci95_hi_ms": mean_hi,
        "p99_ci95_lo_ms": p99_lo,
        "p99_ci95_hi_ms": p99_hi,
    }


def style_ax(ax, ylabel="Density"):
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("RTT (ms)", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)


def draw_hist_kde(ax, values, color, label):
    ax.hist(
        values,
        bins=32,
        density=True,
        color=color,
        alpha=ALPHA,
        edgecolor="white",
        linewidth=0.3,
    )
    if np.unique(values).size > 1:
        kde = stats.gaussian_kde(values)
        xs = np.linspace(np.min(values), np.max(values), 300)
        ax.plot(xs, kde(xs), color="black", linewidth=0.8, alpha=0.75)
    st = robust_stats(values)
    ax.axvline(st["mean_ms"], color=color, lw=1.4, ls="--", label=f"mean={st['mean_ms']:.1f} ms")
    ax.axvline(st["p50_ms"], color="black", lw=1.0, ls="-", label=f"p50={st['p50_ms']:.1f} ms")
    ax.axvline(st["p99_ms"], color=C_ACCENT, lw=1.0, ls=":", label=f"p99={st['p99_ms']:.1f} ms")
    ax.set_title(label, fontsize=7, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.85)


data = {}
sources = {}
for name, candidates in FILE_CANDIDATES.items():
    min_n = N_TARGET_SAMPLES
    values, source = choose_dataset(candidates, min_samples=min_n)
    values = values[:N_TARGET_SAMPLES]
    data[name] = values
    sources[name] = source
stats_rows = []
boot_ci = {}
for name, values in data.items():
    row = {"dataset": name}
    row.update(robust_stats(values))
    ci = bootstrap_summary(values)
    row.update(ci)
    boot_ci[name] = ci
    stats_rows.append(row)

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(FIGS_DIR, "rtt_stats_summary.csv"), index=False)

tests = {
    "shapiro_esp_po_p": float(stats.shapiro(data["esp_po"]).pvalue),
    "shapiro_esp_pso_p": float(stats.shapiro(data["esp_pso"]).pvalue),
    "mann_whitney_u": float(
        stats.mannwhitneyu(data["esp_po"], data["esp_pso"], alternative="two-sided").statistic
    ),
    "mann_whitney_p": float(
        stats.mannwhitneyu(data["esp_po"], data["esp_pso"], alternative="two-sided").pvalue
    ),
    "ks_stat": float(stats.ks_2samp(data["esp_po"], data["esp_pso"]).statistic),
    "ks_p": float(stats.ks_2samp(data["esp_po"], data["esp_pso"]).pvalue),
    "cliffs_delta": float(cliffs_delta(data["esp_po"], data["esp_pso"])),
    "bootstrap": {
        "n_resamples": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
        "ci95": boot_ci,
    },
}

with open(os.path.join(FIGS_DIR, "rtt_stat_tests.json"), "w", encoding="utf-8") as f:
    json.dump(tests, f, indent=2)

print("RTT summary (ms):")
print(stats_df[["dataset", "n", "mean_ms", "std_ms", "p50_ms", "p99_ms", "cv"]].round(3))
print("\n95% bootstrap CIs (mean / p99, ms):")
for key in ("wokwi_po", "wokwi_pso", "esp_po", "esp_pso"):
    c = boot_ci[key]
    print(
        f"  {key}: mean [{c['mean_ci95_lo_ms']:.3f}, {c['mean_ci95_hi_ms']:.3f}]  "
        f"p99 [{c['p99_ci95_lo_ms']:.3f}, {c['p99_ci95_hi_ms']:.3f}]"
    )
print("\nRTT sources:")
for key in ("wokwi_po", "wokwi_pso", "esp_po", "esp_pso"):
    print(f"  {key}: {sources[key]}")
print("\nStatistical tests (ESP32 PO vs PSO):")
print(json.dumps(tests, indent=2))

# Keep existing figure names used by the LaTeX manuscript.
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
draw_hist_kde(ax, data["wokwi_po"], C_PO, "(a) P&O - Wokwi emulator")
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "rtt_po_wokwi.pdf"), dpi=600, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
draw_hist_kde(ax, data["wokwi_pso"], C_PSO, "(b) PSO - Wokwi emulator")
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "rtt_pso_wokwi.pdf"), dpi=600, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
draw_hist_kde(ax, data["esp_po"], C_PO, "(c) P&O - ESP32 real WiFi LAN")
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "rtt_po_esp32.pdf"), dpi=600, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
draw_hist_kde(ax, data["esp_pso"], C_PSO, "(d) PSO - ESP32 real WiFi LAN")
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "rtt_pso_esp32.pdf"), dpi=600, bbox_inches="tight")
plt.close(fig)

# Comparative figure for quick side-by-side interpretation.
fig, ax = plt.subplots(figsize=(2 * FIG_W_IN, 1.4 * FIG_H_IN))
labels = ["Wokwi P&O", "Wokwi PSO", "ESP32 P&O", "ESP32 PSO"]
vals = [data["wokwi_po"], data["wokwi_pso"], data["esp_po"], data["esp_pso"]]
vp = ax.violinplot(vals, showmeans=False, showmedians=True, widths=0.9)
for body in vp["bodies"]:
    body.set_facecolor("#90CAF9")
    body.set_alpha(0.4)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.0)
ax.boxplot(vals, widths=0.22, whis=(5, 95), patch_artist=True, boxprops={"facecolor": "#FFF59D", "alpha": 0.8})
ax.set_xticks(range(1, len(labels) + 1))
ax.set_xticklabels(labels, rotation=13, ha="right", fontsize=7)
ax.set_ylabel("RTT (ms)", fontsize=7)
ax.set_title("RTT distribution comparison (Wokwi vs ESP32, P&O vs PSO)", fontsize=8, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "rtt_violin_comparison.pdf"), dpi=600, bbox_inches="tight")
plt.close(fig)

# Q-Q diagnostic for the real ESP32 datasets.
fig, axes = plt.subplots(1, 2, figsize=(2 * FIG_W_IN, FIG_H_IN))
stats.probplot(data["esp_po"], dist="norm", plot=axes[0])
axes[0].set_title("Q-Q plot ESP32 P&O", fontsize=7, fontweight="bold")
axes[0].tick_params(labelsize=6)
axes[0].spines[["top", "right"]].set_visible(False)
stats.probplot(data["esp_pso"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot ESP32 PSO", fontsize=7, fontweight="bold")
axes[1].tick_params(labelsize=6)
axes[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "rtt_qq_esp32.pdf"), dpi=600, bbox_inches="tight")
plt.close(fig)

print("Saved RTT figures and summary tables in figs/.")
