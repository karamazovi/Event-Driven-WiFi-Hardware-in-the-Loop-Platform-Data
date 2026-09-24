#!/usr/bin/env python3
"""Fail if a number printed in the manuscript tables does not match the raw data.

Recomputes, from data/ and figs/, every value of Tables 3 (ESP32 resources), 4 (factorial),
the RTT table, the delay-sweep table and the Monte Carlo table, formats it as the .tex does,
and asserts it appears in the corresponding section file.

  python scripts/check_numbers.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
S3 = open("sections/sec3_platform_architecture/sec3_platform_architecture_en.tex", encoding="utf-8").read()
S4 = open("sections/sec4_mppt_implementation/sec4_mppt_implementation_en.tex", encoding="utf-8").read()
S6 = open("sections/sec6_results/sec6_results_en.tex", encoding="utf-8").read()
fails = []


def expect(text, s, what):
    if " ".join(s.split()) not in " ".join(text.split()):
        fails.append(f"{what}: '{s}' not found")


# ── Table 4: factorial ────────────────────────────────────────────────────────
runs = pd.DataFrame([json.load(open(f)) for f in glob.glob("data/factorial/runs/*.json")])
runs = runs[runs.delay_ms == 0]
for (a, c, p), g in runs.groupby(["algo", "cond", "profile"]):
    assert len(g) == 5, f"{a}/{c}/{p} has {len(g)} runs"
    e = g.eta_pct
    lab = "A$'$" if c == "D" else c
    expect(S6, f"& {lab} / {p} & ${e.mean():.2f} \\pm {e.std(ddof=1):.2f}$ ({e.min():.2f})", f"T4 eta {a}{c}{p}")
    if c in ("B", "C"):
        expect(S6, f"{g.rtt_p50_ms.median():.1f} & {100 * g.n_fail.sum() / g.n_req.sum():.1f} \\\\", f"T4 rtt/loss {a}{c}{p}")

# ── Table 3: resources from closed-loop C requests ──────────────────────────────
res = json.load(open("data/factorial/resources.json"))
po, pso = res["C_po"], res["C_pso"]
expect(S4, f"${po['t_us_mean']:.1f} \\pm {po['t_us_std']:.1f}$ & ${pso['t_us_mean']:.1f} \\pm {pso['t_us_std']:.1f}$", "T3 t_exec")
expect(S4, f"{po['t_us_min']:.0f} / {po['t_us_p99']:.0f} / {po['t_us_max']:.0f} & {pso['t_us_min']:.0f} / {pso['t_us_p99']:.0f} / {pso['t_us_max']:.0f}", "T3 t_exec range")
expect(S4, f"${po['t_loc_us_mean']/1e3:.2f} \\pm {po['t_loc_us_std']/1e3:.2f}$ & ${pso['t_loc_us_mean']/1e3:.2f} \\pm {pso['t_loc_us_std']/1e3:.2f}$", "T3 T_loc")
hmin = min(po["heap_min_B"], pso["heap_min_B"]) / 1e3
hmax = max(po["heap_max_B"], pso["heap_max_B"]) / 1e3
expect(S4, f"{hmin:.1f}--{hmax:.1f}", "T3 heap")
expect(S4, f"& {po['rtt_mean_ms'] + 0.5:.1f} & {pso['rtt_mean_ms'] + 0.5:.1f} \\\\", "T3 T_s")
expect(S4, f"n={po['n_req'] - po['n_lost']:,}".replace(",", "{,}"), "T3 n P&O")
build = open("data/build_size_esp32.txt").read()
expect(build, "used 44008 bytes", "build RAM")
expect(build, "used 796465 bytes", "build flash")

# ── RTT table (open-loop probe) ───────────────────────────────────────────────
st = pd.read_csv("figs/rtt_stats_summary.csv").set_index("dataset")
row = " & ".join(f"{st.loc[d, 'p50_ms']:.2f}" for d in ("wokwi_po", "wokwi_pso", "esp_po", "esp_pso"))
expect(S3, row, "RTT medians")
row = " & ".join(f"{st.loc[d, 'p99_ms']:.2f}" for d in ("wokwi_po", "wokwi_pso", "esp_po", "esp_pso"))
expect(S3, row, "RTT p99")
for lbl in ("C_po", "C_pso", "B_po", "B_pso"):
    expect(S3, f"{res[lbl]['n_req']}", f"RTT closed-loop exchanges {lbl}")

# ── Delay sweep ───────────────────────────────────────────────────────────────
for r in json.load(open("data/sensitivity_rtt/summary.json")):
    expect(S6, f"{r['latency_ms']:.1f} & ${r['eta_mean']:.2f} \\pm {r['eta_std']:.2f}$", f"sweep {r['algo']} d={r['delay_ms']}")

# ── Monte Carlo ───────────────────────────────────────────────────────────────
mc = pd.read_csv("figs/packet_loss_summary.csv")
for _, r in mc[mc.packet_loss_pct.isin([0, 2, 5, 10, 20])].iterrows():
    expect(S6, f"& {int(r.packet_loss_pct)}{' ' if r.packet_loss_pct < 10 else ''} & {r.mean_eta:.2f} & {r.std_eta:.2f} & [{r.ci95_lo:.2f}, {r.ci95_hi:.2f}]",
           f"MC {r.algorithm} {r.profile} {r.packet_loss_pct}")

print(f"{len(fails)} mismatches")
for f in fails:
    print("  -", f)
raise SystemExit(1 if fails else 0)
