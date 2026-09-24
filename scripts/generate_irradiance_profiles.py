"""
generate_irradiance_profiles.py — Generates the 3 irradiance profiles for MPPT experiments.

Profiles:
  P1  Ramp (V-shape):  1000 → 800 → 400 → 700 → 1000 W/m²  (linear transitions)
  P2  Step:            1000 (constant X₁), then step to 400 W/m² (constant remainder)
  P3  Composite:       1000→400 (ramp), 400→800 (step), 800→200 (ramp),
                       200→600 (step), 600→1000 (ramp)

Time parametrisation:
  5 segments of duration X₁, X₂, X₃, X₄, X₅ (seconds).
  Default: 7 s each → 35 s total (≥ 30 s required by acceptance criteria).

Outputs (relative to project root):
  data/perfil_P1_rampa.csv
  data/perfil_P2_escalon.csv
  data/perfil_P3_compuesto.csv
  figs/perfiles_irradiancia.pdf

Run:
  python scripts/generate_irradiance_profiles.py                     # defaults
  python scripts/generate_irradiance_profiles.py 12 12 12 12 12     # 60 s total

References:
  IEC 61853-1 (2011), IEC 61853-2 (2018), BSRN, NREL SRRL,
  Femia et al. (2005) — see sec5_experimental_design.bib
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

DT = 0.05  # time step [s] → 20 samples/s (50 ms resolution)

# Default segment durations [s]
DEFAULT_X = [7.0, 7.0, 7.0, 7.0, 7.0]  # total 35 s

# Parse optional CLI arguments: X1 X2 X3 X4 X5
if len(sys.argv) == 6:
    X = [float(v) for v in sys.argv[1:6]]
elif len(sys.argv) == 1:
    X = DEFAULT_X
else:
    print("Usage: python generate_irradiance_profiles.py [X1 X2 X3 X4 X5]")
    sys.exit(1)

X1, X2, X3, X4, X5 = X
T_TOTAL = sum(X)

print(f"Segment durations: X1={X1}, X2={X2}, X3={X3}, X4={X4}, X5={X5}")
print(f"Total duration:    {T_TOTAL:.1f} s  ({T_TOTAL/DT:.0f} samples)")

# Breakpoints (cumulative time)
t0 = 0.0
t1 = X1
t2 = X1 + X2
t3 = X1 + X2 + X3
t4 = X1 + X2 + X3 + X4
t5 = T_TOTAL

# Time vector
t = np.arange(0, T_TOTAL + DT / 2, DT)
N = len(t)

# ══════════════════════════════════════════════════════════════════════════════
# Profile P1 — Linear ramp (V-shape: dawn/dusk cycle)
# ══════════════════════════════════════════════════════════════════════════════
# Irradiance levels at breakpoints
# t0→t1 : 1000 → 1000  (constant — initial stabilisation)
# t1→t2 : 1000 → 800   (ramp ↓)
# t2→t3 :  800 → 400   (ramp ↓ — minimum)
# t3→t4 :  400 → 700   (ramp ↑)
# t4→t5 :  700 → 1000  (ramp ↑ — recovery)
#
# Physical justification:
#   IEC 61853-1 defines discrete irradiance levels (200, 400, 600, 800, 1000 W/m²)
#   for performance testing.  The linear ramp is the continuous-time counterpart.
#   Slopes of ~10 W/m²/s are consistent with BSRN measurements during
#   stratiform cloud passages.

G_P1_bp = np.array([1000, 1000, 800, 400, 700, 1000], dtype=float)
t_bp = np.array([t0, t1, t2, t3, t4, t5])

G_P1 = np.interp(t, t_bp, G_P1_bp)

# ══════════════════════════════════════════════════════════════════════════════
# Profile P2 — Step (abrupt cloud shadow)
# ══════════════════════════════════════════════════════════════════════════════
# t0→t1 : 1000  (constant)
# t1     : instantaneous step 1000 → 400  (< 50 ms effective)
# t1→t5 :  400  (constant)
#
# Physical justification:
#   BSRN high-resolution measurements record drops of 500–800 W/m² in < 1 s
#   at cumulus cloud edges.  The step is the standard worst-case test for MPPT
#   (Femia et al., 2005).

G_P2 = np.where(t < t1, 1000.0, 400.0)

# ══════════════════════════════════════════════════════════════════════════════
# Profile P3 — Composite (ramps + steps, partially cloudy day)
# ══════════════════════════════════════════════════════════════════════════════
# Segment  | Transition G       | Type     | Physical phenomenon
# t0 → t1  | 1000 → 400         | Ramp ↓   | Gradual cloud build-up
# t1 → t2  | 400  → 800         | Step ⚡   | Cloud-edge clearing (abrupt)
# t2 → t3  | 800  → 200         | Ramp ↓   | Dense cloud advancing gradually
# t3 → t4  | 200  → 600         | Step ⚡   | Sudden break in cloud cover
# t4 → t5  | 600  → 1000        | Ramp ↑   | Gradual recovery to full sun
#
# The sequence 1000, 400, 800, 200, 600, 1000 covers the operating points of
# the IEC 61853-2 energy rating matrix.  The ramp/step alternation reproduces
# typical partially-cloudy-day variability, validated with NREL SRRL data.

G_P3_bp_start = np.array([1000, 400, 800, 200, 600], dtype=float)
G_P3_bp_end   = np.array([ 400, 800, 200, 600, 1000], dtype=float)
seg_type      = ["ramp", "step", "ramp", "step", "ramp"]

G_P3 = np.empty(N)
for i, (ts, te, gs, ge, stype) in enumerate(zip(
        [t0, t1, t2, t3, t4],
        [t1, t2, t3, t4, t5],
        G_P3_bp_start, G_P3_bp_end, seg_type)):
    mask = (t >= ts) & (t < te) if i < 4 else (t >= ts) & (t <= te)
    if stype == "ramp":
        G_P3[mask] = np.interp(t[mask], [ts, te], [gs, ge])
    else:  # step: instantaneous jump at the start of the segment
        G_P3[mask] = ge

# ══════════════════════════════════════════════════════════════════════════════
# Export CSVs
# ══════════════════════════════════════════════════════════════════════════════

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
FIGS_DIR = os.path.join(ROOT, "figs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

profiles = {
    "perfil_P1_rampa":     G_P1,
    "perfil_P2_escalon":   G_P2,
    "perfil_P3_compuesto": G_P3,
}

for name, G in profiles.items():
    path = os.path.join(DATA_DIR, f"{name}.csv")
    header = "t_s,G_Wm2"
    data = np.column_stack([t, G])
    np.savetxt(path, data, delimiter=",", header=header, comments="",
               fmt=["%.3f", "%.1f"])
    print(f"Exported: {path}  ({len(data)} rows, {t[-1]:.1f} s)")

# ══════════════════════════════════════════════════════════════════════════════
# Figure — 3 profiles overlaid (IEEE single-column style)
# ══════════════════════════════════════════════════════════════════════════════

FIG_W_MM = 88   # IEEE single column width
FIG_H_MM = 60
FIG_W_IN = FIG_W_MM / 25.4
FIG_H_IN = FIG_H_MM / 25.4

COLORS = {
    "P1": "#2196F3",  # blue
    "P2": "#F44336",  # red
    "P3": "#4CAF50",  # green
}

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))

ax.plot(t, G_P1, color=COLORS["P1"], lw=1.2, label="P1 — Rampa lineal")
ax.plot(t, G_P2, color=COLORS["P2"], lw=1.2, label="P2 — Escalón")
ax.plot(t, G_P3, color=COLORS["P3"], lw=1.2, label="P3 — Compuesto")

# Breakpoint vertical guides (light grey dashed)
for tb in [t1, t2, t3, t4]:
    ax.axvline(tb, color="lightgray", ls="--", lw=0.6, zorder=0)

# Breakpoint labels
for i, tb in enumerate([t1, t2, t3, t4]):
    ax.text(tb, 1050, f"$t_{{{i+1}}}$", ha="center", va="bottom",
            fontsize=6.5, color="gray")

# IEC 61853 irradiance levels (horizontal reference lines)
for g_ref in [200, 400, 600, 800, 1000]:
    ax.axhline(g_ref, color="#E0E0E0", ls=":", lw=0.4, zorder=0)

ax.set_xlabel("Tiempo (s)", fontsize=8)
ax.set_ylabel(r"Irradiancia $G$ (W/m$^2$)", fontsize=8)
ax.set_xlim(0, T_TOTAL)
ax.set_ylim(0, 1150)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9, edgecolor="lightgray")
ax.spines[["top", "right"]].set_visible(False)

# X-axis ticks at breakpoints
ax.set_xticks([t0, t1, t2, t3, t4, t5])
ax.set_xticklabels([f"{v:.0f}" for v in [t0, t1, t2, t3, t4, t5]], fontsize=7)

plt.tight_layout(pad=0.4)

out_pdf = os.path.join(FIGS_DIR, "perfiles_irradiancia.pdf")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
print(f"Saved figure: {out_pdf}")

# Also save PNG for quick preview
out_png = os.path.join(FIGS_DIR, "perfiles_irradiancia.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved preview: {out_png}")

print("\n[OK] All profiles generated successfully.")
