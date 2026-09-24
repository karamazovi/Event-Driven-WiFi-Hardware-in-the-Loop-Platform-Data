"""Run ideal MPPT benchmark experiments (zero communication latency).

Generates 6 reference experiments (P&O and PSO-MPPT × 3 irradiance profiles).
PSO is run with 3 seeds [42, 0, 123] and mean ± std are reported.

Simulation duration per profile (normative basis):
- P1 rampa    : 6.5 s  → ramp rate ≈ 102 W/m²/s  (EN 50530:2010+A1 §5.4, max dynamic rate)
- P2 escalon  : 2.0 s  → 1 s pre-step + 1 s post-step (≥ 5× t_settle per IEC 60050-351 §351-37-07)
- P3 compuesto: 3.0 s  → 300 ms per irradiance segment (accelerated composite, per Subudhi 2013)

Metrics:
- η = ∫P_actual dt / ∫P_MPP dt  (EN 50530:2010+A1 §5.4 / IEC 62891:2022)
- Settling time: first crossing at 5% band after step (P2 only)
  Rationale: P&O's inherent ±ΔV oscillation prevents permanent-band settlement;
  first-crossing is the correct measure per Sera et al. (2013) IEEE J. Photovoltaics.
- Ripple RMS of power in last 20% window (steady-state; Tan et al. 2005 IEEE Trans. Ind. Electron.)

Acceptance criterion:
- 6 CSV files are generated
- eta > 99% in all six cases
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow running from any CWD: python scripts/run_ideal_benchmark.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.design.boost_battery_design import BoostBatteryDesign
from core.models.boost_battery_model import BoostBatteryModel
from core.models.pv_cell import PVCell, PVModuleParams
from core.mppt.base_mppt import get_mppt_registry
from core.simulation.engine import SimulationEngine
from core.simulation.profiles import get_irradiance_profile


# Normative simulation durations per profile (see module docstring for references).
T_SIM_PER_PROFILE: dict[str, float] = {
    "rampa":    6.5,   # EN 50530 §5.4: 102 W/m²/s ramp rate
    "escalon":  2.0,   # IEC 60050-351: ≥5× t_settle ≈ 5 × 306ms
    "compuesto": 3.0,  # accelerated composite, 300 ms/segment
}

PSO_SEEDS: list[int] = [42, 0, 123]


@dataclass(frozen=True)
class ExperimentCase:
    algo_label: str
    profile_label: str
    profile_type: str
    t_sim_s: float


@dataclass(frozen=True)
class ExperimentMetrics:
    algo_label: str
    profile_label: str
    eta_pct: float
    eta_std_pct: float        # std over PSO seeds; 0.0 for deterministic algorithms
    settling_time_ms: float | None
    ripple_rms_w: float
    csv_path: Path


def _make_design() -> BoostBatteryDesign:
    """Create standard Boost+Battery design with manual parameters."""
    return BoostBatteryDesign(
        D=0.2458,           # Volt-second balance: 1 - 18.1/24.0 = 0.2458
        Iout_A=3.54,
        IL_avg_A=4.7,
        delta_IL_A=1.5,
        delta_Vc_V=0.5,
        L_H=330e-6,         # L = 330 µH
        C_F=22e-6,          # C = 22 µF (output cap Co)
        Ci_F=22e-6,         # Ci = 22 µF (input cap Ci)
        Rci_ohm=0.006,      # Rci = 6 mΩ
        RL_ohm=0.060,       # RL = 60 mΩ
        ESR_ohm=0.006,      # ESR = 6 mΩ (RCo)
        Vd_V=0.4,           # Schottky diode forward drop 0.4V
        is_CCM=True,
        IL_critical_A=0.1,
        eta_estimated=0.99,
        V_bat=24.0,         # V_bat = 24 V
        R_bat=0.069,        # R_Bat = 69 mΩ
        R_on=0.035,         # Ron = 35 mΩ
        warnings=[]
    )


def _make_mppt(algo_label: str):
    """Instantiate MPPT controller with parameters tuned for ideal benchmark."""
    registry = get_mppt_registry()
    if algo_label == "P&O":
        cls = registry["Perturb & Observe (P&O)"]
        return cls(delta_V=0.2)
    if algo_label == "PSO-MPPT":
        cls = registry["PSO-MPPT"]
        return cls(
            n_particles=5,
            w=0.5,
            c1=2.0,
            c2=2.0,
            max_iterations=10,
            delta_d_tracking=0.002,
            restart_threshold=0.95,   # benchmark irradiancia uniforme: sin reinicio SCAN
        )
    raise ValueError(f"Unsupported algorithm label: {algo_label}")


def _compute_settling_time_ms(
    time_s: np.ndarray,
    p_pv_w: np.ndarray,
    p_mpp_w: np.ndarray,
    threshold_pct: float = 2.0,
) -> float | None:
    """Compute 2% settling time of tracking error after last irradiance change.

    For variable irradiance profiles, settling is measured from the last detectable
    change in P_MPP(t) until the first instant where tracking error stays within
    the 2% band to the end of simulation.
    """
    if len(time_s) == 0:
        return None

    eps = 1e-9
    change_idx = np.flatnonzero(np.abs(np.diff(p_mpp_w)) > eps)
    start_idx = int(change_idx[-1] + 1) if change_idx.size > 0 else 0

    denom = np.maximum(np.abs(p_mpp_w), 1e-12)
    rel_err = np.abs(p_pv_w - p_mpp_w) / denom
    threshold = threshold_pct / 100.0
    within = rel_err <= threshold

    outside_after_start = np.flatnonzero(~within[start_idx:])
    if outside_after_start.size == 0:
        return float(time_s[start_idx] * 1e3)

    last_outside_rel = int(outside_after_start[-1])
    last_outside_abs = start_idx + last_outside_rel
    if last_outside_abs >= len(time_s) - 1:
        return None

    settle_idx = last_outside_abs + 1
    return float(time_s[settle_idx] * 1e3)


# Backward-compatible integration helper for numpy version parity
if hasattr(np, "trapezoid"):
    _integrate = np.trapezoid
else:
    _integrate = np.trapz


def _first_crossing_ms(
    time_s: np.ndarray,
    p_pv_w: np.ndarray,
    p_mpp_w: np.ndarray,
    threshold_pct: float = 5.0,
) -> float | None:
    """Primer cruce del error de tracking por debajo del threshold tras el último cambio de P_MPP.

    Mide velocidad de relocalización del MPP. Solo significativo para perfiles
    con escalón discreto (profile_label == 'escalon').
    """
    if len(time_s) == 0:
        return None
    eps = 1e-9
    change_idx = np.flatnonzero(np.abs(np.diff(p_mpp_w)) > eps)
    start_idx = int(change_idx[-1] + 1) if change_idx.size > 0 else 0

    denom = np.maximum(np.abs(p_mpp_w[start_idx:]), 1e-12)
    rel_err = np.abs(p_pv_w[start_idx:] - p_mpp_w[start_idx:]) / denom
    within = rel_err <= threshold_pct / 100.0

    if not within.any():
        return None
    first_idx = int(np.argmax(within))
    return float(time_s[start_idx + first_idx] * 1e3)


def _compute_metrics(
    time_s: np.ndarray,
    p_pv_w: np.ndarray,
    p_mpp_w: np.ndarray,
    profile_label: str = "",
) -> tuple[float, float | None, float]:
    """Return eta [%], settling time [ms], and ripple RMS [W].

    Settling time solo se calcula para 'escalon' (primer cruce al 5%).
    Para perfiles de irradiancia continua (rampa, compuesto) se retorna None (—).
    """
    energy_actual = float(_integrate(p_pv_w, x=time_s))
    energy_mpp = float(_integrate(p_mpp_w, x=time_s))
    eta_pct = 100.0 * energy_actual / energy_mpp if energy_mpp > 0.0 else 0.0

    if profile_label == "escalon":
        settling_time_ms = _first_crossing_ms(time_s, p_pv_w, p_mpp_w, threshold_pct=5.0)
    else:
        settling_time_ms = None  # no aplica para perfiles de irradiancia continua

    ss_start = min(int(0.8 * len(p_pv_w)), max(len(p_pv_w) - 2, 0))
    p_ss = p_pv_w[ss_start:]
    p_ss_mean = float(np.mean(p_ss)) if len(p_ss) > 0 else 0.0
    ripple_rms_w = float(math.sqrt(np.mean((p_ss - p_ss_mean) ** 2))) if len(p_ss) > 0 else 0.0

    return eta_pct, settling_time_ms, ripple_rms_w


def _write_timeseries_csv(
    out_path: Path,
    time_s: np.ndarray,
    v_pv_v: np.ndarray,
    i_pv_a: np.ndarray,
    p_pv_w: np.ndarray,
    p_mpp_w: np.ndarray,
    duty: np.ndarray,
) -> None:
    """Write benchmark time-series CSV with explicit dynamic P_MPP(t)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_error_pct = 100.0 * np.abs(p_pv_w - p_mpp_w) / np.maximum(np.abs(p_mpp_w), 1e-12)

    data = np.column_stack([
        time_s,
        v_pv_v,
        i_pv_a,
        p_pv_w,
        p_mpp_w,
        tracking_error_pct,
        duty,
    ])

    header = (
        "Time(s),V(t),I(t),P(t),P_MPP(t),"
        "Tracking_Error(%),Duty"
    )
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.8g")


def _write_summary_csv(summary_path: Path, metrics: list[ExperimentMetrics]) -> None:
    """Write summary metrics table for the six benchmark experiments."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm",
            "irradiance_profile",
            "t_sim_s",
            "eta_pct",
            "eta_std_pct",
            "t_settle_5pct_ms",
            "ripple_rms_w",
            "csv_file",
        ])
        for row in metrics:
            writer.writerow([
                row.algo_label,
                row.profile_label,
                f"{T_SIM_PER_PROFILE[row.profile_label]:.1f}",
                f"{row.eta_pct:.6f}",
                f"{row.eta_std_pct:.6f}",
                "" if row.settling_time_ms is None else f"{row.settling_time_ms:.6f}",
                f"{row.ripple_rms_w:.6f}",
                str(row.csv_path.name),
            ])


def _validate_series_integrity(
    *,
    time_s: np.ndarray,
    v_pv_v: np.ndarray,
    i_pv_a: np.ndarray,
    p_pv_w: np.ndarray,
    p_mpp_w: np.ndarray,
    duty: np.ndarray,
    context: str,
) -> None:
    """Validate equal lengths and finite values before CSV export."""
    expected = len(time_s)
    series = {
        "time_s": time_s,
        "v_pv_v": v_pv_v,
        "i_pv_a": i_pv_a,
        "p_pv_w": p_pv_w,
        "p_mpp_w": p_mpp_w,
        "duty": duty,
    }

    for name, values in series.items():
        if len(values) != expected:
            raise ValueError(
                f"{context}: inconsistent length for {name}. "
                f"expected={expected}, got={len(values)}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{context}: non-finite values found in {name}")


def _build_cases(
    algo_filter: str | None = None,
    profile_filter: str | None = None,
) -> list[ExperimentCase]:
    profiles = [
        ("rampa",    "rampa_lenta"),
        ("escalon",  "escalon"),
        ("compuesto","realista_estatico"),
    ]
    algos = ["P&O", "PSO-MPPT"]

    cases = [
        ExperimentCase(
            algo_label=algo,
            profile_label=profile_label,
            profile_type=profile_type,
            t_sim_s=T_SIM_PER_PROFILE[profile_label],
        )
        for algo in algos
        for profile_label, profile_type in profiles
    ]
    if algo_filter:
        cases = [c for c in cases if c.algo_label == algo_filter]
    if profile_filter:
        cases = [c for c in cases if c.profile_label == profile_filter]
    return cases


def _run_single(
    *,
    pv: PVCell,
    engine: SimulationEngine,
    design: BoostBatteryDesign,
    case: ExperimentCase,
    t_mppt_cycles: int,
    seed: int,
    v_oc_init: float,
    fsw_hz: float = 100e3,
    vout_v: float = 24.0,
    g_base: float = 1000.0,
    t_c: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one simulation and return (time, v_pv, i_pv, p_pv, p_mpp, duty)."""
    random.seed(seed)
    np.random.seed(seed)

    g_profile = get_irradiance_profile(case.profile_type, t_sim=case.t_sim_s, G_base=g_base)
    sim_params = {
        "f_sw": fsw_hz,
        "V_bat": vout_v,
        "R_bat": design.R_bat,
        "t_sim": case.t_sim_s,
        "T_mppt_cycles": t_mppt_cycles,
        "G": g_base,
        "T_celsius": t_c,
        "G_profile": g_profile,
        # full_output=True forces short-sim mode (full ODE resolution) regardless of t_sim.
        # Required for accurate η: long-sim mode underestimates P_actual by ~2 pp
        # due to a one-interval stale voltage approximation during convergence.
        "full_output": True,
        "v_c_initial_V": v_oc_init,
    }
    data = engine.run(
        pv_model=pv,
        converter_model=BoostBatteryModel(),
        mppt=_make_mppt(case.algo_label),
        design_result=design,
        params=sim_params,
    )

    time_s = np.asarray(data.time)
    v_pv_v = np.asarray(data.v_pv)
    i_pv_a = np.asarray(data.i_pv)
    p_pv_w = np.asarray(data.p_pv)

    if len(data.p_mpp) == len(time_s):
        p_mpp_w = np.asarray(data.p_mpp)
    else:
        p_mpp_nominal = float(pv.mpp(G=g_base, T_celsius=t_c)[2])
        p_mpp_w = np.full_like(time_s, fill_value=p_mpp_nominal, dtype=float)

    duty = np.asarray(data.duty)
    return time_s, v_pv_v, i_pv_a, p_pv_w, p_mpp_w, duty


def run_benchmark(
    output_dir: Path,
    t_mppt_cycles: int,
    algo_filter: str | None = None,
    profile_filter: str | None = None,
) -> list[ExperimentMetrics]:
    """Run ideal benchmark experiments and return metric rows.

    PSO is executed with PSO_SEEDS = [42, 0, 123]; η mean ± std is reported.
    P&O is deterministic and uses only seed 42.
    Each profile uses its normative t_sim from T_SIM_PER_PROFILE.
    """
    panel_path = PROJECT_ROOT / "data" / "panels" / "generic_85w.json"
    module = PVModuleParams.from_json(panel_path)
    pv = PVCell(module)

    fsw_hz = 100e3
    vout_v = 24.0
    g_base = 1000.0
    t_c = 25.0

    design = _make_design()
    engine = SimulationEngine()
    v_oc_init = float(module.V_oc)

    metrics: list[ExperimentMetrics] = []
    cases = _build_cases(algo_filter=algo_filter, profile_filter=profile_filter)
    total = len(cases)

    for i, case in enumerate(cases):
        t_sim_s = case.t_sim_s
        print(f"\n[{i+1}/{total}] ▶ Algoritmo: {case.algo_label} | Perfil: {case.profile_label} | t_sim={t_sim_s:.1f}s")
        print(f"       Simulando con Fsw={fsw_hz/1e3:.0f} kHz, T_mppt_cycles={t_mppt_cycles}...")

        seeds = PSO_SEEDS if case.algo_label == "PSO-MPPT" else [PSO_SEEDS[0]]
        eta_runs: list[float] = []
        time_s_ref = v_ref = i_ref = p_ref = pm_ref = duty_ref = None

        for run_idx, seed in enumerate(seeds):
            time_s, v_pv_v, i_pv_a, p_pv_w, p_mpp_w, duty = _run_single(
                pv=pv, engine=engine, design=design, case=case,
                t_mppt_cycles=t_mppt_cycles, seed=seed, v_oc_init=v_oc_init,
                fsw_hz=fsw_hz, vout_v=vout_v, g_base=g_base, t_c=t_c,
            )
            _validate_series_integrity(
                time_s=time_s, v_pv_v=v_pv_v, i_pv_a=i_pv_a,
                p_pv_w=p_pv_w, p_mpp_w=p_mpp_w, duty=duty,
                context=f"{case.algo_label}/{case.profile_label}/seed{seed}",
            )
            eta_run, _, _ = _compute_metrics(time_s, p_pv_w, p_mpp_w, profile_label=case.profile_label)
            eta_runs.append(eta_run)
            if run_idx == 0:
                time_s_ref, v_ref, i_ref, p_ref, pm_ref, duty_ref = (
                    time_s, v_pv_v, i_pv_a, p_pv_w, p_mpp_w, duty
                )

        eta_pct = float(np.mean(eta_runs))
        eta_std = float(np.std(eta_runs, ddof=0)) if len(eta_runs) > 1 else 0.0

        _, settling_time_ms, ripple_rms_w = _compute_metrics(
            time_s_ref, p_ref, pm_ref, profile_label=case.profile_label
        )

        if case.profile_label == "escalon":
            settling_txt = "—" if settling_time_ms is None else f"{settling_time_ms:.3f} ms *"
        else:
            settling_txt = "—"
        eta_ok = "✓" if eta_pct > 99.0 else "✗ FALLO"
        seed_info = f"(mean of {len(seeds)} seeds)" if len(seeds) > 1 else ""
        print(f"       η = {eta_pct:.4f}% ±{eta_std:.4f}% {eta_ok} {seed_info}  |  t_settle(5%) = {settling_txt}  |  ripple RMS = {ripple_rms_w:.5f} W")

        algo_slug = "po" if case.algo_label == "P&O" else "pso"
        out_name = f"ideal_benchmark_{algo_slug}_{case.profile_label}.csv"
        out_path = output_dir / out_name

        print(f"       ✓ CSV guardado → {out_path.name}")
        _write_timeseries_csv(
            out_path=out_path,
            time_s=time_s_ref,
            v_pv_v=v_ref,
            i_pv_a=i_ref,
            p_pv_w=p_ref,
            p_mpp_w=pm_ref,
            duty=duty_ref,
        )

        metrics.append(
            ExperimentMetrics(
                algo_label=case.algo_label,
                profile_label=case.profile_label,
                eta_pct=eta_pct,
                eta_std_pct=eta_std,
                settling_time_ms=settling_time_ms,
                ripple_rms_w=ripple_rms_w,
                csv_path=out_path,
            )
        )

    return metrics


def _validate_acceptance(metrics: list[ExperimentMetrics], eta_threshold_pct: float = 99.0, expected_count: int = 6) -> tuple[bool, list[str]]:
    """Validate acceptance: expected CSVs + eta > threshold on all experiments."""
    errors: list[str] = []

    if len(metrics) != expected_count:
        errors.append(f"Expected {expected_count} experiments, got {len(metrics)}")

    for row in metrics:
        if not row.csv_path.exists():
            errors.append(f"Missing CSV: {row.csv_path}")
        if row.eta_pct <= eta_threshold_pct:
            errors.append(
                f"eta <= {eta_threshold_pct:.1f}% for {row.algo_label} + {row.profile_label}: "
                f"{row.eta_pct:.4f}%"
            )

    return (len(errors) == 0), errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ideal MPPT benchmark (6 experiments, normative t_sim per profile).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Per-profile simulation durations (normative):\n"
            f"  rampa     = {T_SIM_PER_PROFILE['rampa']:.1f} s  (EN 50530 §5.4, ~102 W/m²/s)\n"
            f"  escalon   = {T_SIM_PER_PROFILE['escalon']:.1f} s  (IEC 60050-351, ≥5× t_settle)\n"
            f"  compuesto = {T_SIM_PER_PROFILE['compuesto']:.1f} s  (accelerated composite)\n"
            f"PSO seeds: {PSO_SEEDS}  (mean ± std reported)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "simulation_outputs",
        help="Directory for benchmark CSV outputs.",
    )
    parser.add_argument(
        "--t-mppt-cycles",
        type=int,
        default=20,
        help="Switching cycles per MPPT update (T_mppt = cycles / f_sw).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["P&O", "PSO-MPPT"],
        help="Correr solo este algoritmo (omitir = ambos).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["rampa", "escalon", "compuesto"],
        help="Correr solo este perfil de irradiancia (omitir = los tres).",
    )
    args = parser.parse_args()

    metrics = run_benchmark(
        output_dir=args.output_dir,
        t_mppt_cycles=args.t_mppt_cycles,
        algo_filter=args.algo,
        profile_filter=args.profile,
    )

    summary_path = args.output_dir / "ideal_benchmark_summary.csv"
    _write_summary_csv(summary_path, metrics)

    print("\n" + "="*80)
    print("RESUMEN DEL BENCHMARK IDEAL:")
    print(f"  {'Algoritmo':<10} {'Perfil':<10} {'t_sim':>6} {'η (%)':>12} {'±std':>8} {'t_settle 5%':>13} {'Ripple RMS':>12}")
    print("  " + "-"*75)
    for row in metrics:
        if row.profile_label == "escalon":
            settling_txt = "—" if row.settling_time_ms is None else f"{row.settling_time_ms:.3f} ms"
        else:
            settling_txt = "—"
        std_txt = f"±{row.eta_std_pct:.4f}%" if row.eta_std_pct > 0 else "—"
        print(
            f"  {row.algo_label:<10} {row.profile_label:<10} "
            f"{T_SIM_PER_PROFILE[row.profile_label]:>5.1f}s"
            f" {row.eta_pct:>11.4f}% {std_txt:>8} {settling_txt:>13} {row.ripple_rms_w:>11.5f} W"
        )
    print(f"\n  * t_settle: primer cruce al 5% tras el escalón de irradiancia")
    print(f"  — no aplica para perfiles de irradiancia continua (rampa, compuesto)")
    print(f"\n  Resumen CSV → {summary_path.name}")
    print("="*80)

    expected = 6 if (args.algo is None and args.profile is None) else len(metrics)
    ok, errors = _validate_acceptance(metrics, eta_threshold_pct=99.0, expected_count=expected)
    if not ok:
        print("\nCRITERIO DE ACEPTACIÓN FALLIDO:")
        for err in errors:
            print(f"  ✗ {err}")
        return 1

    print("\nCRITERIO DE ACEPTACIÓN SUPERADO: CSVs generados y η > 99% en todos los casos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
