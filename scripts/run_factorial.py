#!/usr/bin/env python3
"""Factorial HIL campaign: {A, B, C} x {P&O, PSO} x {P1, P2, P3}, r repetitions.

Plant: powersim switched Boost+battery model (LSODA, f_sw = 100 kHz, full resolution),
imported read-only from ../powersim. Controller: the ESP32 firmware of
D:/BackUp/mppt-wokwi/ESP32/src/main.cpp, either
  A  — ported to Python below (zero latency, local),
  B  — running inside Wokwi   (http://localhost:8280),
  C  — running on the real ESP32 over the lab WiFi LAN.

Event-driven latency model (same for B and C). The MPPT wrapper is ticked every
T_TICK = 0.5 ms of simulated time. When idle it samples (v, i) and sends one request;
the simulation clock is frozen during the HTTP call (lockstep), the wall-clock RTT_k
is measured, and the old reference is then held for RTT_k (+ optional injected delay)
of simulated time before the new vref is applied. The next request goes out one tick
later, so the effective sampling period is T_s = RTT_k + T_TICK. Condition A applies
vref immediately, i.e. T_s = T_TICK = 0.5 ms.

Usage:
  python scripts/run_factorial.py run --cond C --algo both --profile all --reps 5
  python scripts/run_factorial.py run --cond C --algo po --profile P2 --reps 5 \
         --delay-ms 100 --out data/sensitivity_rtt
  python scripts/run_factorial.py verify-port --host http://192.168.31.143
  python scripts/run_factorial.py aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POWERSIM = Path(os.environ.get("POWERSIM_DIR", ROOT.parent / "powersim"))
sys.path.insert(0, str(POWERSIM))

T_TICK = 0.5e-3          # [s] MPPT wrapper tick; also Condition A sampling period
V_BAT = 24.0             # [V]
F_SW = 100e3             # [Hz]
# B uses 127.0.0.1, not "localhost": on Windows the resolver tries ::1 first (+2 s per connect)
HOSTS = {"B": "http://127.0.0.1:8280", "C": "http://192.168.31.143"}
PROFILES = {"P1": "perfil_P1_rampa", "P2": "perfil_P2_escalon", "P3": "perfil_P3_compuesto"}
# Reference events for t_settle [s] (profiles from generate_irradiance_profiles.py 2 2 2 2 2)
# Irradiance steps only (P1 is all ramps: the 5 % band is never left, t_settle n/a).
# P3 segments: ramp, step@2 s, ramp, step@6 s, ramp.
EVENTS = {"P1": [], "P2": [2.0], "P3": [2.0, 6.0]}
# Client timeout: a reply slower than this counts as lost; the plant ran the whole wait
# with the old vref and the client re-samples and resends. ponytail: fixed 0.5 s, the
# tau_G bound of the paper; sweep it if a reviewer asks about retry policy.
HTTP_TIMEOUT_S = 0.5


# ── Firmware port (main.cpp, line-for-line) ─────────────────────────────────
VOUT_MIN, VOUT_MAX = 5.0, 23.0


class FirmwarePO:
    def __init__(self, seed=None):
        self.reset()

    def reset(self):
        self.p_old, self.vout, self.sign = 0.0, 15.1, 1.0

    def step(self, vpv, ipv):
        pw = vpv * ipv
        if pw <= self.p_old:
            self.sign = -self.sign
        self.vout = min(max(self.vout + 1.0 * self.sign, VOUT_MIN), VOUT_MAX)
        self.p_old = pw
        return self.vout


class FirmwarePSO:
    N, W, C1, C2, VMAX, ITERS, DV_TRACK, RESTART = 5, 0.5, 2.0, 2.0, 3.6, 10, 0.04, 0.15

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)  # stands in for esp_random()
        self.reset()

    def _spread(self):
        span = VOUT_MAX - VOUT_MIN
        self.pos = [VOUT_MIN + span * (i + 0.5) / self.N for i in range(self.N)]
        self.vel = [0.0] * self.N
        self.pbest_pos = list(self.pos)
        self.pbest_fit = [-1.0] * self.N

    def reset(self):  # resetPSOState()
        self._spread()
        self.gbest_pos, self.gbest_fit = self.pos[0], -1.0
        self.particle, self.iteration, self.phase = 0, 0, "SCAN"
        self.p_ref, self.track_dir, self.track_p_prev = -1.0, 1.0, -1.0
        self.track_vout = self.pos[0]

    def _warm_start(self):  # pso_warm_start_reset()
        self._spread()
        if self.gbest_fit > 0.0:
            self.pos[0] = min(max(self.gbest_pos, VOUT_MIN), VOUT_MAX)
            self.pbest_pos[0] = self.pos[0]
        self.particle, self.iteration, self.phase = 0, 0, "SCAN"
        self.p_ref, self.track_p_prev = -1.0, -1.0
        self.track_vout = (min(max(self.gbest_pos, VOUT_MIN), VOUT_MAX)
                           if self.gbest_fit > 0.0 else self.pos[0])

    def step(self, vpv, ipv):
        pw = vpv * ipv
        if self.phase == "SCAN":
            i = self.particle
            if pw > self.pbest_fit[i]:
                self.pbest_fit[i], self.pbest_pos[i] = pw, self.pos[i]
            if pw > self.gbest_fit:
                self.gbest_fit, self.gbest_pos = pw, self.pos[i]
            self.particle = (self.particle + 1) % self.N
            if self.particle == 0:
                self.iteration += 1
                for j in range(self.N):
                    r1, r2 = self.rng.random(), self.rng.random()
                    v = (self.W * self.vel[j] + self.C1 * r1 * (self.pbest_pos[j] - self.pos[j])
                         + self.C2 * r2 * (self.gbest_pos - self.pos[j]))
                    self.vel[j] = min(max(v, -self.VMAX), self.VMAX)
                    self.pos[j] = min(max(self.pos[j] + self.vel[j], VOUT_MIN), VOUT_MAX)
            if self.iteration >= self.ITERS:
                self.phase = "TRACK"
                self.p_ref = self.track_p_prev = self.gbest_fit
                self.track_dir, self.track_vout = 1.0, self.gbest_pos
                return self.gbest_pos
            return self.pos[self.particle]
        if self.p_ref > 0.0 and (self.p_ref - pw) / self.p_ref > self.RESTART:
            self._warm_start()
            return self.pos[0]
        if pw > self.gbest_fit:
            self.gbest_fit, self.gbest_pos, self.p_ref = pw, self.track_vout, pw
        if self.track_p_prev >= 0.0 and pw - self.track_p_prev < 0.0:
            self.track_dir = -self.track_dir
        self.track_p_prev = pw
        self.track_vout = min(max(self.track_vout + self.track_dir * self.DV_TRACK, VOUT_MIN), VOUT_MAX)
        return self.track_vout


PORTS = {"po": FirmwarePO, "pso": FirmwarePSO}


# ── Controller backends ─────────────────────────────────────────────────────
# Condition D (A' in the paper): local port with a FIXED transport delay equal to the median
# closed-loop RTT measured on the real ESP32, no jitter, no loss. Separates "slower sampling"
# from "real channel" in the C-vs-A comparison.
FIXED_DELAY_MS = 28.0


class LocalBackend:
    def __init__(self, algo, seed, fixed_ms=0.0):
        self.ctrl = PORTS[algo](seed)
        self.fixed_ms = fixed_ms

    def reset(self, algo):
        self.ctrl.reset()

    def call(self, vpv, ipv):
        return self.ctrl.step(vpv, ipv), {"ok": 1, "rtt_ms": self.fixed_ms}


class HttpBackend:
    def __init__(self, base, timeout_s=HTTP_TIMEOUT_S):
        import httpx
        self.base = base
        self.httpx = httpx
        self.client = httpx.Client(timeout=timeout_s, headers={"Connection": "keep-alive"})

    def reset(self, algo):
        # Setup calls are not measured: generous timeout and retries (Wokwi RTT tails > 0.5 s)
        for ep in ("config", "reset"):
            for attempt in range(5):
                try:
                    r = self.client.post(f"{self.base}/{ep}", json={"algorithm": algo}, timeout=5.0)
                    r.raise_for_status()
                    break
                except self.httpx.HTTPError:
                    if attempt == 4:
                        raise
                    time.sleep(2.0)

    def call(self, vpv, ipv):
        t0 = time.perf_counter()
        try:
            r = self.client.post(f"{self.base}/mppt", json={"vpv": round(vpv, 6), "ipv": round(ipv, 6)})
            rtt = (time.perf_counter() - t0) * 1e3
            r.raise_for_status()
            d = r.json()
            return float(d["vref"]), {"ok": 1, "rtt_ms": rtt, "t_us": d.get("t_us"),
                                      "t_loc_us": d.get("t_loc_us"), "heap": d.get("heap")}
        except (self.httpx.HTTPError, KeyError, ValueError):
            # Lost/failed transaction: the plant ran for the whole wait with the old vref.
            return None, {"ok": 0, "rtt_ms": (time.perf_counter() - t0) * 1e3}


def _make_mppt(backend, extra_delay_s, log):
    from core.mppt.base_mppt import BaseMPPT

    class EventDrivenMPPT(BaseMPPT):
        """Holds vref for the measured RTT (sim time) before applying the reply."""

        def __init__(self):
            super().__init__(d_min=0.05, d_max=0.95)
            self.last_vref = None
            self.t = 0.0
            self.pending = None
            self.wait = 0
            self.decisions = []  # sim time at which a new vref was applied

        def _duty(self, duty_current):
            return duty_current if self.last_vref is None else self._clamp_duty(1.0 - self.last_vref / V_BAT)

        def update(self, v_pv, i_pv, duty_current):
            self.t += T_TICK
            if self.wait > 0:
                self.wait -= 1
                if self.wait == 0 and self.pending is not None:
                    self.last_vref, self.pending = self.pending, None
                    self.decisions.append(self.t)
                return self._duty(duty_current)
            vref, rec = backend.call(float(v_pv), float(i_pv))
            rec.update(t_sim=self.t, vpv=v_pv, ipv=i_pv, vref=vref)
            log.append(rec)
            lat = rec["rtt_ms"] / 1e3 + (extra_delay_s if rec["rtt_ms"] > 0 else 0.0)
            n = int(round(lat / T_TICK))
            if n == 0:
                if vref is not None:
                    self.last_vref = vref
                    self.decisions.append(self.t)
            else:
                self.pending, self.wait = vref, n
            return self._duty(duty_current)

        def reset(self):
            pass  # controller state is reset explicitly before each run

        @property
        def name(self):
            return "event-driven firmware MPPT"

        @property
        def parameters(self):
            return {}

    return EventDrivenMPPT()


# ── One run ─────────────────────────────────────────────────────────────────
def load_profile(p):
    d = np.loadtxt(ROOT / "data" / f"{PROFILES[p]}.csv", delimiter=",", skiprows=1)
    return [(float(t), float(g)) for t, g in d]


def metrics(t, p, pmpp, decisions, profile):
    eta = 100.0 * trap(p, t) / trap(pmpp, t)
    err = np.abs(p - pmpp) / np.maximum(pmpp, 1e-9)
    settle = []
    for ev in EVENTS[profile]:
        idx = np.flatnonzero((t >= ev) & (err < 0.05))
        settle.append((t[idx[0]] - ev) * 1e3 if idx.size else math.nan)
    t0 = decisions[-11] if len(decisions) > 11 else t[int(0.9 * len(t))]
    w = t >= t0
    sigma = 100.0 * np.std(p[w]) / np.mean(pmpp[w])
    return eta, (float(np.nanmax(settle)) if settle else math.nan), float(sigma)


def trap(y, x):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def run_one(cond, algo, profile, rep, out_dir, delay_ms=0.0, t_sim=None, host=None):
    import scripts_run_ideal as rb  # noqa: provided via sys.modules shim below
    from core.models.boost_battery_model import BoostBatteryModel
    from core.models.pv_cell import PVCell, PVModuleParams
    from core.simulation.engine import SimulationEngine

    tag = f"{cond}_{algo}_{profile}_r{rep}" + (f"_d{int(delay_ms)}" if delay_ms else "")
    module = PVModuleParams.from_json(POWERSIM / "data" / "panels" / "generic_85w.json")
    pv = PVCell(module)
    g_prof = load_profile(profile)
    t_sim = t_sim or g_prof[-1][0]
    seed = 1000 * rep + (0 if algo == "po" else 1)
    if cond in ("A", "D"):
        backend = LocalBackend(algo, seed, FIXED_DELAY_MS if cond == "D" else 0.0)
    else:
        backend = HttpBackend(host or HOSTS[cond])
    backend.reset(algo)
    log = []
    mppt = _make_mppt(backend, delay_ms / 1e3, log)

    params = {"f_sw": F_SW, "V_bat": V_BAT, "R_bat": 0.069, "t_sim": t_sim, "T_mppt_s": T_TICK,
              "G": 1000.0, "T_celsius": 25.0, "G_profile": g_prof, "full_output": True,
              "v_c_initial_V": float(module.V_oc)}
    E_pv = E_mpp = 0.0
    prev = None
    rows = []  # one point per tick (end of each engine micro-interval)
    wall0 = time.time()
    for c in SimulationEngine(solver_method="LSODA").run_generator(
            pv_model=pv, converter_model=BoostBatteryModel(), mppt=mppt,
            design_result=rb._make_design(), params=params):
        tt, pp, pm = c.time, c.p_pv, c.p_mpp
        if prev is not None:
            tt, pp, pm = (np.r_[prev[0], tt], np.r_[prev[1], pp], np.r_[prev[2], pm])
        E_pv += trap(pp, tt)
        E_mpp += trap(pm, tt)
        prev = (tt[-1], pp[-1], pm[-1])
        # tick-average power (switching ripple averaged out) for t_settle / sigma_ss
        rows.append((tt[-1], float(c.v_pv[-1]), float(c.i_pv[-1]),
                     trap(pp, tt) / max(tt[-1] - tt[0], 1e-12), float(pm[-1]), float(c.duty[-1]),
                     float(c.g_irr[-1])))
    tr = np.array(rows)
    eta_tick, t_settle, sigma = metrics(tr[:, 0], tr[:, 3], tr[:, 4], mppt.decisions, profile)
    eta = 100.0 * E_pv / E_mpp

    out = Path(out_dir) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / f"{tag}_trace.csv.gz", tr, delimiter=",", fmt="%.6g",
               header="t_s,v_pv_V,i_pv_A,p_pv_W,p_mpp_W,duty,G_Wm2", comments="")
    keys = ["t_sim", "ok", "rtt_ms", "t_us", "t_loc_us", "heap", "vpv", "ipv", "vref"]
    with open(out / f"{tag}_requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(log)

    rtt = np.array([r["rtt_ms"] for r in log if r["ok"]])
    n_fail = sum(1 for r in log if not r["ok"])
    res = {"tag": tag, "cond": cond, "algo": algo, "profile": profile, "rep": rep,
           "delay_ms": delay_ms, "t_sim_s": t_sim, "eta_pct": eta, "eta_tick_pct": eta_tick,
           "t_settle_ms": t_settle, "sigma_ss_pct": sigma, "n_req": len(log), "n_fail": n_fail,
           "rtt_mean_ms": float(rtt.mean()), "rtt_p50_ms": float(np.median(rtt)),
           "rtt_p99_ms": float(np.percentile(rtt, 99)), "timeout_s": HTTP_TIMEOUT_S, "wall_s": time.time() - wall0}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    (out / f"{tag}.json").write_text(json.dumps(res))
    return res


# powersim's scripts/ is not a package: expose run_ideal_benchmark for _make_design()
def _shim():
    import importlib.util
    spec = importlib.util.spec_from_file_location("scripts_run_ideal", POWERSIM / "scripts" / "run_ideal_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scripts_run_ideal"] = mod
    spec.loader.exec_module(mod)


_shim()


def _run_star(a):
    return run_one(*a)


# ── Aggregation and statistics ──────────────────────────────────────────────
def aggregate(out_dir):
    import pandas as pd
    from scipy import stats

    df = pd.DataFrame([json.loads(f.read_text()) for f in (Path(out_dir) / "runs").glob("*.json")])
    df.to_csv(Path(out_dir) / "runs.csv", index=False, float_format="%.6g")
    # t_settle recomputed from the stored traces so the event definition lives in one place
    for k, row in df.iterrows():
        tr = np.loadtxt(Path(out_dir) / "runs" / f"{row.tag}_trace.csv.gz", delimiter=",", skiprows=1)
        t, p, pm = tr[:, 0], tr[:, 3], tr[:, 4]
        err = np.abs(p - pm) / np.maximum(pm, 1e-9)
        st = []
        for ev in EVENTS[row.profile]:
            idx = np.flatnonzero((t >= ev) & (err < 0.05))
            st.append((t[idx[0]] - ev) * 1e3 if idx.size else math.nan)
        df.loc[k, "t_settle_ms"] = max(st) if st else math.nan
    df.to_csv(Path(out_dir) / "runs.csv", index=False, float_format="%.6g")
    df = df[df.delay_ms == 0]
    g = df.groupby(["algo", "cond", "profile"])
    summ = g.agg(n=("eta_pct", "size"), eta_mean=("eta_pct", "mean"), eta_std=("eta_pct", "std"),
                 t_settle_mean=("t_settle_ms", "mean"), t_settle_std=("t_settle_ms", "std"),
                 sigma_ss_mean=("sigma_ss_pct", "mean"), rtt_mean=("rtt_mean_ms", "mean"),
                 n_req=("n_req", "sum"), n_fail=("n_fail", "sum")).reset_index()
    summ["eta_ci95"] = [stats.t.ppf(0.975, n - 1) * s / math.sqrt(n) if n > 1 else 0.0
                        for n, s in zip(summ.n, summ.eta_std.fillna(0))]
    summ.to_csv(Path(out_dir) / "summary.csv", index=False, float_format="%.4f")

    def cliff(a, b):
        return float(np.mean([np.sign(x - y) for x in a for y in b]))

    tests = []
    for p in sorted(df.profile.unique()):
        for label, a, b in [
            ("C_vs_A_po", ("po", "C"), ("po", "A")), ("C_vs_A_pso", ("pso", "C"), ("pso", "A")),
            ("C_vs_D_po", ("po", "C"), ("po", "D")), ("C_vs_D_pso", ("pso", "C"), ("pso", "D")),
            ("D_vs_A_po", ("po", "D"), ("po", "A")), ("D_vs_A_pso", ("pso", "D"), ("pso", "A")),
            ("C_vs_B_po", ("po", "C"), ("po", "B")), ("C_vs_B_pso", ("pso", "C"), ("pso", "B")),
            ("po_vs_pso_C", ("po", "C"), ("pso", "C")), ("po_vs_pso_B", ("po", "B"), ("pso", "B"))]:
            x = df[(df.algo == a[0]) & (df.cond == a[1]) & (df.profile == p)].eta_pct.to_numpy()
            y = df[(df.algo == b[0]) & (df.cond == b[1]) & (df.profile == p)].eta_pct.to_numpy()
            if len(x) < 2 or len(y) < 1 or (len(y) == 1 and len(x) < 2):
                continue
            if np.ptp(np.r_[x, y]) == 0:
                pval = 1.0
            else:
                pval = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
            tests.append({"profile": p, "test": label, "n_x": len(x), "n_y": len(y),
                          "mean_x": float(x.mean()), "mean_y": float(y.mean()),
                          "diff_pp": float(x.mean() - y.mean()), "mannwhitney_p": pval,
                          "cliffs_delta": cliff(x, y)})
    with open(Path(out_dir) / "stats.json", "w") as f:
        json.dump(tests, f, indent=2)

    # Table 3 source: on-chip timing, heap and link statistics from the closed-loop requests
    res = {}
    for cond in ("B", "C"):
        for algo in ("po", "pso"):
            # chronological order (run completion), so the heap trend follows real time
            files = sorted((Path(out_dir) / "runs").glob(f"{cond}_{algo}_P*_r*_requests.csv"),
                           key=lambda f: f.with_name(f.name.replace("_requests.csv", ".json")).stat().st_mtime)
            if not files:
                continue
            rq = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            ok = rq[rq.ok == 1]
            heap = ok.heap.to_numpy(float)
            slope = float(np.polyfit(np.arange(heap.size), heap, 1)[0]) if heap.size > 2 else math.nan
            d = {"n_req": int(len(rq)), "n_lost": int((rq.ok == 0).sum()),
                 "loss_pct": 100.0 * float((rq.ok == 0).mean()),
                 "rtt_mean_ms": float(ok.rtt_ms.mean()), "rtt_std_ms": float(ok.rtt_ms.std()),
                 "rtt_p50_ms": float(ok.rtt_ms.median()), "rtt_p99_ms": float(ok.rtt_ms.quantile(0.99)),
                 "heap_min_B": int(heap.min()), "heap_max_B": int(heap.max()),
                 "heap_slope_B_per_req": slope}
            for col in ("t_us", "t_loc_us"):
                x = ok[col].to_numpy(float)
                d.update({f"{col}_mean": float(x.mean()), f"{col}_std": float(x.std(ddof=1)),
                          f"{col}_min": float(x.min()), f"{col}_max": float(x.max()),
                          f"{col}_p99": float(np.percentile(x, 99))})
            res[f"{cond}_{algo}"] = d
    with open(Path(out_dir) / "resources.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))
    print(summ.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))


# ── Port fidelity check against a live board ────────────────────────────────
def verify_port(host, n=60):
    """P&O is deterministic: port and firmware must return identical vref sequences.
    For PSO only the first swarm cycle (before any random draw) is deterministic."""
    be = HttpBackend(host)
    rng = np.random.default_rng(0)
    for algo, steps in (("po", n), ("pso", FirmwarePSO.N - 1)):
        be.reset(algo)
        port = PORTS[algo](0)
        for k in range(steps):
            v, i = float(rng.uniform(12, 20)), float(rng.uniform(1, 5))
            v, i = round(v, 6), round(i, 6)
            # the firmware works in float32
            v32, i32 = float(np.float32(v)), float(np.float32(i))
            got, _ = be.call(v, i)
            exp = port.step(v32, i32)
            assert abs(got - exp) < 1e-3, f"{algo} step {k}: firmware {got} != port {exp}"
        print(f"[OK] {algo}: firmware == port over {steps} steps")


def probe(host, n, out_dir, label):
    """Open-loop probe (Condition B = logic verification): fixed pseudo-random (V, I) sequence,
    every reply logged; vref checked against the port (all P&O steps, deterministic PSO steps)."""
    import pandas as pd
    be = HttpBackend(host, timeout_s=5.0)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for algo in ("po", "pso"):
        be.reset(algo)
        port = PORTS[algo](0)
        rng = np.random.default_rng(0)
        rows, mism = [], 0
        for k in range(n):
            v, i = round(float(rng.uniform(12, 20)), 6), round(float(rng.uniform(1, 5)), 6)
            got, rec = be.call(v, i)
            exp = port.step(float(np.float32(v)), float(np.float32(i)))
            checked = algo == "po" or k < FirmwarePSO.N - 1
            if got is not None and checked and abs(got - exp) > 1e-3:
                mism += 1
            rec.update(k=k, vpv=v, ipv=i, vref=got, vref_port=exp if checked else None)
            rows.append(rec)
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{label}_{algo}_probe.csv", index=False)
        ok = df[df.ok == 1]
        summary[algo] = {"n": n, "n_ok": int(len(ok)), "n_lost": int((df.ok == 0).sum()),
                         "vref_checked": int(df.vref_port.notna().sum()), "vref_mismatches": mism,
                         "rtt_mean_ms": float(ok.rtt_ms.mean()), "rtt_std_ms": float(ok.rtt_ms.std()),
                         "rtt_p50_ms": float(ok.rtt_ms.median()), "rtt_p99_ms": float(ok.rtt_ms.quantile(.99)),
                         "t_us_mean": float(ok.t_us.mean()), "t_loc_us_mean": float(ok.t_loc_us.mean()),
                         "heap_min_B": int(ok.heap.min()), "heap_max_B": int(ok.heap.max())}
        print(algo, json.dumps(summary[algo]), flush=True)
    (out / f"{label}_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cond", required=True, choices=["A", "B", "C", "D"])
    r.add_argument("--algo", default="both", choices=["po", "pso", "both"])
    r.add_argument("--profile", default="all", choices=["P1", "P2", "P3", "all"])
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--rep-start", type=int, default=1)
    r.add_argument("--delay-ms", type=float, default=0.0)
    r.add_argument("--t-sim", type=float, default=None)
    r.add_argument("--host", default=None)
    r.add_argument("--jobs", type=int, default=1, help="parallel runs (Condition A only)")
    r.add_argument("--out", default=str(ROOT / "data" / "factorial"))
    v = sub.add_parser("verify-port")
    v.add_argument("--host", default=HOSTS["C"])
    pr = sub.add_parser("probe")
    pr.add_argument("--host", default=HOSTS["B"])
    pr.add_argument("--n", type=int, default=1000)
    pr.add_argument("--label", default="wokwi")
    pr.add_argument("--out", default=str(ROOT / "data" / "rtt_probe"))
    a = sub.add_parser("aggregate")
    a.add_argument("--out", default=str(ROOT / "data" / "factorial"))
    args = ap.parse_args()

    if args.cmd == "verify-port":
        return verify_port(args.host)
    if args.cmd == "probe":
        return probe(args.host, args.n, args.out, args.label)
    if args.cmd == "aggregate":
        return aggregate(args.out)
    algos = ["po", "pso"] if args.algo == "both" else [args.algo]
    profs = list(PROFILES) if args.profile == "all" else [args.profile]
    jobs = [(args.cond, al, p, k, args.out, args.delay_ms, args.t_sim, args.host)
            for k in range(args.rep_start, args.rep_start + args.reps) for p in profs for al in algos]
    # Resume after a reboot: skip runs whose result JSON already exists
    done = {f.stem for f in (Path(args.out) / "runs").glob("*.json")}
    jobs = [j for j in jobs if f"{j[0]}_{j[1]}_{j[2]}_r{j[3]}" + (f"_d{int(j[5])}" if j[5] else "") not in done]
    print(f"[resume] {len(done)} done, {len(jobs)} to run", flush=True)
    if args.cond in ("A", "D") and args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            list(ex.map(_run_star, jobs))
    else:
        assert args.jobs == 1, "B/C share one board state: runs must be sequential"
        for j in jobs:
            run_one(*j)


if __name__ == "__main__":
    main()
