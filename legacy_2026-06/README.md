> **Superseded.** These are the data of release v1.0 (June 2026). The revised article uses the 2026-09 campaign in the repository root; see the top-level README.

# Event-Driven WiFi HIL Platform — Simulation Data

Raw CSV datasets from the 2×3×3 factorial experiment described in:

> **Event-Driven WiFi Hardware-in-the-Loop Platform for MPPT Algorithm Validation in Photovoltaic Systems Using Low-Cost Embedded Hardware**  
> J. E. Zabala Daza, J. A. Gomez Pemberty, J. Mena Palomeque  
> *Measurement* (under review), 2026.

## File naming convention

```
simulation_esp32_<algorithm>_<profile>_<rep>_<hash>.csv
simulation_esp32_<algorithm>_<profile>_<rep>_<hash>_rtt.csv
```

| Field | Values | Description |
|---|---|---|
| `algorithm` | `po`, `pso` | P&O or hybrid PSO (SCAN+TRACK) |
| `profile` | `p1`, `p2`, `p3` | Irradiance profile (step, ramp, partial shading) |
| `rep` | `rep1`–`rep5` | Repetition index |
| `hash` | 8-char hex | Unique run identifier |

Each run produces two files:
- **`*.csv`** — time-series of V_pv, I_pv, P_pv, V_ref (main simulation trace)
- **`*_rtt.csv`** — per-request WiFi HTTP round-trip time measurements

## Conditions

- **Condition C**: Physical ESP32 + real WiFi LAN (data in this repository)
- Algorithms: P&O (perturb & observe) and two-phase hybrid PSO (SCAN+TRACK)
- 240 CSV files total (~163 MB)

## License

Data released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
