# Data and code: Event-Driven WiFi HIL Co-Simulation for MPPT Validation

This repository holds the measurement data, analysis scripts and ESP32 firmware for the article
*"Event-Driven WiFi Hardware-in-the-Loop Co-Simulation for MPPT Validation: Quantifying the Effect of Network Latency and Packet Loss on a Low-Cost ESP32"* (Zabala Daza, Gómez Pemberty, Mena Palomeque).
All data were recorded on 2026-09-23/24. Every number in the article's tables can be recomputed
from the files below.

## Layout

| Path | Content |
|---|---|
| `data/factorial/runs/` | One set of files per closed-loop run: `<cond>_<algo>_<profile>_r<k>.json` (summary metrics), `_requests.csv` (one row per HTTP exchange: simulated time, ok, RTT, on-chip algorithm time `t_us`, local turnaround `t_loc_us`, free heap, inputs, returned `vref`), `_trace.csv.gz` (plant trace at 0.5 ms ticks). |
| `data/factorial/{runs.csv, summary.csv, stats.json, resources.json}` | Aggregates produced by `scripts/run_factorial.py aggregate`. |
| `data/factorial/excluded/` | Discarded runs and the reason for each (`README.md`). |
| `data/sensitivity_rtt/` | Delay-injection sweep on the physical ESP32 (profile P2, +50 to +500 ms, 3 repetitions); `summary.json`. |
| `data/rtt_probe/` | Open-loop RTT probes, identical input sequence on the Wokwi emulator and on the physical ESP32 (1000 requests per algorithm), including the firmware-vs-port reference check. |
| `data/build_size_esp32.txt` | Firmware build sizes and ELF symbol sizes of the algorithm state. |
| `data/perfil_P*.csv` | The three 10 s irradiance profiles (`scripts/generate_irradiance_profiles.py 2 2 2 2 2`). |
| `results/` | RTT statistics and Monte Carlo packet-loss results. |
| `scripts/` | `run_factorial.py` (campaign runner, firmware port, aggregation, probe), `monte_carlo_packet_loss.py`, `generate_irradiance_profiles.py`, `check_numbers.py` (checks the manuscript tables against the data; it needs the LaTeX sources of the article). |
| `figs/` | Scripts that produce the RTT and delay-sweep figures. |
| `firmware/` | ESP32 firmware (PlatformIO, Arduino core). Copy `include/secrets.h.example` to `include/secrets.h`. For the Wokwi runs the WiFi call was `WiFi.begin("Wokwi-GUEST", "", 6)`. |
| `external/` | `powersim_scripts_run_ideal_benchmark.py`, a file from the plant simulator that `run_factorial.py` imports (converter design parameters). It is not committed in the powersim repository at the reference commit. |

## Conditions

| Code | Meaning |
|---|---|
| `A` | Python port of the firmware on the host, no latency (T_s = 0.5 ms) |
| `D` | Condition A′ in the article: same port with a fixed 28 ms transport delay, no jitter, no loss |
| `B` | Unmodified firmware in the Wokwi ESP32 emulator (VS Code extension), reached at `127.0.0.1:8280` |
| `C` | Unmodified firmware on a physical ESP32 over a 2.4 GHz WiFi LAN |

Latency model: the plant runs in 0.5 ms ticks of simulated time. The clock is frozen during each
HTTP exchange, the wall-clock RTT is measured, and the previous voltage reference is held for that
RTT of simulated time before the reply is applied. A reply later than 0.5 s counts as lost.

## Reproducing

Dependencies:
- Python 3.12, numpy 2.4.4, scipy 1.17.1, pandas 3.0.3, httpx.
- PlatformIO Core 6.1.19.
- The plant simulator *powersim* at commit `63f3e71`, placed next to this folder or pointed to with `POWERSIM_DIR`, with `external/powersim_scripts_run_ideal_benchmark.py` copied to `powersim/scripts/run_ideal_benchmark.py`.

```
python scripts/run_factorial.py verify-port --host http://<esp32-ip>      # firmware == port
python scripts/run_factorial.py run --cond C --algo both --profile all --reps 5
python scripts/run_factorial.py run --cond D --algo both --profile all --reps 5 --jobs 6
python scripts/run_factorial.py aggregate
python scripts/run_factorial.py probe --host http://<esp32-ip> --label esp32
python scripts/monte_carlo_packet_loss.py
```

A 10 s profile takes about 10 minutes of wall-clock time, because the switched converter is
integrated at full resolution. Conditions B and C cannot be replayed exactly. That is why every
exchange is published.

## Previous version (v1.0, June 2026)

`legacy_2026-06/` holds the data of release v1.0, from the first Condition C campaign reported
in the version of the manuscript submitted to *Measurement*. That campaign used a different
firmware build, other irradiance profiles and no per-exchange on-chip timing. The results of the
revised article come only from the 2026-09 data above; the June files are kept for traceability.

## License

CC BY 4.0 (data). The firmware folder contains no third-party code. ArduinoJson is fetched by
PlatformIO (`lib_deps`). ESPAsyncWebServer and AsyncTCP were used as local copies in `firmware/lib/`
and are not redistributed here; before building, place the ESPAsyncWebServer and AsyncTCP libraries
(me-no-dev / ESP32Async) in `firmware/lib/`. Each library keeps its own license.
