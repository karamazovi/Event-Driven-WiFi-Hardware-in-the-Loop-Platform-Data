# Corridas excluidas

| Corrida | Motivo |
|---|---|
| C_po_P2_r1 | La placa ESP32 se desconectó ~16:15 del 2026-09-23 (evento físico, no del canal). Las pérdidas se concentran al final (4 entre t=8.0 y 9.7 s). Se repitió. |
| B_po_P1_r1, B_pso_P1_r1 | Corridas con el PC saturado (A con 3 procesos + C + Wokwi): el simulador corrió por debajo de tiempo real, RTT ~600 ms > timeout 0.5 s (7/12 perdidos en PSO). Mide la carga del anfitrión, no el canal. B se repitió al terminar A. |
| B_po_P1_r1 (2.º intento) | Con el PC ya descargado, Wokwi en VS Code siguió dando RTT ~570 ms y 6/14 perdidos, y la sesión se cae cada 15–20 min. Decisión (2026-09-23): B se reporta como verificación lógica (equivalencia de vref + RTT en lazo abierto, `data/rtt_probe/`), no en lazo cerrado. |

**Corrección (2026-09-23 18:30):** la causa real del RTT alto y las pérdidas de las corridas B excluidas fue el cliente, no Wokwi: `localhost` en Windows intenta primero `::1` y pierde ~2 s por conexión. Con `127.0.0.1` Wokwi da RTT ≈ 73 ms y 0/2000 perdidos (`data/rtt_probe/`). Las corridas B se repitieron completas con `127.0.0.1`; las excluidas no se usan.
| C_pso_P3_r2 | Termina con 8 pérdidas consecutivas: la placa se colgó (~18:30) y hubo que reiniciarla por serie. |

**Criterio de exclusión (fijo para toda la campaña):** una corrida B/C se excluye y se repite si termina con ≥3 requests perdidos consecutivos (caída del dispositivo/simulador, no del canal). Rachas internas se conservan: son el comportamiento del canal WiFi.

**Criterio revisado (2026-09-23 19:35, antes de cualquier análisis):** se excluye y repite toda corrida con **≥5 pérdidas consecutivas en cualquier punto** (≥2.5 s sin respuesta = caída del dispositivo o del AP). Las caídas se cuentan aparte como dato de fiabilidad del enlace. Intentos repetidos llevan sufijo `__tryN`.

| C_pso_P3_r2 (intento 2) | Corte de 5 pérdidas consecutivas (11/154 perdidos en total). |
| C_po_P2_r3 | Corte de 12 pérdidas consecutivas a mitad de corrida (17/41 perdidos en total). |

**Nota de validación (2026-09-24):** C_po_P2_r1 (primer intento) tiene como máximo 2 pérdidas consecutivas y no cumpliría el criterio revisado de ≥5; se excluyó antes, con el criterio anterior, por una desconexión física documentada de la placa (~16:15). Se mantiene excluida por esa causa física y se declara aquí.
