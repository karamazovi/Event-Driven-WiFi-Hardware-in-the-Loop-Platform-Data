#include "esp_timer.h"
#include <Arduino.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include <AsyncTCP.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <AsyncJson.h>
#include "secrets.h"   // WIFI_SSID y WIFI_PASSWORD — ver include/secrets.h
#define STACK_SIZE 8192
#define VOUT_MIN 5.0f
#define VOUT_MAX 23.0f
#define dD 1.0f

// PSO parameters
#define N_PARTICLES             5
#define PSO_W                   0.5f    // inertia weight
#define PSO_C1                  2.0f    // cognitive coefficient
#define PSO_C2                  2.0f    // social coefficient
#define PSO_VMAX                3.6f    // max velocity (volts) — 20% of [VOUT_MIN, VOUT_MAX] range
#define PSO_VMIN               -3.6f   // min velocity (volts)
#define PSO_MAX_ITERATIONS      10      // complete swarm cycles in SCAN before switching to TRACK
#define PSO_DELTA_V_TRACKING    0.04f   // fine P&O step in TRACK phase (0.002 duty × 20)
#define PSO_RESTART_THRESHOLD   0.15f   // power drop > 15% triggers warm-start SCAN restart

// Algorithm selection
enum AlgorithmType { ALGO_PO = 0, ALGO_PSO = 1 };
typedef enum { PSO_PHASE_SCAN, PSO_PHASE_TRACK } PsoPhase;
AlgorithmType activeAlgorithm = ALGO_PO;

struct MPPTResult {
    float vout;
    float power;
};

MPPTResult runMPPT(float vpv, float ipv);
MPPTResult runPSO(float vpv, float ipv);
void resetPOState(void);
void resetPSOState(void);
// Benchmark and profiling metrics
volatile uint32_t po_count = 0;
volatile int64_t  po_total_us = 0;
volatile int64_t  po_min_us = 999999;
volatile int64_t  po_max_us = 0;

volatile uint32_t pso_count = 0;
volatile int64_t  pso_total_us = 0;
volatile int64_t  pso_min_us = 999999;
volatile int64_t  pso_max_us = 0;

volatile uint32_t total_requests = 0;

void WiFiTask(void *parameter);
void ServerTask(void *parameter);

AsyncWebServer server(80);

TaskHandle_t wifiTaskHandle  = NULL;
TaskHandle_t serverTaskHandle = NULL;

// P&O state
float P_old = 0.0f;
float Vout  = 15.1f;
float sign  = 1.0f;

// PSO state (global — lives in BSS, not on task stack)
float    pso_pos[N_PARTICLES];
float    pso_vel[N_PARTICLES];
float    pso_pbest_pos[N_PARTICLES];
float    pso_pbest_fit[N_PARTICLES];
float    pso_gbest_pos       = 15.1f;
float    pso_gbest_fit       = -1.0f;
uint8_t  pso_particle        = 0;
bool     pso_initialized     = false;
// Two-phase PSO state
PsoPhase pso_phase           = PSO_PHASE_SCAN;
uint8_t  pso_iteration       = 0;
float    pso_p_ref           = -1.0f;
float    pso_track_direction = 1.0f;
float    pso_track_p_prev    = -1.0f;
float    pso_track_vout      = 0.0f;   // last Vout sent during TRACK phase

void WiFiTask(void *parameter) {
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        vTaskDelay(pdMS_TO_TICKS(500));
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");
    Serial.print("ESP32_IP: ");
    Serial.println(WiFi.localIP());
    Serial.println("Server started at http://" + WiFi.localIP().toString());

    xTaskNotifyGive(serverTaskHandle);
    vTaskDelete(NULL);
}

MPPTResult runMPPT(float vpv, float ipv) {
    MPPTResult result;
    float Pw = vpv * ipv;

    if (Pw <= P_old) {
        sign = sign * (-1);
    }

    Vout = Vout + dD * sign;
    Vout = constrain(Vout, VOUT_MIN, VOUT_MAX);

    P_old = Pw;

    result.vout  = Vout;
    result.power = Pw;
    return result;
}

void resetPOState(void) {
    P_old = 0.0f;
    Vout  = 15.1f;
    sign  = 1.0f;
}

// Warm-start restart: particle[0] goes to last known gbest, others uniform.
// Called when TRACK detects power collapse. Does NOT reset gbest history.
static void pso_warm_start_reset(void) {
    float span = VOUT_MAX - VOUT_MIN;
    for (int i = 0; i < N_PARTICLES; i++) {
        pso_pos[i]       = VOUT_MIN + span * (i + 0.5f) / N_PARTICLES;
        pso_vel[i]       = 0.0f;
        pso_pbest_pos[i] = pso_pos[i];
        pso_pbest_fit[i] = -1.0f;
    }
    if (pso_gbest_fit > 0.0f) {
        pso_pos[0]       = constrain(pso_gbest_pos, VOUT_MIN, VOUT_MAX);
        pso_pbest_pos[0] = pso_pos[0];
    }
    pso_particle         = 0;
    pso_iteration        = 0;
    pso_phase            = PSO_PHASE_SCAN;
    pso_p_ref            = -1.0f;
    pso_track_p_prev     = -1.0f;
    pso_track_vout       = (pso_gbest_fit > 0.0f) 
    ? constrain(pso_gbest_pos, VOUT_MIN, VOUT_MAX) 
    : pso_pos[0];
}

void resetPSOState(void) {
    float span = VOUT_MAX - VOUT_MIN;
    for (int i = 0; i < N_PARTICLES; i++) {
        pso_pos[i]       = VOUT_MIN + span * (i + 0.5f) / N_PARTICLES;
        pso_vel[i]       = 0.0f;
        pso_pbest_pos[i] = pso_pos[i];
        pso_pbest_fit[i] = -1.0f;
    }
    pso_gbest_pos        = pso_pos[0];
    pso_gbest_fit        = -1.0f;
    pso_particle         = 0;
    pso_initialized      = true;
    pso_phase            = PSO_PHASE_SCAN;
    pso_iteration        = 0;
    pso_p_ref            = -1.0f;
    pso_track_direction  = 1.0f;
    pso_track_p_prev     = -1.0f;
    pso_track_vout       = pso_pos[0];
}

// Two-phase PSO MPPT for partial shading.
//
// SCAN phase: cycles through all particles (one per call), runs full PSO
//   velocity/position update after each complete cycle. After PSO_MAX_ITERATIONS
//   cycles, transitions to TRACK.
//
// TRACK phase: fine P&O hill-climbing around gbest with PSO_DELTA_V_TRACKING
//   step. If power drops > PSO_RESTART_THRESHOLD, warm-starts back to SCAN
//   with particle[0] seeded at the last known gbest.
MPPTResult runPSO(float vpv, float ipv) {
    MPPTResult result;
    float pw     = vpv * ipv;
    result.power = pw;

    if (!pso_initialized) {
        resetPSOState();
        result.vout = pso_pos[0];
        return result;
    }

    // ── SCAN Phase ────────────────────────────────────────────────────────
    if (pso_phase == PSO_PHASE_SCAN) {
        uint8_t i = pso_particle;

        // Register fitness for particle i (measured at pso_pos[i] set last call)
        if (pw > pso_pbest_fit[i]) {
            pso_pbest_fit[i] = pw;
            pso_pbest_pos[i] = pso_pos[i];
        }
        if (pw > pso_gbest_fit) {
            pso_gbest_fit = pw;
            pso_gbest_pos = pso_pos[i];
        }

        // Advance particle index
        pso_particle = (pso_particle + 1) % N_PARTICLES;

        if (pso_particle == 0) {
            // Completed one full swarm cycle — run PSO velocity/position update
            pso_iteration++;
            for (int j = 0; j < N_PARTICLES; j++) {
                float r1 = (float)esp_random() / (float)UINT32_MAX;
                float r2 = (float)esp_random() / (float)UINT32_MAX;
                pso_vel[j] = PSO_W * pso_vel[j]
                           + PSO_C1 * r1 * (pso_pbest_pos[j] - pso_pos[j])
                           + PSO_C2 * r2 * (pso_gbest_pos    - pso_pos[j]);
                pso_vel[j] = constrain(pso_vel[j], PSO_VMIN, PSO_VMAX);
                pso_pos[j] = constrain(pso_pos[j] + pso_vel[j], VOUT_MIN, VOUT_MAX);
            }
        }

        if (pso_iteration >= PSO_MAX_ITERATIONS) {
            // Transition to TRACK: lock onto gbest and start fine P&O
            pso_phase           = PSO_PHASE_TRACK;
            pso_p_ref           = pso_gbest_fit;
            pso_track_p_prev    = pso_gbest_fit;  // match Python: init to gbest power, not last particle
            pso_track_direction = 1.0f;
            pso_track_vout      = pso_gbest_pos;  // seed TRACK from the best voltage found in SCAN
            result.vout = pso_gbest_pos;
            return result;
        }

        result.vout = pso_pos[pso_particle];
        return result;
    }

    // ── TRACK Phase ───────────────────────────────────────────────────────
    // Restart if power dropped more than threshold (new shadow or load change)
    if (pso_p_ref > 0.0f && (pso_p_ref - pw) / pso_p_ref > PSO_RESTART_THRESHOLD) {
        pso_warm_start_reset();
        result.vout = pso_pos[0];
        return result;
    }

    // Slide p_ref and gbest upward when power improves
    if (pw > pso_gbest_fit) {
        pso_gbest_fit = pw;
        pso_gbest_pos = pso_track_vout;   // the Vout reference that achieved this power
        pso_p_ref     = pw;
    }

    // P&O hill-climbing: flip direction when power decreases
    if (pso_track_p_prev >= 0.0f && (pw - pso_track_p_prev) < 0.0f) {
        pso_track_direction = -pso_track_direction;
    }
    pso_track_p_prev = pw;

    pso_track_vout = constrain(pso_track_vout + pso_track_direction * PSO_DELTA_V_TRACKING,
                               VOUT_MIN, VOUT_MAX);
    result.vout = pso_track_vout;
    return result;
}

void ServerTask(void *parameter) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    // POST /mppt — run one iteration of the active algorithm.
    // Raw body handler (instead of AsyncCallbackJsonWebHandler) so t_loc_us covers the whole
    // local turnaround once the body is in RAM: JSON parse + algorithm + JSON serialize.
    server.on("/mppt", HTTP_POST, [](AsyncWebServerRequest *request) {}, NULL,
        [](AsyncWebServerRequest *request, uint8_t *data, size_t len, size_t index, size_t total) {
            // ponytail: single-chunk bodies only (payload is ~30 bytes, far below one TCP segment)
            if (index != 0 || len != total) {
                request->send(413, "application/json", "{\"error\":\"chunked body\"}");
                return;
            }
            int64_t t_loc_start = esp_timer_get_time();
            JsonDocument docIn;
            if (deserializeJson(docIn, data, len)) {
                request->send(400, "application/json", "{\"error\":\"bad json\"}");
                return;
            }
            float vpv = docIn["vpv"].as<float>();
            float ipv = docIn["ipv"].as<float>();

            int64_t t_start = esp_timer_get_time();
            MPPTResult result;
            int64_t t_exec_us = 0;
            switch (activeAlgorithm) {
                case ALGO_PSO: {
                    result = runPSO(vpv, ipv);
                    t_exec_us = esp_timer_get_time() - t_start;
                    pso_count++;
                    pso_total_us += t_exec_us;
                    if (t_exec_us < pso_min_us) pso_min_us = t_exec_us;
                    if (t_exec_us > pso_max_us) pso_max_us = t_exec_us;
                    break;
                }
                default: {
                    result = runMPPT(vpv, ipv);
                    t_exec_us = esp_timer_get_time() - t_start;
                    po_count++;
                    po_total_us += t_exec_us;
                    if (t_exec_us < po_min_us) po_min_us = t_exec_us;
                    if (t_exec_us > po_max_us) po_max_us = t_exec_us;
                    break;
                }
            }
            total_requests++;

            JsonDocument docOut;
            docOut["pw"]   = result.power;
            docOut["vref"] = result.vout;
            docOut["t_us"] = (int32_t)t_exec_us;
            docOut["heap"] = esp_get_free_heap_size();
            String response;
            serializeJson(docOut, response);
            // t_loc is known only after serializing; append it without re-serializing the doc.
            int64_t t_loc_us = esp_timer_get_time() - t_loc_start;
            response.remove(response.length() - 1);
            response += ",\"t_loc_us\":" + String((int32_t)t_loc_us) + "}";
            request->send(200, "application/json", response);
        });

    // GET /health — liveness check
    
    // GET /stats — return profiling and resource metrics
    server.on("/stats", HTTP_GET, [](AsyncWebServerRequest *request) {
        StaticJsonDocument<512> doc;
        doc["algorithm"] = (activeAlgorithm == ALGO_PSO) ? "pso" : "po";
        doc["free_heap"] = esp_get_free_heap_size();
        doc["min_free_heap"] = esp_get_minimum_free_heap_size();
        doc["total_requests"] = total_requests;
        doc["po_count"] = po_count;
        doc["po_mean_us"] = po_count > 0 ? (float)po_total_us / (float)po_count : 0.0f;
        doc["po_min_us"] = (po_count > 0 && po_min_us < 999999) ? (float)po_min_us : 0.0f;
        doc["po_max_us"] = (po_count > 0) ? (float)po_max_us : 0.0f;
        doc["pso_count"] = pso_count;
        doc["pso_mean_us"] = pso_count > 0 ? (float)pso_total_us / (float)pso_count : 0.0f;
        doc["pso_min_us"] = (pso_count > 0 && pso_min_us < 999999) ? (float)pso_min_us : 0.0f;
        doc["pso_max_us"] = (pso_count > 0) ? (float)pso_max_us : 0.0f;
        String response;
        serializeJson(doc, response);
        request->send(200, "application/json", response);
    });

    server.on("/health", HTTP_GET, [](AsyncWebServerRequest *request) {
        request->send(200, "application/json", "{\"status\":\"ok\"}");
    });

    // GET /config — query active algorithm
    server.on("/config", HTTP_GET, [](AsyncWebServerRequest *request) {
        StaticJsonDocument<64> doc;
        doc["algorithm"] = (activeAlgorithm == ALGO_PSO) ? "pso" : "po";
        String response;
        serializeJson(doc, response);
        request->send(200, "application/json", response);
    });

    // POST /config — switch active algorithm
    AsyncCallbackJsonWebHandler* configHandler = new AsyncCallbackJsonWebHandler(
        "/config",
        [](AsyncWebServerRequest *request, JsonVariant &json) {
            const char* algo = json["algorithm"] | "";
            if (strcmp(algo, "pso") == 0) {
                activeAlgorithm = ALGO_PSO;
                request->send(200, "application/json", "{\"status\":\"ok\",\"algorithm\":\"pso\"}");
            } else if (strcmp(algo, "po") == 0) {
                activeAlgorithm = ALGO_PO;
                request->send(200, "application/json", "{\"status\":\"ok\",\"algorithm\":\"po\"}");
            } else {
                request->send(400, "application/json", "{\"error\":\"unknown algorithm\"}");
            }
        }
    );
    server.addHandler(configHandler);

    // POST /reset — reset algorithm state to restart optimization
    AsyncCallbackJsonWebHandler* resetHandler = new AsyncCallbackJsonWebHandler(
        "/reset",
        [](AsyncWebServerRequest *request, JsonVariant &json) {
            const char* algo = json["algorithm"] | "";
            if (strcmp(algo, "pso") == 0) {
                resetPSOState();
                request->send(200, "application/json", "{\"status\":\"ok\"}");
            } else if (strcmp(algo, "po") == 0) {
                resetPOState();
                request->send(200, "application/json", "{\"status\":\"ok\"}");
            } else {
                request->send(400, "application/json", "{\"error\":\"unknown algorithm\"}");
            }
        }
    );
    server.addHandler(resetHandler);

    server.begin();

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void setup() {
    Serial.begin(115200);

    xTaskCreate(WiFiTask,   "WiFiTask",   STACK_SIZE, NULL, 2, &wifiTaskHandle);
    xTaskCreate(ServerTask, "ServerTask", STACK_SIZE, NULL, 2, &serverTaskHandle);
}

void loop() {
    vTaskDelete(NULL);
}
