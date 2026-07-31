#define _GNU_SOURCE
#include "obackend.h"
#include "ocapacity.h"
#include "ohttp.h"
#include "ojson.h"
#include "olog.h"
#include "oproxy.h"
#include "oscale.h"
#include "osched.h"
#include "ohost.h"
#include "otext.h"
#include "ovram.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#define MAX_TOKS 4096
#define MAX_MSGS 128

typedef struct {
    osched *sched;
    const char *secret;
    oproxy_target *llm_upstream;
    oproxy_target *image_upstream;
    oproxy_target *art_upstream;
    oproxy_target *birefnet_upstream;
    oproxy_target *tts_upstream;
    oproxy_target *stt_upstream;
    oproxy_target *training_upstream;
    oproxy_target *embedding_upstream;
    oproxy_target *multimodal_upstream;
    oproxy_target *animation_upstream;
    oproxy_target *threed_upstream;
    oproxy_target *aux_upstream;
    int upstream_timeout_ms;
    int llm_permits;
    int image_permits;
    int art_permits;
    int birefnet_permits;
    int tts_permits;
    int stt_permits;
    int training_permits;
    int embedding_permits;
    int multimodal_permits;
    int animation_permits;
    int threed_permits;
    int aux_permits;
    oscale *scale;
    ocapacity *capacity;
    ovram *vram;
} app_state;

extern const char *DOCS_HTML;
extern const char *OPENAPI_JSON;

static bool env_flag(const char *name, int fallback);

static bool query_secret_matches(const ohttp_request *req, const char *secret, size_t secret_len) {
    const char *p = req->query;
    const char *end = p ? p + req->query_len : NULL;
    while (p && p < end) {
        const char *amp = memchr(p, '&', (size_t)(end - p));
        const char *field_end = amp ? amp : end;
        const char *eq = memchr(p, '=', (size_t)(field_end - p));
        if (eq && (size_t)(eq - p) == 6 && memcmp(p, "secret", 6) == 0 &&
            (size_t)(field_end - eq - 1) == secret_len &&
            memcmp(eq + 1, secret, secret_len) == 0) {
            return true;
        }
        p = amp ? amp + 1 : end;
    }
    return false;
}

static bool authorized(const app_state *app, const ohttp_request *req) {
    if (!app->secret || !app->secret[0]) return true;
    size_t len = 0;
    const char *v = ohttp_req_header(req, "secret", &len);
    size_t slen = strlen(app->secret);
    if (v && len == slen && memcmp(v, app->secret, slen) == 0) return true;
    v = ohttp_req_header(req, "Authorization", &len);
    if (v && len > 7 && strncasecmp(v, "Bearer ", 7) == 0 &&
        len - 7 == slen && memcmp(v + 7, app->secret, slen) == 0) return true;
    v = ohttp_req_header(req, "X-API-Key", &len);
    if (v && len == slen && memcmp(v, app->secret, slen) == 0) return true;
    v = ohttp_req_header(req, "X-Rapid-API-Key", &len);
    if (v && len == slen && memcmp(v, app->secret, slen) == 0) return true;
    if (query_secret_matches(req, app->secret, slen)) return true;
    return false;
}

/* Headers a reverse proxy, tunnel or load balancer adds on the way in. Their
 * presence proves the request was relayed, so the loopback peer address is the
 * relay's and says nothing about who originated the call. */
static const char *const relayed_headers[] = {
    "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "Forwarded",
    "X-Real-IP", "CF-Connecting-IP", "CF-Connecting-IPv6", "CF-Ray",
    "CF-IPCountry", "CF-Worker", "CDN-Loop", "True-Client-IP",
    "X-Original-Forwarded-For", "Via",
};
#define RELAYED_HEADER_COUNT (sizeof relayed_headers / sizeof relayed_headers[0])

static bool request_via_proxy(const ohttp_request *req) {
    for (size_t i = 0; i < RELAYED_HEADER_COUNT; i++)
        if (ohttp_req_header(req, relayed_headers[i], NULL)) return true;
    return false;
}

/* "Internal" is a property of the connection, never of a header the client
 * controls. A request qualifies only when it came from this host AND shows no
 * sign of having been relayed, so a public caller cannot buy free GPU time by
 * asserting X-Omniserve-Internal through the Cloudflare tunnel. */
static bool request_is_internal(const ohttp_request *req) {
    return ohttp_req_peer_is_loopback(req) && !request_via_proxy(req);
}

static otier request_tier(const ohttp_request *req) {
    /* Priority is a privilege too: a public caller must not be able to claim
     * the paid lane, so an untrusted X-Omniserve-Tier falls back to default. */
    if (!request_is_internal(req)) return otier_parse_public(NULL, 0);
    size_t len = 0;
    const char *v = ohttp_req_header(req, "X-Omniserve-Tier", &len);
    return otier_parse_public(v, (int)len);
}

static void respond_error(ohttp_request *req, int status, const char *msg) {
    char body[512];
    snprintf(body, sizeof body, "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}", msg);
    ohttp_respond_str(req, status, "application/json", body);
}

/* An embedded model that fell back to CPU still answers, so it cannot be
 * detected by liveness alone. Degraded means "serving, but at a fraction of
 * the paid-for GPU throughput" — worth alerting on, not worth refusing
 * traffic that the proxied workers can still serve. */
static bool gateway_degraded(ollm_placement *placement_out) {
    ollm_placement placement;
    ollm_placement_snapshot(&placement);
    if (placement_out) *placement_out = placement;
    return ollm_ready() && placement.gpu_requested && !placement.on_gpu;
}

static void handle_health(ohttp_request *req) {
    ollm_placement placement;
    bool degraded = gateway_degraded(&placement);
    char body[512];
    snprintf(body, sizeof body,
             "{\"status\":\"%s\",\"gpu\":%s,\"placement\":\"%s\",\"device\":\"%.80s\"}",
             degraded ? "degraded" : "ok",
             placement.gpu_device_present ? "true" : "false",
             placement.on_gpu ? "gpu" : "cpu", placement.device);
    ohttp_force_close(req);
    ohttp_respond_str(req, 200, "application/json", body);
}

/* Prometheus text exposition. The on-call monitor scrapes this rather than
 * parsing logs: a counter it can diff between polls is unambiguous about
 * whether new 5xx happened since the last check. */
static void handle_metrics(ohttp_request *req, app_state *app) {
    ohttp_response_stats responses;
    ohttp_response_snapshot(&responses);
    unsigned long long log_dropped = 0;
    olog_counters(NULL, &log_dropped);
    osched_stats stats;
    osched_snapshot(app->sched, &stats);
    ollm_placement placement;
    ollm_placement_snapshot(&placement);
    bool degraded = ollm_ready() && placement.gpu_requested && !placement.on_gpu;
    double vram_free = -1.0, vram_total = -1.0;
    bool vram_ok = ogpu_memory_gib(&vram_free, &vram_total);

    char body[4096];
    size_t len = (size_t)snprintf(body, sizeof body,
        "# HELP omniserve_responses_total HTTP responses by status class.\n"
        "# TYPE omniserve_responses_total counter\n"
        "omniserve_responses_total{class=\"2xx\"} %llu\n"
        "omniserve_responses_total{class=\"3xx\"} %llu\n"
        "omniserve_responses_total{class=\"4xx\"} %llu\n"
        "omniserve_responses_total{class=\"5xx\"} %llu\n"
        "# HELP omniserve_gpu_degraded Embedded model requested GPU layers but runs on CPU.\n"
        "# TYPE omniserve_gpu_degraded gauge\n"
        "omniserve_gpu_degraded %d\n"
        "# HELP omniserve_vram_free_gib Free device memory, -1 when the driver is unreachable.\n"
        "# TYPE omniserve_vram_free_gib gauge\n"
        "omniserve_vram_free_gib %.3f\n"
        "omniserve_vram_total_gib %.3f\n"
        "# HELP omniserve_access_log_dropped_total Access log lines dropped because the ring was full.\n"
        "# TYPE omniserve_access_log_dropped_total counter\n"
        "omniserve_access_log_dropped_total %llu\n"
        "# HELP omniserve_admission_slots Configured and in-use admission slots.\n"
        "# TYPE omniserve_admission_slots gauge\n"
        "omniserve_admission_slots{state=\"total\"} %d\n"
        "omniserve_admission_slots{state=\"used\"} %d\n"
        "# HELP omniserve_admission_timeouts_total Requests rejected after waiting.\n"
        "# TYPE omniserve_admission_timeouts_total counter\n"
        "omniserve_admission_timeouts_total{tier=\"paid\"} %lu\n"
        "omniserve_admission_timeouts_total{tier=\"sub\"} %lu\n"
        "omniserve_admission_timeouts_total{tier=\"free\"} %lu\n"
        "omniserve_admission_timeouts_total{tier=\"background\"} %lu\n"
        "# HELP omniserve_queue_ms_max Worst observed admission wait.\n"
        "# TYPE omniserve_queue_ms_max gauge\n"
        "omniserve_queue_ms_max{tier=\"paid\"} %.1f\n"
        "omniserve_queue_ms_max{tier=\"free\"} %.1f\n",
        responses.success, responses.redirect, responses.client_error, responses.server_error,
        degraded ? 1 : 0, vram_ok ? vram_free : -1.0, vram_ok ? vram_total : -1.0,
        log_dropped, stats.slots, stats.used_slots,
        (unsigned long)stats.timed_out[TIER_PAID], (unsigned long)stats.timed_out[TIER_SUB],
        (unsigned long)stats.timed_out[TIER_FREE], (unsigned long)stats.timed_out[TIER_BACKGROUND],
        stats.queue_ms_max[TIER_PAID], stats.queue_ms_max[TIER_FREE]);

    /* Rented-capacity spend, so an alert can fire on cost as well as on errors. */
    for (int i = 0; i < oscale_lane_count(app->scale) && len < sizeof body; i++) {
        const oscale_lane *lane = oscale_lane_at(app->scale, i);
        if (!lane) continue;
        len += (size_t)snprintf(body + len, sizeof body - len,
            "omniserve_capacity_spend_rate_usd_hr{lane=\"%.20s\"} %.3f\n"
            "omniserve_capacity_instances{lane=\"%.20s\"} %d\n",
            lane->policy.name, oscale_lane_spend_rate_usd_hr(lane),
            lane->policy.name, oscale_ready_instances(lane));
    }

    /* Headroom, not free memory, is what decides whether a co-tenant can batch,
     * so it is the number worth graphing. A denial rate that climbs while
     * headroom stays high means leases are being held, not that VRAM ran out. */
    if (app->vram && len < sizeof body) {
        ovram_stats vs;
        ovram_snapshot(app->vram, &vs);
        len += (size_t)snprintf(body + len, sizeof body - len,
            "# HELP omniserve_vram_headroom_mb Device memory the broker would grant now.\n"
            "# TYPE omniserve_vram_headroom_mb gauge\n"
            "omniserve_vram_headroom_mb %d\n"
            "omniserve_vram_leased_mb %d\n"
            "omniserve_vram_reserved_mb %d\n"
            "omniserve_vram_leases %d\n"
            "# HELP omniserve_vram_lease_total Broker lease outcomes.\n"
            "# TYPE omniserve_vram_lease_total counter\n"
            "omniserve_vram_lease_total{outcome=\"granted\"} %llu\n"
            "omniserve_vram_lease_total{outcome=\"partial\"} %llu\n"
            "omniserve_vram_lease_total{outcome=\"denied\"} %llu\n"
            "omniserve_vram_lease_total{outcome=\"released\"} %llu\n"
            "omniserve_vram_lease_total{outcome=\"expired\"} %llu\n",
            vs.headroom_mb, vs.leased_mb, vs.reserved_mb, vs.lease_count,
            vs.grants, vs.partial_grants, vs.denials, vs.releases, vs.expirations);
    }
    ohttp_respond(req, 200, "text/plain; version=0.0.4", body, len);
}

/* Recent 5xx with route and age, so an agent has somewhere to start without
 * correlating a shared journal. */
static void handle_errors(ohttp_request *req) {
    ohttp_error_event events[OHTTP_ERROR_RING];
    int count = ohttp_recent_errors(events, OHTTP_ERROR_RING);
    ohttp_response_stats responses;
    ohttp_response_snapshot(&responses);
    char body[4096];
    size_t len = (size_t)snprintf(body, sizeof body,
                                  "{\"server_error_total\":%llu,\"recent\":[",
                                  responses.server_error);
    long now = (long)time(NULL);
    for (int i = 0; i < count && len < sizeof body; i++) {
        len += (size_t)snprintf(body + len, sizeof body - len,
                                "%s{\"status\":%d,\"method\":\"%.7s\",\"path\":\"%.103s\","
                                "\"age_s\":%ld}",
                                i ? "," : "", events[i].status, events[i].method,
                                events[i].path, now - events[i].at_unix);
    }
    if (len < sizeof body) snprintf(body + len, sizeof body - len, "]}");
    ohttp_respond_str(req, 200, "application/json", body);
}

/* Separate from /health so a load balancer can drain a CPU-degraded instance
 * without the liveness probe killing it in a restart loop. */
static void handle_readyz(ohttp_request *req) {
    ollm_placement placement;
    bool degraded = gateway_degraded(&placement);
    char body[512];
    snprintf(body, sizeof body,
             "{\"ready\":%s,\"status\":\"%s\",\"placement\":\"%s\",\"device\":\"%.80s\"}",
             degraded ? "false" : "true", degraded ? "degraded" : "ok",
             placement.on_gpu ? "gpu" : "cpu", placement.device);
    ohttp_force_close(req);
    ohttp_respond_str(req, degraded ? 503 : 200, "application/json", body);
}

static void handle_status(ohttp_request *req, app_state *app) {
    osched_stats stats;
    osched_snapshot(app->sched, &stats);
    oproxy_stats llm_proxy, image_proxy, birefnet_proxy, tts_proxy, stt_proxy;
    oproxy_stats embedding_proxy, multimodal_proxy, animation_proxy, threed_proxy, aux_proxy;
    oproxy_target_snapshot(app->llm_upstream, &llm_proxy);
    oproxy_target_snapshot(app->image_upstream, &image_proxy);
    oproxy_target_snapshot(app->birefnet_upstream, &birefnet_proxy);
    oproxy_target_snapshot(app->tts_upstream, &tts_proxy);
    oproxy_target_snapshot(app->stt_upstream, &stt_proxy);
    oproxy_target_snapshot(app->embedding_upstream, &embedding_proxy);
    oproxy_target_snapshot(app->multimodal_upstream, &multimodal_proxy);
    oproxy_target_snapshot(app->animation_upstream, &animation_proxy);
    oproxy_target_snapshot(app->threed_upstream, &threed_proxy);
    oproxy_target_snapshot(app->aux_upstream, &aux_proxy);
    ollm_placement placement;
    ollm_placement_snapshot(&placement);
    double vram_free = -1.0, vram_total = -1.0;
    bool vram_ok = ogpu_memory_gib(&vram_free, &vram_total);
    /* Zero-initialised because the capacity object below is appended by seeking
     * to strlen(body); starting from a known state keeps that arithmetic sound
     * even if the first format is ever truncated. */
    char body[8192] = {0};
    snprintf(body, sizeof body,
             "{\"vram_free_gib\":%.2f,\"vram_total_gib\":%.2f,\"vram_available\":%s,"
             "\"gpu\":{\"device_present\":%s,\"requested\":%s,\"placement\":\"%s\","
             "\"device\":\"%.80s\",\"kv_type\":\"%s\",\"flash_attn\":%s,\"degraded\":%s},"
             "\"llm\":{\"ready\":%s,\"model\":\"%s\"},"
             "\"embedding\":{\"ready\":%s,\"model\":\"%s\"},"
             "\"diffusion\":{\"ready\":%s,\"model\":\"%s\"},"
             /* Key order must track the argument order below, which matches the
              * permits object: llm, image, birefnet, tts, stt, ... */
             "\"upstreams\":{\"llm\":%s,\"image\":%s,\"art\":%s,\"birefnet\":%s,\"tts\":%s,\"stt\":%s,"
             "\"embedding\":%s,\"multimodal\":%s,\"animation\":%s,\"threed\":%s,\"aux\":%s},"
             "\"permits\":{\"llm\":%d,\"image\":%d,\"art\":%d,\"birefnet\":%d,\"tts\":%d,\"stt\":%d,"
             "\"embedding\":%d,\"multimodal\":%d,\"animation\":%d,\"threed\":%d,\"aux\":%d},"
             "\"proxy_pool\":{"
             "\"llm\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"image\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"birefnet\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"tts\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"stt\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"embedding\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"multimodal\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"animation\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"threed\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu},"
             "\"aux\":{\"idle\":%zu,\"opened\":%llu,\"reused\":%llu,\"failures\":%llu}},"
             "\"admission\":{\"slots\":%d,\"used_slots\":%d,\"active\":%d,"
             "\"waiting\":{\"paid\":%d,\"sub\":%d,\"free\":%d,\"background\":%d},"
             "\"served\":{\"paid\":%ld,\"sub\":%ld,\"free\":%ld,\"background\":%ld},"
             "\"timed_out\":{\"paid\":%lu,\"sub\":%lu,\"free\":%lu,\"background\":%lu},"
             "\"queue_ms_max\":{\"paid\":%.1f,\"sub\":%.1f,\"free\":%.1f,\"background\":%.1f}}}",
             vram_free, vram_total, vram_ok ? "true" : "false",
             placement.gpu_device_present ? "true" : "false",
             placement.gpu_requested ? "true" : "false",
             placement.on_gpu ? "gpu" : "cpu", placement.device,
             placement.kv_type, placement.flash_attn ? "true" : "false",
             (ollm_ready() && placement.gpu_requested && !placement.on_gpu) ? "true" : "false",
             ollm_ready() ? "true" : "false", ollm_model_name(),
             oembed_ready() ? "true" : "false", oembed_model_name(),
             osd_ready() ? "true" : "false", osd_model_name(),
             app->llm_upstream ? "true" : "false", app->image_upstream ? "true" : "false",
             app->art_upstream ? "true" : "false",
             app->birefnet_upstream ? "true" : "false",
             app->tts_upstream ? "true" : "false", app->stt_upstream ? "true" : "false",
             app->embedding_upstream ? "true" : "false",
             app->multimodal_upstream ? "true" : "false",
             app->animation_upstream ? "true" : "false",
             app->threed_upstream ? "true" : "false", app->aux_upstream ? "true" : "false",
             app->llm_permits, app->image_permits, app->art_permits, app->birefnet_permits,
             app->tts_permits, app->stt_permits,
             app->embedding_permits, app->multimodal_permits, app->animation_permits,
             app->threed_permits, app->aux_permits,
             llm_proxy.idle_connections, llm_proxy.connections_opened,
             llm_proxy.connections_reused, llm_proxy.failures,
             image_proxy.idle_connections, image_proxy.connections_opened,
             image_proxy.connections_reused, image_proxy.failures,
             birefnet_proxy.idle_connections, birefnet_proxy.connections_opened,
             birefnet_proxy.connections_reused, birefnet_proxy.failures,
             tts_proxy.idle_connections, tts_proxy.connections_opened,
             tts_proxy.connections_reused, tts_proxy.failures,
             stt_proxy.idle_connections, stt_proxy.connections_opened,
             stt_proxy.connections_reused, stt_proxy.failures,
             embedding_proxy.idle_connections, embedding_proxy.connections_opened,
             embedding_proxy.connections_reused, embedding_proxy.failures,
             multimodal_proxy.idle_connections, multimodal_proxy.connections_opened,
             multimodal_proxy.connections_reused, multimodal_proxy.failures,
             animation_proxy.idle_connections, animation_proxy.connections_opened,
             animation_proxy.connections_reused, animation_proxy.failures,
             threed_proxy.idle_connections, threed_proxy.connections_opened,
             threed_proxy.connections_reused, threed_proxy.failures,
             aux_proxy.idle_connections, aux_proxy.connections_opened,
             aux_proxy.connections_reused, aux_proxy.failures,
             stats.slots, stats.used_slots, stats.active,
             stats.waiting[TIER_PAID], stats.waiting[TIER_SUB],
             stats.waiting[TIER_FREE], stats.waiting[TIER_BACKGROUND],
             stats.served[TIER_PAID], stats.served[TIER_SUB],
             stats.served[TIER_FREE], stats.served[TIER_BACKGROUND],
             (unsigned long)stats.timed_out[TIER_PAID], (unsigned long)stats.timed_out[TIER_SUB],
             (unsigned long)stats.timed_out[TIER_FREE], (unsigned long)stats.timed_out[TIER_BACKGROUND],
             stats.queue_ms_max[TIER_PAID], stats.queue_ms_max[TIER_SUB],
             stats.queue_ms_max[TIER_FREE], stats.queue_ms_max[TIER_BACKGROUND]);
    /* Overflow spend is appended rather than interleaved so the existing
     * status shape stays byte-stable for anything already parsing it. */
    size_t len = strlen(body);
    if (len > 1 && body[len - 1] == '}') {
        char capacity_json[2048];
        ocapacity_status_json(app->capacity, capacity_json, sizeof capacity_json);
        snprintf(body + len - 1, sizeof body - (len - 1),
                 ",\"tune\":{\"class\":\"%s\",\"n_batch\":%d,\"n_ubatch\":%d},"
                 "\"capacity\":%s}",
                 placement.tune_class, placement.n_batch, placement.n_ubatch, capacity_json);
    }
    ohttp_force_close(req);
    ohttp_respond_str(req, 200, "application/json", body);
}

static void handle_models(ohttp_request *req, const app_state *app) {
    char body[2048];
    size_t len = (size_t)snprintf(body, sizeof body, "{\"object\":\"list\",\"data\":[");
    bool comma = false;
#define ADD_MODEL(id, family, engine) do { \
    len += (size_t)snprintf(body + len, sizeof body - len, \
        "%s{\"id\":\"%.120s\",\"object\":\"model\",\"owned_by\":\"omniserve-native\",\"family\":\"%s\",\"engine\":\"%s\"}", \
        comma ? "," : "", (id), (family), (engine)); \
    comma = true; \
} while (0)
    if (ollm_ready()) ADD_MODEL(ollm_model_name(), "llm", "llama.cpp");
    if (oembed_ready()) ADD_MODEL(oembed_model_name(), "embedding", "llama.cpp");
    if (osd_ready()) ADD_MODEL(osd_model_name(), "diffusion", "stable-diffusion.cpp");
    if (app->llm_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_LLM_MODEL") ?:
                                    "upstream-llm", "llm", "proxy");
    if (app->image_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_IMAGE_MODEL") ?:
                                      "upstream-image", "diffusion", "proxy");
    if (app->art_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_ART_MODEL") ?:
                                    "Tongyi-MAI/Z-Image-Turbo", "background-art", "proxy-background");
    if (app->birefnet_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_BIREFNET_MODEL") ?:
                                         "ZhengPeng7/BiRefNet", "background-removal", "proxy-c-hot-path");
    if (app->tts_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_TTS_MODEL") ?:
                                    "upstream-tts", "tts", "proxy");
    if (app->stt_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_STT_MODEL") ?:
                                    "upstream-stt", "stt", "proxy");
    if (app->embedding_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_EMBEDDING_MODEL") ?:
                                          "upstream-embedding", "embedding", "proxy");
    if (app->multimodal_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_MULTIMODAL_MODEL") ?:
                                           "upstream-multimodal", "multimodal", "proxy");
    if (app->animation_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_ANIMATION_MODEL") ?:
                                          "nvidia-ace-animation", "animation", "proxy-background");
    if (app->threed_upstream) ADD_MODEL(getenv("OMNISERVE_NATIVE_3D_MODEL") ?:
                                       "microsoft/TRELLIS.2-4B", "image-to-3d", "proxy-background");
#undef ADD_MODEL
    snprintf(body + len, sizeof body - len, "]}");
    ohttp_respond_str(req, 200, "application/json", body);
}

static bool proxy_header_allowed(const ohttp_header *h) {
    /* X-Omniserve-* carries trust decisions (internal, tier) that this gateway
     * makes from the connection. Relaying an inbound copy would let a client
     * assert them to an upstream that has no way to tell them apart, so they
     * are dropped before the allowlist is consulted rather than merely left
     * out of it — a later addition to the list cannot reintroduce the hole. */
    static const char reserved[] = "X-Omniserve-";
    if (h->name_len >= sizeof reserved - 1 &&
        strncasecmp(h->name, reserved, sizeof reserved - 1) == 0)
        return false;
    static const char *allowed[] = {
        "Accept", "Authorization", "secret", "X-API-Key", "X-Rapid-API-Key",
        "X-Request-ID", "Traceparent", "Tracestate", "OpenAI-Organization",
    };
    for (size_t i = 0; i < sizeof allowed / sizeof allowed[0]; i++) {
        size_t n = strlen(allowed[i]);
        if (h->name_len == n && strncasecmp(h->name, allowed[i], n) == 0) return true;
    }
    return false;
}

static bool proxy_sink_raw(const void *data, size_t len, void *user) {
    return ohttp_raw_write((ohttp_request *)user, data, len);
}

/* Lane configuration reads OMNISERVE_NATIVE_SCALE_<LANE>_<FIELD>. Every field
 * has a conservative default and the lane stays off until _ENABLED is set, so
 * an operator has to opt in per modality before anything can be rented. */
static const char *scale_env(const char *lane, const char *field) {
    char name[128];
    snprintf(name, sizeof name, "OMNISERVE_NATIVE_SCALE_%s_%s", lane, field);
    for (char *p = name; *p; p++) {
        if (*p >= 'a' && *p <= 'z') *p = (char)(*p - 'a' + 'A');
    }
    const char *value = getenv(name);
    return value && value[0] ? value : NULL;
}

static int env_int(const char *name, int fallback) {
    const char *value = getenv(name);
    return value && value[0] ? atoi(value) : fallback;
}

static double scale_env_double(const char *lane, const char *field, double fallback) {
    const char *value = scale_env(lane, field);
    return value ? atof(value) : fallback;
}

static int scale_env_int(const char *lane, const char *field, int fallback) {
    const char *value = scale_env(lane, field);
    return value ? atoi(value) : fallback;
}

static unsigned parse_tier_mask(const char *list) {
    if (!list || !list[0]) return OSCALE_TIERS_PAID_ONLY;
    unsigned mask = 0;
    const char *p = list;
    while (*p) {
        while (*p == ',' || *p == ' ') p++;
        if (!*p) break;
        const char *start = p;
        while (*p && *p != ',') p++;
        int len = (int)(p - start);
        otier tier = otier_parse(start, len);
        /* otier_parse falls back to free for anything unrecognised, so only
         * accept a token that actually names its tier. Silently widening the
         * mask to the free tier would let best-effort traffic rent hardware. */
        if ((len == 4 && strncasecmp(start, "paid", 4) == 0) ||
            (len == 3 && strncasecmp(start, "sub", 3) == 0) ||
            (len == 4 && strncasecmp(start, "free", 4) == 0) ||
            (len == 10 && strncasecmp(start, "background", 10) == 0)) {
            mask |= OSCALE_TIER_BIT(tier);
        }
    }
    return mask ? mask : OSCALE_TIERS_PAID_ONLY;
}

/* Published RunPod list prices, so a lane that does not name a price still
 * costs something in the model rather than looking free. */
static double hardware_price_usd_hr(const char *hardware) {
    if (!hardware) return 0.0;
    if (strcmp(hardware, "gpu-rtx5090") == 0) return 0.89;
    if (strcmp(hardware, "gpu-rtx4090") == 0) return 0.34;
    if (strcmp(hardware, "gpu-rtx3090") == 0) return 0.22;
    if (strcmp(hardware, "gpu-l40s") == 0) return 0.86;
    if (strcmp(hardware, "gpu-a40") == 0) return 0.40;
    if (strcmp(hardware, "gpu-a100") == 0) return 1.64;
    if (strcmp(hardware, "gpu-h100") == 0) return 2.99;
    if (strcmp(hardware, "gpu-t4") == 0) return 0.19;
    return 0.0;
}

static void configure_scale_lanes(app_state *app) {
    /* Text generation is deliberately absent: the local model serves it and
     * best-effort LLM traffic must never rent hardware. */
    static const char *lanes[] = { "tts", "stt", "image", "multimodal" };
    for (size_t i = 0; i < sizeof lanes / sizeof lanes[0]; i++) {
        const char *lane = lanes[i];
        oscale_policy policy;
        oscale_policy_defaults(&policy, lane);
        policy.enabled = scale_env_int(lane, "ENABLED", 0) != 0;
        const char *template_id = scale_env(lane, "TEMPLATE");
        const char *hardware = scale_env(lane, "HARDWARE");
        snprintf(policy.cog_template, sizeof policy.cog_template, "%s",
                 template_id ? template_id : "");
        snprintf(policy.hardware, sizeof policy.hardware, "%s",
                 hardware ? hardware : "gpu-rtx4090");
        policy.tier_mask = parse_tier_mask(scale_env(lane, "TIERS"));
        policy.price_usd_hr = scale_env_double(lane, "PRICE_USD_HR",
                                               hardware_price_usd_hr(policy.hardware));
        policy.revenue_usd_per_req = scale_env_double(lane, "REVENUE_USD_PER_REQ", 0.0);
        policy.seconds_per_req = scale_env_double(lane, "SECONDS_PER_REQ", 5.0);
        policy.margin = scale_env_double(lane, "MARGIN", 1.5);
        policy.scale_up_queue_depth = scale_env_int(lane, "QUEUE_DEPTH", 2);
        policy.scale_up_queue_ms = scale_env_double(lane, "QUEUE_MS", 2000.0);
        policy.max_instances = scale_env_int(lane, "MAX_INSTANCES", 1);
        policy.max_usd_hr = scale_env_double(lane, "MAX_USD_HR", policy.price_usd_hr);
        policy.cooldown_s = scale_env_double(lane, "COOLDOWN_S", 120.0);
        policy.idle_scale_down_s = scale_env_double(lane, "IDLE_S", 180.0);
        policy.max_instance_ttl_s = scale_env_double(lane, "TTL_S", 3600.0);
        /* A lane with no template cannot be warmed, and one with no per-request
         * value can never satisfy the cost gate, so refuse to arm it. */
        if (policy.enabled && (!policy.cog_template[0] || policy.revenue_usd_per_req <= 0.0)) {
            fprintf(stderr,
                    "scale lane %s: ignoring ENABLED without both TEMPLATE and "
                    "REVENUE_USD_PER_REQ\n", lane);
            policy.enabled = false;
        }
        oscale_add_lane(app->scale, &policy);
    }
}

static void unload_embedded_models_for_background(void) {
    fprintf(stderr, "exclusive background window: unloading embedded GPU models\n");
    oembed_shutdown();
    ollm_shutdown();
}

static void reload_embedded_models_after_background(void) {
    const char *gguf = getenv("OMNISERVE_NATIVE_LLM_GGUF");
    if (gguf && gguf[0]) {
        const char *ngl = getenv("OMNISERVE_NATIVE_NGL");
        const char *ctx = getenv("OMNISERVE_NATIVE_CTX");
        const char *contexts = getenv("OMNISERVE_NATIVE_LLM_CONTEXTS");
        const char *slots_env = getenv("OMNISERVE_NATIVE_SLOTS");
        int slots = slots_env ? atoi(slots_env) : 1;
        int parallel_contexts = contexts ? atoi(contexts) : slots;
        if (slots < 1) slots = 1;
        if (parallel_contexts < 1) parallel_contexts = 1;
        if (parallel_contexts > slots) parallel_contexts = slots;
        fprintf(stderr, "exclusive background window: reloading LLM %s\n", gguf);
        if (!ollm_init(gguf, ngl ? atoi(ngl) : 999, ctx ? atoi(ctx) : 8192,
                       parallel_contexts)) {
            fprintf(stderr, "exclusive background window: LLM reload failed\n");
        }
        ollm_placement placement;
        ollm_placement_snapshot(&placement);
        if (ollm_ready() && placement.gpu_requested && !placement.on_gpu) {
            fprintf(stderr,
                    "exclusive background window: LLM reloaded on CPU (device=%s); "
                    "/readyz now reports degraded\n", placement.device);
        }
    }
    const char *embedding_gguf = getenv("OMNISERVE_NATIVE_EMBEDDING_GGUF");
    if (embedding_gguf && embedding_gguf[0]) {
        const char *embed_ngl = getenv("OMNISERVE_NATIVE_EMBEDDING_NGL");
        const char *embed_ctx = getenv("OMNISERVE_NATIVE_EMBEDDING_CTX");
        const char *embed_threads = getenv("OMNISERVE_NATIVE_EMBEDDING_THREADS");
        fprintf(stderr, "exclusive background window: reloading embedding model %s\n", embedding_gguf);
        if (!oembed_init(embedding_gguf, embed_ngl ? atoi(embed_ngl) : 0,
                         embed_ctx ? atoi(embed_ctx) : 512,
                         embed_threads ? atoi(embed_threads) : 8)) {
            fprintf(stderr, "exclusive background window: embedding reload failed\n");
        }
    }
}

static void handle_proxy_as_tier(ohttp_request *req, app_state *app, oproxy_target *upstream,
                                 int modality_permits, const char *default_content_type,
                                 const char *path_override, int tier_override) {
    if (!upstream) {
        respond_error(req, 503, "model backend is not configured");
        return;
    }
    otier tier = tier_override >= 0 ? (otier)tier_override : request_tier(req);
    int permits = tier == TIER_BACKGROUND ? osched_capacity(app->sched) : modality_permits;
    if (!osched_acquire_n(app->sched, tier, permits)) {
        respond_error(req, 503, "admission timeout; retry");
        return;
    }
    bool swap_embedded_models =
        tier == TIER_BACKGROUND &&
        ((upstream == app->threed_upstream &&
          env_flag("OMNISERVE_NATIVE_3D_SWAP_EMBEDDED_MODELS", 1)) ||
         (upstream == app->training_upstream &&
          env_flag("OMNISERVE_NATIVE_TRAINING_SWAP_EMBEDDED_MODELS", 1)));
    if (swap_embedded_models) unload_embedded_models_for_background();

    oproxy_header forwarded[16];
    int forwarded_n = 0;
    int forwarded_limit = tier_override >= 0 ? 13 : 16;
    for (int i = 0; i < req->header_count && forwarded_n < forwarded_limit; i++) {
        if (!proxy_header_allowed(&req->headers[i])) continue;
        forwarded[forwarded_n++] = (oproxy_header){
            .name = req->headers[i].name,
            .name_len = req->headers[i].name_len,
            .value = req->headers[i].value,
            .value_len = req->headers[i].value_len,
        };
    }
    if (tier_override >= 0) {
        forwarded[forwarded_n++] = (oproxy_header){
            .name = "X-Omniserve-Tier", .name_len = sizeof "X-Omniserve-Tier" - 1,
            .value = "background", .value_len = sizeof "background" - 1,
        };
        forwarded[forwarded_n++] = (oproxy_header){
            .name = "X-Animation-Publish", .name_len = sizeof "X-Animation-Publish" - 1,
            .value = "r2-searchable", .value_len = sizeof "r2-searchable" - 1,
        };
        forwarded[forwarded_n++] = (oproxy_header){
            .name = "X-3D-Publish", .name_len = sizeof "X-3D-Publish" - 1,
            .value = "r2-searchable", .value_len = sizeof "r2-searchable" - 1,
        };
    }
    size_t content_type_len = 0;
    const char *content_type = ohttp_req_header(req, "Content-Type", &content_type_len);
    if (!content_type && default_content_type) {
        content_type = default_content_type;
        content_type_len = strlen(default_content_type);
    }

    char error[256];
    oproxy_result result;
    const char *upstream_path = path_override ? path_override : req->path;
    size_t upstream_path_len = path_override ? strlen(path_override) : req->path_len;
    bool ok = oproxy_target_relay(
        upstream,
        req->method, req->method_len,
        upstream_path, upstream_path_len,
        req->query, req->query_len,
        req->body, req->body_len,
        content_type, content_type_len,
        forwarded, forwarded_n,
        app->upstream_timeout_ms,
        proxy_sink_raw, req,
        &result, error, sizeof error);
    if (swap_embedded_models) reload_embedded_models_after_background();
    osched_release_n(app->sched, tier, permits);

    if (!ok) {
        fprintf(stderr, "upstream request failed: %s\n", error);
        if (!result.response_started) respond_error(req, 502, "model backend unavailable");
        else ohttp_force_close(req);
    } else if (result.downstream_close) {
        ohttp_force_close(req);
    }
}

static void handle_proxy_as(ohttp_request *req, app_state *app, oproxy_target *upstream,
                            int modality_permits, const char *default_content_type,
                            const char *path_override) {
    handle_proxy_as_tier(req, app, upstream, modality_permits, default_content_type,
                         path_override, -1);
}

static void handle_proxy(ohttp_request *req, app_state *app, oproxy_target *upstream,
                         int modality_permits, const char *default_content_type) {
    handle_proxy_as(req, app, upstream, modality_permits, default_content_type, NULL);
}

typedef struct {
    ohttp_request *req;
} stream_ctx;

static bool stream_token(const char *piece, size_t len, void *user) {
    stream_ctx *sc = user;
    const char *pre = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"";
    size_t pren = strlen(pre);
    if (len > 256) return false;
    char payload[2048];
    memcpy(payload, pre, pren);
    size_t plen = pren;
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)piece[i];
        switch (c) {
        case '"': payload[plen++] = '\\'; payload[plen++] = '"'; break;
        case '\\': payload[plen++] = '\\'; payload[plen++] = '\\'; break;
        case '\n': payload[plen++] = '\\'; payload[plen++] = 'n'; break;
        case '\r': payload[plen++] = '\\'; payload[plen++] = 'r'; break;
        case '\t': payload[plen++] = '\\'; payload[plen++] = 't'; break;
        default:
            if (c < 0x20) {
                payload[plen++] = '\\';
                payload[plen++] = 'u';
                payload[plen++] = '0';
                payload[plen++] = '0';
                payload[plen++] = hex[c >> 4];
                payload[plen++] = hex[c & 15];
            } else {
                payload[plen++] = (char)c;
            }
        }
    }
    const char *post = "\"}}]}\n\n";
    memcpy(payload + plen, post, strlen(post));
    plen += strlen(post);
    return ohttp_stream_write(sc->req, payload, plen);
}

static int parse_stop_values(const char *js, const oj_tok *toks, int n, int root,
                             const char *key, const char **values, char **owned,
                             int *owned_n, int owned_cap) {
    int stop_tok = oj_obj_get(js, toks, n, root, key);
    if (stop_tok < 0) return 0;
    int count = 0;
    if (toks[stop_tok].type == OJ_STRING) {
        char *value = oj_strdup(js, &toks[stop_tok]);
        if (value && value[0] && *owned_n < owned_cap) {
            owned[(*owned_n)++] = value;
            values[count++] = value;
        } else {
            free(value);
        }
        return count;
    }
    if (toks[stop_tok].type != OJ_ARRAY) return 0;
    for (int i = 0; i < toks[stop_tok].size && count < 16 && *owned_n < owned_cap; i++) {
        int item = oj_arr_at(toks, n, stop_tok, i);
        if (item < 0 || toks[item].type != OJ_STRING) continue;
        char *value = oj_strdup(js, &toks[item]);
        if (!value) continue;
        if (!value[0]) {
            free(value);
            continue;
        }
        owned[(*owned_n)++] = value;
        values[count++] = value;
    }
    return count;
}

static void handle_chat(ohttp_request *req, app_state *app) {
    if (!ollm_ready()) {
        respond_error(req, 503, "no LLM model loaded; start with OMNISERVE_NATIVE_LLM_GGUF");
        return;
    }
    /* The served model is already known here; the inbound "model" field is
     * never parsed, and no JSON parse is added on the hot path just to log it. */
    olog_set_model(ollm_model_name());
    oj_tok *toks = malloc(sizeof(oj_tok) * MAX_TOKS);
    if (!toks) { respond_error(req, 500, "allocation failed"); return; }
    int n = oj_parse(req->body, req->body_len, toks, MAX_TOKS);
    if (n <= 0 || toks[0].type != OJ_OBJECT) { free(toks); respond_error(req, 400, "invalid JSON body"); return; }

    int msgs_tok = oj_obj_get(req->body, toks, n, 0, "messages");
    if (msgs_tok < 0 || toks[msgs_tok].type != OJ_ARRAY) {
        free(toks);
        respond_error(req, 400, "messages array required");
        return;
    }

    ochat_msg msgs[MAX_MSGS];
    char *owned[MAX_MSGS * 2 + 16];
    int owned_n = 0, msg_n = 0;
    for (int i = 0; i < toks[msgs_tok].size && msg_n < MAX_MSGS; i++) {
        int m = oj_arr_at(toks, n, msgs_tok, i);
        if (m < 0 || toks[m].type != OJ_OBJECT) continue;
        int role = oj_obj_get(req->body, toks, n, m, "role");
        int content = oj_obj_get(req->body, toks, n, m, "content");
        if (role < 0 || content < 0 || toks[content].type != OJ_STRING) continue;
        char *r = oj_strdup(req->body, &toks[role]);
        char *c = oj_strdup(req->body, &toks[content]);
        if (!r || !c) {
            free(r);
            free(c);
            for (int j = 0; j < owned_n; j++) free(owned[j]);
            free(toks);
            respond_error(req, 500, "allocation failed");
            return;
        }
        owned[owned_n++] = r;
        owned[owned_n++] = c;
        msgs[msg_n].role = r;
        msgs[msg_n].content = c;
        msg_n++;
    }
    if (msg_n == 0) {
        for (int i = 0; i < owned_n; i++) free(owned[i]);
        free(toks);
        respond_error(req, 400, "no usable messages");
        return;
    }

    ochat_req creq = { .messages = msgs, .message_count = msg_n, .max_tokens = 512,
                       .temperature = 0.8f, .top_p = 0.92f, .top_k = 40,
                       .repetition_penalty = 1.0f, .seed = -1, .enable_thinking = true };
    int t = oj_obj_get(req->body, toks, n, 0, "max_tokens");
    if (t >= 0) creq.max_tokens = (int)oj_number(req->body, &toks[t], 512);
    t = oj_obj_get(req->body, toks, n, 0, "temperature");
    if (t >= 0) creq.temperature = (float)oj_number(req->body, &toks[t], 0.8);
    t = oj_obj_get(req->body, toks, n, 0, "top_p");
    if (t >= 0) creq.top_p = (float)oj_number(req->body, &toks[t], 0.92);
    t = oj_obj_get(req->body, toks, n, 0, "min_p");
    if (t >= 0) creq.min_p = (float)oj_number(req->body, &toks[t], 0.0);
    t = oj_obj_get(req->body, toks, n, 0, "min_probability");
    if (t >= 0) creq.min_probability = (float)oj_number(req->body, &toks[t], 0.0);
    t = oj_obj_get(req->body, toks, n, 0, "top_k");
    if (t >= 0) creq.top_k = (int)oj_number(req->body, &toks[t], 40);
    t = oj_obj_get(req->body, toks, n, 0, "repetition_penalty");
    if (t >= 0) creq.repetition_penalty = (float)oj_number(req->body, &toks[t], 1.0);
    t = oj_obj_get(req->body, toks, n, 0, "seed");
    if (t >= 0) creq.seed = (int64_t)oj_number(req->body, &toks[t], -1);
    t = oj_obj_get(req->body, toks, n, 0, "enable_thinking");
    if (t >= 0) creq.enable_thinking = oj_bool(req->body, &toks[t], true);
    const char *stop_values[16];
    creq.stop_count = parse_stop_values(req->body, toks, n, 0, "stop", stop_values,
                                        owned, &owned_n, MAX_MSGS * 2 + 16);
    creq.stop_sequences = stop_values;
    bool stream = false;
    t = oj_obj_get(req->body, toks, n, 0, "stream");
    if (t >= 0) stream = oj_bool(req->body, &toks[t], false);

    otier tier = request_tier(req);
    int permits = tier == TIER_BACKGROUND ? osched_capacity(app->sched) : app->llm_permits;
    if (!osched_acquire_n(app->sched, tier, permits)) {
        for (int i = 0; i < owned_n; i++) free(owned[i]);
        free(toks);
        respond_error(req, 503, "admission timeout; retry");
        return;
    }

    ochat_result result;
    bool ok;
    if (stream) {
        stream_ctx sc = { .req = req };
        ohttp_stream_begin(req, 200, "text/event-stream");
        ok = ollm_chat(&creq, stream_token, &sc, &result);
        char tail[256];
        snprintf(tail, sizeof tail,
                 "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],"
                 "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}\n\ndata: [DONE]\n\n",
                 ok ? result.finish_reason : "error", ok ? result.prompt_tokens : 0,
                 ok ? result.completion_tokens : 0);
        ohttp_stream_write(req, tail, strlen(tail));
        ohttp_stream_end(req);
        if (ok) ollm_result_free(&result);
    } else {
        ok = ollm_chat(&creq, NULL, NULL, &result);
        if (!ok) {
            respond_error(req, 500, "generation failed");
        } else {
            size_t cap = strlen(result.text) * 6 + 1024;
            char *body = malloc(cap);
            if (!body) {
                ollm_result_free(&result);
                osched_release_n(app->sched, tier, permits);
                for (int i = 0; i < owned_n; i++) free(owned[i]);
                free(toks);
                respond_error(req, 500, "allocation failed");
                return;
            }
            size_t blen = (size_t)snprintf(body, cap,
                "{\"id\":\"chatcmpl-native\",\"object\":\"chat.completion\",\"created\":%ld,"
                "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"",
                time(NULL), ollm_model_name());
            if (!oj_escape_append(&body, &blen, &cap, result.text, strlen(result.text))) {
                free(body);
                ollm_result_free(&result);
                osched_release_n(app->sched, tier, permits);
                for (int i = 0; i < owned_n; i++) free(owned[i]);
                free(toks);
                respond_error(req, 500, "allocation failed");
                return;
            }
            char tail[256];
            snprintf(tail, sizeof tail,
                     "\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,"
                     "\"cached_prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
                     "\"timings\":{\"elapsed_ms\":%.1f}}",
                     result.finish_reason, result.prompt_tokens, result.cached_prompt_tokens,
                     result.completion_tokens, result.prompt_tokens + result.completion_tokens,
                     result.elapsed_ms);
            size_t tlen = strlen(tail);
            if (blen + tlen + 1 > cap) {
                char *grown = realloc(body, blen + tlen + 1);
                if (!grown) {
                    free(body);
                    ollm_result_free(&result);
                    osched_release_n(app->sched, tier, permits);
                    for (int i = 0; i < owned_n; i++) free(owned[i]);
                    free(toks);
                    respond_error(req, 500, "allocation failed");
                    return;
                }
                body = grown;
            }
            memcpy(body + blen, tail, tlen + 1);
            blen += tlen;
            ohttp_respond(req, 200, "application/json", body, blen);
            free(body);
            ollm_result_free(&result);
        }
    }
    osched_release_n(app->sched, tier, permits);
    for (int i = 0; i < owned_n; i++) free(owned[i]);
    free(toks);
}

static bool append_literal(char **buf, size_t *len, size_t *cap, const char *text) {
    size_t n = strlen(text);
    if (*len + n + 1 > *cap) {
        size_t next = *cap ? *cap : 1024;
        while (next < *len + n + 1) next *= 2;
        char *grown = realloc(*buf, next);
        if (!grown) return false;
        *buf = grown;
        *cap = next;
    }
    memcpy(*buf + *len, text, n);
    *len += n;
    (*buf)[*len] = 0;
    return true;
}

static void parse_sampling(const char *js, const oj_tok *toks, int n, int root,
                           ochat_req *out, bool legacy) {
    int t = oj_obj_get(js, toks, n, root, legacy ? "max_length" : "max_tokens");
    if (t >= 0) out->max_tokens = (int)oj_number(js, &toks[t], out->max_tokens);
    t = oj_obj_get(js, toks, n, root, "temperature");
    if (t >= 0) out->temperature = (float)oj_number(js, &toks[t], out->temperature);
    t = oj_obj_get(js, toks, n, root, "top_p");
    if (t >= 0) out->top_p = (float)oj_number(js, &toks[t], out->top_p);
    t = oj_obj_get(js, toks, n, root, "min_p");
    if (t >= 0) out->min_p = (float)oj_number(js, &toks[t], out->min_p);
    t = oj_obj_get(js, toks, n, root, "min_probability");
    if (t >= 0) out->min_probability = (float)oj_number(js, &toks[t], out->min_probability);
    t = oj_obj_get(js, toks, n, root, "top_k");
    if (t >= 0) out->top_k = (int)oj_number(js, &toks[t], out->top_k);
    t = oj_obj_get(js, toks, n, root, "repetition_penalty");
    if (t >= 0) out->repetition_penalty = (float)oj_number(js, &toks[t], out->repetition_penalty);
    t = oj_obj_get(js, toks, n, root, "seed");
    if (t >= 0) out->seed = (int64_t)oj_number(js, &toks[t], out->seed);
}

static void handle_completion(ohttp_request *req, app_state *app, bool legacy, bool autocomplete) {
    if (!ollm_ready()) {
        respond_error(req, 503, "no LLM model loaded or configured upstream");
        return;
    }
    olog_set_model(ollm_model_name());
    oj_tok *toks = malloc(sizeof(oj_tok) * MAX_TOKS);
    if (!toks) { respond_error(req, 500, "allocation failed"); return; }
    int n = oj_parse(req->body, req->body_len, toks, MAX_TOKS);
    if (n <= 0 || toks[0].type != OJ_OBJECT) {
        free(toks);
        respond_error(req, 400, "invalid JSON body");
        return;
    }
    int prompt_tok = oj_obj_get(req->body, toks, n, 0, legacy ? "text" : "prompt");
    if (prompt_tok < 0 || toks[prompt_tok].type != OJ_STRING) {
        free(toks);
        respond_error(req, 400, legacy ? "text is required" : "prompt is required");
        return;
    }
    char *prompt = oj_strdup(req->body, &toks[prompt_tok]);
    if (!prompt) { free(toks); respond_error(req, 500, "allocation failed"); return; }

    char *system_owned = NULL;
    int system_tok = oj_obj_get(req->body, toks, n, 0, "system_message");
    if (system_tok < 0) system_tok = oj_obj_get(req->body, toks, n, 0, "system_prompt");
    if (system_tok >= 0 && toks[system_tok].type == OJ_STRING &&
        toks[system_tok].end > toks[system_tok].start) {
        system_owned = oj_strdup(req->body, &toks[system_tok]);
    }
    ochat_msg messages[2];
    int message_count = 0;
    if (system_owned) messages[message_count++] = (ochat_msg){ .role = "system", .content = system_owned };
    messages[message_count++] = (ochat_msg){ .role = "user", .content = prompt };
    ochat_req creq = {
        .messages = messages, .message_count = message_count,
        .raw_prompt = system_owned ? NULL : prompt, .completion_prefix = prompt,
        .max_tokens = 100,
        .temperature = 0.7f, .top_p = 0.9f, .top_k = 40,
        .repetition_penalty = 1.0f, .seed = -1, .enable_thinking = false,
    };
    if (autocomplete) {
        creq.max_tokens = 32;
        creq.min_probability = 0.4f;
    }
    parse_sampling(req->body, toks, n, 0, &creq, legacy);
    int thinking_tok = oj_obj_get(req->body, toks, n, 0, "enable_thinking");
    if (thinking_tok >= 0) {
        creq.enable_thinking = oj_bool(req->body, &toks[thinking_tok], creq.enable_thinking);
    }
    int sentences_tok = oj_obj_get(req->body, toks, n, 0, "max_sentences");
    if (sentences_tok >= 0) {
        creq.max_sentences = (int)oj_number(req->body, &toks[sentences_tok], 0);
        if (creq.max_sentences < 0) creq.max_sentences = 0;
    }
    const char *stop_values[16];
    char *stop_owned[16];
    int stop_owned_n = 0;
    creq.stop_count = parse_stop_values(
        req->body, toks, n, 0, legacy ? "stop_sequences" : "stop",
        stop_values, stop_owned, &stop_owned_n, 16);
    creq.stop_sequences = stop_values;
    int count = 1;
    int count_tok = oj_obj_get(req->body, toks, n, 0, legacy ? "number_of_results" : "n");
    if (count_tok >= 0) count = (int)oj_number(req->body, &toks[count_tok], 1);
    if (count < 1) count = 1;
    if (count > 8) count = 8;
    bool echo = false;
    int echo_tok = oj_obj_get(req->body, toks, n, 0, "echo");
    if (echo_tok >= 0) echo = oj_bool(req->body, &toks[echo_tok], false);
    free(toks);

    otier tier = request_tier(req);
    int permits = tier == TIER_BACKGROUND ? osched_capacity(app->sched) : app->llm_permits;
    if (!osched_acquire_n(app->sched, tier, permits)) {
        free(system_owned);
        free(prompt);
        for (int i = 0; i < stop_owned_n; i++) free(stop_owned[i]);
        respond_error(req, 503, "admission timeout; retry");
        return;
    }

    char *body = NULL;
    size_t body_len = 0, body_cap = 0;
    bool ok = append_literal(&body, &body_len, &body_cap,
                             legacy ? "[" : "{\"id\":\"cmpl-native\",\"object\":\"text_completion\",\"choices\":[");
    int total_completion_tokens = 0;
    int prompt_tokens = 0;
    for (int i = 0; ok && i < count; i++) {
        ochat_result result;
        if (creq.seed >= 0) creq.seed += i > 0 ? 1 : 0;
        if (!ollm_chat(&creq, NULL, NULL, &result)) {
            ok = false;
            break;
        }
        if (i) ok = append_literal(&body, &body_len, &body_cap, ",");
        if (ok) {
            if (legacy) {
                ok = append_literal(&body, &body_len, &body_cap, "{\"generated_text\":\"");
            } else {
                char choice_head[64];
                snprintf(choice_head, sizeof choice_head, "{\"index\":%d,\"text\":\"", i);
                ok = append_literal(&body, &body_len, &body_cap, choice_head);
            }
        }
        if (ok && (legacy || echo)) {
            ok = oj_escape_append(&body, &body_len, &body_cap, prompt, strlen(prompt));
        }
        if (ok) {
            ok = oj_escape_append(&body, &body_len, &body_cap, result.text, strlen(result.text));
        }
        if (ok) {
            char tail[192];
            if (legacy) {
                const char *reason = result.finish_reason;
                if (strcmp(reason, "length") == 0) reason = "max_length";
                snprintf(tail, sizeof tail,
                         "\",\"stop_reason\":\"%s\",\"thinking_content\":null}",
                         reason);
            } else {
                snprintf(tail, sizeof tail,
                         "\",\"finish_reason\":\"%s\"}", result.finish_reason);
            }
            ok = append_literal(&body, &body_len, &body_cap, tail);
        }
        prompt_tokens = result.prompt_tokens;
        total_completion_tokens += result.completion_tokens;
        ollm_result_free(&result);
    }
    if (ok) {
        if (legacy) {
            ok = append_literal(&body, &body_len, &body_cap, "]");
        } else {
            char tail[256];
            snprintf(tail, sizeof tail,
                     "],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%d,"
                     "\"completion_tokens\":%d,\"total_tokens\":%d}}",
                     ollm_model_name(), prompt_tokens, total_completion_tokens,
                     prompt_tokens + total_completion_tokens);
            ok = append_literal(&body, &body_len, &body_cap, tail);
        }
    }
    osched_release_n(app->sched, tier, permits);
    free(system_owned);
    free(prompt);
    for (int i = 0; i < stop_owned_n; i++) free(stop_owned[i]);
    if (!ok) {
        free(body);
        respond_error(req, 500, "generation failed");
        return;
    }
    ohttp_respond(req, 200, "application/json", body, body_len);
    free(body);
}

static void handle_images(ohttp_request *req, app_state *app) {
    if (!osd_ready()) {
        respond_error(req, 503, "no diffusion model loaded; start with OMNISERVE_NATIVE_SD_MODEL");
        return;
    }
    oj_tok *toks = malloc(sizeof(oj_tok) * MAX_TOKS);
    if (!toks) { respond_error(req, 500, "allocation failed"); return; }
    int n = oj_parse(req->body, req->body_len, toks, MAX_TOKS);
    if (n <= 0 || toks[0].type != OJ_OBJECT) { free(toks); respond_error(req, 400, "invalid JSON body"); return; }
    int p = oj_obj_get(req->body, toks, n, 0, "prompt");
    if (p < 0 || toks[p].type != OJ_STRING) { free(toks); respond_error(req, 400, "prompt required"); return; }
    char *prompt = oj_strdup(req->body, &toks[p]);
    if (!prompt) { free(toks); respond_error(req, 500, "allocation failed"); return; }

    oimg_req ireq = { .prompt = prompt, .width = 768, .height = 768, .steps = 4, .seed = -1 };
    int t = oj_obj_get(req->body, toks, n, 0, "width");
    if (t >= 0) ireq.width = (int)oj_number(req->body, &toks[t], 768);
    t = oj_obj_get(req->body, toks, n, 0, "height");
    if (t >= 0) ireq.height = (int)oj_number(req->body, &toks[t], 768);
    t = oj_obj_get(req->body, toks, n, 0, "steps");
    if (t >= 0) ireq.steps = (int)oj_number(req->body, &toks[t], 4);

    free(toks);
    otier tier = request_tier(req);
    int permits = tier == TIER_BACKGROUND ? osched_capacity(app->sched) : app->image_permits;
    if (!osched_acquire_n(app->sched, tier, permits)) {
        free(prompt);
        respond_error(req, 503, "admission timeout; retry");
        return;
    }
    oimg_result result;
    bool ok = osd_generate(&ireq, &result);
    osched_release_n(app->sched, tier, permits);
    free(prompt);
    if (!ok) { respond_error(req, 500, "image generation failed"); return; }
    ohttp_respond(req, 200, "image/png", (const char *)result.png, result.png_len);
    osd_result_free(&result);
}

static bool path_starts_with(const ohttp_request *req, const char *prefix) {
    size_t n = strlen(prefix);
    return req->path_len >= n && memcmp(req->path, prefix, n) == 0;
}

static bool path_ends_with(const ohttp_request *req, const char *suffix) {
    size_t n = strlen(suffix);
    return req->path_len >= n && memcmp(req->path + req->path_len - n, suffix, n) == 0;
}

static const char *configured_path(const char *name, const char *fallback) {
    const char *path = getenv(name);
    return path && path[0] == '/' ? path : fallback;
}

static bool env_flag(const char *name, int fallback) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback != 0;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' || value[0] == 'y' || value[0] == 'Y';
}

static bool request_forces_local_model(const ohttp_request *req) {
    if (!request_is_internal(req)) return false;
    size_t value_len = 0;
    const char *value = ohttp_req_header(req, "X-Omniserve-Internal", &value_len);
    return value && value_len == 5 && strncasecmp(value, "local", 5) == 0;
}

static void handle_embedding(ohttp_request *req, bool openai_shape) {
    olog_set_model(oembed_model_name());
    oj_tok *toks = malloc(sizeof(oj_tok) * MAX_TOKS);
    if (!toks) { respond_error(req, 500, "allocation failed"); return; }
    int n = oj_parse(req->body, req->body_len, toks, MAX_TOKS);
    if (n <= 0 || toks[0].type != OJ_OBJECT) {
        free(toks);
        respond_error(req, 400, "invalid JSON body");
        return;
    }
    int input_tok = oj_obj_get(req->body, toks, n, 0, openai_shape ? "input" : "text");
    if (input_tok < 0 ||
        (toks[input_tok].type != OJ_STRING &&
         (!openai_shape || toks[input_tok].type != OJ_ARRAY))) {
        free(toks);
        respond_error(req, 400, openai_shape ? "string or string array input required" : "text required");
        return;
    }
    /* Match text-generator.io's FeatureExtractParams default while OpenAI's
     * optional dimensions field keeps the model's full native width. */
    int dimensions = openai_shape ? 0 : 256;
    int dim_tok = oj_obj_get(req->body, toks, n, 0,
                             openai_shape ? "dimensions" : "num_features");
    if (dim_tok >= 0) dimensions = (int)oj_number(req->body, &toks[dim_tok], 0);
    if (dimensions < 0 || dimensions > 4096) {
        free(toks);
        respond_error(req, 400, "invalid dimensions");
        return;
    }

    int input_count = toks[input_tok].type == OJ_ARRAY ? toks[input_tok].size : 1;
    if (input_count < 1 || input_count > 256) {
        free(toks);
        respond_error(req, 400, "input array must contain between 1 and 256 strings");
        return;
    }
    char **inputs = calloc((size_t)input_count, sizeof *inputs);
    if (!inputs) {
        free(toks);
        respond_error(req, 500, "allocation failed");
        return;
    }
    bool inputs_ok = true;
    for (int i = 0; i < input_count; i++) {
        int item = toks[input_tok].type == OJ_ARRAY ? oj_arr_at(toks, n, input_tok, i) : input_tok;
        if (item < 0 || toks[item].type != OJ_STRING || !(inputs[i] = oj_strdup(req->body, &toks[item]))) {
            inputs_ok = false;
            break;
        }
    }
    free(toks);
    if (!inputs_ok) {
        for (int i = 0; i < input_count; i++) free(inputs[i]);
        free(inputs);
        respond_error(req, 400, "input array must contain only strings");
        return;
    }

    oembed_result *results = calloc((size_t)input_count, sizeof *results);
    bool ok = results != NULL;
    int completed = 0;
    int total_tokens = 0;
    double total_elapsed_ms = 0.0;
    size_t cap = 1024;
    for (int i = 0; ok && i < input_count; i++) {
        ok = oembed_text(inputs[i], strlen(inputs[i]), dimensions, &results[i]);
        if (!ok) break;
        completed++;
        total_tokens += results[i].prompt_tokens;
        total_elapsed_ms += results[i].elapsed_ms;
        cap += (size_t)results[i].dimensions * 32 + 128;
    }
    for (int i = 0; i < input_count; i++) free(inputs[i]);
    free(inputs);
    if (!ok) {
        for (int i = 0; i < completed; i++) oembed_result_free(&results[i]);
        free(results);
        respond_error(req, 500, "embedding failed");
        return;
    }

    char *body = malloc(cap);
    if (!body) {
        for (int i = 0; i < input_count; i++) oembed_result_free(&results[i]);
        free(results);
        respond_error(req, 500, "allocation failed");
        return;
    }
    size_t len = 0;
    if (openai_shape) {
        len += (size_t)snprintf(body + len, cap - len,
            "{\"object\":\"list\",\"data\":[");
    } else {
        body[len++] = '[';
    }
    for (int input_index = 0; input_index < input_count; input_index++) {
        oembed_result *result = &results[input_index];
        if (openai_shape) {
            len += (size_t)snprintf(body + len, cap - len,
                "%s{\"object\":\"embedding\",\"index\":%d,\"embedding\":[",
                input_index ? "," : "", input_index);
        }
        for (int i = 0; i < result->dimensions; i++) {
            double value = isfinite(result->values[i]) ? result->values[i] : 0.0;
            len += (size_t)snprintf(body + len, cap - len, "%s%.9g",
                                   i ? "," : "", value);
        }
        if (openai_shape) body[len++] = ']';
        if (openai_shape) body[len++] = '}';
    }
    if (openai_shape) {
        len += (size_t)snprintf(body + len, cap - len,
            "],\"model\":\"%.120s\",\"usage\":{\"prompt_tokens\":%d,"
            "\"total_tokens\":%d},\"timings\":{\"elapsed_ms\":%.3f}}",
            oembed_model_name(), total_tokens, total_tokens, total_elapsed_ms);
    } else {
        body[len++] = ']';
    }
    for (int i = 0; i < input_count; i++) oembed_result_free(&results[i]);
    free(results);
    ohttp_respond(req, 200, "application/json", body, len);
    free(body);
}

static void handle_summarization(ohttp_request *req, app_state *app) {
    if (!ollm_ready()) {
        if (app->aux_upstream) {
            handle_proxy(req, app, app->aux_upstream, app->aux_permits, "application/json");
        } else {
            respond_error(req, 503, "summarization requires an embedded LLM or auxiliary upstream");
        }
        return;
    }
    oj_tok *toks = malloc(sizeof(oj_tok) * MAX_TOKS);
    if (!toks) { respond_error(req, 500, "allocation failed"); return; }
    int n = oj_parse(req->body, req->body_len, toks, MAX_TOKS);
    if (n <= 0 || toks[0].type != OJ_OBJECT) {
        free(toks);
        respond_error(req, 400, "invalid JSON body");
        return;
    }
    int text_tok = oj_obj_get(req->body, toks, n, 0, "text");
    if (text_tok < 0 || toks[text_tok].type != OJ_STRING) {
        free(toks);
        respond_error(req, 400, "text required");
        return;
    }
    int max_tokens = 256;
    int max_tok = oj_obj_get(req->body, toks, n, 0, "max_length");
    if (max_tok >= 0) max_tokens = (int)oj_number(req->body, &toks[max_tok], 256);
    if (max_tokens < 1) max_tokens = 256;
    if (max_tokens > 2048) max_tokens = 2048;
    char *text = oj_strdup(req->body, &toks[text_tok]);
    free(toks);
    if (!text) { respond_error(req, 500, "allocation failed"); return; }

    static const char instruction[] =
        "Summarize the supplied text accurately and concisely. Return only the summary.";
    ochat_msg messages[2] = {
        { .role = "system", .content = instruction },
        { .role = "user", .content = text },
    };
    ochat_req chat = {
        .messages = messages, .message_count = 2, .max_tokens = max_tokens,
        .temperature = 0.2f, .top_p = 0.9f, .top_k = 40,
        .repetition_penalty = 1.05f, .seed = -1, .enable_thinking = false,
    };
    otier tier = request_tier(req);
    if (!osched_acquire_n(app->sched, tier, app->llm_permits)) {
        free(text);
        respond_error(req, 503, "admission timeout; retry");
        return;
    }
    ochat_result result;
    bool ok = ollm_chat(&chat, NULL, NULL, &result);
    osched_release_n(app->sched, tier, app->llm_permits);
    free(text);
    if (!ok) { respond_error(req, 500, "summarization failed"); return; }

    size_t cap = 256, len = 0;
    char *body = malloc(cap);
    static const char prefix[] = "{\"generated_text\":\"";
    if (body) {
        memcpy(body, prefix, sizeof prefix - 1);
        len = sizeof prefix - 1;
    }
    if (!body || !oj_escape_append(&body, &len, &cap, result.text, strlen(result.text))) {
        free(body);
        ollm_result_free(&result);
        respond_error(req, 500, "allocation failed");
        return;
    }
    if (len + 2 > cap) {
        char *grown = realloc(body, len + 2);
        if (!grown) {
            free(body);
            ollm_result_free(&result);
            respond_error(req, 500, "allocation failed");
            return;
        }
        body = grown;
    }
    body[len++] = '"';
    body[len++] = '}';
    ohttp_respond(req, 200, "application/json", body, len);
    free(body);
    ollm_result_free(&result);
}

static void handle_host_memory(ohttp_request *req) {
    char body[512];
    ohost_status_json(body, sizeof body);
    ohttp_respond_str(req, 200, "application/json", body);
}

static void handle_vram_status(ohttp_request *req, app_state *app) {
    if (!app->vram) { respond_error(req, 503, "vram broker is not enabled"); return; }
    char body[512];
    ovram_status_json(app->vram, body, sizeof body);
    ohttp_respond_str(req, 200, "application/json", body);
}

static void handle_vram_lease(ohttp_request *req, app_state *app) {
    if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
    if (!app->vram) { respond_error(req, 503, "vram broker is not enabled"); return; }
    /* Leasing hands one tenant headroom at every other tenant's expense, so it
     * stays on the same trust boundary as the paid tier: loopback and not
     * forwarded. A public caller could otherwise starve the local device by
     * leasing it all and never releasing. */
    if (!request_is_internal(req)) { respond_error(req, 403, "loopback callers only"); return; }

    oj_tok *toks = malloc(sizeof(oj_tok) * 64);
    if (!toks) { respond_error(req, 500, "out of memory"); return; }
    int n = oj_parse(req->body, req->body_len, toks, 64);
    if (n <= 0 || toks[0].type != OJ_OBJECT) {
        free(toks);
        respond_error(req, 400, "body must be a JSON object");
        return;
    }

    char owner[32] = "anon";
    int owner_tok = oj_obj_get(req->body, toks, n, 0, "owner");
    if (owner_tok >= 0) oj_unescape(req->body, &toks[owner_tok], owner, sizeof owner);

    int mb_tok = oj_obj_get(req->body, toks, n, 0, "mb");
    int mb = mb_tok >= 0 ? (int)oj_number(req->body, &toks[mb_tok], 0) : 0;
    int min_tok = oj_obj_get(req->body, toks, n, 0, "min_mb");
    int min_mb = min_tok >= 0 ? (int)oj_number(req->body, &toks[min_tok], 0) : 0;
    int ttl_tok = oj_obj_get(req->body, toks, n, 0, "ttl_s");
    double ttl_s = ttl_tok >= 0 ? oj_number(req->body, &toks[ttl_tok], 0) : 0;

    /* An internal caller may name its own tier here; request_tier already
     * refuses to honour the header for anyone else, and this endpoint is
     * internal-only, so the two agree. */
    otier tier = request_tier(req);
    int tier_tok = oj_obj_get(req->body, toks, n, 0, "tier");
    if (tier_tok >= 0) {
        char name[16] = {0};
        oj_unescape(req->body, &toks[tier_tok], name, sizeof name);
        tier = otier_parse(name, (int)strlen(name));
    }
    free(toks);

    if (mb <= 0) { respond_error(req, 400, "mb must be positive"); return; }

    char id[40] = {0};
    int granted = ovram_lease(app->vram, owner, mb, min_mb, tier, ttl_s, id, sizeof id);
    char body[320];
    if (granted > 0) {
        /* Reported so the holder knows when the broker will take the headroom
         * back; without it a caller cannot tell a live lease from a stale one. */
        double effective_ttl = ttl_s > 0.0 ? ttl_s : ovram_default_ttl_s(app->vram);
        snprintf(body, sizeof body,
                 "{\"granted\":true,\"lease_id\":\"%s\",\"mb\":%d,\"tier\":\"%s\","
                 "\"expires_in_s\":%d}",
                 id, granted, otier_name(tier), (int)effective_ttl);
    } else {
        /* A denial is a normal answer, not a fault: the caller's fallback path
         * is exactly what "no headroom" should trigger, and a 5xx here would
         * make a healthy broker look broken to every monitor watching it. */
        snprintf(body, sizeof body,
                 "{\"granted\":false,\"mb\":0,\"reason\":\"no_headroom\",\"tier\":\"%s\"}",
                 otier_name(tier));
    }
    ohttp_respond_str(req, 200, "application/json", body);
}

static void handle_vram_release(ohttp_request *req, app_state *app) {
    if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
    if (!app->vram) { respond_error(req, 503, "vram broker is not enabled"); return; }
    if (!request_is_internal(req)) { respond_error(req, 403, "loopback callers only"); return; }

    oj_tok *toks = malloc(sizeof(oj_tok) * 32);
    if (!toks) { respond_error(req, 500, "out of memory"); return; }
    int n = oj_parse(req->body, req->body_len, toks, 32);
    char id[40] = {0};
    if (n > 0 && toks[0].type == OJ_OBJECT) {
        int id_tok = oj_obj_get(req->body, toks, n, 0, "lease_id");
        if (id_tok >= 0) oj_unescape(req->body, &toks[id_tok], id, sizeof id);
    }
    free(toks);

    /* Releasing an id the broker has already expired is success from the
     * caller's side: the headroom is back either way, and reporting a failure
     * would push callers into retry loops that cannot achieve anything. */
    ovram_release(app->vram, id);
    ohttp_respond_str(req, 200, "application/json", "{\"released\":true}");
}

static void route(ohttp_request *req, void *user) {
    app_state *app = user;
    if (ohttp_method_is(req, "OPTIONS")) {
        ohttp_respond_str(req, 204, "text/plain", "");
        return;
    }
    if (ohttp_path_is(req, "/health") || ohttp_path_is(req, "/healthz") ||
        ohttp_path_is(req, "/liveness_check")) { handle_health(req); return; }
    if (ohttp_path_is(req, "/readyz") || ohttp_path_is(req, "/readiness_check")) {
        handle_readyz(req); return;
    }
    if (ohttp_path_is(req, "/metrics")) { handle_metrics(req, app); return; }
    if (ohttp_path_is(req, "/errors")) { handle_errors(req); return; }
    if (ohttp_path_is(req, "/status") || ohttp_path_is(req, "/backend_status")) { handle_status(req, app); return; }
    if (ohttp_path_is(req, "/v1/gpu/vram")) { handle_vram_status(req, app); return; }
    if (ohttp_path_is(req, "/v1/host/memory")) { handle_host_memory(req); return; }
    if (ohttp_path_is(req, "/v1/gpu/lease")) { handle_vram_lease(req, app); return; }
    if (ohttp_path_is(req, "/v1/gpu/release")) { handle_vram_release(req, app); return; }
    if (ohttp_path_is(req, "/v1/models")) {
        if (!ohttp_method_is(req, "GET")) { respond_error(req, 405, "GET required"); return; }
        if (app->aux_upstream && env_flag("OMNISERVE_NATIVE_MODELS_COMPAT_PROXY", 0))
            handle_proxy(req, app, app->aux_upstream, app->aux_permits, NULL);
        else
            handle_models(req, app);
        return;
    }
    if (ohttp_path_is(req, "/docs") || ohttp_path_is(req, "/")) {
        ohttp_respond_str(req, 200, "text/html; charset=utf-8", DOCS_HTML);
        return;
    }
    if (ohttp_path_is(req, "/openapi.json")) {
        ohttp_respond_str(req, 200, "application/json", OPENAPI_JSON);
        return;
    }
    if (ohttp_path_is(req, "/v1/chat/completions")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        if (app->aux_upstream && env_flag("OMNISERVE_NATIVE_CHAT_COMPAT_PROXY", 0) &&
            !request_forces_local_model(req)) {
            handle_proxy(req, app, app->aux_upstream, app->llm_permits, "application/json");
        } else if (app->llm_upstream) {
            handle_proxy(req, app, app->llm_upstream, app->llm_permits, "application/json");
        } else {
            handle_chat(req, app);
        }
        return;
    }
    bool engine_completion = path_starts_with(req, "/v1/engines/") &&
                             path_ends_with(req, "/completions");
    bool autocomplete = ohttp_path_is(req, "/api/v1/autocomplete");
    bool legacy_generate = ohttp_path_is(req, "/api/v1/generate") ||
                           ohttp_path_is(req, "/api/v1/generate-large") ||
                           ohttp_path_is(req, "/api/v1/generate-long") ||
                           autocomplete;
    if (ohttp_path_is(req, "/v1/completions") || legacy_generate || engine_completion) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        bool local_autocomplete = autocomplete && ollm_ready() &&
            env_flag("OMNISERVE_NATIVE_AUTOCOMPLETE_LOCAL", 0);
        if (app->llm_upstream && !local_autocomplete) {
            const char *mapped_path = NULL;
            if (engine_completion) mapped_path = "/v1/completions";
            else if (ohttp_path_is(req, "/api/v1/generate-large"))
                mapped_path = configured_path("OMNISERVE_NATIVE_GENERATE_LARGE_PATH", "/api/v1/generate");
            else if (autocomplete)
                mapped_path = configured_path("OMNISERVE_NATIVE_AUTOCOMPLETE_PATH", "/api/v1/generate");
            handle_proxy_as(req, app, app->llm_upstream, app->llm_permits,
                            "application/json", mapped_path);
        } else {
            handle_completion(req, app, legacy_generate, autocomplete);
        }
        return;
    }
    if (ohttp_path_is(req, "/api/v1/summarization")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_summarization(req, app);
        return;
    }
    if (ohttp_path_is(req, "/api/v1/feature-extraction") ||
        ohttp_path_is(req, "/v1/embeddings")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        if (oembed_ready()) {
            otier tier = request_tier(req);
            if (!osched_acquire_n(app->sched, tier, app->embedding_permits)) {
                respond_error(req, 503, "admission timeout; retry");
                return;
            }
            handle_embedding(req, ohttp_path_is(req, "/v1/embeddings"));
            osched_release_n(app->sched, tier, app->embedding_permits);
        } else {
            oproxy_target *target = app->embedding_upstream ? app->embedding_upstream : app->aux_upstream;
            handle_proxy(req, app, target, app->embedding_permits, "application/json");
        }
        return;
    }
    if (ohttp_path_is(req, "/v1/images/generations")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        if (app->image_upstream) {
            handle_proxy_as(req, app, app->image_upstream, app->image_permits,
                            "application/json",
                            configured_path("OMNISERVE_NATIVE_IMAGE_GENERATE_PATH",
                                            "/v1/images/generations"));
        }
        else handle_images(req, app);
        return;
    }
    if (ohttp_path_is(req, "/v1/images/backgrounds") ||
        ohttp_path_is(req, "/api/v1/art")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        /* Batch scene art is never interactive: pin it to the background tier so
         * it only ever runs on slots nothing else wants. */
        handle_proxy_as_tier(
            req, app, app->art_upstream, app->art_permits, "application/json",
            configured_path("OMNISERVE_NATIVE_ART_PATH", "/v1/images/generations"),
            TIER_BACKGROUND);
        return;
    }
    /* Async cutouts: POST /jobs enqueues, GET /jobs/{id} polls. Both are cheap
     * HTTP calls - the worker owns the GPU queue - so they take a single permit
     * at the caller's own tier instead of the whole background capacity, or a
     * poll would block behind the very job it is asking about. */
    if (path_starts_with(req, "/v1/images/background-removals/jobs")) {
        const bool is_post = ohttp_method_is(req, "POST");
        if (!is_post && !ohttp_method_is(req, "GET")) {
            respond_error(req, 405, "GET or POST required");
            return;
        }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        char mapped_path[1024];
        int mapped_len = snprintf(mapped_path, sizeof mapped_path, "%.*s",
                                  (int)req->path_len, req->path);
        if (mapped_len <= 0 || mapped_len >= (int)sizeof mapped_path) {
            respond_error(req, 414, "path too long");
            return;
        }
        handle_proxy_as_tier(req, app, app->birefnet_upstream, 1,
                             is_post ? "application/json" : NULL, mapped_path, -1);
        return;
    }
    if (ohttp_path_is(req, "/v1/images/background-removals") ||
        ohttp_path_is(req, "/api/v1/birefnet")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        const char *mapped_path = configured_path(
            "OMNISERVE_NATIVE_BIREFNET_PATH", "/v1/images/background-removals");
        handle_proxy_as(req, app, app->birefnet_upstream, app->birefnet_permits,
                        NULL, mapped_path);
        return;
    }
    if (ohttp_path_is(req, "/v1/animations/generations")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy_as_tier(
            req, app, app->animation_upstream, app->animation_permits,
            "application/json",
            configured_path("OMNISERVE_NATIVE_ANIMATION_PATH", "/v1/animations/generations"),
            TIER_BACKGROUND);
        return;
    }
    if (ohttp_path_is(req, "/v1/3d/generations")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy_as_tier(
            req, app, app->threed_upstream, app->threed_permits,
            "application/json",
            configured_path("OMNISERVE_NATIVE_3D_PATH", "/v1/3d/generations"),
            TIER_BACKGROUND);
        return;
    }
    if (path_starts_with(req, "/v1/3d/assets/")) {
        if (!ohttp_method_is(req, "GET")) { respond_error(req, 405, "GET required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        const size_t prefix_len = sizeof "/v1/3d/assets/" - 1;
        char mapped_path[1024];
        int mapped_len = snprintf(mapped_path, sizeof mapped_path, "/assets/%.*s",
                                  (int)(req->path_len - prefix_len), req->path + prefix_len);
        if (mapped_len < 0 || (size_t)mapped_len >= sizeof mapped_path) {
            respond_error(req, 414, "asset path too long");
            return;
        }
        handle_proxy_as(req, app, app->threed_upstream, 1, NULL, mapped_path);
        return;
    }
    if (ohttp_path_is(req, "/v1/audio/speech") ||
        ohttp_path_is(req, "/api/v1/generate_speech")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        const char *mapped_path = ohttp_path_is(req, "/v1/audio/speech")
            ? configured_path("OMNISERVE_NATIVE_TTS_OPENAI_PATH", "/v1/audio/speech")
            : configured_path("OMNISERVE_NATIVE_TTS_PATH", "/api/v1/generate_speech");
        handle_proxy_as(req, app, app->tts_upstream, app->tts_permits,
                        "application/json", mapped_path);
        return;
    }
    if (ohttp_path_is(req, "/api/v1/speech/catalog")) {
        if (!ohttp_method_is(req, "GET")) { respond_error(req, 405, "GET required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy(req, app, app->tts_upstream, app->tts_permits, NULL);
        return;
    }
    if (ohttp_path_is(req, "/v1/audio/transcriptions") ||
        ohttp_path_is(req, "/api/v1/audio/transcribe") ||
        ohttp_path_is(req, "/api/v1/audio-file-extraction") ||
        ohttp_path_is(req, "/api/v1/audio-extraction")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        const char *mapped_path = NULL;
        if (ohttp_path_is(req, "/v1/audio/transcriptions")) {
            mapped_path = configured_path("OMNISERVE_NATIVE_STT_OPENAI_PATH", "/v1/audio/transcriptions");
        } else if (ohttp_path_is(req, "/api/v1/audio-file-extraction")) {
            mapped_path = configured_path("OMNISERVE_NATIVE_STT_FILE_PATH", "/api/v1/audio-file-extraction");
        } else if (ohttp_path_is(req, "/api/v1/audio-extraction")) {
            mapped_path = configured_path("OMNISERVE_NATIVE_STT_URL_PATH", "/api/v1/audio-extraction");
        }
        handle_proxy_as(req, app, app->stt_upstream, app->stt_permits, NULL, mapped_path);
        return;
    }
    if (ohttp_path_is(req, "/v1/training/status") ||
        ohttp_path_is(req, "/v1/training/jobs")) {
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy(req, app, app->training_upstream, 1, "application/json");
        return;
    }
    if (ohttp_path_is(req, "/v1/training/jobs/run")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy_as_tier(
            req, app, app->training_upstream, app->training_permits,
            "application/json", NULL, TIER_BACKGROUND);
        return;
    }
    if (ohttp_path_is(req, "/api/v1/image-caption") ||
        ohttp_path_is(req, "/api/v1/video-question") ||
        ohttp_path_is(req, "/api/v1/multimodal-generate") ||
        ohttp_path_is(req, "/api/v1/voice-chat")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        oproxy_target *target = app->multimodal_upstream ? app->multimodal_upstream : app->aux_upstream;
        const char *mapped_path = ohttp_path_is(req, "/api/v1/image-caption")
            ? configured_path("OMNISERVE_NATIVE_CAPTION_PATH", "/api/v1/image-caption") : NULL;
        handle_proxy_as(req, app, target, app->multimodal_permits, NULL, mapped_path);
        return;
    }
    bool image_worker_path =
        ohttp_path_is(req, "/create_and_upload_image") ||
        ohttp_path_is(req, "/inpaint_and_upload_image") ||
        ohttp_path_is(req, "/style_transfer_and_upload_image") ||
        ohttp_path_is(req, "/style_transfer_bytes_and_upload_image") ||
        ohttp_path_is(req, "/generate_image") ||
        ohttp_path_is(req, "/generate_image_b64") ||
        ohttp_path_is(req, "/caption") || ohttp_path_is(req, "/caption_file") ||
        ohttp_path_is(req, "/nsfw_detect") || ohttp_path_is(req, "/nsfw_detect_file") ||
        ohttp_path_is(req, "/aesthetic_score");
    if (image_worker_path) {
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy(req, app, app->image_upstream, app->image_permits, NULL);
        return;
    }
    if (ohttp_path_is(req, "/synthesize")) {
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy(req, app, app->tts_upstream, app->tts_permits, "application/json");
        return;
    }
    if (ohttp_path_is(req, "/transcribe") || ohttp_path_is(req, "/transcribe_file")) {
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy(req, app, app->stt_upstream, app->stt_permits, NULL);
        return;
    }
    if (ohttp_path_is(req, "/api/v1/generate-bulk") ||
        ohttp_path_is(req, "/api/v1/generate-batch-csv") ||
        ohttp_path_is(req, "/api/discord")) {
        if (!ohttp_method_is(req, "POST")) { respond_error(req, 405, "POST required"); return; }
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy(req, app, app->aux_upstream, app->aux_permits, "application/json");
        return;
    }
    if (path_starts_with(req, "/files/") || ohttp_path_is(req, "/config")) {
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        handle_proxy(req, app, app->aux_upstream, app->aux_permits, NULL);
        return;
    }
    if (path_starts_with(req, "/backend/")) {
        if (!authorized(app, req)) { respond_error(req, 401, "invalid secret"); return; }
        if (app->llm_upstream) {
            handle_proxy(req, app, app->llm_upstream, app->llm_permits, "application/json");
        } else {
            /* embedded model is always resident: sleep/ensure are no-ops */
            ohttp_respond_str(req, 200, "application/json",
                              "{\"ok\":true,\"managed\":false}");
        }
        return;
    }
    respond_error(req, 404, "not found");
}

static int configured_permits(const char *name, int fallback, int capacity) {
    const char *value = getenv(name);
    int permits = value ? atoi(value) : fallback;
    if (permits < 1) permits = 1;
    if (permits > capacity) permits = capacity;
    return permits;
}

int main(int argc, char **argv) {
    uint16_t port = 8791;
    const char *port_env = getenv("OMNISERVE_NATIVE_PORT");
    if (port_env) port = (uint16_t)atoi(port_env);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            puts("usage: omniserve-native [--port PORT]\n"
                 "environment: OMNISERVE_NATIVE_LLM_GGUF, _EMBEDDING_GGUF, _SD_MODEL,\n"
                 "             _SECRET, _SLOTS, _LLM_UPSTREAM, _IMAGE_UPSTREAM,\n"
                 "             _ART_UPSTREAM, _ART_PATH, _ART_PERMITS,\n"
                 "             _BIREFNET_UPSTREAM, _TTS_UPSTREAM, _STT_UPSTREAM,\n"
                 "             _EMBEDDING_UPSTREAM, _BIREFNET_PERMITS,\n"
                 "             _MULTIMODAL_UPSTREAM, _ANIMATION_UPSTREAM, _AUX_UPSTREAM");
            return 0;
        }
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = (uint16_t)atoi(argv[++i]);
    }

    app_state app = {0};
    const char *slots_env = getenv("OMNISERVE_NATIVE_SLOTS");
    int slots = slots_env ? atoi(slots_env) : 1;
    if (slots < 1) slots = 1;
    if (slots > 64) slots = 64;
    app.llm_permits = configured_permits("OMNISERVE_NATIVE_LLM_PERMITS", 1, slots);
    app.image_permits = configured_permits("OMNISERVE_NATIVE_IMAGE_PERMITS", slots, slots);
    app.art_permits = configured_permits("OMNISERVE_NATIVE_ART_PERMITS", slots, slots);
    app.birefnet_permits = configured_permits("OMNISERVE_NATIVE_BIREFNET_PERMITS", 1, slots);
    app.tts_permits = configured_permits("OMNISERVE_NATIVE_TTS_PERMITS", 1, slots);
    app.stt_permits = configured_permits("OMNISERVE_NATIVE_STT_PERMITS", 1, slots);
    app.training_permits = configured_permits("OMNISERVE_NATIVE_TRAINING_PERMITS", slots, slots);
    app.embedding_permits = configured_permits("OMNISERVE_NATIVE_EMBEDDING_PERMITS", 1, slots);
    app.multimodal_permits = configured_permits("OMNISERVE_NATIVE_MULTIMODAL_PERMITS", 1, slots);
    app.animation_permits = configured_permits("OMNISERVE_NATIVE_ANIMATION_PERMITS", slots, slots);
    app.threed_permits = configured_permits("OMNISERVE_NATIVE_3D_PERMITS", slots, slots);
    app.aux_permits = configured_permits("OMNISERVE_NATIVE_AUX_PERMITS", 1, slots);
    const char *admission_env = getenv("OMNISERVE_NATIVE_ADMISSION_TIMEOUT_S");
    app.sched = osched_create(slots, admission_env ? atof(admission_env) : 30.0);
    if (env_flag("OMNISERVE_NATIVE_VRAM_BROKER", 1)) {
        const char *keep_env = getenv("OMNISERVE_NATIVE_VRAM_KEEP_FREE_MB");
        const char *ttl_env = getenv("OMNISERVE_NATIVE_VRAM_LEASE_TTL_S");
        app.vram = ovram_create(keep_env ? atoi(keep_env) : 1024,
                                ttl_env ? atof(ttl_env) : 120.0);
        /* Declares memory a co-tenant will grow into but has not taken yet.
         * Memory a tenant already holds must not be listed: the driver's free
         * figure has counted it once already. Format: "name:mb,name:mb". */
        const char *reserve_env = getenv("OMNISERVE_NATIVE_VRAM_RESERVE");
        if (app.vram && reserve_env && reserve_env[0]) {
            char buf[512];
            snprintf(buf, sizeof buf, "%s", reserve_env);
            for (char *save = NULL, *tok = strtok_r(buf, ",", &save); tok;
                 tok = strtok_r(NULL, ",", &save)) {
                char *colon = strchr(tok, ':');
                if (!colon) continue;
                *colon = '\0';
                if (!ovram_reserve(app.vram, tok, atoi(colon + 1))) {
                    fprintf(stderr, "vram reservation rejected: %s\n", tok);
                }
            }
        }
    }

    /* Warm the weights the broker may later ask us to drop. Defaults to the
     * models this process already loads, so the common case needs no config;
     * an explicit list can add files owned by co-tenants. */
    if (env_flag("OMNISERVE_NATIVE_RAM_PREFETCH_ENABLED", 1)) {
        const char *keep_pct_env = getenv("OMNISERVE_NATIVE_RAM_KEEP_FREE_PCT");
        int keep_pct = keep_pct_env ? atoi(keep_pct_env) : 5;
        const char *paths = getenv("OMNISERVE_NATIVE_RAM_PREFETCH");
        char derived[1024];
        if (!paths || !paths[0]) {
            const char *llm = getenv("OMNISERVE_NATIVE_LLM_GGUF");
            const char *embed = getenv("OMNISERVE_NATIVE_EMBEDDING_GGUF");
            snprintf(derived, sizeof derived, "%s%s%s",
                     llm ? llm : "", (llm && embed) ? "," : "", embed ? embed : "");
            paths = derived;
        }
        if (paths[0]) ohost_prefetch_start(paths, keep_pct);
    }
    app.secret = getenv("OMNISERVE_NATIVE_SECRET");
    const char *workers_env = getenv("OMNISERVE_NATIVE_WORKERS");
    int workers = workers_env ? atoi(workers_env) : 32;
    if (workers < 1) workers = 1;
    if (workers > 1024) workers = 1024;
    const char *pool_env = getenv("OMNISERVE_NATIVE_UPSTREAM_IDLE");
    int upstream_idle = pool_env ? atoi(pool_env) : workers;
    if (upstream_idle < 0) upstream_idle = 0;
    if (upstream_idle > 1024) upstream_idle = 1024;
    const char *unified_upstream = getenv("OMNISERVE_NATIVE_UPSTREAM");
    const char *llm_upstream = getenv("OMNISERVE_NATIVE_LLM_UPSTREAM");
    const char *image_upstream = getenv("OMNISERVE_NATIVE_IMAGE_UPSTREAM");
    const char *art_upstream = getenv("OMNISERVE_NATIVE_ART_UPSTREAM");
    const char *birefnet_upstream = getenv("OMNISERVE_NATIVE_BIREFNET_UPSTREAM");
    const char *tts_upstream = getenv("OMNISERVE_NATIVE_TTS_UPSTREAM");
    const char *stt_upstream = getenv("OMNISERVE_NATIVE_STT_UPSTREAM");
    const char *training_upstream = getenv("OMNISERVE_NATIVE_TRAINING_UPSTREAM");
    const char *embedding_upstream = getenv("OMNISERVE_NATIVE_EMBEDDING_UPSTREAM");
    const char *multimodal_upstream = getenv("OMNISERVE_NATIVE_MULTIMODAL_UPSTREAM");
    const char *animation_upstream = getenv("OMNISERVE_NATIVE_ANIMATION_UPSTREAM");
    const char *threed_upstream = getenv("OMNISERVE_NATIVE_3D_UPSTREAM");
    const char *aux_upstream = getenv("OMNISERVE_NATIVE_AUX_UPSTREAM");
    if (!llm_upstream) llm_upstream = unified_upstream;
    if (!image_upstream) image_upstream = unified_upstream;
    if (!art_upstream) art_upstream = image_upstream;
    if (!birefnet_upstream) birefnet_upstream = unified_upstream;
    if (!tts_upstream) tts_upstream = unified_upstream;
    if (!stt_upstream) stt_upstream = unified_upstream;
    if (!aux_upstream) aux_upstream = unified_upstream;
    if (!embedding_upstream) embedding_upstream = aux_upstream;
    if (!multimodal_upstream) multimodal_upstream = aux_upstream ? aux_upstream : image_upstream;
    char upstream_error[256];
#define CREATE_UPSTREAM(field, url, label) do { \
    if ((url) && (url)[0]) { \
        app.field = oproxy_target_create((url), upstream_idle, upstream_error, sizeof upstream_error); \
        if (!app.field) { \
            fprintf(stderr, "invalid %s upstream: %s\n", (label), upstream_error); \
            return 1; \
        } \
    } \
} while (0)
    CREATE_UPSTREAM(llm_upstream, llm_upstream, "LLM");
    CREATE_UPSTREAM(image_upstream, image_upstream, "image");
    CREATE_UPSTREAM(art_upstream, art_upstream, "background art");
    CREATE_UPSTREAM(birefnet_upstream, birefnet_upstream, "birefnet");
    CREATE_UPSTREAM(tts_upstream, tts_upstream, "TTS");
    CREATE_UPSTREAM(stt_upstream, stt_upstream, "STT");
    CREATE_UPSTREAM(training_upstream, training_upstream, "training");
    CREATE_UPSTREAM(embedding_upstream, embedding_upstream, "embedding");
    CREATE_UPSTREAM(multimodal_upstream, multimodal_upstream, "multimodal");
    CREATE_UPSTREAM(animation_upstream, animation_upstream, "animation");
    CREATE_UPSTREAM(threed_upstream, threed_upstream, "3D");
    CREATE_UPSTREAM(aux_upstream, aux_upstream, "auxiliary");
#undef CREATE_UPSTREAM
    const char *upstream_timeout = getenv("OMNISERVE_NATIVE_UPSTREAM_TIMEOUT_MS");
    app.upstream_timeout_ms = upstream_timeout ? atoi(upstream_timeout) : 600000;

    const char *gguf = getenv("OMNISERVE_NATIVE_LLM_GGUF");
    if (gguf && gguf[0]) {
        const char *ngl = getenv("OMNISERVE_NATIVE_NGL");
        const char *ctx = getenv("OMNISERVE_NATIVE_CTX");
        fprintf(stderr, "loading llm %s\n", gguf);
        const char *contexts = getenv("OMNISERVE_NATIVE_LLM_CONTEXTS");
        int parallel_contexts = contexts ? atoi(contexts) : slots;
        if (parallel_contexts < 1) parallel_contexts = 1;
        if (parallel_contexts > slots) parallel_contexts = slots;
        if (!ollm_init(gguf, ngl ? atoi(ngl) : 999, ctx ? atoi(ctx) : 8192,
                       parallel_contexts)) {
            fprintf(stderr, "llm load failed\n");
        }
        ollm_placement placement;
        ollm_placement_snapshot(&placement);
        if (ollm_ready() && placement.gpu_requested && !placement.on_gpu) {
            /* CUDA can be transiently unavailable (driver reload, GPU reset, a
             * peer holding the device). Serving an 8B model from CPU costs
             * roughly an order of magnitude in latency and CPU-hours, so exit
             * and let the supervisor retry rather than silently degrade. */
            fprintf(stderr,
                    "FATAL: %d GPU layers requested but no GPU backend device is "
                    "registered (device=%s); refusing to serve from CPU. Set "
                    "OMNISERVE_NATIVE_ALLOW_CPU_FALLBACK=1 to override.\n",
                    ngl ? atoi(ngl) : 999, placement.device);
            if (!env_flag("OMNISERVE_NATIVE_ALLOW_CPU_FALLBACK", false)) {
                ollm_shutdown();
                return 1;
            }
        }
    }
    const char *embedding_gguf = getenv("OMNISERVE_NATIVE_EMBEDDING_GGUF");
    if (embedding_gguf && embedding_gguf[0]) {
        const char *embed_ngl = getenv("OMNISERVE_NATIVE_EMBEDDING_NGL");
        const char *embed_ctx = getenv("OMNISERVE_NATIVE_EMBEDDING_CTX");
        const char *embed_threads = getenv("OMNISERVE_NATIVE_EMBEDDING_THREADS");
        fprintf(stderr, "loading embedding model %s\n", embedding_gguf);
        if (!oembed_init(embedding_gguf, embed_ngl ? atoi(embed_ngl) : 0,
                         embed_ctx ? atoi(embed_ctx) : 512,
                         embed_threads ? atoi(embed_threads) : 8)) {
            fprintf(stderr, "embedding model load failed\n");
        }
    }
    const char *sd = getenv("OMNISERVE_NATIVE_SD_MODEL");
    if (sd && sd[0]) {
        fprintf(stderr, "loading diffusion %s\n", sd);
        if (!osd_init(sd)) fprintf(stderr, "diffusion load failed\n");
    }

    app.scale = oscale_create();
    if (app.scale) {
        configure_scale_lanes(&app);
        ocapacity_config capacity_cfg = {
            .sched = app.sched,
            .scale = app.scale,
            .control_base = getenv("OMNISERVE_NATIVE_SCALE_CONTROL_BASE"),
            .api_key = getenv("OMNISERVE_NATIVE_SCALE_API_KEY"),
            .poll_interval_s = env_int("OMNISERVE_NATIVE_SCALE_POLL_S", 15),
            .timeout_ms = env_int("OMNISERVE_NATIVE_SCALE_TIMEOUT_MS", 30000),
        };
        app.capacity = ocapacity_start(&capacity_cfg);
        int armed = 0;
        for (int i = 0; i < oscale_lane_count(app.scale); i++) {
            oscale_lane *lane = oscale_lane_at(app.scale, i);
            if (lane && lane->policy.enabled) armed++;
        }
        if (armed) {
            fprintf(stderr, "capacity control: %d lane(s) armed for paid overflow\n", armed);
        }
    }

    /* The router owns the relay-header list and the internal predicate; the log
     * borrows both, so a header added to request_via_proxy is logged the day it
     * is added. Credential and X-Omniserve-* claim headers are a separate list
     * in olog.c and are NOT coupled to this one: adding a new claim header to
     * request_is_internal alone will not make it appear in hdrs=. */
    olog_set_trust_headers(relayed_headers, RELAYED_HEADER_COUNT);
    olog_set_internal_fn(request_is_internal);
    olog_init();
    if (olog_enabled()) fprintf(stderr, "access log: %s\n", olog_path());

    const char *reactors_env = getenv("OMNISERVE_NATIVE_REACTORS");
    ohttp_config cfg = {
        .port = port,
        .bind_addr = getenv("OMNISERVE_NATIVE_BIND"),
        .reactor_threads = reactors_env ? atoi(reactors_env) : 2,
        .worker_threads = workers,
        .handler = route,
        .user = &app,
    };
    ohttp_server *srv = ohttp_start(&cfg);
    if (!srv) {
        fprintf(stderr, "failed to bind port %u\n", port);
        return 1;
    }
    fprintf(stderr, "omniserve-native listening on %u (llm=%s embedding=%s diffusion=%s vram %.1f/%.1f GiB)\n",
            port, ollm_model_name(), oembed_model_name(), osd_model_name(),
            ogpu_free_gib(), ogpu_total_gib());
    int rc = ohttp_join(srv);
    olog_shutdown();
    oproxy_target_destroy(app.llm_upstream);
    oproxy_target_destroy(app.image_upstream);
    oproxy_target_destroy(app.art_upstream);
    oproxy_target_destroy(app.birefnet_upstream);
    oproxy_target_destroy(app.tts_upstream);
    oproxy_target_destroy(app.stt_upstream);
    oproxy_target_destroy(app.training_upstream);
    oproxy_target_destroy(app.embedding_upstream);
    oproxy_target_destroy(app.multimodal_upstream);
    oproxy_target_destroy(app.animation_upstream);
    oproxy_target_destroy(app.threed_upstream);
    oproxy_target_destroy(app.aux_upstream);
    ocapacity_stop(app.capacity);
    oscale_destroy(app.scale);
    oembed_shutdown();
    ollm_shutdown();
    osched_destroy(app.sched);
    ovram_destroy(app.vram);
    return rc;
}
