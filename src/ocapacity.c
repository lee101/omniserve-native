#define _GNU_SOURCE
#include "ocapacity.h"

#include "ojson.h"
#include "oproxy.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct ocapacity {
    ocapacity_config config;
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    bool running;
    /* Rolling arrival estimate per lane, used to size the backlog the cost
     * gate is allowed to count on. */
    long served_baseline[OSCALE_MAX_LANES];
    double baseline_at_s[OSCALE_MAX_LANES];
    double observed_rate_hz[OSCALE_MAX_LANES];
    unsigned long long warm_failures;
    char last_error[192];
};

static double monotonic_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

typedef struct {
    char *buf;
    size_t cap;
    size_t len;
} collect_sink;

static bool collect(const void *data, size_t len, void *user) {
    collect_sink *state = user;
    if (state->len >= state->cap) return true;
    size_t room = state->cap - state->len;
    size_t take = len < room ? len : room;
    memcpy(state->buf + state->len, data, take);
    state->len += take;
    return true;
}

/* Warm a cog through app.nz on loopback. The prediction both provisions the pod
 * and refreshes its idle timer; app.nz records the cost against our key. */
static bool warm_via_appnz(const oscale_policy *policy, char *endpoint, size_t endpoint_cap,
                           char *error, size_t error_cap, void *user) {
    ocapacity *capacity = user;
    if (endpoint && endpoint_cap) endpoint[0] = 0;
    if (!capacity->config.control_base || !capacity->config.control_base[0]) {
        snprintf(error, error_cap, "no control plane configured");
        return false;
    }
    char body[512];
    int body_len = snprintf(body, sizeof body,
                            "{\"template\":\"%s\",\"hardware\":\"%s\",\"input\":{\"warm\":true}}",
                            policy->cog_template, policy->hardware);
    if (body_len < 0 || (size_t)body_len >= sizeof body) {
        snprintf(error, error_cap, "warm body too large");
        return false;
    }

    char response[8192];
    collect_sink state = { response, sizeof response - 1, 0 };

    oproxy_header headers[1];
    headers[0].name = "X-API-Key";
    headers[0].name_len = 9;
    headers[0].value = capacity->config.api_key ? capacity->config.api_key : "";
    headers[0].value_len = strlen(headers[0].value);

    oproxy_result result;
    memset(&result, 0, sizeof result);
    bool ok = oproxy_relay(capacity->config.control_base,
                           "POST", 4, "/api/cogs/run", 13, NULL, 0,
                           body, (size_t)body_len, "application/json", 16,
                           headers, headers[0].value_len ? 1 : 0,
                           capacity->config.timeout_ms > 0 ? capacity->config.timeout_ms : 30000,
                           collect, &state, &result, error, error_cap);
    response[state.len] = 0;
    if (!ok) return false;

    /* app.nz answers with the prediction record; a warm that the control plane
     * rejected is not capacity, so treat a missing id as failure. */
    oj_tok toks[128];
    int ntoks = oj_parse(response, state.len, toks, 128);
    if (ntoks <= 0) {
        snprintf(error, error_cap, "unparseable warm response");
        return false;
    }
    int prediction = oj_obj_get(response, toks, ntoks, 0, "prediction");
    int id = prediction >= 0 ? oj_obj_get(response, toks, ntoks, prediction, "id") : -1;
    if (id < 0) id = oj_obj_get(response, toks, ntoks, 0, "id");
    if (id < 0) {
        snprintf(error, error_cap, "warm rejected by control plane");
        return false;
    }
    /* Overflow traffic is relayed through the same loopback control plane, so
     * the routable handle is the model path, not the pod address: the C data
     * plane has no TLS and must never hold a provider endpoint. */
    if (endpoint && endpoint_cap) {
        snprintf(endpoint, endpoint_cap, "%s", capacity->config.control_base);
    }
    return true;
}

/* Maps a lane onto the tier counters the scheduler already tracks. The
 * scheduler is global rather than per-modality, so a lane sees total admission
 * pressure; that is the correct signal for "the local device is full". */
static void observe(ocapacity *capacity, int lane_index, const oscale_lane *lane,
                    double now_s, oscale_observation *obs) {
    osched_stats stats;
    osched_snapshot(capacity->config.sched, &stats);
    memset(obs, 0, sizeof *obs);
    for (int tier = TIER_PAID; tier <= TIER_BACKGROUND; tier++) {
        obs->queue_depth[tier] = stats.waiting[tier];
        obs->queue_ms_max[tier] = stats.queue_ms_max[tier];
    }
    int free_slots = stats.slots - stats.used_slots;
    obs->local_permits_free = free_slots > 0 ? free_slots : 0;

    long served = 0;
    for (int tier = TIER_PAID; tier <= TIER_BACKGROUND; tier++) {
        if (lane->policy.tier_mask & OSCALE_TIER_BIT(tier)) served += stats.served[tier];
    }
    double elapsed = now_s - capacity->baseline_at_s[lane_index];
    if (capacity->baseline_at_s[lane_index] > 0 && elapsed >= 1.0) {
        long delta = served - capacity->served_baseline[lane_index];
        if (delta < 0) delta = 0;
        capacity->observed_rate_hz[lane_index] = (double)delta / elapsed;
    }
    capacity->served_baseline[lane_index] = served;
    capacity->baseline_at_s[lane_index] = now_s;

    /* Backlog the next hour could plausibly bring: what is already queued plus
     * the recent eligible arrival rate. Estimating from observed traffic keeps
     * the cost gate from renting for a spike that has already passed. */
    obs->backlog_reqs = (double)oscale_eligible_depth(&lane->policy, obs) +
                        capacity->observed_rate_hz[lane_index] * 3600.0;
}

int ocapacity_tick(ocapacity *capacity, double now_s) {
    if (!capacity || !capacity->config.scale || !capacity->config.sched) return 0;
    int changes = 0;
    int lanes = oscale_lane_count(capacity->config.scale);
    for (int i = 0; i < lanes; i++) {
        oscale_lane *lane = oscale_lane_at(capacity->config.scale, i);
        if (!lane) continue;
        oscale_observation obs;
        observe(capacity, i, lane, now_s, &obs);
        oscale_reason reason = OSCALE_REASON_NONE;
        oscale_action action = oscale_decide(lane, &obs, now_s, &reason);
        lane->last_reason = reason;
        if (action == OSCALE_UP) {
            int slot = oscale_begin_instance(lane, now_s);
            if (slot < 0) continue;
            char endpoint[256] = {0};
            char error[160] = {0};
            ocapacity_warm_fn warm = capacity->config.warm ? capacity->config.warm : warm_via_appnz;
            if (warm(&lane->policy, endpoint, sizeof endpoint, error, sizeof error,
                     capacity->config.warm ? capacity->config.warm_user : capacity)) {
                oscale_instance_ready(lane, slot, endpoint, now_s);
                fprintf(stderr, "capacity: lane %s scaled up on %s (%s, $%.3f/hr)\n",
                        lane->policy.name, lane->policy.hardware,
                        oscale_reason_name(reason), lane->policy.price_usd_hr);
            } else {
                /* A warm that failed must not leave a billable placeholder. */
                oscale_release_instance(lane, slot, now_s, OSCALE_REASON_NONE);
                capacity->warm_failures++;
                snprintf(capacity->last_error, sizeof capacity->last_error, "%s: %s",
                         lane->policy.name, error);
                fprintf(stderr, "capacity: lane %s warm failed: %s\n", lane->policy.name, error);
            }
            changes++;
        } else if (action == OSCALE_DOWN) {
            oscale_reason why = OSCALE_REASON_IDLE;
            int slot = oscale_release_candidate(lane, now_s, &why);
            if (slot >= 0) {
                oscale_release_instance(lane, slot, now_s, why);
                fprintf(stderr, "capacity: lane %s released instance (%s)\n",
                        lane->policy.name, oscale_reason_name(why));
                changes++;
            }
        }
    }
    return changes;
}

static void *controller_main(void *user) {
    ocapacity *capacity = user;
    int interval = capacity->config.poll_interval_s > 0 ? capacity->config.poll_interval_s : 15;
    pthread_mutex_lock(&capacity->lock);
    while (capacity->running) {
        struct timespec deadline;
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += interval;
        pthread_cond_timedwait(&capacity->wake, &capacity->lock, &deadline);
        if (!capacity->running) break;
        pthread_mutex_unlock(&capacity->lock);
        ocapacity_tick(capacity, monotonic_s());
        pthread_mutex_lock(&capacity->lock);
    }
    pthread_mutex_unlock(&capacity->lock);
    return NULL;
}

ocapacity *ocapacity_start(const ocapacity_config *config) {
    if (!config || !config->sched || !config->scale) return NULL;
    ocapacity *capacity = calloc(1, sizeof *capacity);
    if (!capacity) return NULL;
    capacity->config = *config;
    pthread_mutex_init(&capacity->lock, NULL);
    pthread_condattr_t attr;
    pthread_condattr_init(&attr);
    pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    pthread_cond_init(&capacity->wake, &attr);
    pthread_condattr_destroy(&attr);
    capacity->running = true;
    if (pthread_create(&capacity->thread, NULL, controller_main, capacity) != 0) {
        capacity->running = false;
        pthread_cond_destroy(&capacity->wake);
        pthread_mutex_destroy(&capacity->lock);
        free(capacity);
        return NULL;
    }
    return capacity;
}

void ocapacity_stop(ocapacity *capacity) {
    if (!capacity) return;
    pthread_mutex_lock(&capacity->lock);
    capacity->running = false;
    pthread_cond_broadcast(&capacity->wake);
    pthread_mutex_unlock(&capacity->lock);
    pthread_join(capacity->thread, NULL);
    /* Releasing on shutdown keeps a restart from orphaning rented capacity. */
    int lanes = oscale_lane_count(capacity->config.scale);
    double now_s = monotonic_s();
    for (int i = 0; i < lanes; i++) {
        oscale_lane *lane = oscale_lane_at(capacity->config.scale, i);
        if (!lane) continue;
        for (int slot = 0; slot < lane->instance_count; slot++) {
            oscale_release_instance(lane, slot, now_s, OSCALE_REASON_NONE);
        }
    }
    pthread_cond_destroy(&capacity->wake);
    pthread_mutex_destroy(&capacity->lock);
    free(capacity);
}

const char *ocapacity_overflow_endpoint(ocapacity *capacity, const char *lane_name, otier tier) {
    if (!capacity || !lane_name) return NULL;
    oscale_lane *lane = oscale_lane_by_name(capacity->config.scale, lane_name);
    if (!lane || !lane->policy.enabled) return NULL;
    if (!(lane->policy.tier_mask & OSCALE_TIER_BIT(tier))) return NULL;
    for (int i = 0; i < lane->instance_count; i++) {
        if (lane->instances[i].active && lane->instances[i].ready &&
            lane->instances[i].endpoint[0]) {
            oscale_instance_touch(lane, i, monotonic_s());
            return lane->instances[i].endpoint;
        }
    }
    return NULL;
}

size_t ocapacity_status_json(ocapacity *capacity, char *out, size_t cap) {
    if (!out || cap == 0) return 0;
    if (!capacity) return (size_t)snprintf(out, cap, "null");
    size_t len = (size_t)snprintf(out, cap, "{\"warm_failures\":%llu,\"lanes\":[",
                                  capacity->warm_failures);
    int lanes = oscale_lane_count(capacity->config.scale);
    for (int i = 0; i < lanes && len < cap; i++) {
        const oscale_lane *lane = oscale_lane_at(capacity->config.scale, i);
        if (!lane) continue;
        len += (size_t)snprintf(out + len, cap > len ? cap - len : 0,
            "%s{\"name\":\"%.20s\",\"enabled\":%s,\"hardware\":\"%.20s\","
            "\"instances\":%d,\"ready\":%d,\"price_usd_hr\":%.3f,\"spend_rate_usd_hr\":%.3f,"
            "\"spend_usd\":%.4f,\"instance_seconds\":%.0f,\"scale_ups\":%llu,"
            "\"scale_downs\":%llu,\"ttl_kills\":%llu,\"last_reason\":\"%s\"}",
            i ? "," : "", lane->policy.name, lane->policy.enabled ? "true" : "false",
            lane->policy.hardware, lane->instance_count, oscale_ready_instances(lane),
            lane->policy.price_usd_hr, oscale_lane_spend_rate_usd_hr(lane),
            lane->spend_usd, lane->instance_seconds, lane->scale_ups,
            lane->scale_downs, lane->ttl_kills, oscale_reason_name(lane->last_reason));
    }
    if (len < cap) len += (size_t)snprintf(out + len, cap - len, "]}");
    return len < cap ? len : cap - 1;
}
