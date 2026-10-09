#include "oresidency.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ORES_MAX_CLAIMS 8
#define ORES_MIN_DEMAND 0.05

typedef struct {
    char owner[32];
    int pid;
    int resident_mb, stream_mb;
    double saved_ms;
    double demand, demand_ts;
    double last_req_s, last_seen_s;
    double admitted_at_s, cooldown_until_s;
    int revokes; /* consecutive short-lived grants: each doubles the next cooldown (max 32x) */
    bool resident, admitted, active;
} claim;

#define ORES_FREE_HIST 64

struct ores {
    pthread_mutex_t lock;
    ores_config cfg;
    claim claims[ORES_MAX_CLAIMS];
    double free_ts[ORES_FREE_HIST];
    int free_mb[ORES_FREE_HIST];
    int free_n, free_head;
    unsigned long long promotions, revocations;
};

void ores_config_defaults(ores_config *cfg) {
    cfg->keep_free_mb = 1024;
    cfg->half_life_s = 300.0;
    cfg->min_hold_s = 120.0;
    cfg->idle_release_s = 900.0;
    cfg->switch_ratio = 1.5;
    cfg->claim_ttl_s = 180.0;
    cfg->reserve_idle_s = 180.0;
    cfg->hysteresis_mb = 1024;
    cfg->free_window_s = 60.0;
}

ores *ores_create(const ores_config *cfg) {
    ores *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    pthread_mutex_init(&r->lock, NULL);
    if (cfg) r->cfg = *cfg;
    else ores_config_defaults(&r->cfg);
    if (r->cfg.half_life_s <= 0) r->cfg.half_life_s = 300.0;
    if (r->cfg.switch_ratio < 1.0) r->cfg.switch_ratio = 1.0;
    if (r->cfg.claim_ttl_s <= 0) r->cfg.claim_ttl_s = 180.0;
    return r;
}

void ores_destroy(ores *r) {
    if (!r) return;
    pthread_mutex_destroy(&r->lock);
    free(r);
}

static void decay(const ores *r, claim *c, double now) {
    if (now > c->demand_ts) c->demand *= exp2(-(now - c->demand_ts) / r->cfg.half_life_s);
    c->demand_ts = now;
}

static double benefit(const claim *c) { return c->demand * c->saved_ms; }

static bool eligible(const ores *r, const claim *c, double now) {
    return c->demand >= ORES_MIN_DEMAND && c->saved_ms > 0 && c->resident_mb > 0 &&
           now - c->last_req_s <= r->cfg.idle_release_s && now >= c->cooldown_until_s;
}

static claim *find_claim(ores *r, const char *owner, double now) {
    claim *slot = NULL;
    for (int i = 0; i < ORES_MAX_CLAIMS; i++) {
        claim *c = &r->claims[i];
        if (c->active && now - c->last_seen_s > r->cfg.claim_ttl_s) c->active = false;
        if (c->active && strcmp(c->owner, owner) == 0) return c;
    }
    for (int i = 0; i < ORES_MAX_CLAIMS && !slot; i++)
        if (!r->claims[i].active) slot = &r->claims[i];
    if (!slot) return NULL;
    memset(slot, 0, sizeof *slot);
    snprintf(slot->owner, sizeof slot->owner, "%s", owner);
    slot->active = true;
    slot->demand_ts = now;
    slot->last_req_s = -1e18;
    return slot;
}

/* Headroom kept for one streamed render of the largest claimant left streamed. A claimant
 * with no request inside reserve_idle_s reserves nothing: a quiet Qwen lane does not keep
 * Z-Image streamed all day; its first request makes it active again. */
static int stream_reserve(const ores *r, const bool *in_set, double now) {
    int m = 0;
    for (int i = 0; i < ORES_MAX_CLAIMS; i++) {
        const claim *c = &r->claims[i];
        if (c->active && !in_set[i] && now - c->last_req_s <= r->cfg.reserve_idle_s && c->stream_mb > m)
            m = c->stream_mb;
    }
    return m;
}

static void note_free(ores *r, double now, int device_free_mb) {
    if (device_free_mb < 0) return;
    r->free_ts[r->free_head] = now;
    r->free_mb[r->free_head] = device_free_mb;
    r->free_head = (r->free_head + 1) % ORES_FREE_HIST;
    if (r->free_n < ORES_FREE_HIST) r->free_n++;
}

static int window_min_free(const ores *r, double now, int device_free_mb) {
    int m = device_free_mb;
    for (int i = 0; i < r->free_n; i++)
        if (now - r->free_ts[i] <= r->cfg.free_window_s && r->free_mb[i] < m) m = r->free_mb[i];
    return m;
}

static void admit_all(ores *r, double now, int device_free_mb) {
    int order[ORES_MAX_CLAIMS], n = 0;
    double score[ORES_MAX_CLAIMS] = {0};
    note_free(r, now, device_free_mb);
    long capacity = device_free_mb;
    long capacity_new = window_min_free(r, now, device_free_mb) - (long)r->cfg.hysteresis_mb;
    for (int i = 0; i < ORES_MAX_CLAIMS; i++) {
        claim *c = &r->claims[i];
        if (c->active && now - c->last_seen_s > r->cfg.claim_ttl_s) c->active = false;
        if (!c->active) continue;
        decay(r, c, now);
        if (c->resident) {
            capacity += c->resident_mb;
            capacity_new += c->resident_mb;
        }
        bool locked = c->admitted && now - c->admitted_at_s < r->cfg.min_hold_s;
        score[i] = benefit(c) * (c->admitted ? r->cfg.switch_ratio : 1.0) + (locked ? 1e30 : 0.0);
        order[n++] = i;
    }
    for (int a = 1; a < n; a++)
        for (int b = a; b > 0 && score[order[b]] > score[order[b - 1]]; b--) {
            int t = order[b]; order[b] = order[b - 1]; order[b - 1] = t;
        }
    bool in_set[ORES_MAX_CLAIMS] = {false};
    long used = 0;
    for (int k = 0; k < n; k++) {
        int i = order[k];
        claim *c = &r->claims[i];
        if (device_free_mb < 0 || !eligible(r, c, now)) continue;
        in_set[i] = true;
        long need = used + c->resident_mb + r->cfg.keep_free_mb + stream_reserve(r, in_set, now);
        if (need <= (c->admitted ? capacity : capacity_new)) used += c->resident_mb;
        else in_set[i] = false;
    }
    for (int i = 0; i < ORES_MAX_CLAIMS; i++) {
        claim *c = &r->claims[i];
        if (!c->active) continue;
        if (in_set[i] && !c->admitted) c->admitted_at_s = now;
        if (!in_set[i] && c->admitted) {
            if (now - c->admitted_at_s >= 4.0 * r->cfg.min_hold_s) c->revokes = 0;
            c->cooldown_until_s = now + r->cfg.min_hold_s * (double)(1 << (c->revokes < 5 ? c->revokes : 5));
            c->revokes++;
            r->revocations++;
        }
        c->admitted = in_set[i];
    }
}

ores_decision ores_report_at(ores *r, const ores_report_in *in, double now_s, int device_free_mb) {
    ores_decision d;
    memset(&d, 0, sizeof d);
    if (!r || !in || !in->owner || !in->owner[0]) {
        snprintf(d.reason, sizeof d.reason, "invalid");
        return d;
    }
    pthread_mutex_lock(&r->lock);
    claim *c = find_claim(r, in->owner, now_s);
    if (!c) {
        pthread_mutex_unlock(&r->lock);
        snprintf(d.reason, sizeof d.reason, "no_slot");
        return d;
    }
    decay(r, c, now_s);
    c->pid = in->pid;
    c->resident_mb = in->resident_mb > 0 ? in->resident_mb : 0;
    c->stream_mb = in->stream_mb > 0 ? in->stream_mb : 0;
    c->saved_ms = in->saved_ms > 0 ? in->saved_ms : 0;
    c->resident = in->resident;
    c->last_seen_s = now_s;
    if (in->requests > 0) {
        c->demand += in->requests;
        c->last_req_s = now_s;
    }
    admit_all(r, now_s, device_free_mb);
    d.admitted = c->admitted;
    d.demand = c->demand;
    d.benefit = benefit(c);
    if (device_free_mb < 0) {
        snprintf(d.reason, sizeof d.reason, "no_device_info");
    } else if (!c->admitted) {
        const char *why = now_s - c->last_req_s > r->cfg.idle_release_s || c->demand < ORES_MIN_DEMAND ? "idle"
                        : now_s < c->cooldown_until_s ? "cooldown"
                        : "outranked_or_no_capacity";
        snprintf(d.reason, sizeof d.reason, "%s", why);
    } else if (c->resident) {
        d.resident = true;
        snprintf(d.reason, sizeof d.reason, "holding");
    } else {
        bool self[ORES_MAX_CLAIMS] = {false};
        for (int i = 0; i < ORES_MAX_CLAIMS; i++) self[i] = r->claims[i].admitted;
        long free_now = (long)device_free_mb - r->cfg.keep_free_mb - stream_reserve(r, self, now_s);
        d.resident = free_now >= c->resident_mb;
        if (d.resident) {
            r->promotions++;
            c->resident = true;
        }
        snprintf(d.reason, sizeof d.reason, "%s", d.resident ? "promote" : "pending_free");
    }
    pthread_mutex_unlock(&r->lock);
    return d;
}

ores_decision ores_report(ores *r, const ores_report_in *in, int device_free_mb) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ores_report_at(r, in, (double)ts.tv_sec + ts.tv_nsec / 1e9, device_free_mb);
}

size_t ores_status_json(ores *r, double now_s, char *out, size_t cap) {
    if (!out || cap < 3) return 0;
    if (!r) return (size_t)snprintf(out, cap, "null");
    pthread_mutex_lock(&r->lock);
    size_t len = (size_t)snprintf(out, cap,
        "{\"keep_free_mb\":%d,\"half_life_s\":%.0f,\"min_hold_s\":%.0f,\"idle_release_s\":%.0f,"
        "\"switch_ratio\":%.2f,\"promotions\":%llu,\"revocations\":%llu,\"claims\":[",
        r->cfg.keep_free_mb, r->cfg.half_life_s, r->cfg.min_hold_s, r->cfg.idle_release_s,
        r->cfg.switch_ratio, r->promotions, r->revocations);
    bool first = true;
    for (int i = 0; i < ORES_MAX_CLAIMS && len < cap; i++) {
        claim *c = &r->claims[i];
        if (!c->active || now_s - c->last_seen_s > r->cfg.claim_ttl_s) continue;
        double demand = c->demand * exp2(-(now_s - c->demand_ts) / r->cfg.half_life_s);
        len += (size_t)snprintf(out + len, cap - len,
            "%s{\"owner\":\"%s\",\"pid\":%d,\"resident\":%s,\"admitted\":%s,\"resident_mb\":%d,"
            "\"stream_mb\":%d,\"saved_ms\":%.0f,\"demand\":%.2f,\"benefit_ms\":%.0f,\"idle_s\":%.0f}",
            first ? "" : ",", c->owner, c->pid, c->resident ? "true" : "false",
            c->admitted ? "true" : "false", c->resident_mb, c->stream_mb, c->saved_ms, demand,
            demand * c->saved_ms, c->last_req_s < 0 ? -1.0 : now_s - c->last_req_s);
        first = false;
    }
    if (len < cap) len += (size_t)snprintf(out + len, cap - len, "]}");
    pthread_mutex_unlock(&r->lock);
    return len < cap ? len : cap - 1;
}
