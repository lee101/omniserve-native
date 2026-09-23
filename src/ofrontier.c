#include "ofrontier.h"
#include "ojson.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>

#define OFR_MAX_FILE (1u << 20)
#define OFR_MAX_TOKS 8192

struct ofrontier {
    char policy_path[512];
    char workload[64];
    FILE *log;
    int port;
    pthread_mutex_t lock;
    pthread_mutex_t log_lock;
    ofr_table table;
    time_t mtime;
    long mtime_ns;
    off_t size;
    double checked_at;
};

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void ofrontier_table_default(ofr_table *t) {
    memset(t, 0, sizeof *t);
    for (int i = 0; i < 4; i++) t->policy[i] = OFR_POLICY_OVERFLOW;
}

const char *ofrontier_policy_name(ofr_policy p) {
    switch (p) {
    case OFR_POLICY_LOCAL: return "local_only";
    case OFR_POLICY_FASTEST: return "fastest";
    case OFR_POLICY_DEADLINE: return "cheapest_within_deadline";
    default: return "overflow_on_busy";
    }
}

static ofr_policy parse_policy(const char *js, const oj_tok *t) {
    if (oj_str_eq(js, t, "local_only") || oj_str_eq(js, t, "local")) return OFR_POLICY_LOCAL;
    if (oj_str_eq(js, t, "fastest")) return OFR_POLICY_FASTEST;
    if (oj_str_eq(js, t, "cheapest_within_deadline")) return OFR_POLICY_DEADLINE;
    return OFR_POLICY_OVERFLOW;
}

bool ofrontier_parse(const char *json, size_t len, const char *workload, ofr_table *out) {
    ofrontier_table_default(out);
    if (!json || !len || !workload) return false;
    oj_tok *toks = malloc(sizeof(oj_tok) * OFR_MAX_TOKS);
    if (!toks) return false;
    bool ok = false;
    int n = oj_parse(json, len, toks, OFR_MAX_TOKS);
    if (n <= 0 || toks[0].type != OJ_OBJECT) goto done;
    int workloads = oj_obj_get(json, toks, n, 0, "workloads");
    int w = oj_obj_get(json, toks, n, workloads, workload);
    int gw = oj_obj_get(json, toks, n, w, "gateway");
    if (gw < 0 || toks[gw].type != OJ_OBJECT) goto done;
    int lp = oj_obj_get(json, toks, n, gw, "local_p50_ms");
    int rp = oj_obj_get(json, toks, n, gw, "remote_p50_ms");
    out->local_p50_ms = lp >= 0 ? oj_number(json, &toks[lp], 0) : 0;
    out->remote_p50_ms = rp >= 0 ? oj_number(json, &toks[rp], 0) : 0;
    if (!(out->local_p50_ms > 0) || !(out->remote_p50_ms > 0)) goto done;
    int tiers = oj_obj_get(json, toks, n, gw, "tiers");
    static const char *names[4] = {"paid", "sub", "free", "background"};
    for (int i = 0; i < 4; i++) {
        int tier = oj_obj_get(json, toks, n, tiers, names[i]);
        if (tier < 0 || toks[tier].type != OJ_OBJECT) continue;
        int p = oj_obj_get(json, toks, n, tier, "policy");
        if (p >= 0 && toks[p].type == OJ_STRING) out->policy[i] = parse_policy(json, &toks[p]);
        int d = oj_obj_get(json, toks, n, tier, "deadline_ms");
        if (d >= 0) out->deadline_ms[i] = oj_number(json, &toks[d], 0);
    }
    out->loaded = true;
    ok = true;
done:
    free(toks);
    if (!ok) ofrontier_table_default(out);
    return ok;
}

double ofrontier_local_wait_ms(const osched_stats *st, otier tier, int permits, double local_p50_ms) {
    if (!st || local_p50_ms <= 0) return 0;
    if (permits < 1) permits = 1;
    int ahead = 0;
    for (int i = TIER_PAID; i <= (int)tier && i <= TIER_BACKGROUND; i++) ahead += st->waiting[i];
    int slots = st->slots > 0 ? st->slots : 1;
    int lanes = slots / permits;
    if (lanes < 1) lanes = 1;
    int running = st->used_slots / permits;
    if (running < 0) running = 0;
    return ((double)(ahead + running) / lanes) * local_p50_ms;
}

ofr_choice ofrontier_decide(const ofr_table *t, otier tier, double local_wait_ms, double deadline_ms) {
    if (!t || !t->loaded || tier < TIER_PAID || tier > TIER_BACKGROUND) return OFR_REMOTE;
    double local_eta = local_wait_ms + t->local_p50_ms;
    double remote_eta = t->remote_p50_ms;
    if (!(deadline_ms > 0)) deadline_ms = t->deadline_ms[tier];
    switch (t->policy[tier]) {
    case OFR_POLICY_LOCAL:
        return OFR_LOCAL;
    case OFR_POLICY_FASTEST:
        return remote_eta < local_eta ? OFR_REMOTE : OFR_LOCAL;
    case OFR_POLICY_DEADLINE:
        if (deadline_ms > 0) {
            if (local_eta <= deadline_ms) return OFR_LOCAL;
            if (remote_eta <= deadline_ms) return OFR_REMOTE;
        }
        return remote_eta < local_eta ? OFR_REMOTE : OFR_LOCAL;
    default:
        return OFR_REMOTE;
    }
}

ofrontier *ofrontier_open(const char *policy_path, const char *workload, const char *log_path, int port) {
    if ((!policy_path || !policy_path[0]) && (!log_path || !log_path[0])) return NULL;
    ofrontier *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    snprintf(f->policy_path, sizeof f->policy_path, "%s", policy_path ? policy_path : "");
    snprintf(f->workload, sizeof f->workload, "%s", workload && workload[0] ? workload : "image");
    pthread_mutex_init(&f->lock, NULL);
    pthread_mutex_init(&f->log_lock, NULL);
    ofrontier_table_default(&f->table);
    f->checked_at = -1e9;
    f->port = port;
    if (log_path && log_path[0]) {
        f->log = fopen(log_path, "a");
        if (!f->log) fprintf(stderr, "frontier log %s unusable\n", log_path);
    }
    return f;
}

void ofrontier_close(ofrontier *f) {
    if (!f) return;
    if (f->log) fclose(f->log);
    pthread_mutex_destroy(&f->lock);
    pthread_mutex_destroy(&f->log_lock);
    free(f);
}

const char *ofrontier_workload(const ofrontier *f) {
    return f ? f->workload : "image";
}

static void reload_locked(ofrontier *f) {
    struct stat sb;
    if (!f->policy_path[0] || stat(f->policy_path, &sb) != 0) {
        ofrontier_table_default(&f->table);
        f->mtime = 0;
        return;
    }
    if (f->table.loaded && sb.st_mtim.tv_sec == f->mtime && sb.st_mtim.tv_nsec == f->mtime_ns &&
        sb.st_size == f->size) return;
    if (sb.st_size <= 0 || (size_t)sb.st_size > OFR_MAX_FILE) return;
    FILE *fp = fopen(f->policy_path, "rb");
    if (!fp) return;
    char *buf = malloc((size_t)sb.st_size + 1);
    size_t got = buf ? fread(buf, 1, (size_t)sb.st_size, fp) : 0;
    fclose(fp);
    if (!buf) return;
    ofr_table next;
    if (got == (size_t)sb.st_size && ofrontier_parse(buf, got, f->workload, &next)) {
        f->table = next;
        f->mtime = sb.st_mtim.tv_sec;
        f->mtime_ns = sb.st_mtim.tv_nsec;
        f->size = sb.st_size;
    }
    free(buf);
}

bool ofrontier_snapshot(ofrontier *f, ofr_table *out) {
    if (!f || !out) return false;
    pthread_mutex_lock(&f->lock);
    double now = mono_s();
    if (now - f->checked_at >= 1.0) {
        f->checked_at = now;
        reload_locked(f);
    }
    *out = f->table;
    pthread_mutex_unlock(&f->lock);
    return out->loaded;
}

void ofrontier_log(ofrontier *f, otier tier, const char *backend, const char *reason,
                   double queue_ms, double exec_ms, double local_wait_ms, int status) {
    if (!f || !f->log) return;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    pthread_mutex_lock(&f->log_lock);
    fprintf(f->log,
            "{\"ts\":%.3f,\"workload\":\"%s\",\"tier\":\"%s\",\"backend\":\"%s\",\"reason\":\"%s\","
            "\"queue_ms\":%.1f,\"exec_ms\":%.1f,\"local_wait_ms\":%.1f,\"status\":%d,\"port\":%d}\n",
            (double)ts.tv_sec + (double)ts.tv_nsec / 1e9, f->workload, otier_name(tier),
            backend ? backend : "-", reason ? reason : "-", queue_ms, exec_ms, local_wait_ms, status, f->port);
    fflush(f->log);
    pthread_mutex_unlock(&f->log_lock);
}
