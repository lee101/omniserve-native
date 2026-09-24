#include "ovram.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OVRAM_MAX_LEASES 64
#define OVRAM_MAX_RESERVATIONS 8
#define OVRAM_MAX_OWNERS 32
#define OVRAM_MAX_PROCS 64
#define OVRAM_OWNER_CAP 32
#define OVRAM_ID_CAP 40
#define OVRAM_MAX_WAITERS 64

typedef struct {
    char owner[OVRAM_OWNER_CAP];
    int mb;
    double expires_s;
    double granted_s;
    char id[OVRAM_ID_CAP];
    otier tier;
    int pid;
    int base_mb; /* pid's device usage at grant, -1 unknown */
    bool active;
} ovram_lease_slot;

typedef struct {
    char owner[OVRAM_OWNER_CAP];
    int mb;
    bool active;
} ovram_reservation;

typedef struct {
    char name[OVRAM_OWNER_CAP];
    int pid;
    double last_active_s;
    int peak_mb;
    unsigned long long grants, denials, waits, timeouts;
    double wait_ms_total, wait_ms_max;
    bool used;
} ovram_owner;

struct ovram {
    pthread_mutex_t lock;
    pthread_cond_t changed;
    int keep_free_mb;
    double default_ttl_s;
    ovram_lease_slot leases[OVRAM_MAX_LEASES];
    ovram_reservation reservations[OVRAM_MAX_RESERVATIONS];
    ovram_owner owners[OVRAM_MAX_OWNERS];
    ogpu_proc procs[OVRAM_MAX_PROCS];
    int nprocs;
    bool procs_valid;
    int waiting[4];
    struct { bool used; otier tier; int need; double since; } waiters[OVRAM_MAX_WAITERS];
    double block_max_s;
    ovram_pressure_fn pressure;
    void *pressure_ctx;
    int pressure_after_ms;
    unsigned long long next_id;
    unsigned long long grants;
    unsigned long long partial_grants;
    unsigned long long denials;
    unsigned long long releases;
    unsigned long long expirations;
    unsigned long long waits;
    unsigned long long wait_timeouts;
    unsigned long long pressure_calls;
    long long pressure_freed_mb;
};

static double monotonic_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* One NVML round trip yields both figures, and the driver call serializes on a
 * process-wide lock, so asking twice for the two halves of the same answer
 * doubles what every /metrics scrape costs the tenants trying to lease. */
static void device_mb_now(int *free_mb, int *total_mb) {
    double free_gib = -1.0, total_gib = -1.0;
    bool ok = ogpu_memory_gib(&free_gib, &total_gib);
    if (free_mb) *free_mb = ok && free_gib >= 0.0 ? (int)(free_gib * 1024.0) : -1;
    if (total_mb) *total_mb = ok && total_gib >= 0.0 ? (int)(total_gib * 1024.0) : -1;
}

static int device_free_mb_now(void) {
    int free_mb = -1;
    device_mb_now(&free_mb, NULL);
    return free_mb;
}

/*
 * How far into the keep-free floor a tier may reach. Background batch work is
 * held to the full floor; interactive paid traffic may spend most of it,
 * because the floor exists to keep unbrokered scratch allocations working and
 * a paid request that fails is worse than a scratch allocation that retries.
 */
static int floor_for_tier(int keep_free_mb, otier tier) {
    switch (tier) {
    case TIER_PAID: return keep_free_mb / 4;
    case TIER_SUB: return keep_free_mb / 2;
    case TIER_FREE: return (keep_free_mb * 3) / 4;
    case TIER_BACKGROUND:
    default: return keep_free_mb;
    }
}

static int proc_used_locked(const ovram *v, int pid) {
    if (pid <= 0) return -1;
    for (int i = 0; i < v->nprocs; i++) {
        if (v->procs[i].pid == pid) return v->procs[i].used_mb;
    }
    /* A readable table without the pid means it holds no device memory yet. */
    return v->procs_valid ? 0 : -1;
}

static ovram_owner *owner_locked(ovram *v, const char *name, int pid) {
    int free_slot = -1, oldest = 0;
    for (int i = 0; i < OVRAM_MAX_OWNERS; i++) {
        if (v->owners[i].used && strcmp(v->owners[i].name, name) == 0) {
            if (pid > 0) v->owners[i].pid = pid;
            return &v->owners[i];
        }
        if (!v->owners[i].used && free_slot < 0) free_slot = i;
        if (v->owners[i].last_active_s < v->owners[oldest].last_active_s) oldest = i;
    }
    int slot = free_slot >= 0 ? free_slot : oldest;
    memset(&v->owners[slot], 0, sizeof v->owners[slot]);
    snprintf(v->owners[slot].name, sizeof v->owners[slot].name, "%s", name);
    v->owners[slot].pid = pid;
    v->owners[slot].used = true;
    return &v->owners[slot];
}

static int expire_locked(ovram *v, double now_s) {
    int reclaimed = 0;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (!v->leases[i].active) continue;
        if (v->leases[i].expires_s > now_s) continue;
        v->leases[i].active = false;
        v->expirations++;
        reclaimed++;
    }
    if (reclaimed) pthread_cond_broadcast(&v->changed);
    return reclaimed;
}

/*
 * What a lease still withholds from others. A lease is permission to grow; once
 * the holder has grown, the driver's free figure already shows it, and
 * charging the lease in full as well would count the same bytes twice (a 10 GB
 * Qwen render would then block every other tenant for 20 GB). Growth is the
 * holder pid's usage now minus its usage when its oldest live lease was
 * granted, handed out to that pid's leases in grant order.
 */
static void charges_locked(const ovram *v, int charge[OVRAM_MAX_LEASES]) {
    int pids[OVRAM_MAX_LEASES], left[OVRAM_MAX_LEASES], order[OVRAM_MAX_LEASES];
    int np = 0, no = 0;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        const ovram_lease_slot *l = &v->leases[i];
        charge[i] = l->active ? l->mb : 0;
        if (!l->active || l->pid <= 0 || l->base_mb < 0) continue;
        order[no++] = i;
        bool seen = false;
        for (int k = 0; k < np; k++) if (pids[k] == l->pid) seen = true;
        if (seen) continue;
        int now_mb = proc_used_locked(v, l->pid);
        if (now_mb < 0) continue;
        int base = l->base_mb;
        for (int j = 0; j < OVRAM_MAX_LEASES; j++) {
            const ovram_lease_slot *o = &v->leases[j];
            if (o->active && o->pid == l->pid && o->base_mb >= 0 && o->base_mb < base) base = o->base_mb;
        }
        pids[np] = l->pid;
        left[np] = now_mb > base ? now_mb - base : 0;
        np++;
    }
    /* Earlier grants absorb growth first. */
    for (int a = 1; a < no; a++) {
        int x = order[a], b = a - 1;
        while (b >= 0 && v->leases[order[b]].granted_s > v->leases[x].granted_s) {
            order[b + 1] = order[b];
            b--;
        }
        order[b + 1] = x;
    }
    for (int a = 0; a < no; a++) {
        int i = order[a];
        for (int k = 0; k < np; k++) {
            if (pids[k] != v->leases[i].pid) continue;
            int credit = left[k] < v->leases[i].mb ? left[k] : v->leases[i].mb;
            left[k] -= credit;
            charge[i] = v->leases[i].mb - credit;
        }
    }
}

static int leased_mb_locked(const ovram *v) {
    int charge[OVRAM_MAX_LEASES];
    charges_locked(v, charge);
    int total = 0;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) total += charge[i];
    return total;
}

static int leased_mb_for_tier_locked(const ovram *v, otier tier) {
    int charge[OVRAM_MAX_LEASES];
    charges_locked(v, charge);
    int total = 0;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (!v->leases[i].active) continue;
        if (v->leases[i].tier <= tier) total += charge[i];
    }
    return total;
}

static int reserved_mb_locked(const ovram *v) {
    int total = 0;
    for (int i = 0; i < OVRAM_MAX_RESERVATIONS; i++) {
        if (v->reservations[i].active) total += v->reservations[i].mb;
    }
    return total;
}

static int headroom_locked(const ovram *v, otier tier, int device_free_mb) {
    if (device_free_mb < 0) return 0;
    int available = device_free_mb - floor_for_tier(v->keep_free_mb, tier)
                    - reserved_mb_locked(v) - leased_mb_for_tier_locked(v, tier);
    return available > 0 ? available : 0;
}

ovram *ovram_create(int keep_free_mb, double default_ttl_s) {
    ovram *v = calloc(1, sizeof *v);
    if (!v) return NULL;
    if (pthread_mutex_init(&v->lock, NULL) != 0) {
        free(v);
        return NULL;
    }
    pthread_condattr_t attr;
    pthread_condattr_init(&attr);
    pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    if (pthread_cond_init(&v->changed, &attr) != 0) {
        pthread_condattr_destroy(&attr);
        pthread_mutex_destroy(&v->lock);
        free(v);
        return NULL;
    }
    pthread_condattr_destroy(&attr);
    v->keep_free_mb = keep_free_mb > 0 ? keep_free_mb : 0;
    v->default_ttl_s = default_ttl_s > 0.0 ? default_ttl_s : 120.0;
    v->pressure_after_ms = 1000;
    v->block_max_s = 15.0;
    v->next_id = 1;
    return v;
}

void ovram_destroy(ovram *v) {
    if (!v) return;
    pthread_cond_destroy(&v->changed);
    pthread_mutex_destroy(&v->lock);
    free(v);
}

void ovram_set_pressure_hook(ovram *v, ovram_pressure_fn fn, void *ctx, int after_ms) {
    if (!v) return;
    pthread_mutex_lock(&v->lock);
    v->pressure = fn;
    v->pressure_ctx = ctx;
    v->pressure_after_ms = after_ms >= 0 ? after_ms : 0;
    pthread_mutex_unlock(&v->lock);
}

void ovram_set_block_max_s(ovram *v, double s) {
    if (!v) return;
    pthread_mutex_lock(&v->lock);
    v->block_max_s = s;
    pthread_mutex_unlock(&v->lock);
}

void ovram_set_procs(ovram *v, const ogpu_proc *procs, int n) {
    if (!v) return;
    pthread_mutex_lock(&v->lock);
    v->nprocs = 0;
    v->procs_valid = procs != NULL && n >= 0;
    for (int i = 0; procs && i < n && i < OVRAM_MAX_PROCS; i++) v->procs[v->nprocs++] = procs[i];
    pthread_mutex_unlock(&v->lock);
}

/* Refresh the per-process table outside the broker lock: NVML serializes on
 * its own lock and must never be waited on while tenants queue on ours. */
static int refresh_procs(ovram *v) {
    ogpu_proc procs[OVRAM_MAX_PROCS];
    int n = ogpu_processes(procs, OVRAM_MAX_PROCS);
    if (n >= 0) ovram_set_procs(v, procs, n);
    return n;
}

bool ovram_reserve(ovram *v, const char *owner, int mb) {
    if (!v || !owner || !owner[0] || mb < 0) return false;
    pthread_mutex_lock(&v->lock);
    int slot = -1;
    for (int i = 0; i < OVRAM_MAX_RESERVATIONS; i++) {
        if (v->reservations[i].active && strcmp(v->reservations[i].owner, owner) == 0) {
            slot = i;
            break;
        }
        if (slot < 0 && !v->reservations[i].active) slot = i;
    }
    if (slot < 0) {
        pthread_mutex_unlock(&v->lock);
        return false;
    }
    snprintf(v->reservations[slot].owner, sizeof v->reservations[slot].owner, "%s", owner);
    v->reservations[slot].mb = mb;
    v->reservations[slot].active = mb > 0;
    pthread_cond_broadcast(&v->changed);
    pthread_mutex_unlock(&v->lock);
    return true;
}

/*
 * A queued higher tier holds back lower tiers only while holding back helps:
 * its need must be coverable by current headroom plus what live leases will
 * hand back, and it must not have been blocking for longer than block_max_s.
 * Otherwise a paid job that cannot fit until a resident is evicted would
 * starve every lower tier (head-of-line blocking) for its whole wait.
 */
static bool higher_tier_waiting_locked(const ovram *v, otier tier, double now_s, int device_free_mb) {
    int leased_total = 0;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (v->leases[i].active) leased_total += v->leases[i].mb;
    }
    for (int i = 0; i < OVRAM_MAX_WAITERS; i++) {
        if (!v->waiters[i].used || v->waiters[i].tier >= tier) continue;
        if (v->block_max_s > 0 && now_s - v->waiters[i].since > v->block_max_s) continue;
        if (device_free_mb >= 0 &&
            v->waiters[i].need > headroom_locked(v, v->waiters[i].tier, device_free_mb) + leased_total) continue;
        return true;
    }
    return false;
}

/* Returns granted MB (0 = does not fit). Caller holds the lock and has expired. */
static int grant_locked(ovram *v, const char *owner, int pid, int mb, int min_mb, otier tier,
                        double ttl_s, double now_s, int device_free_mb,
                        char *id_out, size_t id_cap) {
    int available = headroom_locked(v, tier, device_free_mb);
    int granted = mb < available ? mb : available;
    if (granted < min_mb || granted <= 0) return 0;
    int slot = -1;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (!v->leases[i].active) { slot = i; break; }
    }
    /* Out of slots is a denial, not an error: the caller's fallback path is
     * already the correct behaviour for "no headroom for you". */
    if (slot < 0) return 0;

    const char *name = owner && owner[0] ? owner : "anon";
    ovram_lease_slot *lease = &v->leases[slot];
    snprintf(lease->owner, sizeof lease->owner, "%s", name);
    lease->mb = granted;
    lease->expires_s = now_s + (ttl_s > 0.0 ? ttl_s : v->default_ttl_s);
    lease->granted_s = now_s;
    snprintf(lease->id, sizeof lease->id, "lv%llu", v->next_id++);
    lease->tier = tier;
    lease->pid = pid;
    lease->base_mb = proc_used_locked(v, pid);
    lease->active = true;

    v->grants++;
    if (granted < mb) v->partial_grants++;
    ovram_owner *o = owner_locked(v, name, pid);
    o->grants++;
    o->last_active_s = now_s;
    if (granted > o->peak_mb) o->peak_mb = granted;
    if (id_out && id_cap) snprintf(id_out, id_cap, "%s", lease->id);
    return granted;
}

int ovram_lease_pid_at(ovram *v, const char *owner, int pid, int mb, int min_mb, otier tier,
                       double ttl_s, double now_s, int device_free_mb,
                       char *id_out, size_t id_cap) {
    if (id_out && id_cap) id_out[0] = '\0';
    if (!v || mb <= 0) return 0;
    if (min_mb < 0) min_mb = 0;
    if (min_mb > mb) min_mb = mb;

    pthread_mutex_lock(&v->lock);
    expire_locked(v, now_s);
    /* A lower tier may not take headroom a queued higher tier is waiting for. */
    int granted = higher_tier_waiting_locked(v, tier, now_s, device_free_mb) ? 0
        : grant_locked(v, owner, pid, mb, min_mb, tier, ttl_s, now_s, device_free_mb, id_out, id_cap);
    if (granted <= 0) {
        v->denials++;
        ovram_owner *o = owner_locked(v, owner && owner[0] ? owner : "anon", pid);
        o->denials++;
        o->last_active_s = now_s;
    }
    pthread_mutex_unlock(&v->lock);
    return granted;
}

int ovram_lease_at(ovram *v, const char *owner, int mb, int min_mb, otier tier,
                   double ttl_s, double now_s, int device_free_mb,
                   char *id_out, size_t id_cap) {
    return ovram_lease_pid_at(v, owner, 0, mb, min_mb, tier, ttl_s, now_s, device_free_mb,
                              id_out, id_cap);
}

int ovram_lease(ovram *v, const char *owner, int mb, int min_mb, otier tier,
                double ttl_s, char *id_out, size_t id_cap) {
    return ovram_lease_wait(v, owner, 0, mb, min_mb, tier, ttl_s, 0, id_out, id_cap, NULL);
}

int ovram_lease_wait(ovram *v, const char *owner, int pid, int mb, int min_mb, otier tier,
                     double ttl_s, int wait_ms, char *id_out, size_t id_cap,
                     int *waited_ms_out) {
    if (id_out && id_cap) id_out[0] = '\0';
    if (waited_ms_out) *waited_ms_out = 0;
    if (!v || mb <= 0) return 0;
    if (min_mb < 0) min_mb = 0;
    if (min_mb > mb) min_mb = mb;
    if ((int)tier < TIER_PAID || (int)tier > TIER_BACKGROUND) tier = TIER_BACKGROUND;
    const char *name = owner && owner[0] ? owner : "anon";
    double started = monotonic_s();
    double deadline = started + (wait_ms > 0 ? wait_ms / 1000.0 : 0.0);
    double next_pressure = started + v->pressure_after_ms / 1000.0;
    bool registered = false;
    int wslot = -1;
    int granted = 0;

    for (;;) {
        refresh_procs(v);
        int device_free = device_free_mb_now();
        double now = monotonic_s();
        pthread_mutex_lock(&v->lock);
        expire_locked(v, now);
        if (!higher_tier_waiting_locked(v, tier, now, device_free)) {
            granted = grant_locked(v, name, pid, mb, min_mb, tier, ttl_s, now, device_free,
                                   id_out, id_cap);
        }
        if (granted > 0 || now >= deadline) {
            ovram_owner *o = owner_locked(v, name, pid);
            double waited = (now - started) * 1000.0;
            if (registered) {
                v->waiting[tier]--;
                if (wslot >= 0) v->waiters[wslot].used = false;
                o->waits++;
                o->wait_ms_total += waited;
                if (waited > o->wait_ms_max) o->wait_ms_max = waited;
                pthread_cond_broadcast(&v->changed);
            }
            if (granted <= 0) {
                v->denials++;
                o->denials++;
                o->last_active_s = now;
                if (registered) { v->wait_timeouts++; o->timeouts++; }
            }
            pthread_mutex_unlock(&v->lock);
            if (waited_ms_out) *waited_ms_out = (int)waited;
            return granted;
        }
        if (!registered) {
            registered = true;
            v->waiting[tier]++;
            for (int i = 0; i < OVRAM_MAX_WAITERS; i++) {
                if (v->waiters[i].used) continue;
                v->waiters[i].used = true;
                v->waiters[i].tier = tier;
                v->waiters[i].need = min_mb > 0 ? min_mb : mb;
                v->waiters[i].since = now;
                wslot = i;
                break;
            }
            v->waits++;
        }
        int need = min_mb > 0 ? min_mb : mb;
        int deficit = need - headroom_locked(v, tier, device_free);
        ovram_pressure_fn hook = v->pressure;
        void *hook_ctx = v->pressure_ctx;
        pthread_mutex_unlock(&v->lock);

        if (hook && deficit > 0 && now >= next_pressure) {
            int freed = hook(hook_ctx, tier, deficit);
            pthread_mutex_lock(&v->lock);
            v->pressure_calls++;
            if (freed > 0) v->pressure_freed_mb += freed;
            pthread_mutex_unlock(&v->lock);
            next_pressure = monotonic_s() + 5.0;
            if (freed > 0) continue;
        }

        /* External tenants free memory without telling us, so wake on a short
         * poll as well as on our own releases. */
        double wake = monotonic_s() + 0.2;
        if (wake > deadline) wake = deadline;
        struct timespec ts;
        ts.tv_sec = (time_t)wake;
        ts.tv_nsec = (long)((wake - (double)ts.tv_sec) * 1e9);
        pthread_mutex_lock(&v->lock);
        (void)pthread_cond_timedwait(&v->changed, &v->lock, &ts);
        pthread_mutex_unlock(&v->lock);
    }
}

double ovram_default_ttl_s(const ovram *v) {
    return v ? v->default_ttl_s : 0.0;
}

bool ovram_release(ovram *v, const char *id) {
    if (!v || !id || !id[0]) return false;
    pthread_mutex_lock(&v->lock);
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (!v->leases[i].active) continue;
        if (strcmp(v->leases[i].id, id) != 0) continue;
        v->leases[i].active = false;
        v->releases++;
        ovram_owner *o = owner_locked(v, v->leases[i].owner, v->leases[i].pid);
        o->last_active_s = monotonic_s();
        pthread_cond_broadcast(&v->changed);
        pthread_mutex_unlock(&v->lock);
        return true;
    }
    pthread_mutex_unlock(&v->lock);
    return false;
}

bool ovram_renew_at(ovram *v, const char *id, double ttl_s, double now_s) {
    if (!v || !id || !id[0]) return false;
    pthread_mutex_lock(&v->lock);
    expire_locked(v, now_s);
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (!v->leases[i].active || strcmp(v->leases[i].id, id) != 0) continue;
        v->leases[i].expires_s = now_s + (ttl_s > 0.0 ? ttl_s : v->default_ttl_s);
        pthread_mutex_unlock(&v->lock);
        return true;
    }
    pthread_mutex_unlock(&v->lock);
    return false;
}

bool ovram_renew(ovram *v, const char *id, double ttl_s) {
    return ovram_renew_at(v, id, ttl_s, monotonic_s());
}

int ovram_expire_at(ovram *v, double now_s) {
    if (!v) return 0;
    pthread_mutex_lock(&v->lock);
    int reclaimed = expire_locked(v, now_s);
    pthread_mutex_unlock(&v->lock);
    return reclaimed;
}

int ovram_headroom_at(ovram *v, otier tier, double now_s, int device_free_mb) {
    if (!v) return 0;
    pthread_mutex_lock(&v->lock);
    expire_locked(v, now_s);
    int headroom = headroom_locked(v, tier, device_free_mb);
    pthread_mutex_unlock(&v->lock);
    return headroom;
}

int ovram_headroom(ovram *v, otier tier) {
    if (!v) return 0;
    refresh_procs(v);
    return ovram_headroom_at(v, tier, monotonic_s(), device_free_mb_now());
}

int ovram_waiting(ovram *v, otier tier) {
    if (!v || (int)tier < TIER_PAID || (int)tier > TIER_BACKGROUND) return 0;
    pthread_mutex_lock(&v->lock);
    int n = v->waiting[tier];
    pthread_mutex_unlock(&v->lock);
    return n;
}

void ovram_snapshot(ovram *v, ovram_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!v) return;
    int device_free = -1, total = -1;
    device_mb_now(&device_free, &total);
    refresh_procs(v);
    double now_s = monotonic_s();

    pthread_mutex_lock(&v->lock);
    expire_locked(v, now_s);
    out->total_mb = total;
    out->device_free_mb = device_free;
    out->reserved_mb = reserved_mb_locked(v);
    out->leased_mb = leased_mb_locked(v);
    out->headroom_mb = headroom_locked(v, TIER_BACKGROUND, device_free);
    out->keep_free_mb = v->keep_free_mb;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (v->leases[i].active) out->lease_count++;
    }
    for (int t = 0; t < 4; t++) out->waiting += v->waiting[t];
    out->grants = v->grants;
    out->partial_grants = v->partial_grants;
    out->denials = v->denials;
    out->releases = v->releases;
    out->expirations = v->expirations;
    out->waits = v->waits;
    out->wait_timeouts = v->wait_timeouts;
    out->pressure_calls = v->pressure_calls;
    out->pressure_freed_mb = v->pressure_freed_mb;
    pthread_mutex_unlock(&v->lock);
}

size_t ovram_status_json(ovram *v, char *out, size_t cap) {
    if (!out || cap == 0) return 0;
    ovram_stats s;
    ovram_snapshot(v, &s);
    int written = snprintf(out, cap,
        "{\"total_mb\":%d,\"device_free_mb\":%d,\"reserved_mb\":%d,\"leased_mb\":%d,"
        "\"headroom_mb\":%d,\"keep_free_mb\":%d,\"lease_count\":%d,"
        "\"grants\":%llu,\"partial_grants\":%llu,\"denials\":%llu,"
        "\"releases\":%llu,\"expirations\":%llu}",
        s.total_mb, s.device_free_mb, s.reserved_mb, s.leased_mb,
        s.headroom_mb, s.keep_free_mb, s.lease_count,
        s.grants, s.partial_grants, s.denials, s.releases, s.expirations);
    if (written < 0) return 0;
    return (size_t)written < cap ? (size_t)written : cap - 1;
}

#define APPEND(...) do { \
    if (len < cap) { \
        int n_ = snprintf(out + len, cap - len, __VA_ARGS__); \
        if (n_ > 0) len += (size_t)n_; \
    } \
} while (0)

/* Systemd unit (last cgroup path component) naming a GPU process. */
static void unit_for_pid(int pid, char *out, size_t cap) {
    snprintf(out, cap, "pid%d", pid);
    char path[64];
    snprintf(path, sizeof path, "/proc/%d/cgroup", pid);
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[512];
    if (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        const char *slash = strrchr(line, '/');
        const char *name = slash ? slash + 1 : line;
        if (name[0]) {
            size_t k = 0;
            for (; name[k] && k + 1 < cap; k++) {
                char c = name[k];
                out[k] = (c == '"' || c == '\\' || (unsigned char)c < 0x20) ? '_' : c;
            }
            out[k] = 0;
        }
    }
    fclose(f);
}

static void comm_for_pid(int pid, char *out, size_t cap) {
    out[0] = 0;
    char path[64];
    snprintf(path, sizeof path, "/proc/%d/comm", pid);
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(out, (int)cap, f)) {
        out[strcspn(out, "\r\n")] = 0;
        for (char *c = out; *c; c++) if (*c == '"' || *c == '\\' || (unsigned char)*c < 0x20) *c = '_';
    }
    fclose(f);
}

size_t ovram_ledger_json(ovram *v, char *out, size_t cap) {
    if (!out || cap == 0) return 0;
    out[0] = 0;
    if (!v) return 0;
    ovram_stats s;
    ovram_snapshot(v, &s);
    double now = monotonic_s();
    int device_free = s.device_free_mb;
    size_t len = 0;

    pthread_mutex_lock(&v->lock);
    int charge[OVRAM_MAX_LEASES];
    charges_locked(v, charge);
    APPEND("{\"total_mb\":%d,\"device_free_mb\":%d,\"keep_free_mb\":%d,\"reserved_mb\":%d,"
           "\"leased_mb\":%d,\"headroom_mb\":{\"paid\":%d,\"sub\":%d,\"free\":%d,\"background\":%d},"
           "\"waiting\":{\"paid\":%d,\"sub\":%d,\"free\":%d,\"background\":%d},",
           s.total_mb, device_free, v->keep_free_mb, s.reserved_mb, s.leased_mb,
           headroom_locked(v, TIER_PAID, device_free), headroom_locked(v, TIER_SUB, device_free),
           headroom_locked(v, TIER_FREE, device_free), headroom_locked(v, TIER_BACKGROUND, device_free),
           v->waiting[0], v->waiting[1], v->waiting[2], v->waiting[3]);
    APPEND("\"counters\":{\"grants\":%llu,\"partial\":%llu,\"denials\":%llu,\"releases\":%llu,"
           "\"expirations\":%llu,\"waits\":%llu,\"wait_timeouts\":%llu,\"pressure_calls\":%llu,"
           "\"pressure_freed_mb\":%lld},",
           v->grants, v->partial_grants, v->denials, v->releases, v->expirations,
           v->waits, v->wait_timeouts, v->pressure_calls, v->pressure_freed_mb);
    APPEND("\"leases\":[");
    bool first = true;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        const ovram_lease_slot *l = &v->leases[i];
        if (!l->active) continue;
        APPEND("%s{\"id\":\"%s\",\"owner\":\"%s\",\"pid\":%d,\"tier\":\"%s\",\"mb\":%d,"
               "\"charged_mb\":%d,\"age_s\":%.1f,\"expires_in_s\":%.1f}",
               first ? "" : ",", l->id, l->owner, l->pid, otier_name(l->tier), l->mb,
               charge[i], now - l->granted_s, l->expires_s - now);
        first = false;
    }
    APPEND("],\"reservations\":[");
    first = true;
    for (int i = 0; i < OVRAM_MAX_RESERVATIONS; i++) {
        if (!v->reservations[i].active) continue;
        APPEND("%s{\"owner\":\"%s\",\"mb\":%d}", first ? "" : ",",
               v->reservations[i].owner, v->reservations[i].mb);
        first = false;
    }
    APPEND("],\"owners\":[");
    first = true;
    for (int i = 0; i < OVRAM_MAX_OWNERS; i++) {
        const ovram_owner *o = &v->owners[i];
        if (!o->used) continue;
        APPEND("%s{\"owner\":\"%s\",\"pid\":%d,\"idle_s\":%.1f,\"peak_lease_mb\":%d,"
               "\"grants\":%llu,\"denials\":%llu,\"waits\":%llu,\"timeouts\":%llu,"
               "\"wait_ms_avg\":%.0f,\"wait_ms_max\":%.0f}",
               first ? "" : ",", o->name, o->pid, now - o->last_active_s, o->peak_mb,
               o->grants, o->denials, o->waits, o->timeouts,
               o->waits ? o->wait_ms_total / (double)o->waits : 0.0, o->wait_ms_max);
        first = false;
    }
    ogpu_proc procs[OVRAM_MAX_PROCS];
    int np = v->nprocs;
    memcpy(procs, v->procs, sizeof procs[0] * (size_t)np);
    pthread_mutex_unlock(&v->lock);

    APPEND("],\"tenants\":[");
    for (int i = 0; i < np; i++) {
        char unit[96], comm[40];
        unit_for_pid(procs[i].pid, unit, sizeof unit);
        comm_for_pid(procs[i].pid, comm, sizeof comm);
        APPEND("%s{\"pid\":%d,\"unit\":\"%s\",\"comm\":\"%s\",\"used_mb\":%d}", i ? "," : "",
               procs[i].pid, unit, comm, procs[i].used_mb);
    }
    APPEND("]}");
    if (len >= cap) {
        /* Truncated JSON is worse than none. */
        snprintf(out, cap, "{\"error\":\"ledger too large\"}");
        return strlen(out);
    }
    return len;
}
