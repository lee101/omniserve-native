#include "ovram.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OVRAM_MAX_LEASES 64
#define OVRAM_MAX_RESERVATIONS 8
#define OVRAM_OWNER_CAP 32
#define OVRAM_ID_CAP 40

typedef struct {
    char owner[OVRAM_OWNER_CAP];
    int mb;
    double expires_s;
    char id[OVRAM_ID_CAP];
    otier tier;
    bool active;
} ovram_lease_slot;

typedef struct {
    char owner[OVRAM_OWNER_CAP];
    int mb;
    bool active;
} ovram_reservation;

struct ovram {
    pthread_mutex_t lock;
    int keep_free_mb;
    double default_ttl_s;
    ovram_lease_slot leases[OVRAM_MAX_LEASES];
    ovram_reservation reservations[OVRAM_MAX_RESERVATIONS];
    unsigned long long next_id;
    unsigned long long grants;
    unsigned long long partial_grants;
    unsigned long long denials;
    unsigned long long releases;
    unsigned long long expirations;
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

static int expire_locked(ovram *v, double now_s) {
    int reclaimed = 0;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (!v->leases[i].active) continue;
        if (v->leases[i].expires_s > now_s) continue;
        v->leases[i].active = false;
        v->expirations++;
        reclaimed++;
    }
    return reclaimed;
}

static int leased_mb_locked(const ovram *v) {
    int total = 0;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (v->leases[i].active) total += v->leases[i].mb;
    }
    return total;
}

static int leased_mb_for_tier_locked(const ovram *v, otier tier) {
    int total = 0;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (!v->leases[i].active) continue;
        if (v->leases[i].tier <= tier) total += v->leases[i].mb;
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
    v->keep_free_mb = keep_free_mb > 0 ? keep_free_mb : 0;
    v->default_ttl_s = default_ttl_s > 0.0 ? default_ttl_s : 120.0;
    v->next_id = 1;
    return v;
}

void ovram_destroy(ovram *v) {
    if (!v) return;
    pthread_mutex_destroy(&v->lock);
    free(v);
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
    pthread_mutex_unlock(&v->lock);
    return true;
}

int ovram_lease_at(ovram *v, const char *owner, int mb, int min_mb, otier tier,
                   double ttl_s, double now_s, int device_free_mb,
                   char *id_out, size_t id_cap) {
    if (id_out && id_cap) id_out[0] = '\0';
    if (!v || mb <= 0) return 0;
    if (min_mb < 0) min_mb = 0;
    if (min_mb > mb) min_mb = mb;

    pthread_mutex_lock(&v->lock);
    expire_locked(v, now_s);

    int available = headroom_locked(v, tier, device_free_mb);
    int granted = mb < available ? mb : available;
    if (granted < min_mb || granted <= 0) {
        v->denials++;
        pthread_mutex_unlock(&v->lock);
        return 0;
    }

    int slot = -1;
    for (int i = 0; i < OVRAM_MAX_LEASES; i++) {
        if (!v->leases[i].active) { slot = i; break; }
    }
    if (slot < 0) {
        /* Out of slots is a denial, not an error: the caller's fallback path
         * is already the correct behaviour for "no headroom for you". */
        v->denials++;
        pthread_mutex_unlock(&v->lock);
        return 0;
    }

    ovram_lease_slot *lease = &v->leases[slot];
    snprintf(lease->owner, sizeof lease->owner, "%s", owner && owner[0] ? owner : "anon");
    lease->mb = granted;
    lease->expires_s = now_s + (ttl_s > 0.0 ? ttl_s : v->default_ttl_s);
    snprintf(lease->id, sizeof lease->id, "lv%llu", v->next_id++);
    lease->tier = tier;
    lease->active = true;

    v->grants++;
    if (granted < mb) v->partial_grants++;
    if (id_out && id_cap) snprintf(id_out, id_cap, "%s", lease->id);
    pthread_mutex_unlock(&v->lock);
    return granted;
}

int ovram_lease(ovram *v, const char *owner, int mb, int min_mb, otier tier,
                double ttl_s, char *id_out, size_t id_cap) {
    return ovram_lease_at(v, owner, mb, min_mb, tier, ttl_s, monotonic_s(),
                          device_free_mb_now(), id_out, id_cap);
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

void ovram_snapshot(ovram *v, ovram_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!v) return;
    int device_free = -1, total = -1;
    device_mb_now(&device_free, &total);
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
    out->grants = v->grants;
    out->partial_grants = v->partial_grants;
    out->denials = v->denials;
    out->releases = v->releases;
    out->expirations = v->expirations;
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
