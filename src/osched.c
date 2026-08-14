#define _GNU_SOURCE
#include "osched.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

/*
 * One queued request.
 *
 * The condition variable is per waiter, not shared, and that is the whole point.
 * A single broadcast queue costs O(n) wakeups per release and each woken thread
 * then walks the list to find out whether it is the head, so a release under n
 * queued requests is O(n^2) of contended work on one mutex - and n-1 of those
 * threads go straight back to sleep. Saturation is exactly when the queue is
 * long, so the scheduler got most expensive precisely when it had least to
 * spare. Here a releasing thread grants permits directly to the waiters that fit
 * and signals only those, so a release wakes exactly the threads that are about
 * to run.
 *
 * Allocation is on the waiter's own stack: it is linked in and unlinked again
 * entirely within one osched_acquire_n call, so its lifetime is that frame.
 */
typedef struct waiter {
    otier tier;
    int permits;
    long seq;
    struct timespec queued_at;
    pthread_cond_t cond;
    bool granted;
    struct waiter *next;
} waiter;

struct osched {
    pthread_mutex_t lock;
    /* No shared condition variable: waiters each own one, so a release signals
     * only the threads it just admitted. */
    pthread_condattr_t condattr;
    int slots;
    int used_slots;
    int active;
    double timeout_s;
    long seq;
    waiter *waiters;
    long served[4];
    double queue_ms_total[4];
    double queue_ms_max[4];
    uint64_t timed_out[4];
};

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 +
           (double)(end.tv_nsec - start.tv_nsec) / 1e6;
}

osched *osched_create(int slots, double admission_timeout_s) {
    osched *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    pthread_mutex_init(&s->lock, NULL);
    /* Held for the lifetime of the scheduler because every waiter initialises
     * its condvar from it. The monotonic clock has to match the deadline, or a
     * wall-clock step would move every admission timeout. */
    pthread_condattr_init(&s->condattr);
    pthread_condattr_setclock(&s->condattr, CLOCK_MONOTONIC);
    s->slots = slots > 0 ? slots : 1;
    s->timeout_s = admission_timeout_s > 0 ? admission_timeout_s : 600;
    return s;
}

void osched_destroy(osched *s) {
    if (!s) return;
    pthread_mutex_destroy(&s->lock);
    pthread_condattr_destroy(&s->condattr);
    free(s);
}

otier otier_parse(const char *value, int len) {
    if (!value || len <= 0) return TIER_FREE;
    if (len == 4 && strncasecmp(value, "paid", 4) == 0) return TIER_PAID;
    if (len == 3 && strncasecmp(value, "sub", 3) == 0) return TIER_SUB;
    if (len == 10 && strncasecmp(value, "background", 10) == 0) return TIER_BACKGROUND;
    return TIER_FREE;
}

otier otier_parse_public(const char *value, int len) {
    otier tier = otier_parse(value, len);
    return tier == TIER_BACKGROUND ? TIER_FREE : tier;
}

const char *otier_name(otier t) {
    switch (t) {
    case TIER_PAID: return "paid";
    case TIER_SUB: return "sub";
    case TIER_FREE: return "free";
    default: return "background";
    }
}

static void unlink_waiter(osched *s, waiter *me) {
    waiter **p = &s->waiters;
    while (*p && *p != me) p = &(*p)->next;
    if (*p) *p = me->next;
}

/* Insertion sort into a list kept in service order: tier ascending (paid first),
 * then sequence ascending within a tier. The list is short - it is bounded by
 * the number of in-flight HTTP requests - and keeping it ordered on insert is
 * what makes the head O(1) to find on every release. */
static void link_waiter(osched *s, waiter *me) {
    waiter **p = &s->waiters;
    while (*p && ((*p)->tier < me->tier ||
                  ((*p)->tier == me->tier && (*p)->seq < me->seq))) {
        p = &(*p)->next;
    }
    me->next = *p;
    *p = me;
}

static bool fits_locked(const osched *s, otier tier, int permits) {
    /* Background work reserves the whole device: it is the lane that swaps
     * embedded models out, so it cannot share with a request that expects them
     * loaded. */
    return tier == TIER_BACKGROUND ? (s->active == 0 && s->used_slots == 0)
                                   : (s->used_slots + permits <= s->slots);
}

/* Hands slots to the front of the queue and wakes only those waiters.
 *
 * The loop stops at the first waiter that does not fit rather than skipping it,
 * which preserves the previous head-of-line rule exactly: a cheap request never
 * overtakes an expensive one that is already waiting, so a lane asking for the
 * whole device cannot be starved by a stream of single-permit callers. */
static void promote_locked(osched *s) {
    while (s->waiters) {
        waiter *head = s->waiters;
        if (!fits_locked(s, head->tier, head->permits)) break;
        s->waiters = head->next;
        s->active++;
        s->used_slots += head->permits;
        s->served[head->tier]++;
        head->granted = true;
        pthread_cond_signal(&head->cond);
    }
}

bool osched_try_acquire_n(osched *s, otier tier, int permits) {
    if (!s || tier < TIER_PAID || tier > TIER_BACKGROUND || permits <= 0 || permits > s->slots) {
        return false;
    }
    pthread_mutex_lock(&s->lock);
    /* Deliberately refuses when anyone is queued, even with slots free. This
     * is the admission point for callers that have somewhere else to go, and
     * jumping the queue would let them take the slot the waiter has already
     * paid for in latency. */
    if (s->waiters || !fits_locked(s, tier, permits)) {
        pthread_mutex_unlock(&s->lock);
        return false;
    }
    s->active++;
    s->used_slots += permits;
    s->served[tier]++;
    pthread_mutex_unlock(&s->lock);
    return true;
}

bool osched_acquire_n(osched *s, otier tier, int permits) {
    if (!s || tier < TIER_PAID || tier > TIER_BACKGROUND || permits <= 0 || permits > s->slots) {
        return false;
    }
    pthread_mutex_lock(&s->lock);
    if (!s->waiters && fits_locked(s, tier, permits)) {
        s->active++;
        s->used_slots += permits;
        s->served[tier]++;
        pthread_mutex_unlock(&s->lock);
        return true;
    }

    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    time_t whole = (time_t)s->timeout_s;
    long nanos = (long)((s->timeout_s - (double)whole) * 1e9);
    deadline.tv_sec += whole;
    deadline.tv_nsec += nanos;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }

    waiter me = {0};
    me.tier = tier;
    me.permits = permits;
    me.seq = ++s->seq;
    clock_gettime(CLOCK_MONOTONIC, &me.queued_at);
    if (pthread_cond_init(&me.cond, &s->condattr) != 0) {
        pthread_mutex_unlock(&s->lock);
        return false;
    }
    link_waiter(s, &me);

    bool timed_out = false;
    while (!me.granted) {
        if (pthread_cond_timedwait(&me.cond, &s->lock, &deadline) != 0 && !me.granted) {
            timed_out = true;
            break;
        }
    }

    if (timed_out) {
        /* Still queued, so nothing was handed to us: drop out and let whoever
         * is now at the front take the slot this request was holding a place
         * for. */
        unlink_waiter(s, &me);
        s->timed_out[tier]++;
        promote_locked(s);
        pthread_mutex_unlock(&s->lock);
        pthread_cond_destroy(&me.cond);
        return false;
    }

    /* promote_locked already unlinked us and charged the permits. */
    struct timespec admitted_at;
    clock_gettime(CLOCK_MONOTONIC, &admitted_at);
    double queued_ms = elapsed_ms(me.queued_at, admitted_at);
    s->queue_ms_total[tier] += queued_ms;
    if (queued_ms > s->queue_ms_max[tier]) s->queue_ms_max[tier] = queued_ms;
    pthread_mutex_unlock(&s->lock);
    pthread_cond_destroy(&me.cond);
    return true;
}

bool osched_acquire(osched *s, otier tier) {
    return osched_acquire_n(s, tier, 1);
}

void osched_release_n(osched *s, otier tier, int permits) {
    (void)tier;
    if (!s || permits <= 0) return;
    pthread_mutex_lock(&s->lock);
    if (s->active > 0) s->active--;
    s->used_slots = s->used_slots >= permits ? s->used_slots - permits : 0;
    promote_locked(s);
    pthread_mutex_unlock(&s->lock);
}

void osched_release(osched *s, otier tier) {
    osched_release_n(s, tier, 1);
}

int osched_capacity(const osched *s) {
    return s ? s->slots : 0;
}

void osched_snapshot(const osched *s, osched_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!s) return;
    pthread_mutex_lock((pthread_mutex_t *)&s->lock);
    out->slots = s->slots;
    out->used_slots = s->used_slots;
    out->active = s->active;
    for (const waiter *w = s->waiters; w; w = w->next) out->waiting[w->tier]++;
    memcpy(out->served, s->served, sizeof out->served);
    memcpy(out->queue_ms_total, s->queue_ms_total, sizeof out->queue_ms_total);
    memcpy(out->queue_ms_max, s->queue_ms_max, sizeof out->queue_ms_max);
    memcpy(out->timed_out, s->timed_out, sizeof out->timed_out);
    pthread_mutex_unlock((pthread_mutex_t *)&s->lock);
}

int osched_active(const osched *s) {
    osched_stats st;
    osched_snapshot(s, &st);
    return st.active;
}

int osched_waiting(const osched *s, otier tier) {
    osched_stats st;
    osched_snapshot(s, &st);
    return tier >= TIER_PAID && tier <= TIER_BACKGROUND ? st.waiting[tier] : 0;
}

long osched_served(const osched *s, otier tier) {
    osched_stats st;
    osched_snapshot(s, &st);
    return tier >= TIER_PAID && tier <= TIER_BACKGROUND ? st.served[tier] : 0;
}

typedef int (*nvml_init_fn)(void);
typedef int (*nvml_handle_fn)(unsigned, void *);
typedef int (*nvml_mem_fn)(void *, void *);

static pthread_mutex_t nvml_lock = PTHREAD_MUTEX_INITIALIZER;
static nvml_init_fn nvml_init;
static nvml_handle_fn nvml_handle_by_index;
static nvml_mem_fn nvml_mem_info;
static void *nvml_lib;
static void *nvml_dev;
static double nvml_next_retry_s;

struct nvml_memory { unsigned long long total, free_b, used; };

static double monotonic_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* The driver stack can be briefly unavailable (module reload, GPU reset, a
 * restart racing another process' exclusive hold). A one-shot load would
 * poison free-VRAM gating for the lifetime of the process, so retry with a
 * bounded backoff instead. */
static void nvml_load_locked(void) {
    double now = monotonic_s();
    if (nvml_dev && nvml_mem_info) return;
    if (now < nvml_next_retry_s) return;
    nvml_next_retry_s = now + 5.0;
    if (!nvml_lib) {
        nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!nvml_lib) return;
        nvml_init = (nvml_init_fn)dlsym(nvml_lib, "nvmlInit_v2");
        nvml_handle_by_index = (nvml_handle_fn)dlsym(nvml_lib, "nvmlDeviceGetHandleByIndex_v2");
        nvml_mem_info = (nvml_mem_fn)dlsym(nvml_lib, "nvmlDeviceGetMemoryInfo");
    }
    if (!nvml_init || !nvml_handle_by_index || nvml_init() != 0) return;
    nvml_dev = NULL;
    nvml_handle_by_index(0, &nvml_dev);
}

static bool nvml_query(struct nvml_memory *mem) {
    pthread_mutex_lock(&nvml_lock);
    nvml_load_locked();
    bool ok = nvml_dev && nvml_mem_info && nvml_mem_info(nvml_dev, mem) == 0;
    if (!ok) nvml_dev = NULL; /* force a re-handle on the next retry window */
    pthread_mutex_unlock(&nvml_lock);
    return ok;
}

bool ogpu_memory_gib(double *free_gib, double *total_gib) {
    struct nvml_memory mem;
    if (!nvml_query(&mem)) {
        if (free_gib) *free_gib = -1.0;
        if (total_gib) *total_gib = -1.0;
        return false;
    }
    if (free_gib) *free_gib = (double)mem.free_b / (1024.0 * 1024.0 * 1024.0);
    if (total_gib) *total_gib = (double)mem.total / (1024.0 * 1024.0 * 1024.0);
    return true;
}

double ogpu_free_gib(void) {
    double free_gib = 0.0;
    ogpu_memory_gib(&free_gib, NULL);
    return free_gib > 0.0 ? free_gib : 0.0;
}

double ogpu_total_gib(void) {
    double total_gib = 0.0;
    ogpu_memory_gib(NULL, &total_gib);
    return total_gib > 0.0 ? total_gib : 0.0;
}
