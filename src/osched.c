#define _GNU_SOURCE
#include "osched.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

typedef struct waiter {
    otier tier;
    int permits;
    long seq;
    struct timespec queued_at;
    struct waiter *next;
} waiter;

struct osched {
    pthread_mutex_t lock;
    pthread_cond_t cond;
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
    pthread_condattr_t attr;
    pthread_condattr_init(&attr);
    pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    pthread_cond_init(&s->cond, &attr);
    pthread_condattr_destroy(&attr);
    s->slots = slots > 0 ? slots : 1;
    s->timeout_s = admission_timeout_s > 0 ? admission_timeout_s : 600;
    return s;
}

void osched_destroy(osched *s) {
    if (!s) return;
    pthread_mutex_destroy(&s->lock);
    pthread_cond_destroy(&s->cond);
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

static bool is_head(const osched *s, const waiter *me) {
    for (const waiter *w = s->waiters; w; w = w->next) {
        if (w->tier < me->tier) return false;
        if (w->tier == me->tier && w->seq < me->seq) return false;
    }
    return true;
}

static void unlink_waiter(osched *s, waiter *me) {
    waiter **p = &s->waiters;
    while (*p && *p != me) p = &(*p)->next;
    if (*p) *p = me->next;
}

bool osched_acquire_n(osched *s, otier tier, int permits) {
    if (!s || tier < TIER_PAID || tier > TIER_BACKGROUND || permits <= 0 || permits > s->slots) {
        return false;
    }
    pthread_mutex_lock(&s->lock);
    bool immediately_available = tier == TIER_BACKGROUND
        ? s->active == 0 && s->used_slots == 0
        : s->used_slots + permits <= s->slots;
    if (!s->waiters && immediately_available) {
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

    waiter *me = calloc(1, sizeof *me);
    if (!me) {
        pthread_mutex_unlock(&s->lock);
        return false;
    }
    me->tier = tier;
    me->permits = permits;
    clock_gettime(CLOCK_MONOTONIC, &me->queued_at);
    me->seq = ++s->seq;
    me->next = s->waiters;
    s->waiters = me;

    for (;;) {
        bool slot_free = tier == TIER_BACKGROUND
            ? s->active == 0 && s->used_slots == 0
            : s->used_slots + permits <= s->slots;
        if (slot_free && is_head(s, me)) break;
        int rc = pthread_cond_timedwait(&s->cond, &s->lock, &deadline);
        if (rc != 0) {
            unlink_waiter(s, me);
            free(me);
            s->timed_out[tier]++;
            if (s->waiters) pthread_cond_broadcast(&s->cond);
            pthread_mutex_unlock(&s->lock);
            return false;
        }
    }
    unlink_waiter(s, me);
    s->active++;
    s->used_slots += permits;
    s->served[tier]++;
    struct timespec admitted_at;
    clock_gettime(CLOCK_MONOTONIC, &admitted_at);
    double queued_ms = elapsed_ms(me->queued_at, admitted_at);
    free(me);
    s->queue_ms_total[tier] += queued_ms;
    if (queued_ms > s->queue_ms_max[tier]) s->queue_ms_max[tier] = queued_ms;
    if (s->waiters) pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->lock);
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
    if (s->waiters) pthread_cond_broadcast(&s->cond);
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
