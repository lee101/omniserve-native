#ifndef OSCHED_H
#define OSCHED_H

#include <stdbool.h>
#include <stdint.h>

typedef enum { TIER_PAID = 0, TIER_SUB = 1, TIER_FREE = 2, TIER_BACKGROUND = 3 } otier;

typedef struct osched osched;

osched *osched_create(int slots, double admission_timeout_s);
void osched_destroy(osched *s);

otier otier_parse(const char *value, int len);
otier otier_parse_public(const char *value, int len);
const char *otier_name(otier t);

bool osched_acquire(osched *s, otier tier);
void osched_release(osched *s, otier tier);

/* Weighted permits let latency-friendly text/audio calls share configured GPU
 * capacity while diffusion/video reserve the entire device. */
bool osched_acquire_n(osched *s, otier tier, int permits);
void osched_release_n(osched *s, otier tier, int permits);

/* Non-blocking admission for callers with an alternative, so a saturated local
 * device sends them to overflow immediately instead of after a queue wait. */
bool osched_try_acquire_n(osched *s, otier tier, int permits);
int osched_capacity(const osched *s);

typedef struct {
    int slots;
    int used_slots;
    int active;
    int waiting[4];
    long served[4];
    double queue_ms_total[4];
    double queue_ms_max[4];
    uint64_t timed_out[4];
} osched_stats;

void osched_snapshot(const osched *s, osched_stats *out);

int osched_active(const osched *s);
int osched_waiting(const osched *s, otier tier);
long osched_served(const osched *s, otier tier);

double ogpu_free_gib(void);
double ogpu_total_gib(void);

/* Returns false when the driver stack is unreachable, writing -1 to the
 * outputs. A real 0 GiB free and "cannot ask the driver" are different
 * conditions and must not both surface as 0.00. */
bool ogpu_memory_gib(double *free_gib, double *total_gib);

#endif
