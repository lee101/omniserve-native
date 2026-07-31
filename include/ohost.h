#ifndef OHOST_H
#define OHOST_H

#include <stdbool.h>
#include <stddef.h>

/*
 * Host page-cache warming for model files.
 *
 * The VRAM broker only makes sense if giving memory back is cheap. Unloading a
 * model to admit a bigger image batch is a good trade at a two second reload
 * and a bad one at twenty, and the difference is entirely whether the weights
 * come from page cache or from disk. So the eviction policy and this module are
 * the same decision viewed from two sides: warm the file first, and dropping it
 * from the device stops being expensive enough to avoid.
 *
 * Warming is advisory (posix_fadvise WILLNEED), never MAP_POPULATE. The kernel
 * stays free to drop these pages under real pressure, which is the correct
 * outcome: a prefetch that can evict a running process's working set has turned
 * an optimisation into an outage. Between chunks the loop rechecks
 * MemAvailable and stops rather than pushing the machine toward swap.
 */

typedef struct {
    long mem_total_kb;
    long mem_available_kb;
    long swap_total_kb;
    long swap_free_kb;
} ohost_meminfo;

typedef struct {
    bool enabled;
    int keep_free_pct;
    long long bytes_requested;   /* handed to the kernel as WILLNEED */
    int files_done;
    int files_skipped;           /* unreadable, or no headroom when reached */
    int stops_for_pressure;      /* chunk loops cut short by the floor */
    long mem_available_kb;
    long mem_total_kb;
} ohost_stats;

bool ohost_read_meminfo(ohost_meminfo *out);

/* Available bytes above the configured floor, or 0 when at or below it. */
long long ohost_headroom_bytes(const ohost_meminfo *mi, int keep_free_pct);

/*
 * Starts a detached warming pass over a comma-separated path list. Returns
 * false only when the thread could not be started; an individual unreadable
 * path is counted and skipped, because a stale entry in a config list is not a
 * reason to refuse to warm the rest.
 */
bool ohost_prefetch_start(const char *paths, int keep_free_pct);

void ohost_snapshot(ohost_stats *out);
size_t ohost_status_json(char *out, size_t cap);

/* Deterministic core, exposed so the pressure policy is testable without
 * touching the page cache. Returns bytes it would request for one file. */
long long ohost_plan_bytes(long long file_size, long long headroom_bytes);

#endif
