#ifndef OVRAM_H
#define OVRAM_H

#include <stdbool.h>
#include <stddef.h>

#include "osched.h"

/*
 * Device-memory broker for tenants that share one GPU.
 *
 * Four processes hold VRAM on this box and each one sizes its own workload from
 * cudaMemGetInfo. That read is a lie in two directions at once. It counts a
 * caching allocator's retained blocks as used, so a tenant under-sizes against
 * memory that is in fact reusable; and it is stale the instant two tenants read
 * it concurrently, so both can size a batch against the same free bytes and the
 * second one OOMs. The observed symptom was the image server pinned to a live
 * microbatch of 1 for 40k consecutive jobs.
 *
 * The fix is not more accurate measurement, it is arbitration: a tenant asks
 * for headroom and is told a number it may rely on until the lease expires.
 * Granted-but-not-yet-materialised memory is subtracted from what the next
 * caller sees, which is the part cudaMemGetInfo structurally cannot do.
 *
 * Every lease carries a TTL. A tenant that crashes between lease and release
 * cannot strand headroom, so no tenant needs to trust another tenant's
 * shutdown path. Expiry is evaluated on every call rather than on a timer:
 * a broker whose reclaim depends on its own liveness is a broker that leaks
 * exactly when it is least able to say so.
 *
 * The broker does not allocate, free, or touch device memory. It hands out
 * permission. Enforcement stays with the tenant that owns the allocation,
 * because a broker that could evict another process's memory would need
 * privileges no inference server should hold.
 */

typedef struct ovram ovram;

typedef struct {
    int total_mb;
    int device_free_mb;    /* -1 when the driver is unreachable */
    int reserved_mb;       /* declared future growth, not yet on the device */
    int leased_mb;         /* granted and unexpired */
    int headroom_mb;       /* what a background-tier caller would be granted now */
    int keep_free_mb;
    int lease_count;
    unsigned long long grants;
    unsigned long long partial_grants;
    unsigned long long denials;
    unsigned long long releases;
    unsigned long long expirations;
} ovram_stats;

/* keep_free_mb is the floor left untouched for tenants that allocate scratch
 * without asking, notably the search server's per-query score buffer. */
ovram *ovram_create(int keep_free_mb, double default_ttl_s);
void ovram_destroy(ovram *v);

/*
 * Declare memory a tenant will grow into but has not allocated yet. Memory a
 * tenant already holds must NOT be declared: the driver's free figure has
 * already accounted for it, and declaring it again would subtract it twice.
 * Re-registering an owner replaces its previous figure.
 */
bool ovram_reserve(ovram *v, const char *owner, int mb);

/*
 * Grants between min_mb and mb, or 0 when even min_mb does not fit. Higher
 * tiers may dip further into the keep-free floor, so a paid request is not
 * starved by background batch work that is already holding leases.
 *
 * Returns the granted megabytes and writes an opaque lease id. A grant of 0
 * leaves id_out an empty string and is a normal outcome, not an error.
 */
int ovram_lease(ovram *v, const char *owner, int mb, int min_mb, otier tier,
                double ttl_s, char *id_out, size_t id_cap);

bool ovram_release(ovram *v, const char *id);

/* The TTL a lease gets when the caller does not name one. Callers need this to
 * report the expiry back, and a lease whose expiry the holder cannot see is one
 * it cannot renew before losing. */
double ovram_default_ttl_s(const ovram *v);

void ovram_snapshot(ovram *v, ovram_stats *out);
size_t ovram_status_json(ovram *v, char *out, size_t cap);

/* Deterministic entry points so the policy can be tested without a GPU or a
 * clock. now_s is a monotonic seconds value; device_free_mb of -1 means the
 * driver could not be read, which denies every request rather than guessing. */
int ovram_lease_at(ovram *v, const char *owner, int mb, int min_mb, otier tier,
                   double ttl_s, double now_s, int device_free_mb,
                   char *id_out, size_t id_cap);
int ovram_expire_at(ovram *v, double now_s);
int ovram_headroom_at(ovram *v, otier tier, double now_s, int device_free_mb);

#endif
