#ifndef OCAPACITY_H
#define OCAPACITY_H

#include <stdbool.h>
#include <stddef.h>

#include "oscale.h"
#include "osched.h"

/*
 * The impure half of capacity control: a background thread that samples the
 * admission scheduler, asks oscale what to do, and drives the control plane.
 *
 * Pod lifecycle deliberately stays in app.nz, which already owns provisioning,
 * per-second billing, the idle reap, and the orphan reconciler. This controller
 * only decides *whether* overflow capacity is worth paying for and asks app.nz
 * to warm it; it never talks to a GPU provider directly. Scaling down is
 * therefore "stop routing here and let the idle reap release the pod", which
 * cannot leak a running instance if this process dies.
 */

typedef struct ocapacity ocapacity;

/* Returns true when the control plane accepted a warm request for the lane,
 * writing a proxyable base URL into endpoint when one is available. */
typedef bool (*ocapacity_warm_fn)(const oscale_policy *policy, char *endpoint,
                                  size_t endpoint_cap, char *error, size_t error_cap,
                                  void *user);

typedef struct {
    osched *sched;
    oscale *scale;
    const char *control_base;  /* app.nz base, loopback plain HTTP */
    const char *api_key;
    int poll_interval_s;
    int timeout_ms;
    /* Overridable so tests can drive the controller without a control plane. */
    ocapacity_warm_fn warm;
    void *warm_user;
} ocapacity_config;

ocapacity *ocapacity_start(const ocapacity_config *config);
void ocapacity_stop(ocapacity *capacity);

/* One controller pass. Exposed so a test can step the clock deterministically
 * instead of sleeping. Returns the number of state changes applied. */
int ocapacity_tick(ocapacity *capacity, double now_s);

size_t ocapacity_status_json(ocapacity *capacity, char *out, size_t cap);

/* Base URL of a ready overflow instance for the lane, or NULL. The tier is
 * checked against the lane policy, so an ineligible tier can never be routed
 * to rented hardware even if an instance happens to be up. */
const char *ocapacity_overflow_endpoint(ocapacity *capacity, const char *lane_name, otier tier);

#endif
