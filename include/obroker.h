#ifndef OBROKER_H
#define OBROKER_H

#include <stdbool.h>
#include <stddef.h>

#include "osched.h"

/*
 * Client for the VRAM broker of another omniserve-native process on the same
 * host (the :8791 gateway owns the box-wide ledger). base_url is
 * "http://127.0.0.1:PORT". The call blocks while the broker queues the lease.
 * Returns granted MB, 0 when denied after the wait, -1 when the broker could
 * not be reached (callers fail open: a dead broker must not stop serving).
 */
int obroker_lease(const char *base_url, const char *owner, int pid, int mb, int min_mb,
                  otier tier, double ttl_s, int wait_ms, char *id_out, size_t id_cap,
                  int *waited_ms_out);
bool obroker_release(const char *base_url, const char *lease_id);

#endif
