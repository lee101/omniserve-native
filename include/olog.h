#ifndef OLOG_H
#define OLOG_H

#include <stdbool.h>
#include <stddef.h>

#include "ohttp.h"

/*
 * Per-request access log.
 *
 * /metrics answers "how many 5xx" and /errors answers "which route, recently";
 * neither answers "who called, with what trust markers, how long did it take" —
 * the question that actually matters after a bypass or an abuse report. This
 * log answers that one, and deliberately nothing more: names of trust-relevant
 * headers, never their values, never the query string, never the body.
 *
 * Formatting happens on the worker thread into a fixed stack buffer (no
 * allocation), the line is copied into a bounded ring, and a single drain
 * thread does the write(2). A worker that finds the ring full drops the line
 * and bumps a counter rather than waiting on disk: losing a log line is
 * cheaper than stalling an inference request behind a slow filesystem.
 *
 * Retention is enforced in-process, not by logrotate: OLOG_MAX_FILE_BYTES per
 * file and OLOG_KEEP_FILES generations, so the access log can never occupy
 * more than 128 MiB no matter how long the gateway runs. The size is chosen
 * against the question this log exists to answer: at the observed ~185 B/line
 * and ~25k requests/day that is roughly a month of history. A smaller cap
 * would expire at about the same age as Cloudflare's 7-day edge window, which
 * is precisely the blind spot that made the bypass audit inconclusive.
 */

#define OLOG_MAX_FILE_BYTES (32u << 20) /* 32 MiB per file */
#define OLOG_KEEP_FILES 4               /* access.log + .1 .2 .3 = 128 MiB total */
#define OLOG_RING_SLOTS 4096u          /* power of two */
#define OLOG_SLOT_BYTES 512u           /* 2 MiB of ring, fixed at startup */

/* Reads OMNISERVE_ACCESS_LOG (default on), OMNISERVE_ACCESS_LOG_DIR and
 * OMNISERVE_ACCESS_LOG_MAX_BYTES, then starts the drain thread. Safe to call
 * more than once; later calls are ignored. */
void olog_init(void);
void olog_shutdown(void);
bool olog_enabled(void);
const char *olog_path(void);

/* The trust-relevant header list and the "is this internal" predicate live in
 * the router, next to the code that acts on them. Registering them here keeps
 * one definition: a header added to the bypass check is logged automatically. */
void olog_set_trust_headers(const char *const *names, size_t count);
void olog_set_internal_fn(bool (*fn)(const ohttp_request *req));

/* Records the model for the current request only. Thread-local because one
 * worker thread owns a connection for the whole handler call. */
void olog_set_model(const char *name);

void olog_request(const ohttp_request *req, int status, double duration_ms);
void olog_counters(unsigned long long *emitted, unsigned long long *dropped);

/* Blocks until the drain thread has written everything queued so far. For
 * tests; the serving path never calls it. */
void olog_flush(void);

#endif
