#define _GNU_SOURCE
#include "olog.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define OLOG_PATH_MAX 512
#define OLOG_PATH_FIELD 192 /* escaped bytes of request path kept per line */
#define OLOG_HDR_FIELD 224  /* bytes of the present-header name list kept */
#define OLOG_MODEL_MAX 80

typedef struct {
    atomic_uint seq;
    unsigned len;
    char line[OLOG_SLOT_BYTES];
} olog_slot;

static olog_slot ring[OLOG_RING_SLOTS];
static atomic_uint ring_head;
static atomic_uint ring_tail;
static atomic_ullong olog_emitted;
static atomic_ullong olog_dropped;
/* Drained from the ring but not written: rotation or the disk failed. Kept
 * separate from olog_emitted, which olog_flush waits on and which must still
 * advance when the disk is gone or a flush would never return. */
static atomic_ullong olog_unwritten;

static atomic_bool olog_on;
static atomic_bool olog_stop;
static pthread_t drain_thread;
static bool drain_running;
static pthread_once_t init_once = PTHREAD_ONCE_INIT;

static char log_path[OLOG_PATH_MAX];
static size_t max_file_bytes = OLOG_MAX_FILE_BYTES;

static const char *const *trust_headers;
static size_t trust_header_count;
static bool (*internal_fn)(const ohttp_request *req);

/* Credential-bearing headers are listed so their PRESENCE is visible; the
 * value is never read, only the name is ever written to the file. */
static const char *const credential_headers[] = {
    "Authorization", "X-API-Key", "X-Rapid-API-Key", "secret", "Cookie",
    "X-Omniserve-Internal", "X-Omniserve-Tier",
};

static _Thread_local char tls_model[OLOG_MODEL_MAX];

void olog_set_trust_headers(const char *const *names, size_t count) {
    trust_headers = names;
    trust_header_count = count;
}

void olog_set_internal_fn(bool (*fn)(const ohttp_request *req)) {
    internal_fn = fn;
}

void olog_set_model(const char *name) {
    if (!name || !name[0]) {
        tls_model[0] = 0;
        return;
    }
    size_t n = strlen(name);
    if (n >= sizeof tls_model) n = sizeof tls_model - 1;
    memcpy(tls_model, name, n);
    tls_model[n] = 0;
}

bool olog_enabled(void) {
    return atomic_load_explicit(&olog_on, memory_order_relaxed);
}

const char *olog_path(void) {
    return log_path;
}

void olog_counters(unsigned long long *emitted, unsigned long long *dropped) {
    if (emitted) *emitted = atomic_load_explicit(&olog_emitted, memory_order_relaxed);
    if (dropped) *dropped = atomic_load_explicit(&olog_dropped, memory_order_relaxed);
}

/* Everything outside a conservative printable set becomes \xHH, so a path of
 * "/a\r\nGET /b" cannot terminate the line early and forge a second record.
 * Quote and backslash are escaped for the same reason. */
static size_t escape_into(char *out, size_t cap, const char *s, size_t len, bool *truncated) {
    size_t w = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char ch = (unsigned char)s[i];
        char unit[5];
        size_t unit_len;
        if (ch == '"' || ch == '\\') {
            unit[0] = '\\';
            unit[1] = (char)ch;
            unit_len = 2;
        } else if (ch >= 0x20 && ch < 0x7f) {
            unit[0] = (char)ch;
            unit_len = 1;
        } else {
            static const char hex[] = "0123456789abcdef";
            unit[0] = '\\';
            unit[1] = 'x';
            unit[2] = hex[ch >> 4];
            unit[3] = hex[ch & 0xf];
            unit_len = 4;
        }
        if (w + unit_len > cap) {
            if (truncated) *truncated = true;
            break;
        }
        memcpy(out + w, unit, unit_len);
        w += unit_len;
    }
    return w;
}

static void timestamp_utc(char *out, size_t cap) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    /* Clamped so the formatted width is provably fixed: a line whose length can
     * surprise the caller is exactly what the escaping above exists to avoid. */
    int year = (tm.tm_year + 1900) % 10000;
    if (year < 0) year = 0;
    snprintf(out, cap, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
             year, (tm.tm_mon + 1) % 100, tm.tm_mday % 100,
             tm.tm_hour % 100, tm.tm_min % 100, tm.tm_sec % 100,
             (int)(ts.tv_nsec / 1000000) % 1000);
}

static size_t append_header_names(char *out, size_t cap, const ohttp_request *req,
                                  const char *const *names, size_t count, size_t written) {
    for (size_t i = 0; i < count; i++) {
        if (!ohttp_req_header(req, names[i], NULL)) continue;
        size_t nlen = strlen(names[i]);
        size_t need = nlen + (written ? 1 : 0);
        if (written + need >= cap) break;
        if (written) out[written++] = ',';
        memcpy(out + written, names[i], nlen);
        written += nlen;
    }
    return written;
}

/* Single-producer-per-slot bounded ring: reserve a slot with a CAS on the head,
 * fill it, then publish. A producer that finds the ring full returns instead of
 * waiting, so a stalled disk can never back-pressure a request thread. */
static void ring_push(const char *line, size_t len) {
    if (len >= OLOG_SLOT_BYTES) len = OLOG_SLOT_BYTES - 1;
    unsigned pos = atomic_load_explicit(&ring_head, memory_order_relaxed);
    olog_slot *slot;
    for (;;) {
        slot = &ring[pos & (OLOG_RING_SLOTS - 1)];
        unsigned seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
        int diff = (int)(seq - pos);
        if (diff == 0) {
            if (atomic_compare_exchange_weak_explicit(&ring_head, &pos, pos + 1,
                                                      memory_order_relaxed,
                                                      memory_order_relaxed)) break;
        } else if (diff < 0) {
            atomic_fetch_add_explicit(&olog_dropped, 1, memory_order_relaxed);
            return;
        } else {
            pos = atomic_load_explicit(&ring_head, memory_order_relaxed);
        }
    }
    memcpy(slot->line, line, len);
    slot->len = (unsigned)len;
    atomic_store_explicit(&slot->seq, pos + 1, memory_order_release);
}

void olog_request(const ohttp_request *req, int status, double duration_ms) {
    if (!req || !olog_enabled()) return;

    char stamp[32];
    timestamp_utc(stamp, sizeof stamp);

    char path[OLOG_PATH_FIELD + 8];
    bool path_truncated = false;
    size_t path_len = escape_into(path, OLOG_PATH_FIELD, req->path, req->path_len,
                                  &path_truncated);
    if (path_truncated && path_len < sizeof path - 1) path[path_len++] = '+';
    path[path_len] = 0;

    char headers[OLOG_HDR_FIELD];
    size_t hdr_len = append_header_names(headers, sizeof headers, req,
                                         trust_headers, trust_header_count, 0);
    hdr_len = append_header_names(headers, sizeof headers, req, credential_headers,
                                 sizeof credential_headers / sizeof credential_headers[0],
                                 hdr_len);
    headers[hdr_len] = 0;

    bool internal = internal_fn ? internal_fn(req)
                                : (ohttp_req_peer_is_loopback(req) && hdr_len == 0);
    char method[16];
    size_t method_len = escape_into(method, sizeof method - 1, req->method,
                                    req->method_len, NULL);
    method[method_len] = 0;

    /* Path is last on purpose. It is the only field a caller controls the
     * length of, and snprintf truncates from the right: with path in the
     * middle, a long enough request target would push internal= and hdrs= off
     * the end of the line and delete exactly the bypass evidence this log
     * exists to keep. */
    char line[OLOG_SLOT_BYTES];
    int n = snprintf(line, sizeof line,
                     "%s peer=%s method=%s model=%s status=%d "
                     "dur_ms=%.2f internal=%d hdrs=%s path=\"%s\"\n",
                     stamp, ohttp_req_peer_addr(req), method_len ? method : "-",
                     tls_model[0] ? tls_model : "-", status, duration_ms,
                     internal ? 1 : 0, hdr_len ? headers : "-", path);
    if (n < 0) return;
    size_t len = (size_t)n < sizeof line ? (size_t)n : sizeof line - 1;
    if (line[len - 1] != '\n') line[len - 1] = '\n';
    ring_push(line, len);
}

/* ---- drain side: only the drain thread touches the file ---- */

static int log_fd = -1;
static size_t log_size;

static void log_open(void) {
    log_fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0640);
    if (log_fd < 0) {
        log_size = 0;
        return;
    }
    off_t end = lseek(log_fd, 0, SEEK_END);
    log_size = end > 0 ? (size_t)end : 0;
}

/* access.log -> .1 -> .2 -> .3, oldest overwritten. Rotation is in-process on
 * purpose: an external logrotate that is not installed on a given box is a cap
 * that silently does not exist. */
static void log_rotate(void) {
    if (log_fd >= 0) close(log_fd);
    log_fd = -1;
    char from[OLOG_PATH_MAX + 16], to[OLOG_PATH_MAX + 16];
    bool renamed = false;
    for (int i = OLOG_KEEP_FILES - 1; i >= 1; i--) {
        snprintf(to, sizeof to, "%s.%d", log_path, i);
        if (i == 1) snprintf(from, sizeof from, "%s", log_path);
        else snprintf(from, sizeof from, "%s.%d", log_path, i - 1);
        if (rename(from, to) == 0 && i == 1) renamed = true;
    }
    log_open();
    /* A cap that a failed rename can lift is not a cap. If access.log is still
     * the same full file - read-only directory, ENOSPC, someone holding the
     * name - discard it rather than append forever. Losing the oldest lines is
     * the lesser failure against filling the disk. */
    if (!renamed && log_fd >= 0 && log_size >= max_file_bytes) {
        if (ftruncate(log_fd, 0) == 0) log_size = 0;
    }
}

static bool log_write(const char *data, size_t len) {
    if (log_fd < 0) return false;
    if (log_size + len > max_file_bytes) log_rotate();
    if (log_fd < 0) return false;
    /* Rotation could not make room, so writing would breach the cap. */
    if (log_size + len > max_file_bytes) return false;
    while (len) {
        ssize_t w = write(log_fd, data, len);
        if (w <= 0) {
            if (errno == EINTR) continue;
            return false;
        }
        data += w;
        len -= (size_t)w;
        log_size += (size_t)w;
    }
    return true;
}

static bool ring_pop(char *out, size_t *len) {
    unsigned pos = atomic_load_explicit(&ring_tail, memory_order_relaxed);
    olog_slot *slot = &ring[pos & (OLOG_RING_SLOTS - 1)];
    unsigned seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
    if ((int)(seq - (pos + 1)) != 0) return false;
    *len = slot->len;
    memcpy(out, slot->line, slot->len);
    atomic_store_explicit(&ring_tail, pos + 1, memory_order_relaxed);
    atomic_store_explicit(&slot->seq, pos + OLOG_RING_SLOTS, memory_order_release);
    return true;
}

static void emit_drops_if_changed(unsigned long long *reported) {
    unsigned long long dropped = atomic_load_explicit(&olog_dropped, memory_order_relaxed);
    if (dropped == *reported) return;
    *reported = dropped;
    char stamp[32];
    timestamp_utc(stamp, sizeof stamp);
    char line[160];
    int n = snprintf(line, sizeof line,
                     "%s event=access_log_drops dropped=%llu emitted=%llu unwritten=%llu\n",
                     stamp, dropped,
                     atomic_load_explicit(&olog_emitted, memory_order_relaxed),
                     atomic_load_explicit(&olog_unwritten, memory_order_relaxed));
    if (n > 0) log_write(line, (size_t)n);
}

static void *drain_main(void *arg) {
    (void)arg;
    unsigned long long reported_drops = 0;
    static char batch[32u << 10];
    struct timespec idle = { .tv_sec = 0, .tv_nsec = 1000L * 1000 };
    time_t next_drop_report = time(NULL) + 10;
    for (;;) {
        char line[OLOG_SLOT_BYTES];
        size_t len = 0;
        bool did_work = false;
        /* Batched so a burst costs one write(2) per 32 KiB rather than one per
         * line; the ring is what absorbs the burst, and a syscall per record
         * is what makes it overflow. Never larger than the rotation cap, so a
         * single write cannot overshoot the file size limit. */
        size_t batch_cap = max_file_bytes < sizeof batch ? max_file_bytes : sizeof batch;
        size_t used = 0, batched = 0;
        while (ring_pop(line, &len)) {
            if (used + len > batch_cap) {
                if (!log_write(batch, used))
                    atomic_fetch_add_explicit(&olog_unwritten, batched, memory_order_relaxed);
                /* Counted only once on disk: olog_flush waits on this. */
                atomic_fetch_add_explicit(&olog_emitted, batched, memory_order_relaxed);
                used = 0;
                batched = 0;
            }
            memcpy(batch + used, line, len);
            used += len;
            batched++;
            did_work = true;
        }
        if (used) {
            if (!log_write(batch, used))
                atomic_fetch_add_explicit(&olog_unwritten, batched, memory_order_relaxed);
            atomic_fetch_add_explicit(&olog_emitted, batched, memory_order_relaxed);
        }
        /* Drops are only useful if they surface, so they go into the same file
         * the operator is already reading, on a slow timer. */
        time_t now = time(NULL);
        if (now >= next_drop_report) {
            next_drop_report = now + 10;
            emit_drops_if_changed(&reported_drops);
        }
        if (atomic_load_explicit(&olog_stop, memory_order_acquire) && !did_work) break;
        if (!did_work) nanosleep(&idle, NULL);
    }
    emit_drops_if_changed(&reported_drops);
    if (log_fd >= 0) close(log_fd);
    log_fd = -1;
    return NULL;
}

static bool env_off(const char *name) {
    const char *v = getenv(name);
    return v && (v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
}

static void resolve_path(void) {
    const char *dir = getenv("OMNISERVE_ACCESS_LOG_DIR");
    if (!dir || !dir[0]) dir = "/var/log/omniserve";
    mkdir(dir, 0750);
    snprintf(log_path, sizeof log_path, "%s/access.log", dir);
    log_open();
    if (log_fd < 0) {
        /* A default directory that does not exist must not silently disable the
         * log; fall back somewhere always writable and say so once. */
        fprintf(stderr, "access log: %s unusable (%s), falling back to /tmp\n",
                log_path, strerror(errno));
        snprintf(log_path, sizeof log_path, "/tmp/omniserve-access.log");
        log_open();
    }
}

static void olog_init_once(void) {
    if (env_off("OMNISERVE_ACCESS_LOG")) return;
    const char *cap = getenv("OMNISERVE_ACCESS_LOG_MAX_BYTES");
    if (cap && cap[0]) {
        unsigned long long v = strtoull(cap, NULL, 10);
        /* The env var may only tighten the cap: the 32 MiB ceiling is a
         * property of the box, not a suggestion. */
        if (v >= 4096 && v < max_file_bytes) max_file_bytes = (size_t)v;
    }
    for (unsigned i = 0; i < OLOG_RING_SLOTS; i++) atomic_store(&ring[i].seq, i);
    resolve_path();
    if (log_fd < 0) return;
    atomic_store(&olog_stop, false);
    if (pthread_create(&drain_thread, NULL, drain_main, NULL) != 0) {
        close(log_fd);
        log_fd = -1;
        return;
    }
    drain_running = true;
    atomic_store(&olog_on, true);
}

void olog_init(void) {
    pthread_once(&init_once, olog_init_once);
}

void olog_shutdown(void) {
    if (!drain_running) return;
    atomic_store(&olog_on, false);
    atomic_store_explicit(&olog_stop, true, memory_order_release);
    pthread_join(drain_thread, NULL);
    drain_running = false;
}

void olog_flush(void) {
    for (int i = 0; i < 400; i++) {
        /* Compare against emitted, not the read cursor: a slot is dequeued
         * before its write(2) completes. */
        unsigned long long pushed = atomic_load_explicit(&ring_head, memory_order_relaxed);
        if (atomic_load_explicit(&olog_emitted, memory_order_relaxed) >= pushed) return;
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 5L * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
}
