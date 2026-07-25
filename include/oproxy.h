#ifndef OPROXY_H
#define OPROXY_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    const char *name;
    size_t name_len;
    const char *value;
    size_t value_len;
} oproxy_header;

typedef bool (*oproxy_sink)(const void *data, size_t len, void *user);

typedef struct {
    bool response_started;
    bool downstream_close;
    bool upstream_reused;
    size_t bytes_relayed;
} oproxy_result;

typedef struct oproxy_target oproxy_target;

typedef struct {
    size_t idle_connections;
    unsigned long long connections_opened;
    unsigned long long connections_reused;
    unsigned long long failures;
} oproxy_stats;

/* Resolve an upstream once and retain up to ``max_idle`` HTTP/1.1
 * connections. A target is safe to share between worker threads. */
oproxy_target *oproxy_target_create(const char *base_url, int max_idle,
                                    char *error, size_t error_cap);
void oproxy_target_destroy(oproxy_target *target);
void oproxy_target_snapshot(oproxy_target *target, oproxy_stats *out);

bool oproxy_target_relay(oproxy_target *target,
                         const char *method, size_t method_len,
                         const char *path, size_t path_len,
                         const char *query, size_t query_len,
                         const char *body, size_t body_len,
                         const char *content_type, size_t content_type_len,
                         const oproxy_header *headers, int header_count,
                         int timeout_ms,
                         oproxy_sink sink, void *sink_user,
                         oproxy_result *out,
                         char *error, size_t error_cap);

/*
 * Convenience one-shot relay. Long-lived servers should create one
 * ``oproxy_target`` per upstream and call ``oproxy_target_relay`` so DNS and
 * established connections are reused.
 *
 * ``base_url`` accepts http://host[:port][/prefix]. TLS is intentionally not
 * implemented: model workers should be on loopback or a private service mesh,
 * with TLS terminated at the public edge.
 */
bool oproxy_relay(const char *base_url,
                  const char *method, size_t method_len,
                  const char *path, size_t path_len,
                  const char *query, size_t query_len,
                  const char *body, size_t body_len,
                  const char *content_type, size_t content_type_len,
                  const oproxy_header *headers, int header_count,
                  int timeout_ms,
                  oproxy_sink sink, void *sink_user,
                  oproxy_result *out,
                  char *error, size_t error_cap);

#endif
