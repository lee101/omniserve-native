#ifndef OHTTP_H
#define OHTTP_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define OHTTP_MAX_HEADERS 48
/* Match the public nginx API limit so multipart STT/video requests accepted at
 * the edge are not rejected by the native gateway. Buffers still grow lazily. */
#define OHTTP_MAX_BODY (80u << 20)

typedef struct {
    const char *name;
    size_t name_len;
    const char *value;
    size_t value_len;
} ohttp_header;

typedef struct ohttp_conn ohttp_conn;

typedef struct {
    const char *method;
    size_t method_len;
    const char *path;
    size_t path_len;
    const char *query;
    size_t query_len;
    ohttp_header headers[OHTTP_MAX_HEADERS];
    int header_count;
    const char *body;
    size_t body_len;
    ohttp_conn *conn;
} ohttp_request;

typedef void (*ohttp_handler)(ohttp_request *req, void *user);

typedef struct {
    uint16_t port;
    const char *bind_addr;
    int reactor_threads;
    int worker_threads;
    ohttp_handler handler;
    void *user;
} ohttp_config;

typedef struct ohttp_server ohttp_server;

ohttp_server *ohttp_start(const ohttp_config *cfg);
/* Wait for ``ohttp_stop`` and release all server resources. The server pointer
 * is invalid after this returns. */
int ohttp_join(ohttp_server *srv);
void ohttp_stop(ohttp_server *srv);

const char *ohttp_req_header(const ohttp_request *req, const char *name, size_t *len);
bool ohttp_path_is(const ohttp_request *req, const char *path);
bool ohttp_method_is(const ohttp_request *req, const char *method);

void ohttp_respond(ohttp_request *req, int status, const char *content_type,
                   const char *body, size_t body_len);
void ohttp_respond_str(ohttp_request *req, int status, const char *content_type,
                       const char *body);

void ohttp_stream_begin(ohttp_request *req, int status, const char *content_type);
bool ohttp_stream_write(ohttp_request *req, const char *data, size_t len);
void ohttp_stream_end(ohttp_request *req);

/* Reverse-proxy fast path: relay an upstream HTTP response verbatim. The proxy
 * must call ``ohttp_force_close`` when upstream framing requires EOF. */
bool ohttp_raw_write(ohttp_request *req, const void *data, size_t len);
void ohttp_force_close(ohttp_request *req);

/*
 * Response accounting.
 *
 * An on-call agent needs to know that the gateway is returning 5xx, on which
 * route, and how recently — without scraping logs, which is both slow and
 * ambiguous once several services share a journal. Own responses are recorded
 * where the status is written; proxied responses are recorded by reading the
 * status line of the first relayed write, so an upstream 500 is not invisible
 * just because the bytes passed through verbatim.
 */

#define OHTTP_ERROR_RING 32

typedef struct {
    unsigned long long total;
    unsigned long long informational; /* 1xx */
    unsigned long long success;       /* 2xx */
    unsigned long long redirect;      /* 3xx */
    unsigned long long client_error;  /* 4xx */
    unsigned long long server_error;  /* 5xx */
} ohttp_response_stats;

typedef struct {
    int status;
    long at_unix;
    char method[8];
    char path[104];
} ohttp_error_event;

void ohttp_response_snapshot(ohttp_response_stats *out);

/* Copies up to max recent 5xx events, newest first. Returns the count. */
int ohttp_recent_errors(ohttp_error_event *out, int max);

#endif
