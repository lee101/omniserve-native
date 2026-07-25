#define _GNU_SOURCE
#include "ohttp.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <time.h>

#define OHTTP_INLINE_BUFFER 4096
#define OHTTP_MAX_BUFFER (OHTTP_MAX_BODY + (64u << 10))

typedef enum { CONN_READING, CONN_DISPATCHED, CONN_CLOSED } conn_state;

/* Process-global, like the NVML handle in osched.c: one gateway per process, so
 * threading a server pointer through every response path would buy nothing. */
static atomic_ullong resp_total;
static atomic_ullong resp_by_class[6]; /* index by status/100, 0 unused */
static pthread_mutex_t error_ring_lock = PTHREAD_MUTEX_INITIALIZER;
static ohttp_error_event error_ring[OHTTP_ERROR_RING];
static unsigned error_ring_next;
static unsigned long long error_ring_seen;

static void record_status(const ohttp_request *req, int status) {
    if (status < 100 || status > 599) return;
    atomic_fetch_add_explicit(&resp_total, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&resp_by_class[status / 100], 1, memory_order_relaxed);
    if (status < 500) return;
    pthread_mutex_lock(&error_ring_lock);
    ohttp_error_event *slot = &error_ring[error_ring_next % OHTTP_ERROR_RING];
    memset(slot, 0, sizeof *slot);
    slot->status = status;
    slot->at_unix = (long)time(NULL);
    if (req) {
        size_t method_len = req->method_len < sizeof slot->method - 1
            ? req->method_len : sizeof slot->method - 1;
        memcpy(slot->method, req->method, method_len);
        size_t path_len = req->path_len < sizeof slot->path - 1
            ? req->path_len : sizeof slot->path - 1;
        memcpy(slot->path, req->path, path_len);
    }
    error_ring_next++;
    error_ring_seen++;
    pthread_mutex_unlock(&error_ring_lock);
}

/* Proxied responses are relayed verbatim, so the only place the status appears
 * is the first bytes of the upstream response. */
static void record_relayed_status(ohttp_request *req, const void *data, size_t len) {
    if (len < 12 || memcmp(data, "HTTP/1.", 7) != 0) return;
    const char *p = (const char *)data + 8;
    while (p < (const char *)data + len && *p == ' ') p++;
    if (p + 3 > (const char *)data + len) return;
    if (p[0] < '0' || p[0] > '9' || p[1] < '0' || p[1] > '9' || p[2] < '0' || p[2] > '9') return;
    record_status(req, (p[0] - '0') * 100 + (p[1] - '0') * 10 + (p[2] - '0'));
}

void ohttp_response_snapshot(ohttp_response_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    out->total = atomic_load_explicit(&resp_total, memory_order_relaxed);
    out->informational = atomic_load_explicit(&resp_by_class[1], memory_order_relaxed);
    out->success = atomic_load_explicit(&resp_by_class[2], memory_order_relaxed);
    out->redirect = atomic_load_explicit(&resp_by_class[3], memory_order_relaxed);
    out->client_error = atomic_load_explicit(&resp_by_class[4], memory_order_relaxed);
    out->server_error = atomic_load_explicit(&resp_by_class[5], memory_order_relaxed);
}

int ohttp_recent_errors(ohttp_error_event *out, int max) {
    if (!out || max <= 0) return 0;
    pthread_mutex_lock(&error_ring_lock);
    int have = (int)(error_ring_seen < OHTTP_ERROR_RING ? error_ring_seen : OHTTP_ERROR_RING);
    int count = have < max ? have : max;
    for (int i = 0; i < count; i++) {
        /* Newest first: walk backwards from the write cursor. */
        unsigned index = (error_ring_next - 1u - (unsigned)i) % OHTTP_ERROR_RING;
        out[i] = error_ring[index];
    }
    pthread_mutex_unlock(&error_ring_lock);
    return count;
}

struct ohttp_conn {
    int fd;
    conn_state state;
    char *buf;
    char inline_buf[OHTTP_INLINE_BUFFER];
    size_t buf_cap;
    size_t buf_len;
    size_t body_start;
    size_t need_total;
    bool keep_alive;
    bool continue_sent;
    bool streaming;
    bool chunked_out;
    bool status_recorded; /* one status per request, even across many raw writes */
    pthread_mutex_t ownership_lock;
    struct ohttp_server *srv;
    struct ohttp_conn *next_job;
    struct ohttp_conn *next_all;
};

struct ohttp_server {
    ohttp_config cfg;
    int listen_fd;
    int epoll_fd;
    atomic_bool running;
    pthread_t *reactors;
    pthread_t *workers;
    pthread_mutex_t q_lock;
    pthread_cond_t q_cond;
    struct ohttp_conn *q_head, *q_tail;
    pthread_mutex_t all_lock;
    struct ohttp_conn *all_head;
};

static void server_free_empty(struct ohttp_server *srv) {
    if (srv->listen_fd >= 0) close(srv->listen_fd);
    if (srv->epoll_fd >= 0) close(srv->epoll_fd);
    pthread_mutex_destroy(&srv->all_lock);
    pthread_mutex_destroy(&srv->q_lock);
    pthread_cond_destroy(&srv->q_cond);
    free(srv->reactors);
    free(srv->workers);
    free(srv);
}

static int set_nonblock(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int set_block(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
}

static void conn_free(struct ohttp_conn *c) {
    struct ohttp_server *srv = c->srv;
    if (srv) {
        pthread_mutex_lock(&srv->all_lock);
        struct ohttp_conn **current = &srv->all_head;
        while (*current && *current != c) current = &(*current)->next_all;
        if (*current) *current = c->next_all;
        pthread_mutex_unlock(&srv->all_lock);
    }
    if (c->fd >= 0) close(c->fd);
    if (c->buf != c->inline_buf) free(c->buf);
    pthread_mutex_destroy(&c->ownership_lock);
    free(c);
}

static bool write_all(int fd, const char *data, size_t len) {
    while (len > 0) {
        ssize_t n = write(fd, data, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        data += n;
        len -= (size_t)n;
    }
    return true;
}

static bool writev_all(int fd, struct iovec *iov, int iov_count) {
    while (iov_count > 0) {
        ssize_t n = writev(fd, iov, iov_count);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        size_t written = (size_t)n;
        while (iov_count > 0 && written >= iov[0].iov_len) {
            written -= iov[0].iov_len;
            iov++;
            iov_count--;
        }
        if (iov_count > 0 && written) {
            iov[0].iov_base = (char *)iov[0].iov_base + written;
            iov[0].iov_len -= written;
        }
    }
    return true;
}

static const char *status_text(int status) {
    switch (status) {
    case 200: return "OK";
    case 202: return "Accepted";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    case 507: return "Insufficient Storage";
    default: return "OK";
    }
}

static bool header_has_token(const char *begin, const char *end, const char *token) {
    size_t token_len = strlen(token);
    while (begin < end) {
        while (begin < end && (*begin == ' ' || *begin == '\t' || *begin == ',')) begin++;
        const char *token_end = begin;
        while (token_end < end && *token_end != ',') token_end++;
        const char *trimmed = token_end;
        while (trimmed > begin && (trimmed[-1] == ' ' || trimmed[-1] == '\t')) trimmed--;
        if ((size_t)(trimmed - begin) == token_len && strncasecmp(begin, token, token_len) == 0) {
            return true;
        }
        begin = token_end < end ? token_end + 1 : end;
    }
    return false;
}

static bool parse_content_length(const char *begin, const char *end, size_t *out) {
    while (begin < end && (*begin == ' ' || *begin == '\t')) begin++;
    while (end > begin && (end[-1] == ' ' || end[-1] == '\t')) end--;
    if (begin == end) return false;
    size_t value = 0;
    for (const char *p = begin; p < end; p++) {
        if (*p < '0' || *p > '9') return false;
        unsigned digit = (unsigned)(*p - '0');
        if (value > (SIZE_MAX - digit) / 10) return false;
        value = value * 10 + digit;
    }
    *out = value;
    return true;
}

const char *ohttp_req_header(const ohttp_request *req, const char *name, size_t *len) {
    size_t nlen = strlen(name);
    for (int i = 0; i < req->header_count; i++) {
        if (req->headers[i].name_len == nlen &&
            strncasecmp(req->headers[i].name, name, nlen) == 0) {
            if (len) *len = req->headers[i].value_len;
            return req->headers[i].value;
        }
    }
    if (len) *len = 0;
    return NULL;
}

bool ohttp_path_is(const ohttp_request *req, const char *path) {
    size_t plen = strlen(path);
    return req->path_len == plen && memcmp(req->path, path, plen) == 0;
}

bool ohttp_method_is(const ohttp_request *req, const char *method) {
    size_t mlen = strlen(method);
    return req->method_len == mlen && memcmp(req->method, method, mlen) == 0;
}

void ohttp_respond(ohttp_request *req, int status, const char *content_type,
                   const char *body, size_t body_len) {
    struct ohttp_conn *c = req->conn;
    char head[512];
    int hn = snprintf(head, sizeof head,
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: %s\r\n"
                      "Content-Length: %zu\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                      "Access-Control-Allow-Headers: Authorization, Content-Type, secret, X-API-Key, X-Rapid-API-Key, X-Omniserve-Tier\r\n"
                      "Connection: %s\r\n\r\n",
                      status, status_text(status), content_type, body_len,
                      c->keep_alive ? "keep-alive" : "close");
    struct iovec iov[2] = {
        { .iov_base = head, .iov_len = (size_t)hn },
        { .iov_base = (void *)body, .iov_len = body_len },
    };
    if (!writev_all(c->fd, iov, body_len ? 2 : 1)) c->keep_alive = false;
    record_status(req, status);
}

void ohttp_respond_str(ohttp_request *req, int status, const char *content_type,
                       const char *body) {
    ohttp_respond(req, status, content_type, body, strlen(body));
}

void ohttp_stream_begin(ohttp_request *req, int status, const char *content_type) {
    struct ohttp_conn *c = req->conn;
    c->streaming = true;
    c->chunked_out = true;
    char head[512];
    int hn = snprintf(head, sizeof head,
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: %s\r\n"
                      "Transfer-Encoding: chunked\r\n"
                      "Cache-Control: no-cache\r\n"
                      "Access-Control-Allow-Origin: *\r\n"
                      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                      "Access-Control-Allow-Headers: Authorization, Content-Type, secret, X-API-Key, X-Rapid-API-Key, X-Omniserve-Tier\r\n"
                      "Connection: %s\r\n\r\n",
                      status, status_text(status), content_type,
                      c->keep_alive ? "keep-alive" : "close");
    write_all(c->fd, head, (size_t)hn);
    record_status(req, status);
}

bool ohttp_stream_write(ohttp_request *req, const char *data, size_t len) {
    struct ohttp_conn *c = req->conn;
    if (!len) return true;
    char sz[24];
    int sn = snprintf(sz, sizeof sz, "%zx\r\n", len);
    struct iovec iov[3] = {
        { .iov_base = sz, .iov_len = (size_t)sn },
        { .iov_base = (void *)data, .iov_len = len },
        { .iov_base = (void *)"\r\n", .iov_len = 2 },
    };
    bool ok = writev_all(c->fd, iov, 3);
    if (!ok) c->keep_alive = false;
    return ok;
}

void ohttp_stream_end(ohttp_request *req) {
    write_all(req->conn->fd, "0\r\n\r\n", 5);
}

bool ohttp_raw_write(ohttp_request *req, const void *data, size_t len) {
    if (!req || !req->conn) return false;
    if (!req->conn->status_recorded) {
        req->conn->status_recorded = true;
        record_relayed_status(req, data, len);
    }
    return write_all(req->conn->fd, data, len);
}

void ohttp_force_close(ohttp_request *req) {
    if (req && req->conn) req->conn->keep_alive = false;
}

static bool parse_request(struct ohttp_conn *c, ohttp_request *req) {
    char *end = memmem(c->buf, c->buf_len, "\r\n\r\n", 4);
    if (!end) {
        if (c->buf_len > (64u << 10)) return false;
        c->need_total = 0;
        return true;
    }
    size_t head_len = (size_t)(end - c->buf) + 4;

    memset(req, 0, sizeof *req);
    req->conn = c;
    char *p = c->buf;
    char *line_end = memmem(p, head_len, "\r\n", 2);
    if (!line_end) return false;

    req->method = p;
    char *sp = memchr(p, ' ', (size_t)(line_end - p));
    if (!sp) return false;
    req->method_len = (size_t)(sp - p);
    p = sp + 1;
    sp = memchr(p, ' ', (size_t)(line_end - p));
    if (!sp) return false;
    req->path = p;
    req->path_len = (size_t)(sp - p);
    char *q = memchr(req->path, '?', req->path_len);
    if (q) {
        req->query = q + 1;
        req->query_len = req->path_len - (size_t)(q - req->path) - 1;
        req->path_len = (size_t)(q - req->path);
    }

    const char *version = sp + 1;
    size_t version_len = (size_t)(line_end - version);
    bool http_10 = version_len == 8 && memcmp(version, "HTTP/1.0", 8) == 0;
    if (!http_10 && !(version_len == 8 && memcmp(version, "HTTP/1.1", 8) == 0)) return false;

    c->keep_alive = !http_10;
    size_t content_length = 0;
    bool have_content_length = false;
    bool expect_continue = false;
    p = line_end + 2;
    while (p < c->buf + head_len - 2) {
        line_end = memmem(p, head_len - (size_t)(p - c->buf), "\r\n", 2);
        if (!line_end || line_end == p) break;
        char *colon = memchr(p, ':', (size_t)(line_end - p));
        if (colon) {
            if (req->header_count >= OHTTP_MAX_HEADERS) return false;
            ohttp_header *h = &req->headers[req->header_count++];
            h->name = p;
            h->name_len = (size_t)(colon - p);
            char *v = colon + 1;
            while (v < line_end && (*v == ' ' || *v == '\t')) v++;
            h->value = v;
            h->value_len = (size_t)(line_end - v);
            if (h->name_len == 14 && strncasecmp(h->name, "Content-Length", 14) == 0) {
                size_t parsed = 0;
                if (!parse_content_length(v, line_end, &parsed) ||
                    (have_content_length && parsed != content_length)) return false;
                content_length = parsed;
                have_content_length = true;
            } else if (h->name_len == 17 && strncasecmp(h->name, "Transfer-Encoding", 17) == 0) {
                /* The edge must dechunk requests. Treating an encoded body as a
                 * second request would be a request-smuggling vulnerability. */
                return false;
            } else if (h->name_len == 10 && strncasecmp(h->name, "Connection", 10) == 0) {
                if (header_has_token(v, line_end, "close")) c->keep_alive = false;
                if (http_10 && header_has_token(v, line_end, "keep-alive")) c->keep_alive = true;
            } else if (h->name_len == 6 && strncasecmp(h->name, "Expect", 6) == 0) {
                if (!header_has_token(v, line_end, "100-continue")) return false;
                expect_continue = true;
            }
        }
        p = line_end + 2;
    }

    if (content_length > OHTTP_MAX_BODY) return false;
    c->body_start = head_len;
    c->need_total = head_len + content_length;
    if (c->buf_len < c->need_total) {
        if (expect_continue && content_length && !c->continue_sent) {
            static const char response[] = "HTTP/1.1 100 Continue\r\n\r\n";
            ssize_t sent = send(c->fd, response, sizeof response - 1, MSG_NOSIGNAL | MSG_DONTWAIT);
            if (sent != (ssize_t)(sizeof response - 1)) return false;
            c->continue_sent = true;
        }
        return true;
    }

    req->body = c->buf + head_len;
    req->body_len = content_length;
    return true;
}

static void queue_push(struct ohttp_server *srv, struct ohttp_conn *c) {
    pthread_mutex_lock(&srv->q_lock);
    c->next_job = NULL;
    if (srv->q_tail) srv->q_tail->next_job = c;
    else srv->q_head = c;
    srv->q_tail = c;
    pthread_cond_signal(&srv->q_cond);
    pthread_mutex_unlock(&srv->q_lock);
}

static struct ohttp_conn *queue_pop(struct ohttp_server *srv) {
    pthread_mutex_lock(&srv->q_lock);
    while (!srv->q_head && atomic_load(&srv->running)) {
        pthread_cond_wait(&srv->q_cond, &srv->q_lock);
    }
    struct ohttp_conn *c = srv->q_head;
    if (c) {
        srv->q_head = c->next_job;
        if (!srv->q_head) srv->q_tail = NULL;
    }
    pthread_mutex_unlock(&srv->q_lock);
    return c;
}

static bool connection_rearm(struct ohttp_server *srv, struct ohttp_conn *c) {
    struct epoll_event event = { .events = EPOLLIN | EPOLLONESHOT, .data.ptr = c };
    return epoll_ctl(srv->epoll_fd, EPOLL_CTL_MOD, c->fd, &event) == 0;
}

static void *worker_main(void *arg) {
    struct ohttp_server *srv = arg;
    while (atomic_load(&srv->running)) {
        struct ohttp_conn *c = queue_pop(srv);
        if (!c) break;
        pthread_mutex_lock(&c->ownership_lock);
        set_block(c->fd);
        bool alive = true;
        for (;;) {
            ohttp_request req;
            if (!parse_request(c, &req) || c->need_total == 0 || c->buf_len < c->need_total) {
                alive = false;
                break;
            }
            size_t consumed = c->need_total;
            c->status_recorded = false;
            srv->cfg.handler(&req, srv->cfg.user);
            if (!c->keep_alive) {
                alive = false;
                break;
            }
            size_t remaining = c->buf_len - consumed;
            if (remaining) memmove(c->buf, c->buf + consumed, remaining);
            c->buf_len = remaining;
            c->need_total = 0;
            c->body_start = 0;
            c->continue_sent = false;
            c->streaming = false;
            c->chunked_out = false;
            c->state = CONN_READING;
            if (!remaining) break;
            ohttp_request probe;
            if (!parse_request(c, &probe)) {
                alive = false;
                break;
            }
            if (c->need_total == 0 || c->buf_len < c->need_total) break;
        }
        if (alive) {
            set_nonblock(c->fd);
            alive = connection_rearm(srv, c);
        }
        pthread_mutex_unlock(&c->ownership_lock);
        if (!alive) conn_free(c);
    }
    return NULL;
}

static bool handle_readable(struct ohttp_server *srv, struct ohttp_conn *c) {
    for (;;) {
        if (c->buf_len == c->buf_cap) {
            if (c->buf_cap >= OHTTP_MAX_BUFFER) return false;
            size_t ncap = c->buf_cap ? c->buf_cap * 2 : 16384;
            if (ncap > OHTTP_MAX_BUFFER) ncap = OHTTP_MAX_BUFFER;
            char *nb;
            if (c->buf == c->inline_buf) {
                nb = malloc(ncap);
                if (nb) memcpy(nb, c->inline_buf, c->buf_len);
            } else {
                nb = realloc(c->buf, ncap);
            }
            if (!nb) return false;
            c->buf = nb;
            c->buf_cap = ncap;
        }
        ssize_t n = read(c->fd, c->buf + c->buf_len, c->buf_cap - c->buf_len);
        if (n > 0) {
            c->buf_len += (size_t)n;
            continue;
        }
        if (n == 0) return false;
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        if (errno == EINTR) continue;
        return false;
    }

    ohttp_request probe;
    if (!parse_request(c, &probe)) return false;
    bool complete = c->need_total > 0 && c->buf_len >= c->need_total;
    if (complete) {
        c->state = CONN_DISPATCHED;
        queue_push(srv, c);
        return true;
    } else {
        return connection_rearm(srv, c);
    }
}

static void *reactor_main(void *arg) {
    struct ohttp_server *srv = arg;
    struct epoll_event events[128];
    while (atomic_load(&srv->running)) {
        int n = epoll_wait(srv->epoll_fd, events, 128, 500);
        for (int i = 0; i < n; i++) {
            if (events[i].data.ptr == NULL) {
                for (;;) {
                    int fd = accept4(srv->listen_fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
                    if (fd < 0) break;
                    int one = 1;
                    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                    struct ohttp_conn *c = calloc(1, sizeof *c);
                    if (!c) { close(fd); continue; }
                    pthread_mutex_init(&c->ownership_lock, NULL);
                    pthread_mutex_lock(&c->ownership_lock);
                    c->fd = fd;
                    c->buf = c->inline_buf;
                    c->buf_cap = sizeof c->inline_buf;
                    c->srv = srv;
                    c->state = CONN_READING;
                    pthread_mutex_lock(&srv->all_lock);
                    c->next_all = srv->all_head;
                    srv->all_head = c;
                    pthread_mutex_unlock(&srv->all_lock);
                    struct epoll_event ev = { .events = EPOLLIN | EPOLLONESHOT, .data.ptr = c };
                    bool added = epoll_ctl(srv->epoll_fd, EPOLL_CTL_ADD, fd, &ev) == 0;
                    pthread_mutex_unlock(&c->ownership_lock);
                    if (!added) conn_free(c);
                }
                struct epoll_event lev = { .events = EPOLLIN | EPOLLONESHOT, .data.ptr = NULL };
                epoll_ctl(srv->epoll_fd, EPOLL_CTL_MOD, srv->listen_fd, &lev);
            } else {
                struct ohttp_conn *c = events[i].data.ptr;
                pthread_mutex_lock(&c->ownership_lock);
                bool alive = !(events[i].events & (EPOLLHUP | EPOLLERR)) &&
                             handle_readable(srv, c);
                pthread_mutex_unlock(&c->ownership_lock);
                if (!alive) conn_free(c);
            }
        }
    }
    return NULL;
}

ohttp_server *ohttp_start(const ohttp_config *cfg) {
    if (!cfg || !cfg->handler || cfg->port == 0) return NULL;
    signal(SIGPIPE, SIG_IGN);
    ohttp_server *srv = calloc(1, sizeof *srv);
    if (!srv) return NULL;
    srv->listen_fd = -1;
    srv->epoll_fd = -1;
    srv->cfg = *cfg;
    if (srv->cfg.reactor_threads <= 0) srv->cfg.reactor_threads = 1;
    if (srv->cfg.worker_threads <= 0) srv->cfg.worker_threads = 8;
    atomic_store(&srv->running, true);
    pthread_mutex_init(&srv->q_lock, NULL);
    pthread_cond_init(&srv->q_cond, NULL);
    pthread_mutex_init(&srv->all_lock, NULL);

    srv->listen_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (srv->listen_fd < 0) {
        server_free_empty(srv);
        return NULL;
    }
    int one = 1;
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof one);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(cfg->port);
    if (cfg->bind_addr) {
        if (inet_pton(AF_INET, cfg->bind_addr, &addr.sin_addr) != 1) {
            server_free_empty(srv);
            return NULL;
        }
    } else {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }
    if (bind(srv->listen_fd, (struct sockaddr *)&addr, sizeof addr) != 0 ||
        listen(srv->listen_fd, 1024) != 0) {
        server_free_empty(srv);
        return NULL;
    }

    srv->epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (srv->epoll_fd < 0) {
        server_free_empty(srv);
        return NULL;
    }
    struct epoll_event lev = { .events = EPOLLIN | EPOLLONESHOT, .data.ptr = NULL };
    if (epoll_ctl(srv->epoll_fd, EPOLL_CTL_ADD, srv->listen_fd, &lev) != 0) {
        server_free_empty(srv);
        return NULL;
    }

    srv->reactors = calloc((size_t)srv->cfg.reactor_threads, sizeof(pthread_t));
    srv->workers = calloc((size_t)srv->cfg.worker_threads, sizeof(pthread_t));
    if (!srv->reactors || !srv->workers) {
        server_free_empty(srv);
        return NULL;
    }
    for (int i = 0; i < srv->cfg.reactor_threads; i++) {
        if (pthread_create(&srv->reactors[i], NULL, reactor_main, srv) != 0) {
            srv->cfg.reactor_threads = i;
            srv->cfg.worker_threads = 0;
            ohttp_stop(srv);
            ohttp_join(srv);
            return NULL;
        }
    }
    for (int i = 0; i < srv->cfg.worker_threads; i++) {
        if (pthread_create(&srv->workers[i], NULL, worker_main, srv) != 0) {
            srv->cfg.worker_threads = i;
            ohttp_stop(srv);
            ohttp_join(srv);
            return NULL;
        }
    }
    return srv;
}

int ohttp_join(ohttp_server *srv) {
    if (!srv) return -1;
    for (int i = 0; i < srv->cfg.reactor_threads; i++) pthread_join(srv->reactors[i], NULL);
    pthread_mutex_lock(&srv->q_lock);
    pthread_cond_broadcast(&srv->q_cond);
    pthread_mutex_unlock(&srv->q_lock);
    for (int i = 0; i < srv->cfg.worker_threads; i++) pthread_join(srv->workers[i], NULL);

    pthread_mutex_lock(&srv->all_lock);
    struct ohttp_conn *connection = srv->all_head;
    srv->all_head = NULL;
    pthread_mutex_unlock(&srv->all_lock);
    while (connection) {
        struct ohttp_conn *next = connection->next_all;
        connection->srv = NULL;
        conn_free(connection);
        connection = next;
    }
    if (srv->listen_fd >= 0) close(srv->listen_fd);
    if (srv->epoll_fd >= 0) close(srv->epoll_fd);
    pthread_mutex_destroy(&srv->all_lock);
    pthread_mutex_destroy(&srv->q_lock);
    pthread_cond_destroy(&srv->q_cond);
    free(srv->reactors);
    free(srv->workers);
    free(srv);
    return 0;
}

void ohttp_stop(ohttp_server *srv) {
    if (!srv || !atomic_exchange(&srv->running, false)) return;
    shutdown(srv->listen_fd, SHUT_RDWR);
    pthread_mutex_lock(&srv->q_lock);
    pthread_cond_broadcast(&srv->q_cond);
    pthread_mutex_unlock(&srv->q_lock);
}
