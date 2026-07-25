#define _GNU_SOURCE
#include "oproxy.h"

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define OPROXY_HOST_MAX 256
#define OPROXY_PREFIX_MAX 1024
#define OPROXY_IO_CHUNK (64u << 10)
#define OPROXY_INITIAL_HEADER (8u << 10)
#define OPROXY_MAX_HEADER (64u << 10)

typedef struct {
    struct sockaddr_storage addr;
    socklen_t addr_len;
    int family;
    int socktype;
    int protocol;
} proxy_addr;

struct oproxy_target {
    char host[OPROXY_HOST_MAX];
    char port[8];
    char prefix[OPROXY_PREFIX_MAX];
    proxy_addr *addresses;
    size_t address_count;
    pthread_mutex_t pool_lock;
    int *idle_fds;
    size_t idle_count;
    size_t max_idle;
    atomic_ullong connections_opened;
    atomic_ullong connections_reused;
    atomic_ullong failures;
};

typedef struct {
    bool chunked;
    bool has_content_length;
    bool no_body;
    bool connection_close;
    size_t content_length;
} response_framing;

typedef enum {
    CHUNK_SIZE,
    CHUNK_EXTENSION,
    CHUNK_SIZE_LF,
    CHUNK_DATA,
    CHUNK_DATA_CR,
    CHUNK_DATA_LF,
    CHUNK_TRAILER_START,
    CHUNK_TRAILER,
    CHUNK_TRAILER_LF,
    CHUNK_FINAL_LF,
    CHUNK_DONE,
} chunk_state;

typedef struct {
    chunk_state state;
    uint64_t value;
    uint64_t remaining;
    size_t line_len;
    bool have_digit;
} chunk_parser;

static void set_error(char *dst, size_t cap, const char *fmt, ...) {
    if (!dst || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dst, cap, fmt, ap);
    va_end(ap);
}

static int64_t monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static bool parse_base_url(const char *url, oproxy_target *out,
                           char *error, size_t error_cap) {
    static const char scheme[] = "http://";
    if (!url || strncmp(url, scheme, sizeof(scheme) - 1) != 0) {
        set_error(error, error_cap, "upstream must use http://");
        return false;
    }
    const char *authority = url + sizeof(scheme) - 1;
    const char *slash = strchr(authority, '/');
    const char *authority_end = slash ? slash : authority + strlen(authority);
    const char *host_begin = authority;
    const char *host_end = authority_end;
    const char *port_begin = NULL;

    if (host_begin < authority_end && *host_begin == '[') {
        const char *close = memchr(host_begin, ']', (size_t)(authority_end - host_begin));
        if (!close || (close + 1 < authority_end && close[1] != ':')) {
            set_error(error, error_cap, "invalid bracketed upstream host");
            return false;
        }
        host_begin++;
        host_end = close;
        if (close + 1 < authority_end) port_begin = close + 2;
    } else {
        const char *colon = memchr(authority, ':', (size_t)(authority_end - authority));
        if (colon) {
            host_end = colon;
            port_begin = colon + 1;
        }
    }

    size_t host_len = (size_t)(host_end - host_begin);
    size_t port_len = port_begin ? (size_t)(authority_end - port_begin) : 2;
    if (host_len == 0 || host_len >= sizeof out->host ||
        port_len == 0 || port_len >= sizeof out->port) {
        set_error(error, error_cap, "invalid upstream host or port");
        return false;
    }
    memcpy(out->host, host_begin, host_len);
    out->host[host_len] = 0;
    if (port_begin) memcpy(out->port, port_begin, port_len);
    else memcpy(out->port, "80", 2);
    out->port[port_len] = 0;

    if (slash) {
        size_t prefix_len = strlen(slash);
        while (prefix_len > 0 && slash[prefix_len - 1] == '/') prefix_len--;
        if (prefix_len >= sizeof out->prefix) {
            set_error(error, error_cap, "upstream URL prefix is too long");
            return false;
        }
        memcpy(out->prefix, slash, prefix_len);
        out->prefix[prefix_len] = 0;
    }
    return true;
}

oproxy_target *oproxy_target_create(const char *base_url, int max_idle,
                                    char *error, size_t error_cap) {
    if (error && error_cap) error[0] = 0;
    oproxy_target *target = calloc(1, sizeof *target);
    if (!target) {
        set_error(error, error_cap, "could not allocate upstream target");
        return NULL;
    }
    if (!parse_base_url(base_url, target, error, error_cap)) {
        free(target);
        return NULL;
    }

    struct addrinfo hints = {0};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *resolved = NULL;
    int gai = getaddrinfo(target->host, target->port, &hints, &resolved);
    if (gai != 0) {
        set_error(error, error_cap, "upstream DNS failed: %s", gai_strerror(gai));
        free(target);
        return NULL;
    }
    size_t count = 0;
    for (const struct addrinfo *ai = resolved; ai; ai = ai->ai_next) {
        if (ai->ai_addrlen <= sizeof(struct sockaddr_storage)) count++;
    }
    /* Resolution can succeed while every result is unusable. Failing here beats
     * handing back a target that looks valid and can never connect. */
    if (count == 0) {
        set_error(error, error_cap, "upstream resolved to no usable addresses");
        freeaddrinfo(resolved);
        free(target);
        return NULL;
    }
    target->addresses = calloc(count, sizeof *target->addresses);
    if (!target->addresses) {
        set_error(error, error_cap, "could not allocate resolved upstream addresses");
        freeaddrinfo(resolved);
        free(target);
        return NULL;
    }
    for (const struct addrinfo *ai = resolved; ai; ai = ai->ai_next) {
        if (ai->ai_addrlen > sizeof(struct sockaddr_storage)) continue;
        proxy_addr *dst = &target->addresses[target->address_count++];
        memcpy(&dst->addr, ai->ai_addr, ai->ai_addrlen);
        dst->addr_len = (socklen_t)ai->ai_addrlen;
        dst->family = ai->ai_family;
        dst->socktype = ai->ai_socktype;
        dst->protocol = ai->ai_protocol;
    }
    freeaddrinfo(resolved);
    if (target->address_count == 0) {
        set_error(error, error_cap, "upstream resolved to no usable addresses");
        free(target->addresses);
        free(target);
        return NULL;
    }

    if (max_idle < 0) max_idle = 0;
    if (max_idle > 1024) max_idle = 1024;
    target->max_idle = (size_t)max_idle;
    if (target->max_idle) {
        target->idle_fds = malloc(target->max_idle * sizeof *target->idle_fds);
        if (!target->idle_fds) {
            set_error(error, error_cap, "could not allocate upstream connection pool");
            free(target->addresses);
            free(target);
            return NULL;
        }
    }
    pthread_mutex_init(&target->pool_lock, NULL);
    return target;
}

void oproxy_target_destroy(oproxy_target *target) {
    if (!target) return;
    pthread_mutex_lock(&target->pool_lock);
    for (size_t i = 0; i < target->idle_count; i++) close(target->idle_fds[i]);
    target->idle_count = 0;
    pthread_mutex_unlock(&target->pool_lock);
    pthread_mutex_destroy(&target->pool_lock);
    free(target->idle_fds);
    free(target->addresses);
    free(target);
}

void oproxy_target_snapshot(oproxy_target *target, oproxy_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!target) return;
    pthread_mutex_lock(&target->pool_lock);
    out->idle_connections = target->idle_count;
    pthread_mutex_unlock(&target->pool_lock);
    out->connections_opened = atomic_load_explicit(&target->connections_opened, memory_order_relaxed);
    out->connections_reused = atomic_load_explicit(&target->connections_reused, memory_order_relaxed);
    out->failures = atomic_load_explicit(&target->failures, memory_order_relaxed);
}

static bool wait_fd(int fd, short events, int64_t deadline) {
    for (;;) {
        int64_t remaining = deadline - monotonic_ms();
        if (remaining <= 0) return false;
        struct pollfd pfd = { .fd = fd, .events = events };
        int rc = poll(&pfd, 1, remaining > INT32_MAX ? INT32_MAX : (int)remaining);
        if (rc > 0) return (pfd.revents & (events | POLLHUP)) != 0;
        if (rc < 0 && errno == EINTR) continue;
        return false;
    }
}

static int connect_deadline(oproxy_target *target, int64_t deadline,
                            char *error, size_t error_cap) {
    int fd = -1;
    for (size_t i = 0; i < target->address_count; i++) {
        const proxy_addr *address = &target->addresses[i];
        fd = socket(address->family, address->socktype | SOCK_NONBLOCK, address->protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
        int rc = connect(fd, (const struct sockaddr *)&address->addr, address->addr_len);
        if (rc == 0) break;
        if (errno != EINPROGRESS || !wait_fd(fd, POLLOUT, deadline)) {
            close(fd);
            fd = -1;
            continue;
        }
        int socket_error = 0;
        socklen_t socket_error_len = sizeof socket_error;
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &socket_error_len) != 0 ||
            socket_error != 0) {
            close(fd);
            fd = -1;
            continue;
        }
        break;
    }
    if (fd < 0) {
        set_error(error, error_cap, "could not connect to upstream %s:%s",
                  target->host, target->port);
    } else {
        atomic_fetch_add_explicit(&target->connections_opened, 1, memory_order_relaxed);
    }
    return fd;
}

static bool idle_connection_healthy(int fd) {
    struct pollfd pfd = { .fd = fd, .events = POLLIN };
    int rc;
    do {
        rc = poll(&pfd, 1, 0);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0 || (rc > 0 && (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)))) return false;
    if (rc == 0) return true;
    char byte;
    ssize_t n = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    return n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
}

static int target_take_connection(oproxy_target *target, int64_t deadline,
                                  bool *reused, char *error, size_t error_cap) {
    *reused = false;
    for (;;) {
        pthread_mutex_lock(&target->pool_lock);
        int fd = target->idle_count ? target->idle_fds[--target->idle_count] : -1;
        pthread_mutex_unlock(&target->pool_lock);
        if (fd < 0) return connect_deadline(target, deadline, error, error_cap);
        if (idle_connection_healthy(fd)) {
            *reused = true;
            atomic_fetch_add_explicit(&target->connections_reused, 1, memory_order_relaxed);
            return fd;
        }
        close(fd);
    }
}

static void target_return_connection(oproxy_target *target, int fd) {
    if (fd < 0 || !idle_connection_healthy(fd)) {
        if (fd >= 0) close(fd);
        return;
    }
    pthread_mutex_lock(&target->pool_lock);
    if (target->idle_count < target->max_idle) {
        target->idle_fds[target->idle_count++] = fd;
        fd = -1;
    }
    pthread_mutex_unlock(&target->pool_lock);
    if (fd >= 0) close(fd);
}

static bool socket_write_all(int fd, const void *data, size_t len, int64_t deadline) {
    const char *p = data;
    while (len) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            if ((errno == EAGAIN || errno == EWOULDBLOCK) && wait_fd(fd, POLLOUT, deadline)) continue;
            return false;
        }
        if (n == 0) return false;
        p += n;
        len -= (size_t)n;
    }
    return true;
}

static ssize_t socket_read(int fd, void *data, size_t len, int64_t deadline) {
    for (;;) {
        ssize_t n = recv(fd, data, len, 0);
        if (n >= 0) return n;
        if (errno == EINTR) continue;
        if ((errno == EAGAIN || errno == EWOULDBLOCK) && wait_fd(fd, POLLIN, deadline)) continue;
        return -1;
    }
}

static bool reserve_bytes(char **buf, size_t needed, size_t *cap) {
    if (needed <= *cap) return true;
    size_t next = *cap ? *cap : 2048;
    while (next < needed) {
        if (next > SIZE_MAX / 2) return false;
        next *= 2;
    }
    char *grown = realloc(*buf, next);
    if (!grown) return false;
    *buf = grown;
    *cap = next;
    return true;
}

static bool append_bytes(char **buf, size_t *len, size_t *cap,
                         const void *data, size_t data_len) {
    if (data_len > SIZE_MAX - *len - 1 || !reserve_bytes(buf, *len + data_len + 1, cap)) return false;
    memcpy(*buf + *len, data, data_len);
    *len += data_len;
    (*buf)[*len] = 0;
    return true;
}

static bool append_fmt(char **buf, size_t *len, size_t *cap, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int needed = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (needed < 0 || (size_t)needed > SIZE_MAX - *len - 1 ||
        !reserve_bytes(buf, *len + (size_t)needed + 1, cap)) {
        va_end(ap);
        return false;
    }
    vsnprintf(*buf + *len, (size_t)needed + 1, fmt, ap);
    *len += (size_t)needed;
    va_end(ap);
    return true;
}

static bool safe_header(const oproxy_header *header) {
    if (!header || !header->name || !header->value || header->name_len == 0) return false;
    return memchr(header->name, '\r', header->name_len) == NULL &&
           memchr(header->name, '\n', header->name_len) == NULL &&
           memchr(header->value, '\r', header->value_len) == NULL &&
           memchr(header->value, '\n', header->value_len) == NULL;
}

static bool span_token_is(const char *begin, const char *end, const char *token) {
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

static bool parse_size_span(const char *begin, const char *end, size_t *out) {
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

static bool parse_response_headers(const char *data, size_t header_len,
                                   response_framing *out) {
    memset(out, 0, sizeof *out);
    const char *end = data + header_len;
    const char *line_end = memmem(data, header_len, "\r\n", 2);
    if (!line_end || line_end - data < 12 || memcmp(data, "HTTP/1.", 7) != 0) return false;
    bool http_10 = data[7] == '0';
    const char *space = memchr(data, ' ', (size_t)(line_end - data));
    if (!space || line_end - space < 4 || space[1] < '0' || space[1] > '9' ||
        space[2] < '0' || space[2] > '9' || space[3] < '0' || space[3] > '9') return false;
    int status = (space[1] - '0') * 100 + (space[2] - '0') * 10 + (space[3] - '0');
    out->no_body = (status >= 100 && status < 200) || status == 204 || status == 304;
    out->connection_close = http_10;

    bool connection_keep_alive = false;
    const char *line = line_end + 2;
    while (line < end - 2) {
        line_end = memmem(line, (size_t)(end - line), "\r\n", 2);
        if (!line_end || line_end == line) break;
        const char *colon = memchr(line, ':', (size_t)(line_end - line));
        if (!colon) return false;
        const char *value = colon + 1;
        while (value < line_end && (*value == ' ' || *value == '\t')) value++;
        const char *value_end = line_end;
        while (value_end > value && (value_end[-1] == ' ' || value_end[-1] == '\t')) value_end--;
        size_t name_len = (size_t)(colon - line);
        if (name_len == 14 && strncasecmp(line, "Content-Length", 14) == 0) {
            size_t parsed = 0;
            if (!parse_size_span(value, value_end, &parsed) ||
                (out->has_content_length && parsed != out->content_length)) return false;
            out->has_content_length = true;
            out->content_length = parsed;
        } else if (name_len == 17 && strncasecmp(line, "Transfer-Encoding", 17) == 0) {
            if (span_token_is(value, value_end, "chunked")) out->chunked = true;
        } else if (name_len == 10 && strncasecmp(line, "Connection", 10) == 0) {
            if (span_token_is(value, value_end, "close")) out->connection_close = true;
            if (span_token_is(value, value_end, "keep-alive")) connection_keep_alive = true;
        }
        line = line_end + 2;
    }
    if (http_10 && connection_keep_alive) out->connection_close = false;
    if (out->chunked) out->has_content_length = false;
    return true;
}

static int hex_value(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool chunk_feed(chunk_parser *parser, const char *data, size_t len,
                       size_t *consumed) {
    size_t i = 0;
    while (i < len && parser->state != CHUNK_DONE) {
        unsigned char c = (unsigned char)data[i];
        switch (parser->state) {
        case CHUNK_SIZE: {
            int digit = hex_value(c);
            if (digit >= 0) {
                if (parser->value > (UINT64_MAX - (unsigned)digit) / 16) return false;
                parser->value = parser->value * 16 + (unsigned)digit;
                parser->have_digit = true;
                parser->line_len++;
                i++;
            } else if (c == ';' && parser->have_digit) {
                parser->state = CHUNK_EXTENSION;
                parser->line_len++;
                i++;
            } else if (c == '\r' && parser->have_digit) {
                parser->state = CHUNK_SIZE_LF;
                i++;
            } else {
                return false;
            }
            break;
        }
        case CHUNK_EXTENSION:
            if (++parser->line_len > 8192 || c == '\n') return false;
            if (c == '\r') parser->state = CHUNK_SIZE_LF;
            i++;
            break;
        case CHUNK_SIZE_LF:
            if (c != '\n') return false;
            i++;
            if (parser->value == 0) {
                parser->state = CHUNK_TRAILER_START;
            } else {
                parser->remaining = parser->value;
                parser->state = CHUNK_DATA;
            }
            break;
        case CHUNK_DATA: {
            size_t available = len - i;
            size_t take = parser->remaining < available ? (size_t)parser->remaining : available;
            parser->remaining -= take;
            i += take;
            if (parser->remaining == 0) parser->state = CHUNK_DATA_CR;
            break;
        }
        case CHUNK_DATA_CR:
            if (c != '\r') return false;
            parser->state = CHUNK_DATA_LF;
            i++;
            break;
        case CHUNK_DATA_LF:
            if (c != '\n') return false;
            parser->state = CHUNK_SIZE;
            parser->value = 0;
            parser->line_len = 0;
            parser->have_digit = false;
            i++;
            break;
        case CHUNK_TRAILER_START:
            parser->line_len = 0;
            if (c == '\r') parser->state = CHUNK_FINAL_LF;
            else if (c == '\n') return false;
            else parser->state = CHUNK_TRAILER;
            i++;
            break;
        case CHUNK_TRAILER:
            if (++parser->line_len > OPROXY_MAX_HEADER || c == '\n') return false;
            if (c == '\r') parser->state = CHUNK_TRAILER_LF;
            i++;
            break;
        case CHUNK_TRAILER_LF:
            if (c != '\n') return false;
            parser->state = CHUNK_TRAILER_START;
            i++;
            break;
        case CHUNK_FINAL_LF:
            if (c != '\n') return false;
            parser->state = CHUNK_DONE;
            i++;
            break;
        case CHUNK_DONE:
            break;
        }
    }
    *consumed = i;
    return true;
}

static bool relay_bytes(oproxy_sink sink, void *sink_user, const void *data, size_t len,
                        oproxy_result *out, char *error, size_t error_cap) {
    if (!len) return true;
    if (!sink(data, len, sink_user)) {
        set_error(error, error_cap, "downstream disconnected");
        return false;
    }
    if (out) out->bytes_relayed += len;
    return true;
}

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
                         char *error, size_t error_cap) {
    if (out) memset(out, 0, sizeof *out);
    if (error && error_cap) error[0] = 0;
    if (!target || !sink || !method || !path || method_len == 0 || path_len == 0) {
        set_error(error, error_cap, "invalid proxy request");
        return false;
    }
    int64_t deadline = monotonic_ms() + (timeout_ms > 0 ? timeout_ms : 600000);
    bool reused = false;
    int fd = target_take_connection(target, deadline, &reused, error, error_cap);
    if (fd < 0) {
        atomic_fetch_add_explicit(&target->failures, 1, memory_order_relaxed);
        return false;
    }
    if (out) out->upstream_reused = reused;

    char *head = NULL;
    size_t head_len = 0, head_cap = 0;
    bool built = append_fmt(&head, &head_len, &head_cap, "%.*s %s%.*s",
                            (int)method_len, method, target->prefix, (int)path_len, path);
    if (built && query && query_len) {
        built = append_fmt(&head, &head_len, &head_cap, "?%.*s", (int)query_len, query);
    }
    if (built) {
        bool ipv6 = strchr(target->host, ':') != NULL;
        built = append_fmt(&head, &head_len, &head_cap,
                           " HTTP/1.1\r\nHost: %s%s%s:%s\r\n"
                           "Connection: keep-alive\r\nContent-Length: %zu\r\n",
                           ipv6 ? "[" : "", target->host, ipv6 ? "]" : "", target->port,
                           body_len);
    }
    if (built && content_type && content_type_len) {
        built = append_fmt(&head, &head_len, &head_cap, "Content-Type: %.*s\r\n",
                           (int)content_type_len, content_type);
    }
    for (int i = 0; built && i < header_count; i++) {
        if (!safe_header(&headers[i])) continue;
        built = append_fmt(&head, &head_len, &head_cap, "%.*s: %.*s\r\n",
                           (int)headers[i].name_len, headers[i].name,
                           (int)headers[i].value_len, headers[i].value);
    }
    if (built) built = append_bytes(&head, &head_len, &head_cap, "\r\n", 2);
    if (!built) {
        set_error(error, error_cap, "could not allocate upstream request");
        free(head);
        close(fd);
        atomic_fetch_add_explicit(&target->failures, 1, memory_order_relaxed);
        return false;
    }

    bool ok = socket_write_all(fd, head, head_len, deadline) &&
              (!body_len || socket_write_all(fd, body, body_len, deadline));
    free(head);
    if (!ok) {
        set_error(error, error_cap, "timed out writing upstream request");
        close(fd);
        atomic_fetch_add_explicit(&target->failures, 1, memory_order_relaxed);
        return false;
    }

    char initial[OPROXY_INITIAL_HEADER + 1];
    char *response = initial;
    size_t response_cap = OPROXY_INITIAL_HEADER;
    size_t response_len = 0;
    char *header_end = NULL;
    while (!header_end) {
        if (response_len == response_cap) {
            if (response_cap >= OPROXY_MAX_HEADER) {
                set_error(error, error_cap, "upstream response headers are too large");
                ok = false;
                break;
            }
            size_t next_cap = response_cap * 2;
            if (next_cap > OPROXY_MAX_HEADER) next_cap = OPROXY_MAX_HEADER;
            char *grown = malloc(next_cap + 1);
            if (!grown) {
                set_error(error, error_cap, "could not allocate upstream response headers");
                ok = false;
                break;
            }
            memcpy(grown, response, response_len);
            if (response != initial) free(response);
            response = grown;
            response_cap = next_cap;
        }
        ssize_t n = socket_read(fd, response + response_len, response_cap - response_len, deadline);
        if (n <= 0) {
            set_error(error, error_cap, n == 0 ? "upstream closed without a response" :
                      "upstream response timed out");
            ok = false;
            break;
        }
        response_len += (size_t)n;
        response[response_len] = 0;
        header_end = memmem(response, response_len, "\r\n\r\n", 4);
    }

    response_framing framing;
    size_t response_header_len = header_end ? (size_t)(header_end - response) + 4 : 0;
    if (ok && !parse_response_headers(response, response_header_len, &framing)) {
        set_error(error, error_cap, "invalid upstream HTTP response");
        ok = false;
    }
    if (!ok) {
        if (response != initial) free(response);
        close(fd);
        atomic_fetch_add_explicit(&target->failures, 1, memory_order_relaxed);
        return false;
    }

    if (out) {
        out->response_started = true;
        out->downstream_close = framing.connection_close ||
                                (!framing.no_body && !framing.chunked && !framing.has_content_length);
    }
    ok = relay_bytes(sink, sink_user, response, response_header_len, out, error, error_cap);
    size_t buffered_body = response_len - response_header_len;
    const char *buffered = response + response_header_len;
    bool complete = framing.no_body;
    bool reusable = !framing.connection_close;
    size_t remaining = framing.has_content_length ? framing.content_length : 0;
    chunk_parser chunks = { .state = CHUNK_SIZE };

    if (ok && !complete && framing.has_content_length) {
        size_t take = buffered_body < remaining ? buffered_body : remaining;
        ok = relay_bytes(sink, sink_user, buffered, take, out, error, error_cap);
        remaining -= take;
        if (buffered_body != take) {
            set_error(error, error_cap, "upstream sent bytes beyond Content-Length");
            ok = false;
        }
        complete = remaining == 0;
    } else if (ok && !complete && framing.chunked) {
        size_t consumed = 0;
        ok = chunk_feed(&chunks, buffered, buffered_body, &consumed) &&
             relay_bytes(sink, sink_user, buffered, consumed, out, error, error_cap);
        if (ok && consumed != buffered_body) {
            set_error(error, error_cap, "upstream sent bytes beyond chunked response");
            ok = false;
        }
        complete = chunks.state == CHUNK_DONE;
    } else if (ok && !complete && buffered_body) {
        ok = relay_bytes(sink, sink_user, buffered, buffered_body, out, error, error_cap);
    } else if (ok && complete && buffered_body) {
        set_error(error, error_cap, "upstream sent a body for a bodyless response");
        ok = false;
    }
    if (response != initial) free(response);

    char *chunk = NULL;
    if (ok && !complete) {
        chunk = malloc(OPROXY_IO_CHUNK);
        if (!chunk) {
            set_error(error, error_cap, "could not allocate proxy buffer");
            ok = false;
        }
    }
    while (ok && !complete) {
        size_t wanted = OPROXY_IO_CHUNK;
        if (framing.has_content_length && remaining < wanted) wanted = remaining;
        ssize_t n = socket_read(fd, chunk, wanted, deadline);
        if (n < 0) {
            set_error(error, error_cap, "upstream response timed out");
            ok = false;
            break;
        }
        if (n == 0) {
            if (!framing.has_content_length && !framing.chunked) {
                complete = true;
                reusable = false;
            } else {
                set_error(error, error_cap, "upstream closed during a framed response");
                ok = false;
            }
            break;
        }
        size_t relay_len = (size_t)n;
        if (framing.chunked) {
            size_t consumed = 0;
            if (!chunk_feed(&chunks, chunk, (size_t)n, &consumed)) {
                set_error(error, error_cap, "invalid upstream chunked response");
                ok = false;
                break;
            }
            relay_len = consumed;
            complete = chunks.state == CHUNK_DONE;
            if (complete && consumed != (size_t)n) {
                set_error(error, error_cap, "upstream sent bytes beyond chunked response");
                ok = false;
                break;
            }
        } else if (framing.has_content_length) {
            remaining -= (size_t)n;
            complete = remaining == 0;
        }
        ok = relay_bytes(sink, sink_user, chunk, relay_len, out, error, error_cap);
    }
    free(chunk);

    if (ok && complete && reusable) target_return_connection(target, fd);
    else close(fd);
    if (!ok) atomic_fetch_add_explicit(&target->failures, 1, memory_order_relaxed);
    return ok;
}

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
                  char *error, size_t error_cap) {
    oproxy_target *target = oproxy_target_create(base_url, 0, error, error_cap);
    if (!target) return false;
    bool ok = oproxy_target_relay(target, method, method_len, path, path_len,
                                  query, query_len, body, body_len,
                                  content_type, content_type_len, headers, header_count,
                                  timeout_ms, sink, sink_user, out, error, error_cap);
    oproxy_target_destroy(target);
    return ok;
}
