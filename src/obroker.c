#include "obroker.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

static bool parse_url(const char *url, char *host, size_t host_cap, int *port) {
    if (!url || strncmp(url, "http://", 7) != 0) return false;
    const char *h = url + 7;
    const char *colon = strchr(h, ':');
    const char *slash = strchr(h, '/');
    size_t hl = colon ? (size_t)(colon - h) : slash ? (size_t)(slash - h) : strlen(h);
    if (hl == 0 || hl >= host_cap) return false;
    memcpy(host, h, hl);
    host[hl] = 0;
    *port = colon ? atoi(colon + 1) : 80;
    return *port > 0 && *port < 65536;
}

/* One blocking HTTP/1.0 POST; returns status or -1 on transport failure. */
static int post_json(const char *base_url, const char *path, const char *body,
                     int timeout_ms, char *resp, size_t resp_cap) {
    char host[64];
    int port = 0;
    if (!parse_url(base_url, host, sizeof host, &port)) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (strcmp(host, "localhost") == 0) snprintf(host, sizeof host, "127.0.0.1");
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) return -1;
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    struct timeval tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    struct timeval ctv = { 1, 0 };
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &ctv, sizeof ctv);
    if (connect(fd, (struct sockaddr *)&addr, sizeof addr) != 0) { close(fd); return -1; }
    char req[1024];
    int n = snprintf(req, sizeof req,
                     "POST %s HTTP/1.0\r\nHost: %s\r\nContent-Type: application/json\r\n"
                     "Content-Length: %zu\r\nConnection: close\r\n\r\n%s",
                     path, host, strlen(body), body);
    if (n <= 0 || (size_t)n >= sizeof req || send(fd, req, (size_t)n, MSG_NOSIGNAL) != n) {
        close(fd);
        return -1;
    }
    size_t got = 0;
    char buf[4096];
    for (;;) {
        ssize_t r = recv(fd, buf + got, sizeof buf - 1 - got, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        got += (size_t)r;
        if (got >= sizeof buf - 1) break;
    }
    close(fd);
    buf[got] = 0;
    int status = 0;
    if (got < 12 || sscanf(buf, "HTTP/%*s %d", &status) != 1) return -1;
    const char *b = strstr(buf, "\r\n\r\n");
    if (resp && resp_cap) snprintf(resp, resp_cap, "%s", b ? b + 4 : "");
    return status;
}

static bool json_bool(const char *s, const char *key) {
    const char *p = strstr(s, key);
    if (!p) return false;
    p += strlen(key);
    while (*p == '"' || *p == ':' || *p == ' ') p++;
    return strncmp(p, "true", 4) == 0;
}

static long json_int(const char *s, const char *key, long fallback) {
    const char *p = strstr(s, key);
    if (!p) return fallback;
    p += strlen(key);
    while (*p == '"' || *p == ':' || *p == ' ') p++;
    return strtol(p, NULL, 10);
}

int obroker_lease(const char *base_url, const char *owner, int pid, int mb, int min_mb,
                  otier tier, double ttl_s, int wait_ms, char *id_out, size_t id_cap,
                  int *waited_ms_out) {
    if (id_out && id_cap) id_out[0] = 0;
    if (waited_ms_out) *waited_ms_out = 0;
    char safe_owner[32];
    size_t k = 0;
    for (const char *p = owner ? owner : "anon"; *p && k + 1 < sizeof safe_owner; p++) {
        safe_owner[k++] = (*p == '"' || *p == '\\' || (unsigned char)*p < 0x20) ? '_' : *p;
    }
    safe_owner[k] = 0;
    char body[256];
    snprintf(body, sizeof body,
             "{\"owner\":\"%s\",\"pid\":%d,\"mb\":%d,\"min_mb\":%d,\"tier\":\"%s\","
             "\"ttl_s\":%.0f,\"wait_ms\":%d}",
             safe_owner, pid, mb, min_mb, otier_name(tier), ttl_s, wait_ms > 0 ? wait_ms : 0);
    char resp[512];
    int status = post_json(base_url, "/v1/gpu/lease", body, (wait_ms > 0 ? wait_ms : 0) + 5000,
                           resp, sizeof resp);
    if (status != 200) return -1;
    if (waited_ms_out) *waited_ms_out = (int)json_int(resp, "\"waited_ms\"", 0);
    if (!json_bool(resp, "\"granted\"")) return 0;
    const char *id = strstr(resp, "\"lease_id\":\"");
    if (id && id_out && id_cap) {
        id += 12;
        size_t i = 0;
        while (id[i] && id[i] != '"' && i + 1 < id_cap) { id_out[i] = id[i]; i++; }
        id_out[i] = 0;
    }
    long granted = json_int(resp, "\"mb\"", 0);
    return granted > 0 ? (int)granted : 0;
}

bool obroker_release(const char *base_url, const char *lease_id) {
    if (!lease_id || !lease_id[0]) return false;
    char body[128];
    snprintf(body, sizeof body, "{\"lease_id\":\"%s\"}", lease_id);
    return post_json(base_url, "/v1/gpu/release", body, 3000, NULL, 0) == 200;
}
