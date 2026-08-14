#define _GNU_SOURCE

#include "ohttp.h"
#include "ojson.h"
#include "oproxy.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define ASR_CAPTURE_LIMIT (OHTTP_MAX_BODY + (256u << 10))
#define ASR_HEALTH_LIMIT (1u << 20)

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    size_t limit;
} capture;

typedef struct {
    oproxy_target *local;
    oproxy_target *fallback;
    oproxy_target *gateway;
    char gateway_path[256];
    double min_free_vram_gib;
    int timeout_ms;
    int health_timeout_ms;
} asr_router;

static bool capture_sink(const void *data, size_t len, void *user) {
    capture *out = user;
    if (!out || len > out->limit - out->len) return false;
    size_t needed = out->len + len + 1;
    if (needed > out->cap) {
        size_t next = out->cap ? out->cap : 4096;
        while (next < needed) {
            if (next > out->limit / 2) {
                next = out->limit + 1;
                break;
            }
            next *= 2;
        }
        if (next > out->limit + 1) next = out->limit + 1;
        char *grown = realloc(out->data, next);
        if (!grown) return false;
        out->data = grown;
        out->cap = next;
    }
    memcpy(out->data + out->len, data, len);
    out->len += len;
    out->data[out->len] = 0;
    return true;
}

static void capture_release(capture *value) {
    if (!value) return;
    free(value->data);
    memset(value, 0, sizeof *value);
}

static int response_status(const capture *response) {
    if (!response || !response->data || response->len < 12 ||
        memcmp(response->data, "HTTP/1.", 7) != 0) {
        return 0;
    }
    const char *line_end = memmem(response->data, response->len, "\r\n", 2);
    const char *space = line_end ? memchr(response->data, ' ', (size_t)(line_end - response->data))
                                 : NULL;
    if (!space || line_end - space < 4 || space[1] < '0' || space[1] > '9' ||
        space[2] < '0' || space[2] > '9' || space[3] < '0' || space[3] > '9') {
        return 0;
    }
    return (space[1] - '0') * 100 + (space[2] - '0') * 10 + (space[3] - '0');
}

static const char *response_body(const capture *response, size_t *body_len) {
    if (body_len) *body_len = 0;
    if (!response || !response->data) return NULL;
    const char *separator = memmem(response->data, response->len, "\r\n\r\n", 4);
    if (!separator) return NULL;
    const char *body = separator + 4;
    if (body_len) *body_len = response->len - (size_t)(body - response->data);
    return body;
}

static bool relay_capture(oproxy_target *target, const char *method, const char *path,
                          size_t path_len, const char *query, size_t query_len,
                          const char *body, size_t body_len,
                          const char *content_type, const oproxy_header *headers,
                          int header_count, int timeout_ms, size_t limit, capture *out,
                          oproxy_result *result, char *error, size_t error_cap) {
    memset(out, 0, sizeof *out);
    out->limit = limit;
    return oproxy_target_relay(
        target, method, strlen(method), path, path_len, query, query_len,
        body, body_len, content_type, content_type ? strlen(content_type) : 0, headers,
        header_count, timeout_ms, capture_sink, out, result, error, error_cap);
}

static bool header_is(const ohttp_header *header, const char *name) {
    size_t len = strlen(name);
    return header->name_len == len && strncasecmp(header->name, name, len) == 0;
}

static bool forwarded_header(const ohttp_header *header) {
    static const char *const blocked[] = {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
        "content-type", "x-omniserve-internal",
    };
    for (size_t i = 0; i < sizeof blocked / sizeof blocked[0]; i++) {
        if (header_is(header, blocked[i])) return false;
    }
    return true;
}

static int request_headers(const ohttp_request *req, oproxy_header *out, int cap) {
    int count = 0;
    for (int i = 0; i < req->header_count && count < cap - 1; i++) {
        if (!forwarded_header(&req->headers[i])) continue;
        out[count++] = (oproxy_header){
            .name = req->headers[i].name,
            .name_len = req->headers[i].name_len,
            .value = req->headers[i].value,
            .value_len = req->headers[i].value_len,
        };
    }
    static const char internal_name[] = "X-Omniserve-Internal";
    static const char internal_value[] = "local";
    out[count++] = (oproxy_header){
        .name = internal_name,
        .name_len = sizeof internal_name - 1,
        .value = internal_value,
        .value_len = sizeof internal_value - 1,
    };
    return count;
}

static bool json_value(const char *json, size_t len, const char *key, oj_tok *tokens,
                       int *token_count, int *value) {
    if (*token_count < 0) *token_count = oj_parse(json, len, tokens, 48);
    if (*token_count <= 0 || tokens[0].type != OJ_OBJECT) return false;
    *value = oj_obj_get(json, tokens, *token_count, 0, key);
    return *value >= 0;
}

static bool response_json(const capture *response, const char **json, size_t *json_len) {
    if (response_status(response) != 200) return false;
    *json = response_body(response, json_len);
    return *json != NULL;
}

static bool health_request(oproxy_target *target, const char *path, int timeout_ms,
                           capture *response) {
    char error[192];
    oproxy_result result;
    return target && relay_capture(target, "GET", path, strlen(path), NULL, 0, NULL, 0,
                                   NULL, NULL, 0,
                                   timeout_ms, ASR_HEALTH_LIMIT, response, &result, error,
                                   sizeof error);
}

static bool local_capacity(asr_router *router, const char **reason) {
    capture health = {0};
    if (!health_request(router->local, "/health", router->health_timeout_ms, &health)) {
        capture_release(&health);
        *reason = "local_unhealthy";
        return false;
    }
    const char *json = NULL;
    size_t json_len = 0;
    if (!response_json(&health, &json, &json_len)) {
        capture_release(&health);
        *reason = "local_unhealthy";
        return false;
    }
    oj_tok tokens[48];
    int ntokens = -1;
    int value = -1;
    bool ready = json_value(json, json_len, "ready", tokens, &ntokens, &value) &&
                 oj_bool(json, &tokens[value], false);
    bool held = json_value(json, json_len, "held", tokens, &ntokens, &value) &&
                oj_bool(json, &tokens[value], false);
    int device_token = -1;
    bool cuda_device = json_value(json, json_len, "device", tokens, &ntokens, &device_token) &&
                       oj_str_eq(json, &tokens[device_token], "cuda");
    double required = 0.0;
    if (json_value(json, json_len, "vram_required_gib", tokens, &ntokens, &value)) {
        required = oj_number(json, &tokens[value], 0.0);
    }
    capture_release(&health);
    if (!ready || held) {
        *reason = "local_not_ready";
        return false;
    }
    if (!cuda_device) {
        *reason = "local_cpu";
        return true;
    }

    capture gateway = {0};
    if (!health_request(router->gateway, router->gateway_path, router->health_timeout_ms,
                        &gateway) ||
        !response_json(&gateway, &json, &json_len)) {
        capture_release(&gateway);
        *reason = "gateway_unhealthy";
        return false;
    }
    ntokens = -1;
    bool available = json_value(json, json_len, "vram_available", tokens, &ntokens, &value) &&
                     oj_bool(json, &tokens[value], false);
    double free_gib = 0.0;
    if (json_value(json, json_len, "vram_free_gib", tokens, &ntokens, &value)) {
        free_gib = oj_number(json, &tokens[value], 0.0);
    }
    capture_release(&gateway);
    if (!available) {
        *reason = "vram_unknown";
        return false;
    }
    if (required < router->min_free_vram_gib) required = router->min_free_vram_gib;
    if (free_gib < required) {
        *reason = "vram_busy";
        return false;
    }
    *reason = "local_gpu";
    return true;
}

static bool transient_status(int status) {
    return status == 429 || status == 502 || status == 503 || status == 504;
}

static void respond_json(ohttp_request *req, int status, const char *body) {
    ohttp_respond_str(req, status, "application/json", body);
}

static bool forward_response(ohttp_request *req, const capture *response, const char *backend,
                             const char *attempts, bool downstream_close) {
    const char *separator = response && response->data
        ? memmem(response->data, response->len, "\r\n\r\n", 4) : NULL;
    if (!separator) return false;
    size_t head_len = (size_t)(separator - response->data);
    char added[320];
    int added_len = snprintf(added, sizeof added,
                             "\r\nX-Omniserve-ASR-Backend: %s"
                             "\r\nX-Omniserve-ASR-Attempts: %s",
                             backend, attempts);
    if (added_len < 0 || (size_t)added_len >= sizeof added) return false;
    bool ok = ohttp_raw_write(req, response->data, head_len) &&
              ohttp_raw_write(req, added, (size_t)added_len) &&
              ohttp_raw_write(req, separator, response->len - head_len);
    if (downstream_close) ohttp_force_close(req);
    return ok;
}

static void handle_get(ohttp_request *req, asr_router *router) {
    if (!ohttp_path_is(req, "/health") && !ohttp_path_is(req, "/status")) {
        respond_json(req, 404, "{\"error\":\"not found\"}");
        return;
    }
    const char *reason = NULL;
    bool usable = local_capacity(router, &reason);
    char body[256];
    snprintf(body, sizeof body,
             "{\"status\":\"ok\",\"local_usable\":%s,\"reason\":\"%s\","
             "\"fallback_configured\":%s}",
             usable ? "true" : "false", reason, router->fallback ? "true" : "false");
    respond_json(req, 200, body);
}

static bool attempt_relay(ohttp_request *req, oproxy_target *target, asr_router *router,
                          capture *response, oproxy_result *result) {
    oproxy_header headers[OHTTP_MAX_HEADERS + 1];
    int header_count = request_headers(req, headers, OHTTP_MAX_HEADERS + 1);
    size_t content_type_len = 0;
    const char *content_type = ohttp_req_header(req, "Content-Type", &content_type_len);
    char content_type_copy[256];
    if (content_type && content_type_len < sizeof content_type_copy) {
        memcpy(content_type_copy, content_type, content_type_len);
        content_type_copy[content_type_len] = 0;
        content_type = content_type_copy;
    } else if (content_type) {
        content_type = "application/octet-stream";
    }
    char error[192];
    return relay_capture(target, "POST", req->path, req->path_len, req->query,
                         req->query_len, req->body, req->body_len, content_type, headers,
                         header_count, router->timeout_ms,
                         ASR_CAPTURE_LIMIT, response, result, error, sizeof error);
}

static void handle_post(ohttp_request *req, asr_router *router) {
    if (req->body_len == 0) {
        respond_json(req, 400, "{\"error\":\"empty request\"}");
        return;
    }
    const char *reason = NULL;
    bool usable = local_capacity(router, &reason);
    capture response = {0};
    oproxy_result result = {0};
    char attempts[192];
    int status = 0;

    if (usable && attempt_relay(req, router->local, router, &response, &result)) {
        status = response_status(&response);
        snprintf(attempts, sizeof attempts, "local:%d", status);
        if (status && !transient_status(status)) {
            if (!forward_response(req, &response, "local", attempts, result.downstream_close)) {
                respond_json(req, 502, "{\"error\":\"invalid local ASR response\"}");
            }
            capture_release(&response);
            return;
        }
        capture_release(&response);
    } else if (usable) {
        snprintf(attempts, sizeof attempts, "local:unavailable");
        capture_release(&response);
    } else {
        snprintf(attempts, sizeof attempts, "local:%s", reason);
    }

    if (!router->fallback) {
        char body[384];
        snprintf(body, sizeof body,
                 "{\"error\":\"local ASR unavailable and no fallback configured\","
                 "\"attempts\":\"%s\"}", attempts);
        respond_json(req, 503, body);
        return;
    }

    capture fallback = {0};
    oproxy_result fallback_result = {0};
    if (!attempt_relay(req, router->fallback, router, &fallback, &fallback_result)) {
        char body[384];
        size_t used = strlen(attempts);
        snprintf(attempts + used, sizeof attempts - used, ",fallback:unavailable");
        snprintf(body, sizeof body,
                 "{\"error\":\"all ASR backends unavailable\",\"attempts\":\"%s\"}",
                 attempts);
        capture_release(&fallback);
        respond_json(req, 502, body);
        return;
    }
    status = response_status(&fallback);
    size_t used = strlen(attempts);
    snprintf(attempts + used, sizeof attempts - used, ",fallback:%d", status);
    if (!status || !forward_response(req, &fallback, "fallback", attempts,
                                     fallback_result.downstream_close)) {
        capture_release(&fallback);
        respond_json(req, 502, "{\"error\":\"invalid fallback ASR response\"}");
        return;
    }
    capture_release(&fallback);
}

static void route(ohttp_request *req, void *user) {
    asr_router *router = user;
    if (ohttp_method_is(req, "GET")) {
        handle_get(req, router);
    } else if (ohttp_method_is(req, "POST")) {
        handle_post(req, router);
    } else {
        respond_json(req, 405, "{\"error\":\"method not allowed\"}");
    }
}

static double env_double(const char *name, double fallback, double min, double max) {
    const char *raw = getenv(name);
    if (!raw || !*raw) return fallback;
    char *end = NULL;
    errno = 0;
    double value = strtod(raw, &end);
    if (errno || end == raw || *end || value < min || value > max) return fallback;
    return value;
}

static int env_int(const char *name, int fallback, int min, int max) {
    double value = env_double(name, fallback, min, max);
    return (int)value;
}

static bool split_status_url(const char *url, char **base, char *path, size_t path_cap) {
    const char *scheme = strstr(url, "://");
    if (!scheme) return false;
    const char *slash = strchr(scheme + 3, '/');
    size_t base_len = slash ? (size_t)(slash - url) : strlen(url);
    *base = malloc(base_len + 1);
    if (!*base) return false;
    memcpy(*base, url, base_len);
    (*base)[base_len] = 0;
    const char *status_path = slash ? slash : "/status";
    if (strlen(status_path) >= path_cap) {
        free(*base);
        *base = NULL;
        return false;
    }
    strcpy(path, status_path);
    return true;
}

static oproxy_target *make_target(const char *url, int idle, const char *label) {
    if (!url || !*url) return NULL;
    char error[256];
    oproxy_target *target = oproxy_target_create(url, idle, error, sizeof error);
    if (!target) fprintf(stderr, "asr-router: invalid %s upstream: %s\n", label, error);
    return target;
}

int main(void) {
    const char *local_url = getenv("OMNISERVE_ASR_LOCAL_UPSTREAM");
    if (!local_url || !*local_url) local_url = "http://127.0.0.1:9097";
    const char *fallback_url = getenv("OMNISERVE_ASR_FALLBACK_UPSTREAM");
    const char *gateway_url = getenv("OMNISERVE_ASR_GATEWAY_STATUS");
    if (!gateway_url || !*gateway_url) gateway_url = "http://127.0.0.1:8791/status";

    asr_router router = {
        .local = make_target(local_url, 4, "local"),
        .fallback = make_target(fallback_url, 4, "fallback"),
        .min_free_vram_gib = env_double("OMNISERVE_ASR_MIN_FREE_VRAM_GIB", 3.0, 0.0, 1024.0),
        .timeout_ms = (int)(1000.0 * env_double("OMNISERVE_ASR_TIMEOUT_S", 600.0, 0.1, 3600.0)),
        .health_timeout_ms = (int)(1000.0 * env_double("OMNISERVE_ASR_HEALTH_TIMEOUT_S", 1.5, 0.05, 60.0)),
    };
    if (!router.local) return 2;

    char *gateway_base = NULL;
    if (!split_status_url(gateway_url, &gateway_base, router.gateway_path,
                          sizeof router.gateway_path)) {
        fprintf(stderr, "asr-router: invalid gateway status URL\n");
        oproxy_target_destroy(router.local);
        oproxy_target_destroy(router.fallback);
        return 2;
    }
    router.gateway = make_target(gateway_base, 2, "gateway status");
    free(gateway_base);
    if (!router.gateway) {
        oproxy_target_destroy(router.local);
        oproxy_target_destroy(router.fallback);
        return 2;
    }

    const char *bind = getenv("OMNISERVE_ASR_BIND");
    if (!bind || !*bind) bind = "127.0.0.1";
    int port = env_int("OMNISERVE_ASR_PORT", 9096, 1, 65535);
    ohttp_config config = {
        .port = (unsigned short)port,
        .bind_addr = bind,
        .reactor_threads = env_int("OMNISERVE_ASR_REACTORS", 1, 1, 64),
        .worker_threads = env_int("OMNISERVE_ASR_WORKERS", 8, 1, 256),
        .handler = route,
        .user = &router,
    };
    ohttp_server *server = ohttp_start(&config);
    if (!server) {
        fprintf(stderr, "asr-router: could not listen on %s:%d\n", bind, port);
        oproxy_target_destroy(router.gateway);
        oproxy_target_destroy(router.fallback);
        oproxy_target_destroy(router.local);
        return 1;
    }
    fprintf(stderr, "omniserve native ASR router listening on %s:%d\n", bind, port);
    int result = ohttp_join(server);
    oproxy_target_destroy(router.gateway);
    oproxy_target_destroy(router.fallback);
    oproxy_target_destroy(router.local);
    return result;
}
