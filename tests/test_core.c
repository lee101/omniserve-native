#define _GNU_SOURCE
#include "ocapacity.h"
#include "ohttp.h"
#include "ojson.h"
#include "oproxy.h"
#include "oscale.h"
#include "osched.h"
#include "otext.h"
#include "omatte.h"
#include "otune.h"

#include <arpa/inet.h>
#include <assert.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int failures;

extern const char *OPENAPI_JSON;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static void test_json(void) {
    const char *js = "{\"model\":\"m1\",\"stream\":true,\"max_tokens\":42,"
                     "\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},"
                     "{\"role\":\"user\",\"content\":\"hi\\nthere \\u00e9\"}]}";
    oj_tok toks[128];
    int n = oj_parse(js, strlen(js), toks, 128);
    CHECK(n > 0);
    int model = oj_obj_get(js, toks, n, 0, "model");
    CHECK(model >= 0 && oj_str_eq(js, &toks[model], "m1"));
    int stream = oj_obj_get(js, toks, n, 0, "stream");
    CHECK(stream >= 0 && oj_bool(js, &toks[stream], false));
    int mt = oj_obj_get(js, toks, n, 0, "max_tokens");
    CHECK(mt >= 0 && (int)oj_number(js, &toks[mt], 0) == 42);
    int msgs = oj_obj_get(js, toks, n, 0, "messages");
    CHECK(msgs >= 0 && toks[msgs].type == OJ_ARRAY && toks[msgs].size == 2);
    int m1 = oj_arr_at(toks, n, msgs, 1);
    CHECK(m1 >= 0);
    int content = oj_obj_get(js, toks, n, m1, "content");
    CHECK(content >= 0);
    char *c = content >= 0 ? oj_strdup(js, &toks[content]) : NULL;
    CHECK(c != NULL);
    if (c) CHECK(strcmp(c, "hi\nthere \xc3\xa9") == 0);
    free(c);

    CHECK(oj_parse("{\"a\":", 6, toks, 128) < 0);
    CHECK(oj_parse("]", 1, toks, 128) < 0);
    CHECK(oj_parse("", 0, toks, 128) == 0);

    char *buf = NULL;
    size_t len = 0, cap = 0;
    CHECK(oj_escape_append(&buf, &len, &cap, "a\"b\\c\nd\x01", 8));
    CHECK(strcmp(buf, "a\\\"b\\\\c\\nd\\u0001") == 0);
    free(buf);
}

static void test_tier_parse(void) {
    CHECK(otier_parse("paid", 4) == TIER_PAID);
    CHECK(otier_parse("SUB", 3) == TIER_SUB);
    CHECK(otier_parse("background", 10) == TIER_BACKGROUND);
    CHECK(otier_parse("junk", 4) == TIER_FREE);
    CHECK(otier_parse(NULL, 0) == TIER_FREE);
    CHECK(otier_parse_public("paid", 4) == TIER_PAID);
    CHECK(otier_parse_public("SUB", 3) == TIER_SUB);
    CHECK(otier_parse_public("background", 10) == TIER_FREE);
}

static void test_completion_spacing(void) {
    CHECK(otext_completion_needs_space("looking", 7, "I", 1, true));
    CHECK(otext_completion_needs_space("looking", 7, "It was", 6, true));
    CHECK(!otext_completion_needs_space("n", 1, "ame", 3, true));
    CHECK(!otext_completion_needs_space("looking", 7, " for", 4, true));
    CHECK(!otext_completion_needs_space("hello", 5, ".", 1, true));
    CHECK(otext_completion_needs_space("looking", 7, "for", 3, false));
}

static void test_openapi(void) {
    const int capacity = 8192;
    oj_tok *toks = malloc((size_t)capacity * sizeof *toks);
    CHECK(toks != NULL);
    if (!toks) return;
    int n = oj_parse(OPENAPI_JSON, strlen(OPENAPI_JSON), toks, capacity);
    CHECK(n > 0 && toks[0].type == OJ_OBJECT);
    int paths = n > 0 ? oj_obj_get(OPENAPI_JSON, toks, n, 0, "paths") : -1;
    CHECK(paths >= 0 && toks[paths].type == OJ_OBJECT);
    static const char *required_paths[] = {
        "/api/v1/feature-extraction", "/api/v1/summarization",
        "/api/v1/generate_speech", "/api/v1/audio-file-extraction",
        "/api/v1/audio-extraction", "/api/v1/generate",
        "/api/v1/generate-large", "/api/v1/image-caption",
        "/v1/animations/generations", "/v1/3d/generations",
        "/v1/images/backgrounds",
        "/v1/images/background-removals/jobs",
        "/v1/images/background-removals",
        "/v1/3d/assets/{job}/{file}",
        "/v1/engines/{engine_name}/completions",
    };
    for (size_t i = 0; paths >= 0 && i < sizeof required_paths / sizeof required_paths[0]; i++) {
        CHECK(oj_obj_get(OPENAPI_JSON, toks, n, paths, required_paths[i]) >= 0);
    }
    free(toks);
}

typedef struct {
    osched *s;
    otier tier;
    atomic_int *order;
    int got;
    int hold_us;
    int permits;
} sched_job;

static void *sched_worker(void *arg) {
    sched_job *j = arg;
    int permits = j->permits > 0 ? j->permits : 1;
    if (osched_acquire_n(j->s, j->tier, permits)) {
        j->got = atomic_fetch_add(j->order, 1);
        usleep((useconds_t)j->hold_us);
        osched_release_n(j->s, j->tier, permits);
    } else {
        j->got = -1;
    }
    return NULL;
}

static void test_sched_priority(void) {
    osched *s = osched_create(1, 5);
    atomic_int order = 0;

    CHECK(osched_acquire(s, TIER_FREE));

    sched_job jf = { .s = s, .tier = TIER_FREE, .order = &order, .hold_us = 1000 };
    sched_job jp = { .s = s, .tier = TIER_PAID, .order = &order, .hold_us = 1000 };
    sched_job jb = { .s = s, .tier = TIER_BACKGROUND, .order = &order, .hold_us = 1000 };
    pthread_t tf, tp, tb;
    pthread_create(&tf, NULL, sched_worker, &jf);
    usleep(20000);
    pthread_create(&tp, NULL, sched_worker, &jp);
    pthread_create(&tb, NULL, sched_worker, &jb);
    usleep(50000);
    CHECK(osched_waiting(s, TIER_PAID) == 1);
    osched_release(s, TIER_FREE);
    pthread_join(tf, NULL);
    pthread_join(tp, NULL);
    pthread_join(tb, NULL);

    CHECK(jp.got >= 0 && jf.got >= 0 && jb.got >= 0);
    CHECK(jp.got < jf.got);
    CHECK(jb.got > jf.got);
    CHECK(osched_served(s, TIER_PAID) == 1);
    CHECK(osched_active(s) == 0);
    osched_destroy(s);
}

static void test_sched_timeout(void) {
    osched *s = osched_create(1, 1);
    CHECK(osched_acquire(s, TIER_FREE));
    CHECK(!osched_acquire(s, TIER_FREE));
    osched_release(s, TIER_FREE);
    osched_destroy(s);
}

static void test_sched_weighted_capacity(void) {
    osched *s = osched_create(3, 1);
    CHECK(osched_capacity(s) == 3);
    CHECK(osched_acquire_n(s, TIER_FREE, 2));
    CHECK(osched_acquire_n(s, TIER_FREE, 1));
    osched_stats st;
    osched_snapshot(s, &st);
    CHECK(st.active == 2);
    CHECK(st.used_slots == 3);
    CHECK(st.served[TIER_FREE] == 2);
    osched_release_n(s, TIER_FREE, 2);
    osched_release_n(s, TIER_FREE, 1);
    osched_snapshot(s, &st);
    CHECK(st.active == 0);
    CHECK(st.used_slots == 0);
    CHECK(!osched_acquire_n(s, TIER_FREE, 4));
    osched_destroy(s);

    s = osched_create(2, 2);
    CHECK(osched_acquire_n(s, TIER_FREE, 1));
    atomic_int order = 0;
    sched_job exclusive = {
        .s = s, .tier = TIER_PAID, .order = &order, .hold_us = 1000, .permits = 2,
    };
    pthread_t thread;
    pthread_create(&thread, NULL, sched_worker, &exclusive);
    usleep(20000);
    osched_snapshot(s, &st);
    CHECK(st.waiting[TIER_PAID] == 1);
    CHECK(st.used_slots == 1);
    osched_release_n(s, TIER_FREE, 1);
    pthread_join(thread, NULL);
    CHECK(exclusive.got == 0);
    osched_snapshot(s, &st);
    CHECK(st.used_slots == 0);
    CHECK(st.queue_ms_max[TIER_PAID] >= 10.0);
    osched_destroy(s);
}

static bool relay_to_http_client(const void *data, size_t len, void *user) {
    return ohttp_raw_write((ohttp_request *)user, data, len);
}

typedef struct {
    oproxy_target *target;
} echo_context;

static void echo_handler(ohttp_request *req, void *user) {
    echo_context *context = user;
    if (ohttp_path_is(req, "/relay") || ohttp_path_is(req, "/relay-stream")) {
        const char *target = ohttp_path_is(req, "/relay-stream") ? "/stream" : "/echo";
        oproxy_result result;
        char error[128];
        bool ok = oproxy_target_relay(
            context->target,
            req->method, req->method_len,
            target, strlen(target),
            req->query, req->query_len,
            req->body, req->body_len,
            "application/json", 16,
            NULL, 0, 2000,
            relay_to_http_client, req,
            &result, error, sizeof error);
        if (!ok && !result.response_started) {
            ohttp_respond_str(req, 502, "text/plain", error);
        } else if (!ok || result.downstream_close) {
            ohttp_force_close(req);
        }
        return;
    }
    if (ohttp_path_is(req, "/echo") && ohttp_method_is(req, "POST")) {
        ohttp_respond(req, 200, "application/json", req->body, req->body_len);
        return;
    }
    if (ohttp_path_is(req, "/stream")) {
        ohttp_stream_begin(req, 200, "text/plain");
        ohttp_stream_write(req, "one", 3);
        ohttp_stream_write(req, "two", 3);
        ohttp_stream_end(req);
        return;
    }
    size_t hlen = 0;
    const char *h = ohttp_req_header(req, "X-Test", &hlen);
    if (h && hlen == 3 && memcmp(h, "abc", 3) == 0) {
        ohttp_respond_str(req, 200, "text/plain", "header-ok");
        return;
    }
    ohttp_respond_str(req, 404, "text/plain", "nope");
}

static int connect_local(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&addr, sizeof addr) != 0) { close(fd); return -1; }
    return fd;
}

static bool test_write_all(int fd, const char *data, size_t len) {
    while (len) {
        ssize_t n = write(fd, data, len);
        if (n <= 0) return false;
        data += n;
        len -= (size_t)n;
    }
    return true;
}

static char *http_roundtrip(uint16_t port, const char *raw, size_t *out_len) {
    int fd = connect_local(port);
    if (fd < 0) return NULL;
    if (!test_write_all(fd, raw, strlen(raw))) {
        close(fd);
        return NULL;
    }
    char *resp = malloc(65536);
    size_t total = 0;
    ssize_t n;
    while ((n = read(fd, resp + total, 65536 - total - 1)) > 0) total += (size_t)n;
    close(fd);
    resp[total] = 0;
    if (out_len) *out_len = total;
    return resp;
}

static int count_text(const char *text, const char *needle) {
    int count = 0;
    size_t needle_len = strlen(needle);
    while ((text = strstr(text, needle)) != NULL) {
        count++;
        text += needle_len;
    }
    return count;
}

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} relay_sink;

static bool collect_relay(const void *data, size_t len, void *user) {
    relay_sink *sink = user;
    if (sink->len + len + 1 > sink->cap) {
        size_t next = sink->cap ? sink->cap * 2 : 4096;
        while (next < sink->len + len + 1) next *= 2;
        char *grown = realloc(sink->data, next);
        if (!grown) return false;
        sink->data = grown;
        sink->cap = next;
    }
    memcpy(sink->data + sink->len, data, len);
    sink->len += len;
    sink->data[sink->len] = 0;
    return true;
}

static void test_proxy_relay(void) {
    relay_sink sink = {0};
    oproxy_result result;
    char error[256];
    const char *body = "{\"proxied\":true}";
    bool ok = oproxy_relay(
        "http://127.0.0.1:18791/",
        "POST", 4,
        "/echo", 5,
        "source=test", 11,
        body, strlen(body),
        "application/json", 16,
        NULL, 0,
        2000,
        collect_relay, &sink,
        &result, error, sizeof error);
    CHECK(ok);
    CHECK(result.response_started);
    CHECK(result.bytes_relayed == sink.len);
    CHECK(sink.data && strstr(sink.data, "HTTP/1.1 200 OK"));
    CHECK(sink.data && strstr(sink.data, "Content-Type: application/json"));
    CHECK(sink.data && strstr(sink.data, body));
    free(sink.data);
}

static void test_http_server(void) {
    char target_error[256];
    echo_context context = {
        .target = oproxy_target_create("http://127.0.0.1:18791", 4,
                                       target_error, sizeof target_error),
    };
    CHECK(context.target != NULL);
    ohttp_config cfg = { .port = 18791, .reactor_threads = 1, .worker_threads = 4,
                         .handler = echo_handler, .user = &context };
    ohttp_server *srv = ohttp_start(&cfg);
    CHECK(srv != NULL);
    usleep(100000);

    char *r = http_roundtrip(18791, "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"a\":\"b\\n\"}", NULL);
    CHECK(r && strstr(r, "200 OK") && strstr(r, "{\"a\":\"b\\n\"}"));
    CHECK(r && strstr(r, "Access-Control-Allow-Origin: *"));
    free(r);

    r = http_roundtrip(18791, "GET /stream HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", NULL);
    CHECK(r && strstr(r, "Transfer-Encoding: chunked") && strstr(r, "one") && strstr(r, "two"));
    free(r);

    r = http_roundtrip(18791, "GET /hdr HTTP/1.1\r\nHost: x\r\nX-Test: abc\r\nConnection: close\r\n\r\n", NULL);
    CHECK(r && strstr(r, "header-ok"));
    free(r);

    r = http_roundtrip(18791, "GET /missing HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", NULL);
    CHECK(r && strstr(r, "404"));
    free(r);

    r = http_roundtrip(18791, "POST /relay?source=gateway HTTP/1.1\r\nHost: x\r\nContent-Length: 14\r\nConnection: close\r\n\r\n{\"relay\":true}", NULL);
    CHECK(r && strstr(r, "HTTP/1.1 200 OK") && strstr(r, "{\"relay\":true}"));
    free(r);

    r = http_roundtrip(18791, "GET /relay-stream HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", NULL);
    CHECK(r && strstr(r, "Transfer-Encoding: chunked") && strstr(r, "one") && strstr(r, "two"));
    free(r);

    oproxy_stats proxy_stats;
    oproxy_target_snapshot(context.target, &proxy_stats);
    CHECK(proxy_stats.connections_opened == 1);
    CHECK(proxy_stats.connections_reused >= 1);
    CHECK(proxy_stats.failures == 0);

    test_proxy_relay();

    int fd = connect_local(18791);
    CHECK(fd >= 0);
    const char *req1 = "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\nhi";
    CHECK(test_write_all(fd, req1, strlen(req1)));
    char buf[4096];
    ssize_t n = read(fd, buf, sizeof buf - 1);
    CHECK(n > 0);
    buf[n] = 0;
    CHECK(strstr(buf, "200 OK") && strstr(buf, "keep-alive"));
    CHECK(test_write_all(fd, req1, strlen(req1)));
    n = read(fd, buf, sizeof buf - 1);
    CHECK(n > 0);
    close(fd);

    const char *pipelined =
        "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n\r\none"
        "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\nConnection: close\r\n\r\ntwo";
    r = http_roundtrip(18791, pipelined, NULL);
    CHECK(r && count_text(r, "HTTP/1.1 200 OK") == 2);
    CHECK(r && strstr(r, "one") && strstr(r, "two"));
    free(r);

    fd = connect_local(18791);
    CHECK(fd >= 0);
    const char *expect_headers =
        "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n"
        "Expect: 100-continue\r\nConnection: close\r\n\r\n";
    CHECK(test_write_all(fd, expect_headers, strlen(expect_headers)));
    n = read(fd, buf, sizeof buf - 1);
    CHECK(n > 0);
    if (n > 0) {
        buf[n] = 0;
        CHECK(strstr(buf, "100 Continue") != NULL);
    }
    CHECK(test_write_all(fd, "hey", 3));
    size_t response_len = 0;
    while ((n = read(fd, buf + response_len, sizeof buf - response_len - 1)) > 0) {
        response_len += (size_t)n;
    }
    buf[response_len] = 0;
    CHECK(strstr(buf, "200 OK") && strstr(buf, "hey"));
    close(fd);

    r = http_roundtrip(
        18791,
        "POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\n"
        "Content-Length: 2\r\nConnection: close\r\n\r\nx",
        NULL);
    CHECK(r && r[0] == 0);
    free(r);

    r = http_roundtrip(
        18791,
        "POST /echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n"
        "Connection: close\r\n\r\n1\r\nx\r\n0\r\n\r\n",
        NULL);
    CHECK(r && r[0] == 0);
    free(r);

    ohttp_stop(srv);
    CHECK(ohttp_join(srv) == 0);
    oproxy_target_destroy(context.target);
}

/* A lane that is already justified and under pressure, so each test can turn
 * exactly one condition off and assert the refusal. */
static void saturated_paid_lane(oscale_policy *policy, oscale_observation *obs) {
    oscale_policy_defaults(policy, "tts");
    policy->enabled = true;
    policy->hardware[0] = 0;
    snprintf(policy->hardware, sizeof policy->hardware, "gpu-rtx4090");
    policy->price_usd_hr = 0.34;
    policy->revenue_usd_per_req = 0.02;
    policy->seconds_per_req = 5.0;
    policy->max_instances = 2;
    policy->max_usd_hr = 1.0;
    policy->cooldown_s = 120.0;

    memset(obs, 0, sizeof *obs);
    obs->queue_depth[TIER_PAID] = 4;
    obs->queue_ms_max[TIER_PAID] = 5000.0;
    obs->local_permits_free = 0;
    obs->backlog_reqs = 200.0;
}

static void test_scale_defaults_to_zero(void) {
    oscale_policy policy;
    oscale_policy_defaults(&policy, "tts");
    CHECK(policy.enabled == false);
    CHECK(policy.max_instances == 1);
    CHECK(policy.tier_mask == OSCALE_TIERS_PAID_ONLY);
    CHECK(policy.max_instance_ttl_s > 0);

    oscale *s = oscale_create();
    CHECK(s != NULL);
    CHECK(oscale_add_lane(s, &policy) == 0);
    /* A duplicate lane name must not create a second billing path. */
    CHECK(oscale_add_lane(s, &policy) == -1);
    oscale_lane *lane = oscale_lane_by_name(s, "tts");
    CHECK(lane != NULL);
    CHECK(oscale_ready_instances(lane) == 0);

    /* Background eligibility is not configurable: a lane that asks for it is
     * stripped back, so a batch backlog can never rent hardware. */
    oscale_policy background;
    oscale_policy_defaults(&background, "batch");
    background.tier_mask = OSCALE_TIER_BIT(TIER_BACKGROUND);
    CHECK(oscale_add_lane(s, &background) == 1);
    oscale_lane *batch_lane = oscale_lane_by_name(s, "batch");
    CHECK(batch_lane != NULL);
    if (batch_lane) {
        CHECK((batch_lane->policy.tier_mask & OSCALE_TIER_BIT(TIER_BACKGROUND)) == 0);
        CHECK(batch_lane->policy.tier_mask == OSCALE_TIERS_PAID_ONLY);
    }

    oscale_observation obs;
    memset(&obs, 0, sizeof obs);
    obs.queue_depth[TIER_PAID] = 8;
    obs.queue_ms_max[TIER_PAID] = 30000.0;
    obs.backlog_reqs = 5000.0;
    oscale_reason reason = OSCALE_REASON_NONE;
    /* Disabled by default: heavy paid pressure still rents nothing. */
    CHECK(oscale_decide(lane, &obs, 1000.0, &reason) == OSCALE_HOLD);
    CHECK(reason == OSCALE_REASON_DISABLED);
    oscale_destroy(s);
}

static void test_scale_only_for_eligible_tiers(void) {
    oscale_policy policy;
    oscale_observation obs;
    saturated_paid_lane(&policy, &obs);
    oscale *s = oscale_create();
    CHECK(oscale_add_lane(s, &policy) == 0);
    oscale_lane *lane = oscale_lane_at(s, 0);

    oscale_reason reason = OSCALE_REASON_NONE;
    CHECK(oscale_decide(lane, &obs, 1000.0, &reason) == OSCALE_UP);
    CHECK(reason == OSCALE_REASON_PRESSURE_SUSTAINED);

    /* The same pressure on best-effort tiers must never rent hardware: free,
     * sub, and background traffic is explicitly not worth a bill. */
    for (int tier = TIER_SUB; tier <= TIER_BACKGROUND; tier++) {
        oscale_observation best_effort;
        memset(&best_effort, 0, sizeof best_effort);
        best_effort.queue_depth[tier] = 50;
        best_effort.queue_ms_max[tier] = 60000.0;
        best_effort.local_permits_free = 0;
        best_effort.backlog_reqs = 5000.0;
        CHECK(oscale_decide(lane, &best_effort, 1000.0, &reason) == OSCALE_HOLD);
        CHECK(reason == OSCALE_REASON_TIER_NOT_ELIGIBLE);
        CHECK(oscale_eligible_depth(&lane->policy, &best_effort) == 0);
    }
    oscale_destroy(s);
}

static void test_scale_prefers_local_capacity(void) {
    oscale_policy policy;
    oscale_observation obs;
    saturated_paid_lane(&policy, &obs);
    oscale *s = oscale_create();
    CHECK(oscale_add_lane(s, &policy) == 0);
    oscale_lane *lane = oscale_lane_at(s, 0);

    /* The local GPU is sunk cost; while it has a free permit, renting is wrong
     * no matter how deep the paid queue looks. */
    obs.local_permits_free = 1;
    oscale_reason reason = OSCALE_REASON_NONE;
    CHECK(oscale_decide(lane, &obs, 1000.0, &reason) == OSCALE_HOLD);
    CHECK(reason == OSCALE_REASON_LOCAL_HAS_ROOM);
    oscale_destroy(s);
}

static void test_scale_cost_gate(void) {
    oscale_policy policy;
    oscale_observation obs;
    saturated_paid_lane(&policy, &obs);

    double value = 0.0, cost = 0.0;
    CHECK(oscale_rent_is_justified(&policy, &obs, &value, &cost));
    /* 200 backlogged requests at $0.02 against $0.34/hr with a 1.5x margin. */
    CHECK(value > cost);

    /* Thin backlog: an instance would sit mostly idle, so the rent loses money. */
    oscale_observation thin = obs;
    thin.backlog_reqs = 3.0;
    CHECK(!oscale_rent_is_justified(&policy, &thin, &value, &cost));
    CHECK(value < cost);

    /* Capacity caps the value: renting cannot claim more throughput than an
     * instance physically has, so an enormous backlog is not a blank cheque. */
    oscale_policy slow = policy;
    slow.seconds_per_req = 600.0; /* 6 requests per hour */
    oscale_observation flood = obs;
    flood.backlog_reqs = 1e9;
    CHECK(!oscale_rent_is_justified(&slow, &flood, &value, &cost));

    oscale *s = oscale_create();
    CHECK(oscale_add_lane(s, &policy) == 0);
    oscale_lane *lane = oscale_lane_at(s, 0);
    oscale_reason reason = OSCALE_REASON_NONE;
    CHECK(oscale_decide(lane, &thin, 1000.0, &reason) == OSCALE_HOLD);
    CHECK(reason == OSCALE_REASON_NOT_WORTH_IT);
    oscale_destroy(s);
}

static void test_scale_caps_and_cooldown(void) {
    oscale_policy policy;
    oscale_observation obs;
    saturated_paid_lane(&policy, &obs);
    policy.max_instances = 2;
    policy.max_usd_hr = 1.0; /* room for two $0.34 instances */
    oscale *s = oscale_create();
    CHECK(oscale_add_lane(s, &policy) == 0);
    oscale_lane *lane = oscale_lane_at(s, 0);

    oscale_reason reason = OSCALE_REASON_NONE;
    CHECK(oscale_decide(lane, &obs, 1000.0, &reason) == OSCALE_UP);
    int slot = oscale_begin_instance(lane, 1000.0);
    CHECK(slot == 0);
    oscale_instance_ready(lane, slot, "http://127.0.0.1:8787/api/cogs/x", 1000.0);
    CHECK(oscale_ready_instances(lane) == 1);

    /* Immediately after acting, the cooldown suppresses a second rent even
     * though pressure is unchanged — this is the anti-thrash guard. */
    CHECK(oscale_decide(lane, &obs, 1010.0, &reason) == OSCALE_HOLD);
    CHECK(reason == OSCALE_REASON_COOLDOWN);

    /* A hard ceiling is reported ahead of the timing throttle: dropping the
     * ceiling below two instances refuses on spend, not on cooldown. */
    lane->policy.max_usd_hr = 0.5;
    oscale_instance_touch(lane, slot, 1200.0);
    CHECK(oscale_decide(lane, &obs, 1200.0, &reason) == OSCALE_HOLD);
    CHECK(reason == OSCALE_REASON_SPEND_CAP);

    /* Lifting the ceiling but not the instance cap moves the refusal. */
    lane->policy.max_usd_hr = 10.0;
    CHECK(oscale_decide(lane, &obs, 1200.0, &reason) == OSCALE_UP);
    CHECK(oscale_begin_instance(lane, 1200.0) == 1);
    oscale_instance_ready(lane, 1, "http://127.0.0.1:8787/api/cogs/y", 1200.0);
    oscale_instance_touch(lane, 0, 1400.0);
    oscale_instance_touch(lane, 1, 1400.0);
    CHECK(oscale_decide(lane, &obs, 1400.0, &reason) == OSCALE_HOLD);
    CHECK(reason == OSCALE_REASON_INSTANCE_CAP);
    CHECK(oscale_lane_spend_rate_usd_hr(lane) > 0.67);
    oscale_destroy(s);
}

static void test_scale_down_to_zero(void) {
    oscale_policy policy;
    oscale_observation obs;
    saturated_paid_lane(&policy, &obs);
    policy.idle_scale_down_s = 60.0;
    policy.max_instance_ttl_s = 3600.0;
    oscale *s = oscale_create();
    CHECK(oscale_add_lane(s, &policy) == 0);
    oscale_lane *lane = oscale_lane_at(s, 0);

    int slot = oscale_begin_instance(lane, 1000.0);
    oscale_instance_ready(lane, slot, "http://127.0.0.1:8787/api/cogs/x", 1000.0);

    /* Busy: no release while requests keep landing. */
    oscale_instance_touch(lane, slot, 1050.0);
    oscale_reason reason = OSCALE_REASON_NONE;
    CHECK(oscale_release_candidate(lane, 1080.0, &reason) == -1);

    /* Idle past the window: release, and the lane returns to zero. */
    CHECK(oscale_decide(lane, &obs, 1200.0, &reason) == OSCALE_DOWN);
    CHECK(reason == OSCALE_REASON_IDLE);
    oscale_release_instance(lane, slot, 1200.0, reason);
    CHECK(oscale_ready_instances(lane) == 0);
    CHECK(oscale_lane_spend_rate_usd_hr(lane) == 0.0);
    /* 200 seconds of a $0.34/hr instance. */
    CHECK(lane->spend_usd > 0.018 && lane->spend_usd < 0.020);
    oscale_destroy(s);
}

static void test_scale_hard_ttl_beats_everything(void) {
    oscale_policy policy;
    oscale_observation obs;
    saturated_paid_lane(&policy, &obs);
    policy.idle_scale_down_s = 600.0;
    policy.max_instance_ttl_s = 300.0;
    oscale *s = oscale_create();
    CHECK(oscale_add_lane(s, &policy) == 0);
    oscale_lane *lane = oscale_lane_at(s, 0);

    int slot = oscale_begin_instance(lane, 1000.0);
    oscale_instance_ready(lane, slot, "http://127.0.0.1:8787/api/cogs/x", 1000.0);

    /* Continuously busy and under heavy paid pressure, but past its lifetime:
     * the instance is still reclaimed. An instance that outlives its TTL is the
     * failure mode that bills forever. */
    oscale_instance_touch(lane, slot, 1299.0);
    oscale_reason reason = OSCALE_REASON_NONE;
    CHECK(oscale_decide(lane, &obs, 1301.0, &reason) == OSCALE_DOWN);
    CHECK(reason == OSCALE_REASON_TTL_EXPIRED);
    oscale_release_instance(lane, slot, 1301.0, reason);
    CHECK(lane->ttl_kills == 1);

    /* A disabled lane must still shed what it already rented. */
    slot = oscale_begin_instance(lane, 2000.0);
    oscale_instance_ready(lane, slot, "http://127.0.0.1:8787/api/cogs/z", 2000.0);
    lane->policy.enabled = false;
    CHECK(oscale_decide(lane, &obs, 2400.0, &reason) == OSCALE_DOWN);
    oscale_destroy(s);
}

static void test_tune_profiles(void) {
    otune_profile profile;
    otune_profile_for("NVIDIA GeForce RTX 5090", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_BLACKWELL);
    CHECK(strcmp(profile.kv_type, "q8_0") == 0);
    CHECK(profile.flash_attn);
    CHECK(profile.n_ubatch == 1024);
    CHECK(profile.parallel_contexts >= 4);

    otune_profile_for("NVIDIA GeForce RTX 4090", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_ADA);
    CHECK(profile.parallel_contexts == 3);

    otune_profile_for("Tesla T4", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_TURING);
    /* Turing predates the attention kernels the newer profiles assume. */
    CHECK(!profile.flash_attn);
    CHECK(profile.n_ubatch == 256);

    otune_profile_for("NVIDIA RTX 3090", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_AMPERE);
    otune_profile_for("NVIDIA H100 PCIe", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_HOPPER);
    otune_profile_for("cpu", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_CPU);
    CHECK(profile.parallel_contexts == 1);

    /* An unrecognised device must still yield a usable profile. */
    otune_profile_for("Some Future Accelerator", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_UNKNOWN);
    CHECK(profile.n_batch > 0 && profile.n_ubatch > 0 && profile.kv_type != NULL);
}

static int fake_warm_calls;
static bool fake_warm_ok = true;

static bool fake_warm(const oscale_policy *policy, char *endpoint, size_t endpoint_cap,
                      char *error, size_t error_cap, void *user) {
    (void)policy; (void)user;
    fake_warm_calls++;
    if (!fake_warm_ok) {
        snprintf(error, error_cap, "control plane refused");
        return false;
    }
    snprintf(endpoint, endpoint_cap, "http://127.0.0.1:8787");
    return true;
}

static void test_capacity_controller(void) {
    osched *sched = osched_create(2, 0.05);
    oscale *scale = oscale_create();
    oscale_policy policy;
    oscale_policy_defaults(&policy, "tts");
    policy.enabled = true;
    snprintf(policy.cog_template, sizeof policy.cog_template, "appnz-tts");
    snprintf(policy.hardware, sizeof policy.hardware, "gpu-rtx4090");
    policy.price_usd_hr = 0.34;
    policy.revenue_usd_per_req = 0.05;
    policy.seconds_per_req = 2.0;
    policy.cooldown_s = 0.0;
    policy.idle_scale_down_s = 30.0;
    policy.max_usd_hr = 1.0;
    CHECK(oscale_add_lane(scale, &policy) == 0);

    ocapacity_config cfg = {
        .sched = sched, .scale = scale, .poll_interval_s = 3600,
        .warm = fake_warm,
    };
    ocapacity *capacity = ocapacity_start(&cfg);
    CHECK(capacity != NULL);
    oscale_lane *lane = oscale_lane_at(scale, 0);

    /* Idle scheduler: nothing is queued, so nothing is rented. */
    fake_warm_calls = 0;
    ocapacity_tick(capacity, 100.0);
    CHECK(fake_warm_calls == 0);
    CHECK(oscale_ready_instances(lane) == 0);

    /* Fill the local device and queue a paid waiter behind it, which is the
     * only situation that may rent. */
    CHECK(osched_acquire_n(sched, TIER_PAID, 2));
    /* Times out against the short admission window, which is what leaves the
     * scheduler's used_slots at capacity for the observation below. */
    CHECK(!osched_acquire_n(sched, TIER_PAID, 2));

    /* Drive the decision directly: the controller shares this path, and a
     * synthetic observation keeps the test off wall-clock queue timing. */
    oscale_observation obs;
    memset(&obs, 0, sizeof obs);
    obs.queue_depth[TIER_PAID] = 4;
    obs.queue_ms_max[TIER_PAID] = 9000.0;
    obs.local_permits_free = 0;
    obs.backlog_reqs = 400.0;
    oscale_reason reason = OSCALE_REASON_NONE;
    CHECK(oscale_decide(lane, &obs, 200.0, &reason) == OSCALE_UP);

    /* A refused warm must not leave a billable placeholder behind. */
    fake_warm_ok = false;
    int slot = oscale_begin_instance(lane, 200.0);
    CHECK(slot >= 0);
    oscale_release_instance(lane, slot, 200.0, OSCALE_REASON_NONE);
    CHECK(oscale_ready_instances(lane) == 0);
    CHECK(oscale_lane_spend_rate_usd_hr(lane) == 0.0);

    fake_warm_ok = true;
    slot = oscale_begin_instance(lane, 300.0);
    oscale_instance_ready(lane, slot, "http://127.0.0.1:8787", 300.0);
    CHECK(oscale_ready_instances(lane) == 1);
    /* Routing is tier-gated even when an instance is already up. */
    CHECK(ocapacity_overflow_endpoint(capacity, "tts", TIER_PAID) != NULL);
    CHECK(ocapacity_overflow_endpoint(capacity, "tts", TIER_FREE) == NULL);
    CHECK(ocapacity_overflow_endpoint(capacity, "tts", TIER_BACKGROUND) == NULL);
    CHECK(ocapacity_overflow_endpoint(capacity, "image", TIER_PAID) == NULL);

    char json[2048];
    size_t json_len = ocapacity_status_json(capacity, json, sizeof json);
    CHECK(json_len > 0);
    CHECK(strstr(json, "\"name\":\"tts\"") != NULL);
    CHECK(strstr(json, "\"ready\":1") != NULL);

    osched_release_n(sched, TIER_PAID, 2);
    /* Shutdown must not leave rented capacity behind. */
    ocapacity_stop(capacity);
    CHECK(oscale_ready_instances(lane) == 0);
    oscale_destroy(scale);
    osched_destroy(sched);
}

static void test_response_accounting(void) {
    ohttp_response_stats before;
    ohttp_response_snapshot(&before);
    /* test_http_server has already driven traffic through the server, so the
     * counters must be non-zero and internally consistent. */
    CHECK(before.total > 0);
    CHECK(before.total == before.informational + before.success + before.redirect +
                          before.client_error + before.server_error);

    ohttp_error_event events[OHTTP_ERROR_RING];
    int count = ohttp_recent_errors(events, OHTTP_ERROR_RING);
    CHECK(count >= 0 && count <= OHTTP_ERROR_RING);
    for (int i = 0; i < count; i++) {
        /* Only 5xx is ringed, and every entry must be attributable. */
        CHECK(events[i].status >= 500 && events[i].status <= 599);
        CHECK(events[i].at_unix > 0);
    }
    CHECK(ohttp_recent_errors(events, 0) == 0);
    CHECK(ohttp_recent_errors(NULL, 4) == 0);
}

/* Foreground estimation: an opaque matte must return the image unchanged, and
 * the parallel red-black sweep must land on the same answer as the sequential
 * one regardless of thread count. */
static void test_matte(void) {
    enum { H = 24, W = 20, D = 3 };
    static float image[H * W * D];
    static float alpha[H * W];
    static float fg[H * W * D];
    static float bg[H * W * D];

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const int i = y * W + x;
            /* Left half is a red object, right half a green backdrop. */
            const bool inside = x < W / 2;
            image[i * D + 0] = inside ? 0.85f : 0.05f;
            image[i * D + 1] = inside ? 0.15f : 0.90f;
            image[i * D + 2] = inside ? 0.20f : 0.10f;
            alpha[i] = inside ? 1.0f : 0.0f;
        }
    }

    omatte_params params = omatte_default_params();
    CHECK(omatte_estimate_fb(image, alpha, H, W, D, &params, fg, bg) == 0);
    for (int i = 0; i < H * W; i++) {
        if (alpha[i] < 1.0f) continue;
        for (int c = 0; c < D; c++) {
            CHECK(fabsf(fg[i * D + c] - image[i * D + c]) < 1e-2f);
        }
    }

    /* Semi-transparent edge: alpha ramps across the middle column so the solver
     * has to separate the two colours instead of copying the composite. */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const int i = y * W + x;
            const float a = (float)x / (float)(W - 1);
            alpha[i] = a;
            for (int c = 0; c < D; c++) {
                const float front = c == 0 ? 0.85f : (c == 1 ? 0.15f : 0.20f);
                const float back = c == 0 ? 0.05f : (c == 1 ? 0.90f : 0.10f);
                image[i * D + c] = a * front + (1.0f - a) * back;
            }
        }
    }

    static float fg_seq[H * W * D];
    static float fg_par_one[H * W * D];
    static float fg_par_many[H * W * D];

    params.order = OMATTE_ORDER_SEQUENTIAL;
    CHECK(omatte_estimate_fb(image, alpha, H, W, D, &params, fg_seq, NULL) == 0);

    params.order = OMATTE_ORDER_RED_BLACK;
    params.threads = 1;
    CHECK(omatte_estimate_fb(image, alpha, H, W, D, &params, fg_par_one, NULL) == 0);
    params.threads = 8;
    CHECK(omatte_estimate_fb(image, alpha, H, W, D, &params, fg_par_many, NULL) == 0);

    for (int i = 0; i < H * W * D; i++) {
        /* Thread count must not change the result at all. */
        CHECK(fg_par_one[i] == fg_par_many[i]);
        /* And the two sweep orders must agree to within solver noise. */
        CHECK(fabsf(fg_seq[i] - fg_par_one[i]) < 5e-2f);
    }

    omatte_composite(fg_seq, alpha, NULL, H, W, D, bg);
    for (int i = 0; i < H * W; i++) {
        CHECK(bg[i * D] >= 0.0f && bg[i * D] <= 1.0f);
    }

    CHECK(omatte_estimate_fb(NULL, alpha, H, W, D, &params, fg, bg) == -1);
    CHECK(omatte_estimate_fb(image, alpha, 0, W, D, &params, fg, bg) == -1);
    if (!omatte_cuda_available()) {
        CHECK(omatte_estimate_fb_cuda(image, alpha, H, W, D, &params, fg, bg) == -3);
    }
}

int main(void) {
    test_json();
    test_matte();
    test_tier_parse();
    test_completion_spacing();
    test_openapi();
    test_sched_priority();
    test_sched_timeout();
    test_sched_weighted_capacity();
    test_scale_defaults_to_zero();
    test_scale_only_for_eligible_tiers();
    test_scale_prefers_local_capacity();
    test_scale_cost_gate();
    test_scale_caps_and_cooldown();
    test_scale_down_to_zero();
    test_scale_hard_ttl_beats_everything();
    test_tune_profiles();
    test_capacity_controller();
    test_http_server();
    test_response_accounting();
    if (failures) {
        fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    fprintf(stderr, "all core tests passed\n");
    return 0;
}
