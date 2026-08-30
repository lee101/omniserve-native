#define _GNU_SOURCE
#include "ocapacity.h"
#include "ohttp.h"
#include "oimage.h"
#include "ojson.h"
#include "olog.h"
#include "oproxy.h"
#include "oscale.h"
#include "osched.h"
#include "otext.h"
#include "omatte.h"
#include "otune.h"
#include "ovram.h"
#include "ohost.h"
#include "ospec.h"

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
#include <sys/stat.h>
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

static void test_image_contract(void) {
    char headroom_error[160] = {0};
    unsetenv("OMNISERVE_NATIVE_SD_MIN_FREE_MB");
    CHECK(oimage_gpu_headroom_mb() == 4096);
    CHECK(oimage_gpu_headroom_ok(4.0, headroom_error, sizeof headroom_error));
    CHECK(!oimage_gpu_headroom_ok(2.0, headroom_error, sizeof headroom_error));
    CHECK(strstr(headroom_error, "4096 MB") != NULL);
    CHECK(strstr(headroom_error, "2048 MB") != NULL);
    setenv("OMNISERVE_NATIVE_SD_MIN_FREE_MB", "0", 1);
    CHECK(oimage_gpu_headroom_mb() == 0);
    CHECK(oimage_gpu_headroom_ok(0.1, headroom_error, sizeof headroom_error));
    setenv("OMNISERVE_NATIVE_SD_MIN_FREE_MB", "8192", 1);
    CHECK(oimage_gpu_headroom_mb() == 8192);
    CHECK(!oimage_gpu_headroom_ok(7.99, headroom_error, sizeof headroom_error));
    unsetenv("OMNISERVE_NATIVE_SD_MIN_FREE_MB");

    const char *body = "{\"prompt\":\"red cube\",\"negative_prompt\":\"blur\","
                       "\"size\":\"768x512\",\"width\":1024,\"steps\":8,"
                       "\"guidance_scale\":1.5,\"seed\":42,\"n\":1}";
    oimage_request request;
    char error[160];
    CHECK(oimage_request_parse(body, strlen(body), &request, error, sizeof error));
    CHECK(request.generation.prompt && strcmp(request.generation.prompt, "red cube") == 0);
    CHECK(request.generation.negative_prompt && strcmp(request.generation.negative_prompt, "blur") == 0);
    CHECK(request.generation.width == 1024 && request.generation.height == 512);
    CHECK(request.generation.steps == 8);
    CHECK(fabsf(request.generation.guidance_scale - 1.5f) < 0.0001f);
    CHECK(request.generation.seed == 42);
    CHECK(request.generation.lora_count == 0);
    oimage_request_free(&request);

    const char *loras = "{\"prompt\":\"cat\",\"loras\":["
                        "{\"path\":\"/models/a.safetensors\",\"scale\":0.75},"
                        "\"/models/b.safetensors\"]}";
    CHECK(oimage_request_parse(loras, strlen(loras), &request, error, sizeof error));
    CHECK(request.direct_lora_paths);
    CHECK(request.generation.lora_count == 2);
    CHECK(strcmp(request.generation.loras[0].path, "/models/a.safetensors") == 0);
    CHECK(fabsf(request.generation.loras[0].scale - 0.75f) < 0.0001f);
    CHECK(strcmp(request.generation.loras[1].path, "/models/b.safetensors") == 0);
    CHECK(request.generation.loras[1].scale == 1.0f);
    oimage_request_free(&request);

    char lora_dir[] = "/tmp/omniserve-lora-XXXXXX";
    CHECK(mkdtemp(lora_dir) != NULL);
    char lora_path[512];
    snprintf(lora_path, sizeof lora_path, "%s/pixel_art.safetensors", lora_dir);
    FILE *lora_file = fopen(lora_path, "wb");
    CHECK(lora_file != NULL);
    if (lora_file) fclose(lora_file);
    setenv("OMNISERVE_NATIVE_LORA_DIR", lora_dir, 1);
    const char *lora_id = "{\"prompt\":\"cat\",\"lora_id\":\"pixel_art\",\"lora_scale\":0.8}";
    CHECK(oimage_request_parse(lora_id, strlen(lora_id), &request, error, sizeof error));
    CHECK(!request.direct_lora_paths);
    CHECK(request.generation.lora_count == 1);
    CHECK(strcmp(request.generation.loras[0].path, lora_path) == 0);
    CHECK(fabsf(request.generation.loras[0].scale - 0.8f) < 0.0001f);
    oimage_request_free(&request);
    char named_lora_path[512];
    snprintf(named_lora_path, sizeof named_lora_path, "%s/Z Image.safetensors", lora_dir);
    lora_file = fopen(named_lora_path, "wb");
    CHECK(lora_file != NULL);
    if (lora_file) fclose(lora_file);
    const char *named_lora = "{\"prompt\":\"cat\",\"lora_id\":\"z_style\","
                             "\"lora_filename\":\"Z Image.safetensors\"}";
    CHECK(oimage_request_parse(named_lora, strlen(named_lora), &request, error, sizeof error));
    CHECK(request.generation.lora_count == 1);
    CHECK(strcmp(request.generation.loras[0].path, named_lora_path) == 0);
    oimage_request_free(&request);
    char registry_lora_path[512];
    snprintf(registry_lora_path, sizeof registry_lora_path, "%s/Z-Image_360.safetensors", lora_dir);
    lora_file = fopen(registry_lora_path, "wb");
    CHECK(lora_file != NULL);
    if (lora_file) fclose(lora_file);
    char registry_path[512];
    snprintf(registry_path, sizeof registry_path, "%s/lora_registry.json", lora_dir);
    FILE *registry_file = fopen(registry_path, "wb");
    CHECK(registry_file != NULL);
    if (registry_file) {
        fprintf(registry_file,
                "[{\"id\":\"zimage_360\",\"path\":\"%s/Z-Image_360.safetensors\"}]",
                lora_dir);
        fclose(registry_file);
    }
    setenv("OMNISERVE_NATIVE_LORA_REGISTRY", registry_path, 1);
    const char *registry_lora = "{\"prompt\":\"cat\",\"lora_id\":\"zimage_360\"}";
    CHECK(oimage_request_parse(registry_lora, strlen(registry_lora),
                               &request, error, sizeof error));
    CHECK(request.generation.lora_count == 1);
    CHECK(strcmp(request.generation.loras[0].path, registry_lora_path) == 0);
    oimage_request_free(&request);
    const char *unsafe_filename = "{\"prompt\":\"cat\",\"lora_id\":\"z_style\","
                                  "\"lora_filename\":\"../secret.safetensors\"}";
    CHECK(!oimage_request_parse(unsafe_filename, strlen(unsafe_filename),
                                &request, error, sizeof error));
    char escaped_lora_path[512];
    snprintf(escaped_lora_path, sizeof escaped_lora_path, "%s/escape.safetensors", lora_dir);
    CHECK(symlink("/etc/passwd", escaped_lora_path) == 0);
    const char *escaped_lora = "{\"prompt\":\"cat\",\"lora_id\":\"z_style\","
                               "\"lora_filename\":\"escape.safetensors\"}";
    CHECK(!oimage_request_parse(escaped_lora, strlen(escaped_lora),
                                &request, error, sizeof error));
    const char *unsafe_lora_id = "{\"prompt\":\"cat\",\"lora_id\":\"../secret\"}";
    CHECK(!oimage_request_parse(unsafe_lora_id, strlen(unsafe_lora_id),
                                &request, error, sizeof error));
    unsetenv("OMNISERVE_NATIVE_LORA_REGISTRY");
    unsetenv("OMNISERVE_NATIVE_LORA_DIR");
    unlink(escaped_lora_path);
    unlink(registry_path);
    unlink(registry_lora_path);
    unlink(named_lora_path);
    unlink(lora_path);
    rmdir(lora_dir);

    const char *exact_seed = "{\"prompt\":\"cat\",\"guidance_scale\":0,"
                             "\"seed\":9007199254740993}";
    CHECK(oimage_request_parse(exact_seed, strlen(exact_seed), &request, error, sizeof error));
    CHECK(request.generation.guidance_scale == 0.0f);
    CHECK(request.generation.seed == INT64_C(9007199254740993));
    oimage_request_free(&request);

    const char *aliases = "{\"prompt\":\"cat\",\"num_inference_steps\":7}";
    CHECK(oimage_request_parse(aliases, strlen(aliases), &request, error, sizeof error));
    CHECK(request.generation.width == 1024 && request.generation.height == 1024);
    CHECK(request.generation.steps == 7 && request.generation.seed == 0);
    oimage_request_free(&request);

    const char *bad_size = "{\"prompt\":\"cat\",\"size\":\"513x512\"}";
    CHECK(!oimage_request_parse(bad_size, strlen(bad_size), &request, error, sizeof error));
    CHECK(strstr(error, "64-pixel") != NULL);
    const char *bad_batch = "{\"prompt\":\"cat\",\"n\":2}";
    CHECK(!oimage_request_parse(bad_batch, strlen(bad_batch), &request, error, sizeof error));
    setenv("OMNISERVE_NATIVE_SD_MAX_BATCH", "4", 1);
    CHECK(oimage_request_parse(bad_batch, strlen(bad_batch), &request, error, sizeof error));
    CHECK(request.count == 2 && request.generation.batch_count == 2);
    oimage_request_free(&request);
    unsetenv("OMNISERVE_NATIVE_SD_MAX_BATCH");
    const char *bad_guidance = "{\"prompt\":\"cat\",\"guidance_scale\":NaN}";
    CHECK(!oimage_request_parse(bad_guidance, strlen(bad_guidance), &request, error, sizeof error));
    const char *bad_seed = "{\"prompt\":\"cat\",\"seed\":9223372036854775808}";
    CHECK(!oimage_request_parse(bad_seed, strlen(bad_seed), &request, error, sizeof error));
    const char *bad_negative = "{\"prompt\":\"cat\",\"negative_prompt\":42}";
    CHECK(!oimage_request_parse(bad_negative, strlen(bad_negative), &request, error, sizeof error));
    const char *bad_teleport = "{\"prompt\":\"cat\",\"teleport\":\"true\"}";
    CHECK(!oimage_request_parse(bad_teleport, strlen(bad_teleport), &request, error, sizeof error));
    const char *bad_teleport_step =
        "{\"prompt\":\"cat\",\"steps\":9,\"teleport_start_step\":0}";
    CHECK(!oimage_request_parse(bad_teleport_step, strlen(bad_teleport_step),
                                &request, error, sizeof error));

    const char *teleport = "{\"prompt\":\"cat\",\"steps\":9,\"teleport\":true,\"teleport_start_step\":7}";
    CHECK(oimage_request_parse(teleport, strlen(teleport), &request, error, sizeof error));
    CHECK(request.generation.teleport && request.generation.teleport_start_step == 7);
    oimage_request_free(&request);

    const unsigned char image[] = {'a', 'b', 'c'};
    oimg_result image_result = {
        .png = (unsigned char *)image,
        .png_len = sizeof image,
        .elapsed_ms = 12.6,
        .teleport_requested = true,
        .teleport_used = true,
        .teleport_cache_hit = true,
        .teleport_result_cache_hit = true,
        .teleport_capture_step = 6,
        .teleport_resume_step = 7,
    };
    char *json = NULL;
    size_t json_len = 0;
    CHECK(oimage_openai_response(&image_result, "z-image", 42,
                                 &json, &json_len));
    CHECK(json && json_len == strlen(json));
    CHECK(strstr(json, "\"b64_json\":\"YWJj\"") != NULL);
    CHECK(strstr(json, "\"seed\":42") != NULL);
    CHECK(strstr(json, "\"inference_time_ms\":13") != NULL);
    CHECK(strstr(json, "\"cache_hit\":true") != NULL);
    CHECK(strstr(json, "\"method\":\"exact_prompt_result_cache\"") != NULL);
    CHECK(strstr(json, "\"resume_step\":7") != NULL);
    oj_tok response_tokens[32];
    CHECK(oj_parse(json, json_len, response_tokens, 32) > 0);
    free(json);

    unsigned char *batch_images[] = {(unsigned char *)image, (unsigned char *)image};
    size_t batch_lens[] = {sizeof image, sizeof image};
    oimg_result batch_result = {
        .images = batch_images,
        .image_lens = batch_lens,
        .image_count = 2,
        .format = "webp",
        .elapsed_ms = 10.0,
    };
    CHECK(oimage_openai_response(&batch_result, "z-image", 7, &json, &json_len));
    CHECK(strstr(json, "\"format\":\"webp\"") != NULL);
    CHECK(strstr(json, "\"seed\":7") != NULL);
    CHECK(strstr(json, "\"seed\":8") != NULL);
    free(json);
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
        "/v1/images/segmentations", "/v1/images/text-layers", "/v1/images/edits",
        "/v1/images/foreground-generations/jobs",
        "/v1/images/foreground-generations/jobs/{job_id}",
        "/v1/images/background-removals/jobs",
        "/v1/images/background-removals",
        "/v1/images/captions", "/v1/images/classifications",
        "/v1/classifications", "/v1/video/generations", "/loras",
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

/* Saturation is the case the queue exists for, so it is the case worth testing.
 * Many more threads than slots, mixed tiers, every one of them hammering the
 * admission path: the cap must hold exactly, and nobody may be lost - a handoff
 * that signals the wrong waiter shows up here as a thread that never wakes. */
typedef struct {
    osched *s;
    otier tier;
    int rounds;
    _Atomic int *inflight;
    _Atomic int *peak;
    _Atomic int *admitted;
} sched_stress_job;

static void *sched_stress_worker(void *arg) {
    sched_stress_job *j = arg;
    for (int i = 0; i < j->rounds; i++) {
        if (!osched_acquire_n(j->s, j->tier, 1)) continue;
        int now = atomic_fetch_add(j->inflight, 1) + 1;
        int seen = atomic_load(j->peak);
        while (now > seen && !atomic_compare_exchange_weak(j->peak, &seen, now)) {
        }
        atomic_fetch_add(j->admitted, 1);
        atomic_fetch_sub(j->inflight, 1);
        osched_release_n(j->s, j->tier, 1);
    }
    return NULL;
}

static void test_sched_saturation(void) {
    enum { THREADS = 48, SLOTS = 4, ROUNDS = 200 };
    osched *s = osched_create(SLOTS, 30);
    _Atomic int inflight = 0, peak = 0, admitted = 0;
    static const otier tiers[3] = {TIER_PAID, TIER_SUB, TIER_FREE};

    pthread_t workers[THREADS];
    sched_stress_job jobs[THREADS];
    for (int i = 0; i < THREADS; i++) {
        jobs[i] = (sched_stress_job){ .s = s, .tier = tiers[i % 3], .rounds = ROUNDS,
                                      .inflight = &inflight, .peak = &peak,
                                      .admitted = &admitted };
        CHECK(pthread_create(&workers[i], NULL, sched_stress_worker, &jobs[i]) == 0);
    }
    for (int i = 0; i < THREADS; i++) pthread_join(workers[i], NULL);

    CHECK(atomic_load(&admitted) == THREADS * ROUNDS);
    CHECK(atomic_load(&peak) <= SLOTS);
    CHECK(atomic_load(&inflight) == 0);
    CHECK(osched_active(s) == 0);

    osched_stats st;
    osched_snapshot(s, &st);
    CHECK(st.used_slots == 0);
    CHECK(st.waiting[TIER_PAID] == 0 && st.waiting[TIER_FREE] == 0);
    CHECK(st.served[TIER_PAID] + st.served[TIER_SUB] + st.served[TIER_FREE] ==
          (long)THREADS * ROUNDS);
    CHECK(st.timed_out[TIER_PAID] == 0 && st.timed_out[TIER_FREE] == 0);
    osched_destroy(s);
}

/* A waiter that gives up must not take the slot it was queued for with it, and
 * must leave the queue in a state where the next one still gets promoted. */
static void test_sched_timeout_releases_the_queue(void) {
    osched *s = osched_create(1, 0.05);
    CHECK(osched_acquire(s, TIER_FREE));

    atomic_int order = 0;
    sched_job late = { .s = s, .tier = TIER_FREE, .order = &order, .hold_us = 0 };
    pthread_t thread;
    pthread_create(&thread, NULL, sched_worker, &late);
    pthread_join(thread, NULL);
    CHECK(late.got == -1);  /* timed out while the slot was held */

    osched_stats st;
    osched_snapshot(s, &st);
    CHECK(st.timed_out[TIER_FREE] == 1);
    CHECK(st.waiting[TIER_FREE] == 0);
    CHECK(st.used_slots == 1);  /* still exactly the one live holder */

    osched_release(s, TIER_FREE);
    CHECK(osched_acquire(s, TIER_PAID));
    osched_release(s, TIER_PAID);
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
    oproxy_target *bare_target;
} echo_context;

typedef struct {
    uint16_t port;
    const char *response;
    atomic_bool ready;
    atomic_bool failed;
} raw_http_context;

static void *raw_http_server(void *arg) {
    raw_http_context *context = arg;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        atomic_store(&context->failed, true);
        atomic_store(&context->ready, true);
        return NULL;
    }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(context->port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) != 0 || listen(fd, 1) != 0) {
        close(fd);
        atomic_store(&context->failed, true);
        atomic_store(&context->ready, true);
        return NULL;
    }
    atomic_store(&context->ready, true);
    int client = accept(fd, NULL, NULL);
    if (client >= 0) {
        char buf[1024];
        ssize_t ignored = read(client, buf, sizeof buf);
        (void)ignored;
        const char *p = context->response;
        size_t remaining = strlen(context->response);
        while (remaining) {
            ssize_t n = write(client, p, remaining);
            if (n <= 0) break;
            p += n;
            remaining -= (size_t)n;
        }
        close(client);
    } else {
        atomic_store(&context->failed, true);
    }
    close(fd);
    return NULL;
}

static void echo_handler(ohttp_request *req, void *user) {
    echo_context *context = user;
    if (ohttp_path_is(req, "/relay-bare")) {
        oproxy_result result;
        char error[128];
        bool ok = oproxy_target_relay(
            context->bare_target,
            req->method, req->method_len,
            "/bare", 5,
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
    if (ohttp_path_is(req, "/close")) {
        ohttp_force_close(req);
        ohttp_respond_str(req, 200, "text/plain", "bye");
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
        .bare_target = oproxy_target_create("http://127.0.0.1:18792", 1,
                                            target_error, sizeof target_error),
    };
    CHECK(context.target != NULL);
    CHECK(context.bare_target != NULL);
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

    raw_http_context raw_context = {
        .port = 18792,
        .response = "HTTP/1.1 401 Unauthorized\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: 20\r\n"
                    "Connection: close\r\n\r\n"
                    "{\"detail\":\"invalid\"}",
    };
    pthread_t raw_thread;
    pthread_create(&raw_thread, NULL, raw_http_server, &raw_context);
    while (!atomic_load(&raw_context.ready)) usleep(1000);
    CHECK(!atomic_load(&raw_context.failed));
    r = http_roundtrip(18791, "POST /relay-bare HTTP/1.1\r\nHost: x\r\nOrigin: https://text-generator.io\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}", NULL);
    pthread_join(raw_thread, NULL);
    CHECK(r && strstr(r, "HTTP/1.1 401 Unauthorized"));
    CHECK(r && strstr(r, "Access-Control-Allow-Origin: *"));
    CHECK(r && strstr(r, "Content-Type: application/json\r\n"));
    CHECK(r && strstr(r, "{\"detail\":\"invalid\"}"));
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

    fd = connect_local(18791);
    CHECK(fd >= 0);
    const char *close_req =
        "GET /close HTTP/1.1\r\nHost: x\r\n\r\n"
        "GET /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\nhi";
    CHECK(test_write_all(fd, close_req, strlen(close_req)));
    n = read(fd, buf, sizeof buf - 1);
    CHECK(n > 0);
    if (n > 0) {
        buf[n] = 0;
        CHECK(strstr(buf, "200 OK") && strstr(buf, "Connection: close"));
        CHECK(count_text(buf, "HTTP/1.1 200 OK") == 1);
    }
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
    oproxy_target_destroy(context.bare_target);
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
    /* A 3090 hits the same 24 GB wall as a 4090 and runs the same quantized
     * flash-attention kernels, so it gets the same KV cache. A profile that
     * enables a quantized cache without flash attention would fail to load a
     * context at all: llama.cpp requires FA for a quantized V. */
    CHECK(strcmp(profile.kv_type, "q8_0") == 0);
    CHECK(profile.flash_attn);
    CHECK(profile.parallel_contexts == 3);

    otune_profile_for("NVIDIA H100 PCIe", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_HOPPER);
    otune_profile_for("cpu", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_CPU);
    CHECK(profile.parallel_contexts == 1);

    /* An unrecognised device must still yield a usable profile. */
    otune_profile_for("Some Future Accelerator", &profile);
    CHECK(profile.device_class == OTUNE_DEVICE_UNKNOWN);
    CHECK(profile.n_batch > 0 && profile.n_ubatch > 0 && profile.kv_type != NULL);

    /* Holds for every profile, not just the ones spelled out above: a
     * quantized V without flash attention is a context llama.cpp refuses to
     * create, and a micro-batch wider than the batch is a config it silently
     * clamps. Both are easy to introduce while tuning one card. */
    static const char *const devices[] = {
        "NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090", "NVIDIA RTX 3090",
        "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "Tesla T4", "cpu",
        "Some Future Accelerator",
    };
    for (size_t i = 0; i < sizeof devices / sizeof *devices; i++) {
        otune_profile_for(devices[i], &profile);
        CHECK(profile.n_ubatch > 0 && profile.n_ubatch <= profile.n_batch);
        CHECK(profile.parallel_contexts >= 1);
        if (strcmp(profile.kv_type, "f16") != 0 && strcmp(profile.kv_type, "bf16") != 0) {
            CHECK(profile.flash_attn);
        }
    }
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

/* Reads the whole access-log set (current file plus rotated generations) so an
 * assertion cannot pass just because the interesting line rotated away. */
static char *read_log_set(const char *dir, size_t *out_len) {
    size_t cap = 1 << 20, len = 0;
    char *all = malloc(cap);
    if (!all) return NULL;
    all[0] = 0;
    for (int i = 0; i < OLOG_KEEP_FILES; i++) {
        char path[512];
        if (i == 0) snprintf(path, sizeof path, "%s/access.log", dir);
        else snprintf(path, sizeof path, "%s/access.log.%d", dir, i);
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        size_t n = fread(all + len, 1, cap - len - 1, f);
        len += n;
        all[len] = 0;
        fclose(f);
    }
    if (out_len) *out_len = len;
    return all;
}

static int log_files_present(const char *dir) {
    int present = 0;
    for (int i = 0; i <= OLOG_KEEP_FILES; i++) {
        char path[512];
        if (i == 0) snprintf(path, sizeof path, "%s/access.log", dir);
        else snprintf(path, sizeof path, "%s/access.log.%d", dir, i);
        struct stat st;
        if (stat(path, &st) == 0) present++;
    }
    return present;
}

static int unused_loopback_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    socklen_t len = sizeof addr;
    int port = 0;
    if (bind(fd, (struct sockaddr *)&addr, sizeof addr) == 0 &&
        getsockname(fd, (struct sockaddr *)&addr, &len) == 0) {
        port = ntohs(addr.sin_port);
    }
    close(fd);
    return port;
}

/* The access log is the one place a credential could leak to disk by accident,
 * so the redaction is pinned here rather than left to review. */
static void test_access_log(void) {
    char dir[] = "/tmp/onative-accesslog-XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    setenv("OMNISERVE_ACCESS_LOG_DIR", dir, 1);
    setenv("OMNISERVE_ACCESS_LOG_MAX_BYTES", "4096", 1);
    setenv("OMNISERVE_ACCESS_LOG", "1", 1);
    /* The gateway registers its real bypass-header list here; the test
     * registers a stand-in to prove a registered header is logged by name. */
    static const char *const test_trust_headers[] = { "X-Forwarded-For", "Via" };
    olog_set_trust_headers(test_trust_headers, 2);
    olog_init();
    CHECK(olog_enabled());

    echo_context context = { .target = NULL };
    int port = unused_loopback_port();
    CHECK(port > 0);
    ohttp_config cfg = { .port = port, .reactor_threads = 1, .worker_threads = 2,
                         .handler = echo_handler, .user = &context };
    ohttp_server *srv = ohttp_start(&cfg);
    CHECK(srv != NULL);
    usleep(100000);

    char *r = http_roundtrip(port,
        "POST /echo?token=querysecret789 HTTP/1.1\r\nHost: x\r\n"
        "Authorization: Bearer supersecret123\r\nX-API-Key: topsecretkey456\r\n"
        "X-Forwarded-For: 8.8.8.8\r\nX-Omniserve-Tier: paid\r\n"
        "Content-Length: 2\r\nConnection: close\r\n\r\nhi", NULL);
    CHECK(r && strstr(r, "200 OK"));
    free(r);

    /* A bare CR, a quote and a control byte in the path: one request must stay
     * one line no matter what the client puts in the request target. */
    r = http_roundtrip(port,
        "GET /inject\"\rmarker\x01x HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", NULL);
    free(r);

    olog_flush();
    size_t len = 0;
    char *log = read_log_set(dir, &len);
    CHECK(log != NULL);
    if (log) {
        CHECK(strstr(log, "supersecret123") == NULL);
        CHECK(strstr(log, "topsecretkey456") == NULL);
        CHECK(strstr(log, "querysecret789") == NULL);
        CHECK(strstr(log, "token=") == NULL);
        CHECK(strstr(log, "Authorization") != NULL);
        CHECK(strstr(log, "X-API-Key") != NULL);
        CHECK(strstr(log, "X-Forwarded-For") != NULL);
        CHECK(strstr(log, "X-Omniserve-Tier") != NULL);
        CHECK(strstr(log, "path=\"/echo\"") != NULL);
        CHECK(strstr(log, "status=200") != NULL);
        /* Escaped, so the forged bytes cannot start a second record. */
        CHECK(count_text(log, "/inject") == 1);
        CHECK(strstr(log, "\\x0d") != NULL && strstr(log, "\\x01") != NULL);
        CHECK(strstr(log, "marker\n") == NULL);
        int lines = count_text(log, "\n");
        CHECK(lines == 2);
    }
    free(log);

    /* Rotation: 4 KiB cap, so a few hundred requests must still leave at most
     * OLOG_KEEP_FILES files behind. */
    for (int i = 0; i < 400; i++) {
        char raw[256];
        snprintf(raw, sizeof raw,
                 "GET /rotate/%d HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", i);
        free(http_roundtrip(port, raw, NULL));
    }
    olog_flush();
    usleep(50000);
    CHECK(log_files_present(dir) <= OLOG_KEEP_FILES);
    for (int i = 0; i <= OLOG_KEEP_FILES; i++) {
        char path[512];
        if (i == 0) snprintf(path, sizeof path, "%s/access.log", dir);
        else snprintf(path, sizeof path, "%s/access.log.%d", dir, i);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        CHECK(i < OLOG_KEEP_FILES);
        CHECK((size_t)st.st_size <= 4096 + OLOG_SLOT_BYTES);
    }
    unsigned long long emitted = 0, dropped = 0;
    olog_counters(&emitted, &dropped);
    CHECK(emitted + dropped >= 400);

    ohttp_stop(srv);
    CHECK(ohttp_join(srv) == 0);
    olog_shutdown();
    for (int i = 0; i <= OLOG_KEEP_FILES; i++) {
        char path[512];
        if (i == 0) snprintf(path, sizeof path, "%s/access.log", dir);
        else snprintf(path, sizeof path, "%s/access.log.%d", dir, i);
        unlink(path);
    }
    rmdir(dir);
}

/* The draft policy takes plain token ids, so what it proposes is provable
 * without a model behind it. */
static void test_spec_draft(void) {
    ospec_config cfg;
    ospec_config_default(&cfg);
    int32_t draft[8];

    /* "the quick brown fox ... the quick brown" continues "fox". Everything
     * after the match is offered, not just the part that is obviously
     * predictive: a wrong tail costs a batch slot that was already paid for,
     * and the caller's cap is the real bound. */
    const int32_t repeat[] = {10, 11, 12, 13, 14, 15, 10, 11, 12};
    int n = ospec_draft(&cfg, repeat, 9, draft, 8);
    CHECK(n == 6);
    CHECK(draft[0] == 13 && draft[1] == 14 && draft[2] == 15);

    /* Capped by the caller's budget, not by what is available. */
    CHECK(ospec_draft(&cfg, repeat, 9, draft, 2) == 2);
    CHECK(draft[0] == 13 && draft[1] == 14);

    /* No repeat, so nothing is proposed. Returning a guess here would be worse
     * than returning none: it would be rejected and cost a wider batch. */
    const int32_t unique[] = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(ospec_draft(&cfg, unique, 8, draft, 8) == 0);

    /* A match shorter than min_ngram is coincidence and must not be drafted
     * from; with min_ngram 3 a repeated pair is not enough. */
    const int32_t pair[] = {1, 2, 9, 9, 9, 1, 2};
    CHECK(ospec_draft(&cfg, pair, 7, draft, 8) == 0);

    /* The suffix must not match itself: the last n tokens are the needle. */
    const int32_t tail_only[] = {4, 5, 6};
    CHECK(ospec_draft(&cfg, tail_only, 3, draft, 8) == 0);

    /* A pattern that has already repeated once is predicted to repeat again:
     * the match at the start is followed by the pattern itself. */
    const int32_t at_end[] = {7, 8, 9, 7, 8, 9};
    n = ospec_draft(&cfg, at_end, 6, draft, 8);
    CHECK(n == 3 && draft[0] == 7 && draft[1] == 8 && draft[2] == 9);

    /* Two candidate matches: the longer one wins even though the shorter one
     * sits closer to the end. Here the 4-gram 2,3,4,5 predicts 99, while the
     * 3-gram 3,4,5 also occurs later predicting 77. */
    const int32_t both[] = {2, 3, 4, 5, 99, 0, 3, 4, 5, 77, 0, 2, 3, 4, 5};
    n = ospec_draft(&cfg, both, 15, draft, 4);
    CHECK(n >= 1 && draft[0] == 99);

    /* Among equally long matches the most recent wins. */
    const int32_t recent[] = {1, 2, 3, 50, 0, 0, 1, 2, 3, 60, 0, 0, 1, 2, 3};
    n = ospec_draft(&cfg, recent, 15, draft, 1);
    CHECK(n == 1 && draft[0] == 60);

    /* Degenerate inputs must not read out of bounds. */
    CHECK(ospec_draft(&cfg, NULL, 5, draft, 8) == 0);
    CHECK(ospec_draft(&cfg, repeat, 0, draft, 8) == 0);
    CHECK(ospec_draft(&cfg, repeat, 9, draft, 0) == 0);
    CHECK(ospec_draft(NULL, repeat, 9, draft, 8) == 0);
}

/* The governor is what keeps a bad draft source from costing anything, so its
 * back-off and its recovery are both worth pinning down. */
static void test_spec_governor(void) {
    ospec_governor g;
    ospec_governor_init(&g, 4, 8, 1);

    /* Starts optimistic: the first rounds are the cheapest place to find out
     * whether this request copies from its context. */
    CHECK(ospec_governor_next(&g) == 4);

    /* Everything accepted: the draft was too short for the round trip. */
    ospec_governor_observe(&g, 4, 4);
    CHECK(g.length == 4);  /* already at max */

    /* Partial acceptance settles on the length that actually landed. */
    ospec_governor_observe(&g, 4, 2);
    CHECK(ospec_governor_next(&g) == 2);
    ospec_governor_observe(&g, 2, 2);
    CHECK(ospec_governor_next(&g) == 3);  /* all accepted, grow */

    /* A miss turns speculation off outright rather than tapering. Composing
     * text stays composing text for many tokens, and each wasted round is a
     * wider batch that produced one token. */
    ospec_governor_observe(&g, 3, 0);
    CHECK(g.length == 0);

    /* Off means free: for a whole interval no draft is even requested, so a
     * request that never copies stops paying for speculation entirely. Each
     * next() is one round of the probe clock, so the count below is the
     * interval exactly. */
    int probes = 0;
    for (int i = 0; i < 8; i++) {
        if (ospec_governor_next(&g) > 0) probes++;
        ospec_governor_observe(&g, 0, 0);
    }
    CHECK(probes == 0);

    /* Then exactly one probe comes due. */
    CHECK(ospec_governor_next(&g) == 1);

    /* A probe that lands brings speculation back, cautiously, so a request
     * that starts inventing and later starts quoting is not stuck at one
     * token per model call. */
    ospec_governor_observe(&g, 1, 1);
    CHECK(g.length == 1);
    CHECK(ospec_governor_next(&g) == 1);
    ospec_governor_observe(&g, 1, 1);
    CHECK(g.length == 2);

    /* Patience is the one hardware-dependent number here: on a device where a
     * wrong guess is nearly free, quitting after one miss forfeits most of the
     * win. With patience 3 the first two misses shorten the draft instead of
     * abandoning it. */
    ospec_governor patient;
    ospec_governor_init(&patient, 4, 8, 3);
    CHECK(ospec_governor_next(&patient) == 4);
    ospec_governor_observe(&patient, 4, 0);
    CHECK(patient.length == 2);   /* halved, still speculating */
    ospec_governor_observe(&patient, 2, 0);
    CHECK(patient.length == 1);
    ospec_governor_observe(&patient, 1, 0);
    CHECK(patient.length == 0);   /* third strike */

    /* A landed draft anywhere in between clears the miss count, so speculation
     * survives an isolated miss inside a passage that is otherwise quoted. */
    ospec_governor_init(&patient, 4, 8, 3);
    ospec_governor_observe(&patient, 4, 0);
    ospec_governor_observe(&patient, 2, 2);
    CHECK(patient.misses == 0);
    ospec_governor_observe(&patient, 3, 0);
    ospec_governor_observe(&patient, 1, 0);
    CHECK(patient.length > 0);    /* two misses, patience 3, still alive */

    /* Accounting: accepted tokens are model calls that did not happen. */
    ospec_governor g2;
    ospec_governor_init(&g2, 4, 8, 1);
    ospec_governor_observe(&g2, 4, 3);
    ospec_governor_observe(&g2, 3, 1);
    CHECK(g2.drafted == 7 && g2.accepted == 4 && g2.saved_calls == 4);
    CHECK(ospec_acceptance_rate(&g2) > 0.57 && ospec_acceptance_rate(&g2) < 0.58);

    /* Nothing drafted yet is a rate of zero, not a division by zero. */
    ospec_governor g3;
    ospec_governor_init(&g3, 4, 8, 1);
    CHECK(ospec_acceptance_rate(&g3) == 0.0);
    CHECK(ospec_acceptance_rate(NULL) == 0.0);

    /* An accept count larger than the draft is a caller bug; clamp rather than
     * letting it inflate the saved-call figure the dashboards read. */
    ospec_governor_observe(&g3, 2, 9);
    CHECK(g3.accepted == 2 && g3.saved_calls == 2);
}

/* The deterministic entry points take the clock and the device figure as
 * arguments precisely so the arbitration policy is provable without a GPU. */
static void test_vram_arbitration(void) {
    char id_a[40], id_b[40];
    ovram *v = ovram_create(1024, 60.0);
    CHECK(v != NULL);

    /* 8192 free, 1024 floor for background: 7168 grantable. */
    CHECK(ovram_headroom_at(v, TIER_BACKGROUND, 100.0, 8192) == 7168);

    /* The embedded image lane leases its whole scratch floor, not a partial
     * best effort.  While it is active no second background tenant can size
     * itself against those same physical bytes. */
    char image_id[40], competing_id[40];
    CHECK(ovram_lease_at(v, "embedded-zimage", 4096, 4096,
                         TIER_BACKGROUND, 60.0, 90.0, 8192,
                         image_id, sizeof image_id) == 4096);
    CHECK(ovram_lease_at(v, "competing-gpu-tenant", 4096, 4096,
                         TIER_BACKGROUND, 60.0, 90.0, 8192,
                         competing_id, sizeof competing_id) == 0);
    CHECK(competing_id[0] == '\0');
    CHECK(ovram_release(v, image_id));

    /* The whole point: a granted lease is subtracted from what the next caller
     * sees, so two tenants cannot size against the same free bytes. */
    int a = ovram_lease_at(v, "zimage", 4096, 1024, TIER_BACKGROUND, 60.0, 100.0, 8192,
                           id_a, sizeof id_a);
    CHECK(a == 4096);
    CHECK(ovram_headroom_at(v, TIER_BACKGROUND, 100.0, 8192) == 3072);
    CHECK(ovram_renew_at(v, id_a, 120.0, 110.0));
    CHECK(!ovram_renew_at(v, "missing", 120.0, 110.0));

    /* Asking beyond headroom yields a partial grant when min_mb still fits. */
    int b = ovram_lease_at(v, "other", 8192, 1024, TIER_BACKGROUND, 60.0, 100.0, 8192,
                           id_b, sizeof id_b);
    CHECK(b == 3072);
    CHECK(ovram_headroom_at(v, TIER_BACKGROUND, 100.0, 8192) == 0);

    /* Below min_mb is a denial, and a denial must not mint a lease id. */
    char id_c[40] = "dirty";
    CHECK(ovram_lease_at(v, "third", 2048, 2048, TIER_BACKGROUND, 60.0, 100.0, 8192,
                         id_c, sizeof id_c) == 0);
    CHECK(id_c[0] == '\0');

    /* Higher-priority traffic is not starved by background leases. Those
     * leases still protect background tenants from each other, but normal
     * interactive work sizes from the real device headroom and its own floor. */
    CHECK(ovram_headroom_at(v, TIER_FREE, 100.0, 8192) == 7424);
    CHECK(ovram_headroom_at(v, TIER_PAID, 100.0, 8192) == 7936);

    CHECK(ovram_release(v, id_a));
    CHECK(!ovram_release(v, id_a));
    CHECK(ovram_headroom_at(v, TIER_BACKGROUND, 100.0, 8192) == 4096);

    /* A tenant that dies between lease and release must not strand headroom. */
    CHECK(ovram_expire_at(v, 100.0 + 61.0) == 1);
    CHECK(ovram_headroom_at(v, TIER_BACKGROUND, 200.0, 8192) == 7168);

    /* An unreadable driver denies rather than guessing a free figure. */
    CHECK(ovram_headroom_at(v, TIER_PAID, 200.0, -1) == 0);
    CHECK(ovram_lease_at(v, "zimage", 512, 0, TIER_PAID, 60.0, 200.0, -1,
                         id_a, sizeof id_a) == 0);

    /* Declared future growth is withheld from everyone. */
    CHECK(ovram_reserve(v, "netwrck", 2048));
    CHECK(ovram_headroom_at(v, TIER_BACKGROUND, 200.0, 8192) == 5120);
    CHECK(ovram_reserve(v, "netwrck", 0));
    CHECK(ovram_headroom_at(v, TIER_BACKGROUND, 200.0, 8192) == 7168);

    /* id_a was released and id_b expired above, so nothing is outstanding.
     * That matters because ovram_snapshot reads the real clock: a fake-clock
     * lease still held here would be reaped by it and skew the counters. */
    ovram_stats st;
    ovram_snapshot(v, &st);
    CHECK(st.grants == 3 && st.partial_grants == 1 && st.denials == 3);
    CHECK(st.releases == 2 && st.expirations == 1);
    CHECK(st.lease_count == 0);

    ovram_destroy(v);
}

static void test_host_prefetch_policy(void) {
    ohost_meminfo mi = {0};
    mi.mem_total_kb = 256L * 1024 * 1024;      /* 256 GiB */
    mi.mem_available_kb = 128L * 1024 * 1024;  /* 128 GiB */

    /* 5% floor of 256 GiB is 12.8 GiB, leaving 115.2 GiB spare. */
    long long headroom = ohost_headroom_bytes(&mi, 5);
    CHECK(headroom == (128LL * 1024 * 1024 - 256LL * 1024 * 1024 * 5 / 100) * 1024);

    /* A file smaller than the headroom is warmed whole; a larger one is
     * truncated to the headroom rather than warmed until the box swaps. */
    CHECK(ohost_plan_bytes(8LL * 1024 * 1024 * 1024, headroom) == 8LL * 1024 * 1024 * 1024);
    CHECK(ohost_plan_bytes(headroom * 4, headroom) == headroom);

    /* At or under the floor nothing is warmed at all, and a zero-percent floor
     * still cannot plan past what is actually available. */
    mi.mem_available_kb = 4L * 1024 * 1024;
    CHECK(ohost_headroom_bytes(&mi, 5) == 0);
    CHECK(ohost_plan_bytes(1024, ohost_headroom_bytes(&mi, 5)) == 0);
    CHECK(ohost_headroom_bytes(&mi, 0) == 4LL * 1024 * 1024 * 1024);

    /* Reading the real /proc must agree with itself. */
    ohost_meminfo live;
    if (ohost_read_meminfo(&live)) {
        CHECK(live.mem_total_kb > 0);
        CHECK(live.mem_available_kb <= live.mem_total_kb);
    }
}

/* Overflow routing hinges entirely on try-acquire refusing rather than waiting,
 * so its refusal conditions are the contract worth pinning down. */
static void test_sched_try_acquire(void) {
    osched *s = osched_create(2, 5.0);
    CHECK(osched_try_acquire_n(s, TIER_PAID, 2));
    /* Saturated: the caller is expected to go elsewhere, not to block. */
    CHECK(!osched_try_acquire_n(s, TIER_PAID, 1));
    osched_release_n(s, TIER_PAID, 2);
    CHECK(osched_try_acquire_n(s, TIER_PAID, 1));

    /* Background still demands an idle device, not merely a free slot. */
    CHECK(!osched_try_acquire_n(s, TIER_BACKGROUND, 2));
    osched_release_n(s, TIER_PAID, 1);
    CHECK(osched_try_acquire_n(s, TIER_BACKGROUND, 2));
    osched_release_n(s, TIER_BACKGROUND, 2);

    CHECK(!osched_try_acquire_n(s, TIER_PAID, 3));  /* more than total slots */
    CHECK(!osched_try_acquire_n(s, TIER_PAID, 0));
    CHECK(!osched_try_acquire_n(NULL, TIER_PAID, 1));
    osched_destroy(s);
}

int main(void) {
    test_sched_try_acquire();
    test_spec_draft();
    test_spec_governor();
    test_host_prefetch_policy();
    test_vram_arbitration();
    test_json();
    test_image_contract();
    test_matte();
    test_tier_parse();
    test_completion_spacing();
    test_openapi();
    test_sched_priority();
    test_sched_timeout();
    test_sched_timeout_releases_the_queue();
    test_sched_saturation();
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
    test_access_log();
    if (failures) {
        fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    fprintf(stderr, "all core tests passed\n");
    return 0;
}
