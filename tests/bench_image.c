#include "obackend.h"
#include "oimage.h"
#include "stb_image_write.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static char *b64(const unsigned char *in, size_t len) {
    char *out = malloc((len + 2) / 3 * 4 + 1);
    size_t o = 0;
    for (size_t i = 0; i < len; i += 3) {
        unsigned v = (unsigned)in[i] << 16;
        if (i + 1 < len) v |= (unsigned)in[i + 1] << 8;
        if (i + 2 < len) v |= in[i + 2];
        out[o++] = B64[v >> 18 & 63];
        out[o++] = B64[v >> 12 & 63];
        out[o++] = i + 1 < len ? B64[v >> 6 & 63] : '=';
        out[o++] = i + 2 < len ? B64[v & 63] : '=';
    }
    out[o] = 0;
    return out;
}

typedef struct { unsigned char *data; size_t len; } sink;

static void sink_write(void *ctx, void *data, int size) {
    sink *s = ctx;
    s->data = realloc(s->data, s->len + (size_t)size);
    memcpy(s->data + s->len, data, (size_t)size);
    s->len += (size_t)size;
}

extern int stbi_write_png_compression_level;

#define REPORT(name, iters, t0) \
    printf("%-28s %10.1f us/op\n", name, (now_ms() - (t0)) * 1000.0 / (iters))

int main(void) {
    const char *stub = getenv("OMNISERVE_SD_STUB");
    if (!stub) { fputs("set OMNISERVE_SD_STUB\n", stderr); return 2; }
    setenv("OMNISERVE_NATIVE_SD_LIB", stub, 1);
    setenv("OMNISERVE_STUB_DELAY_MS", "0", 1);
    setenv("OMNISERVE_STUB_SIZE", "1024", 1);
    setenv("OMNISERVE_NATIVE_SD_TELEPORT_CACHE_SIZE", "64", 1);
    if (getenv("OMNISERVE_BENCH_PNG_LEVEL")) stbi_write_png_compression_level = atoi(getenv("OMNISERVE_BENCH_PNG_LEVEL"));
    if (!osd_init("bench.gguf")) { fputs("osd_init failed\n", stderr); return 1; }

    enum { W = 1024, H = 1024 };
    unsigned char *pix = malloc((size_t)W * H * 3);
    uint32_t x = 12345;
    for (size_t i = 0; i < (size_t)W * H * 3; ++i) {
        x = x * 1664525u + 1013904223u;
        pix[i] = (unsigned char)(((i / 3) % W) / 4 + ((x >> 24) & 15));
    }
    sink png = {0};
    stbi_write_png_to_func(sink_write, &png, W, H, 3, pix, W * 3);
    char *image_b64 = b64(png.data, png.len);
    printf("image png %zu bytes, base64 %zu bytes\n", png.len, strlen(image_b64));

    int n = 30;
    double t = now_ms();
    for (int i = 0; i < n; ++i) {
        oimg_req req = {0};
        req.image_base64 = image_b64;
        if (!osd_prepare_image(&req)) { fputs("prepare failed\n", stderr); return 1; }
        free(req.image_pixels);
    }
    REPORT("prepare_image png 1024^2", n, t);

    t = now_ms();
    for (int i = 0; i < n; ++i) {
        oimg_req req = {0};
        req.image_base64 = image_b64;
        req.cache = true;
        if (!osd_prepare_image(&req)) { fputs("prepare failed\n", stderr); return 1; }
        free(req.image_pixels);
    }
    REPORT("prepare_image +cache hash", n, t);

    size_t body_cap = strlen(image_b64) + 256;
    char *body = malloc(body_cap);
    int body_len = snprintf(body, body_cap,
        "{\"prompt\":\"a cat\",\"size\":\"1024x1024\",\"steps\":4,\"seed\":7,\"cache\":true,"
        "\"image_base64\":\"%s\"}", image_b64);
    n = 300;
    t = now_ms();
    for (int i = 0; i < n; ++i) {
        oimage_request r;
        char err[128];
        if (!oimage_request_parse(body, (size_t)body_len, &r, err, sizeof err)) {
            fprintf(stderr, "parse: %s\n", err);
            return 1;
        }
        oimage_request_free(&r);
    }
    REPORT("request_parse w/ image", n, t);

    oimage_request base;
    char err[128];
    if (!oimage_request_parse(body, (size_t)body_len, &base, err, sizeof err)) return 1;
    oimg_req *req = &base.generation;
    if (!osd_prepare_image(req)) return 1;

    n = 64;
    t = now_ms();
    for (int i = 0; i < n; ++i) {
        req->seed = i;
        oimg_result out;
        if (!osd_generate(req, &out)) { fputs("generate failed\n", stderr); return 1; }
        osd_result_free(&out);
    }
    REPORT("generate miss+encode+insert", n, t);

    n = 20000;
    t = now_ms();
    for (int i = 0; i < n; ++i) {
        req->seed = i % 64;
        oimg_result out;
        if (!osd_try_cached_result(req, &out)) { fputs("expected hit\n", stderr); return 1; }
        osd_result_free(&out);
    }
    REPORT("try_cached hit (64 entries)", n, t);

    n = 2000;
    t = now_ms();
    for (int i = 0; i < n; ++i) {
        req->seed = 1000 + i;
        oimg_result out;
        if (osd_try_cached_result(req, &out)) return 1;
    }
    REPORT("try_cached miss", n, t);

    req->cache = false;
    req->seed = 5;
    n = 64;
    t = now_ms();
    for (int i = 0; i < n; ++i) {
        oimg_result out;
        if (!osd_generate(req, &out)) return 1;
        osd_result_free(&out);
    }
    REPORT("generate nocache+encode", n, t);

    oimg_result out;
    req->cache = true;
    req->seed = 3;
    if (!osd_try_cached_result(req, &out)) return 1;
    n = 300;
    t = now_ms();
    for (int i = 0; i < n; ++i) {
        char *json = NULL;
        size_t json_len = 0;
        if (!oimage_openai_response(&out, "m", 3, &json, &json_len)) return 1;
        free(json);
    }
    REPORT("openai_response", n, t);
    printf("result %zu bytes format %s\n", out.image_lens[0], out.format);
    osd_result_free(&out);
    oimage_request_free(&base);
    return 0;
}
