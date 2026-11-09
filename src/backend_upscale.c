#define _GNU_SOURCE
#include "oupscale.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_SD

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_write.h"

bool osd_ready(void);

typedef upscaler_ctx_t *(*fn_new_up)(const char *, bool, int, int, const char *, const char *);
typedef bool (*fn_up)(upscaler_ctx_t *, sd_image_t, uint32_t, sd_image_t **, int *);
typedef int (*fn_up_factor)(upscaler_ctx_t *);
typedef void (*fn_free_up)(upscaler_ctx_t *);
typedef void (*fn_set_log)(sd_log_cb_t, void *);

static fn_new_up p_new_up;
static fn_up p_up;
static fn_up_factor p_up_factor;
static fn_free_up p_free_up;
static void (*p_set_tile)(upscaler_ctx_t *, int);

typedef size_t (*fn_webp_enc)(const unsigned char *, int, int, int, float, unsigned char **);
typedef unsigned char *(*fn_webp_dec_into)(const unsigned char *, size_t, unsigned char *, size_t, int);
typedef int (*fn_webp_info)(const unsigned char *, size_t, int *, int *);
static fn_webp_enc p_webp_rgb, p_webp_rgba;
static fn_webp_dec_into p_webp_dec_rgba;
static fn_webp_info p_webp_info;
static void (*p_webp_free)(void *);
static int (*p_oenc_png)(const unsigned char *, int, int, int, int, int, unsigned char **, size_t *);
static int (*p_oenc_webp)(const unsigned char *, int, int, int, float, int, int, int, int, unsigned char **, size_t *);
static bool g_direct;

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_lib_ok;

enum { NET_X4_GENERAL, NET_X4_ANIME, NET_X2_GENERAL, NET_COUNT };
static const char *const net_env[NET_COUNT] = {
    "OMNISERVE_NATIVE_ESRGAN_GENERAL", "OMNISERVE_NATIVE_ESRGAN_ANIME", "OMNISERVE_NATIVE_ESRGAN_GENERAL_X2",
};
static const char *const net_label[NET_COUNT] = {"x4plus", "x4plus_anime_6B", "x2plus"};
static upscaler_ctx_t *g_ctx[NET_COUNT];
static int g_factor[NET_COUNT];
static int g_tile[NET_COUNT];
static bool g_failed[NET_COUNT];
static unsigned long long g_runs[NET_COUNT];
static double g_last_ms[NET_COUNT];

#define OUP_BYTES_PER_OUT_PX 1700.0

static void up_log(enum sd_log_level_t level, const char *text, void *data) {
    (void)data;
    if (level >= SD_LOG_WARN && text) fputs(text, stderr);
}

static int env_int(const char *name, int fallback);

static void *dlopen_near_exe(const char *name) {
    void *h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (h || strchr(name, '/')) return h;
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return NULL;
    exe[n] = 0;
    char *slash = strrchr(exe, '/');
    if (!slash) return NULL;
    snprintf(slash + 1, sizeof exe - (size_t)(slash + 1 - exe), "%s", name);
    return dlopen(exe, RTLD_NOW | RTLD_LOCAL);
}

/* ggml's optional cuDNN conv (ggml-cuda/conv2d.cu) only runs for direct convs; without cuDNN the
 * direct kernel is slower than im2col, so direct is enabled only when cuDNN loads. */
static void cudnn_probe(void) {
    const char *on = getenv("GGML_CUDA_CUDNN_CONV");
    if (!env_int("OMNISERVE_NATIVE_ESRGAN_CUDNN", 1) || (on && strcmp(on, "1"))) return;
    const char *lib = getenv("GGML_CUDA_CUDNN_LIB");
    void *h = dlopen(lib && lib[0] ? lib : "libcudnn.so.9", RTLD_NOW | RTLD_LOCAL);
    if (!h || !dlsym(h, "cudnnConvolutionForward")) {
        fprintf(stderr, "[upscale] cuDNN unavailable, im2col conv\n");
        return;
    }
    setenv("GGML_CUDA_CUDNN_CONV", "1", 0);
    setenv("GGML_CUDA_CUDNN_CONV_F32ACC", "1", 0);
    g_direct = true;
}

static void lib_load(void) {
    cudnn_probe();
    const char *path = getenv("OMNISERVE_NATIVE_SD_LIB");
    if (!path || !path[0]) {
#ifdef OMNISERVE_SD_LIB_DEFAULT
        path = OMNISERVE_SD_LIB_DEFAULT;
#else
        path = "libstable-diffusion.so";
#endif
    }
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!lib) {
        fprintf(stderr, "[upscale] sd dlopen failed: %s\n", dlerror());
        return;
    }
    p_new_up = (fn_new_up)dlsym(lib, "new_upscaler_ctx");
    p_up = (fn_up)dlsym(lib, "upscale");
    p_up_factor = (fn_up_factor)dlsym(lib, "get_upscale_factor");
    p_free_up = (fn_free_up)dlsym(lib, "free_upscaler_ctx");
    *(void **)&p_set_tile = dlsym(lib, "sd_upscaler_set_tile_size");
    if (!osd_ready()) {
        fn_set_log set_log = (fn_set_log)dlsym(lib, "sd_set_log_callback");
        if (set_log) set_log(up_log, NULL);
    }
    g_lib_ok = p_new_up && p_up && p_up_factor;
    static const char *const webp[] = {"libwebp.so.7", "libwebp.so.6", "libwebp.so", NULL};
    for (int i = 0; webp[i]; ++i) {
        void *h = dlopen(webp[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        p_webp_rgb = (fn_webp_enc)dlsym(h, "WebPEncodeRGB");
        p_webp_rgba = (fn_webp_enc)dlsym(h, "WebPEncodeRGBA");
        p_webp_dec_rgba = (fn_webp_dec_into)dlsym(h, "WebPDecodeRGBAInto");
        p_webp_info = (fn_webp_info)dlsym(h, "WebPGetInfo");
        *(void **)&p_webp_free = dlsym(h, "WebPFree");
        if (p_webp_rgb && p_webp_free) break;
        dlclose(h);
    }
    const char *oenc = getenv("OMNISERVE_NATIVE_OENCODE_LIB");
    void *oh = dlopen_near_exe(oenc && oenc[0] ? oenc : "liboencode.so");
    if (oh) {
        *(void **)&p_oenc_png = dlsym(oh, "oenc_png");
        *(void **)&p_oenc_webp = dlsym(oh, "oenc_webp");
    }
    fprintf(stderr, "[upscale] conv=%s png=%s webp=%s\n", g_direct ? "cudnn" : "im2col", p_oenc_png ? "libdeflate" : "stb",
            p_oenc_webp ? "oencode" : "libwebp-simple");
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static const char *net_path(int net) {
    const char *p = getenv(net_env[net]);
    struct stat st;
    return p && p[0] && stat(p, &st) == 0 && S_ISREG(st.st_mode) ? p : NULL;
}

static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    return v && v[0] ? atoi(v) : fallback;
}

static int pick_net(oup_model m, int scale) {
    if (m == OUP_ANIME) return NET_X4_ANIME;
    if (scale == 2 && net_path(NET_X2_GENERAL)) return NET_X2_GENERAL;
    return NET_X4_GENERAL;
}

bool oup_backend_available(oup_model m, int scale) {
    pthread_once(&g_once, lib_load);
    int net = pick_net(m, scale);
    return g_lib_ok && net_path(net) && !g_failed[net];
}

static int vram_budget_mb(void) {
    int mb = env_int("OMNISERVE_NATIVE_ESRGAN_VRAM_MB", 4096);
    return mb < 512 ? 512 : mb;
}

static int net_factor(int net) { return net == NET_X2_GENERAL ? 2 : 4; }

/* Tile edge in input pixels, 0 = untiled. OMNISERVE_NATIVE_ESRGAN_TILE > 0
 * forces it, < 0 forces untiled; 0 (default) runs untiled while the compute
 * buffer (~1.7 KB per output pixel, measured for x4plus/x2plus/anime) fits
 * OMNISERVE_NATIVE_ESRGAN_VRAM_MB, else the smallest tile count per axis whose
 * square tile (sd.cpp overlaps tiles by 25%) fits it. A library without
 * sd_upscaler_set_tile_size keeps the budget's largest tile for every image. */
/* Compute bytes per output pixel: im2col ~1700 (all nets); cuDNN direct conv measured ~950 (x2plus)
 * and ~1440 (x4 nets) peak process VRAM, sized here with ~15% margin. */
static double out_px_bytes(int f) {
    if (!g_direct) return OUP_BYTES_PER_OUT_PX;
    return f == 2 ? 1100.0 : 1650.0;
}

static int tile_max(int f) {
    int t = (int)(sqrt((double)vram_budget_mb() * 1048576.0 / out_px_bytes(f)) / f) & ~1;
    return t < 64 ? 64 : t;
}

static int tile_pick(int w, int h, int f) {
    int forced = env_int("OMNISERVE_NATIVE_ESRGAN_TILE", 0);
    if (forced < 0) return 0;
    if (forced > 0) return forced < 32 ? 32 : forced & ~1;
    int tmax = tile_max(f);
    if (!p_set_tile) return w <= tmax && h <= tmax ? 0 : tmax;
    if ((double)w * h * f * f * out_px_bytes(f) <= (double)vram_budget_mb() * 1048576.0) return 0;
    int side = w > h ? w : h;
    for (int n = 2; n <= 256; ++n) {
        int t = ((int)ceil(side / (1.0 + (n - 1) * 0.75)) + 1) & ~1;
        if (t <= tmax) return t < 32 ? 32 : t;
    }
    return tmax;
}

int oup_backend_lease_mb(int w, int h, oup_model m, int scale) {
    pthread_once(&g_once, lib_load);
    int f = net_factor(pick_net(m, scale));
    w += w & 1;
    h += h & 1;
    int t = tile_pick(w, h, f);
    double px = (t && (w > t || h > t)) ? (double)t * t : (double)w * h;
    return (int)(px * f * f * out_px_bytes(f) / 1048576.0) + env_int("OMNISERVE_NATIVE_ESRGAN_OVERHEAD_MB", 512);
}

static upscaler_ctx_t *ctx_get(int net) {
    if (g_ctx[net] || g_failed[net]) return g_ctx[net];
    const char *path = net_path(net);
    if (!path) return NULL;
    double t0 = now_ms();
    int threads = env_int("OMNISERVE_NATIVE_ESRGAN_THREADS", 4);
    g_tile[net] = tile_pick(1 << 20, 1 << 20, net_factor(net));
    g_ctx[net] = p_new_up(path, g_direct, threads, g_tile[net], NULL, NULL);
    if (!g_ctx[net]) {
        g_failed[net] = true;
        fprintf(stderr, "[upscale] load failed: %s\n", path);
        return NULL;
    }
    g_factor[net] = p_up_factor(g_ctx[net]);
    unsigned char px[64 * 64 * 3];
    memset(px, 128, sizeof px);
    sd_image_t *o = NULL;
    int n = 0;
    if (p_up(g_ctx[net], (sd_image_t){64, 64, 3, px}, (uint32_t)g_factor[net], &o, &n) && o) {
        free(o[0].data);
        free(o);
    }
    fprintf(stderr, "[upscale] loaded %s factor=%d tile=%d in %.0f ms (incl. warmup)\n", net_label[net],
            g_factor[net], g_tile[net], now_ms() - t0);
    return g_ctx[net];
}

static void *preload_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&g_lock);
    for (int net = 0; net < NET_COUNT; ++net)
        if (net_path(net)) (void)ctx_get(net);
    pthread_mutex_unlock(&g_lock);
    return NULL;
}

void oup_backend_preload(void) {
    pthread_once(&g_once, lib_load);
    if (!g_lib_ok) return;
    bool any = false;
    for (int net = 0; net < NET_COUNT; ++net) any |= net_path(net) != NULL;
    if (!any) return;
    pthread_t th;
    if (pthread_create(&th, NULL, preload_main, NULL) == 0) pthread_detach(th);
}

typedef struct {
    unsigned char *data;
    size_t len, cap;
    bool failed;
} sink;

static void sink_write(void *ctx, void *data, int size) {
    sink *s = ctx;
    if (s->failed || size <= 0) return;
    if (s->len + (size_t)size > s->cap) {
        size_t nc = s->cap ? s->cap : 1u << 20;
        while (nc < s->len + (size_t)size) nc *= 2;
        unsigned char *g = realloc(s->data, nc);
        if (!g) {
            s->failed = true;
            return;
        }
        s->data = g;
        s->cap = nc;
    }
    memcpy(s->data + s->len, data, (size_t)size);
    s->len += (size_t)size;
}

static unsigned char *decode(const unsigned char *b, size_t n, int *w, int *h, int *ch) {
    int c = 0;
    unsigned char *px = NULL;
    if (stbi_info_from_memory(b, (int)n, w, h, &c)) {
        int want = (c == 2 || c == 4) ? 4 : 3;
        px = stbi_load_from_memory(b, (int)n, w, h, &c, want);
        *ch = want;
    } else if (n > 12 && !memcmp(b, "RIFF", 4) && !memcmp(b + 8, "WEBP", 4) && p_webp_info &&
               p_webp_dec_rgba && p_webp_info(b, n, w, h) && *w > 0 && *h > 0) {
        size_t sz = (size_t)*w * *h * 4;
        px = malloc(sz);
        if (px && !p_webp_dec_rgba(b, n, px, sz, *w * 4)) {
            free(px);
            px = NULL;
        }
        *ch = 4;
    }
    if (px && *ch == 4) {
        size_t np = (size_t)*w * *h;
        bool opaque = true;
        for (size_t i = 0; i < np && opaque; ++i) opaque = px[i * 4 + 3] == 255;
        if (opaque) {
            for (size_t i = 0; i < np; ++i) memmove(px + i * 3, px + i * 4, 3);
            *ch = 3;
        }
    }
    return px;
}

static bool encode(const unsigned char *px, int w, int h, int ch, oup_format f, unsigned char **out, size_t *len) {
    int q = env_int("OMNISERVE_NATIVE_UPSCALE_QUALITY", 92);
    int threads = env_int("OMNISERVE_NATIVE_UPSCALE_ENCODE_THREADS", 4);
    if (f == OUP_PNG && p_oenc_png)
        return p_oenc_png(px, w, h, ch, env_int("OMNISERVE_NATIVE_UPSCALE_PNG_LEVEL", 1), threads, out, len) == 0;
    if (f == OUP_WEBP && p_oenc_webp) {
        unsigned char *rgba = (unsigned char *)px;
        if (ch == 3) {
            rgba = malloc((size_t)w * h * 4);
            if (!rgba) return false;
            for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
                memcpy(rgba + i * 4, px + i * 3, 3);
                rgba[i * 4 + 3] = 255;
            }
        }
        int r = p_oenc_webp(rgba, w, h, w * 4, (float)env_int("OMNISERVE_NATIVE_UPSCALE_WEBP_QUALITY", 94),
                            env_int("OMNISERVE_NATIVE_UPSCALE_WEBP_METHOD", 2), 1, 0, threads, out, len);
        if (rgba != px) free(rgba);
        return r == 0;
    }
    if (f == OUP_WEBP) {
        fn_webp_enc enc = ch == 4 ? p_webp_rgba : p_webp_rgb;
        if (!enc || !p_webp_free) return false;
        unsigned char *wb = NULL;
        size_t n = enc(px, w, h, w * ch, (float)q, &wb);
        if (!n || !wb) return false;
        *out = malloc(n);
        if (*out) memcpy(*out, wb, n);
        p_webp_free(wb);
        *len = n;
        return *out != NULL;
    }
    sink s = {0};
    int ok = f == OUP_JPEG ? stbi_write_jpg_to_func(sink_write, &s, w, h, ch, px, q)
                           : stbi_write_png_to_func(sink_write, &s, w, h, ch, px, w * ch);
    if (!ok || s.failed || !s.data) {
        free(s.data);
        return false;
    }
    *out = s.data;
    *len = s.len;
    return true;
}

static void set_err(char *err, size_t cap, const char *msg) {
    if (err && cap) snprintf(err, cap, "%s", msg);
}

oup_status oup_backend_run(const unsigned char *bytes, size_t len, const oup_req *req,
                           oup_result *out, char *err, size_t err_cap) {
    memset(out, 0, sizeof *out);
    pthread_once(&g_once, lib_load);
    int net = pick_net(req->model, req->scale);
    if (!g_lib_ok || !net_path(net) || g_failed[net]) {
        set_err(err, err_cap, "upscale model weights are not configured on this server");
        return OUP_ERR_UNAVAILABLE;
    }
    int w = 0, h = 0, ch = 0;
    unsigned char *src = decode(bytes, len, &w, &h, &ch);
    if (!src) {
        set_err(err, err_cap, "image_url is not a decodable PNG, JPEG or WebP");
        return OUP_ERR_INPUT;
    }
    if (oup_check_limits(w, h, req->scale, err, err_cap) == 413) {
        free(src);
        return OUP_ERR_TOO_LARGE;
    }
    int pw = w + (w & 1), ph = h + (h & 1);
    unsigned char *rgb = malloc((size_t)pw * ph * 3);
    if (!rgb) {
        free(src);
        return OUP_ERR_INTERNAL;
    }
    for (int y = 0; y < ph; ++y) {
        const unsigned char *row = src + (size_t)(y < h ? y : h - 1) * w * ch;
        unsigned char *d = rgb + (size_t)y * pw * 3;
        for (int x = 0; x < pw; ++x) memcpy(d + (size_t)x * 3, row + (size_t)(x < w ? x : w - 1) * ch, 3);
    }
    pthread_mutex_lock(&g_lock);
    upscaler_ctx_t *ctx = ctx_get(net);
    if (!ctx) {
        pthread_mutex_unlock(&g_lock);
        free(src);
        free(rgb);
        set_err(err, err_cap, "upscale model failed to load");
        return OUP_ERR_UNAVAILABLE;
    }
    int f = g_factor[net];
    int tile = tile_pick(pw, ph, f);
    if (p_set_tile) p_set_tile(ctx, tile);
    else tile = g_tile[net] && (pw > g_tile[net] || ph > g_tile[net]) ? g_tile[net] : 0;
    double t0 = now_ms();
    sd_image_t *res = NULL;
    int nres = 0;
    bool ok = p_up(ctx, (sd_image_t){(uint32_t)pw, (uint32_t)ph, 3, rgb}, (uint32_t)f, &res, &nres) &&
              res && res[0].data && (int)res[0].width == pw * f && (int)res[0].height == ph * f;
    double run_ms = now_ms() - t0;
    if (ok) {
        g_runs[net]++;
        g_last_ms[net] = run_ms;
    }
    pthread_mutex_unlock(&g_lock);
    free(rgb);
    if (!ok) {
        if (res) {
            free(res[0].data);
            free(res);
        }
        free(src);
        set_err(err, err_cap, "upscale failed");
        return OUP_ERR_INTERNAL;
    }
    unsigned char *up = res[0].data;
    int uw = pw * f, uh = ph * f;
    free(res);
    if (f == 4 && req->scale == 2) {
        unsigned char *half = malloc((size_t)(uw / 2) * (uh / 2) * 3);
        if (!half) {
            free(up);
            free(src);
            return OUP_ERR_INTERNAL;
        }
        oup_box_down2(up, uw, uh, 3, half);
        free(up);
        up = half;
        uw /= 2;
        uh /= 2;
    }
    int ow = w * req->scale, oh = h * req->scale;
    int och = ch == 4 && req->format != OUP_JPEG ? 4 : 3;
    unsigned char *img = malloc((size_t)ow * oh * och);
    if (!img) {
        free(up);
        free(src);
        return OUP_ERR_INTERNAL;
    }
    for (int y = 0; y < oh; ++y) {
        const unsigned char *s = up + (size_t)y * uw * 3;
        unsigned char *d = img + (size_t)y * ow * och;
        if (och == 3) memcpy(d, s, (size_t)ow * 3);
        else
            for (int x = 0; x < ow; ++x) memcpy(d + (size_t)x * 4, s + (size_t)x * 3, 3);
    }
    if (och == 4) oup_resize_plane_bilinear(src + 3, w, h, 4, img + 3, ow, oh, 4);
    if (ch == 4 && och == 3) {
        for (int y = 0; y < oh; ++y) {
            int sy = y / req->scale;
            for (int x = 0; x < ow; ++x) {
                int a = src[((size_t)sy * w + x / req->scale) * 4 + 3];
                unsigned char *p = img + ((size_t)y * ow + x) * 3;
                for (int c = 0; c < 3; ++c) p[c] = (unsigned char)((p[c] * a + 255 * (255 - a) + 127) / 255);
            }
        }
    }
    free(up);
    free(src);
    bool enc = encode(img, ow, oh, och, req->format, &out->data, &out->len);
    free(img);
    if (!enc) {
        set_err(err, err_cap, "image encoding failed");
        return OUP_ERR_INTERNAL;
    }
    out->width = ow;
    out->height = oh;
    out->scale = req->scale;
    out->model = oup_model_name(req->model);
    out->ms = run_ms;
    fprintf(stderr, "[upscale] %s %dx%d -> %dx%d net=%s tile=%d ch=%d run=%.0f ms\n", out->model, w, h, ow, oh,
            net_label[net], tile, ch, run_ms);
    return OUP_OK;
}

size_t oup_backend_status_json(char *out, size_t cap) {
    pthread_once(&g_once, lib_load);
    size_t n = 0;
    int r = snprintf(out, cap, "{\"lib\":%s,\"vram_budget_mb\":%d,\"nets\":{", g_lib_ok ? "true" : "false",
                     vram_budget_mb());
    if (r > 0) n = (size_t)r < cap ? (size_t)r : cap - 1;
    for (int net = 0; net < NET_COUNT && n < cap; ++net) {
        r = snprintf(out + n, cap - n, "%s\"%s\":{\"configured\":%s,\"loaded\":%s,\"tile\":%d,\"failed\":%s,\"runs\":%llu,\"last_ms\":%.0f}",
                     net ? "," : "", net_label[net], net_path(net) ? "true" : "false", g_ctx[net] ? "true" : "false", g_tile[net],
                     g_failed[net] ? "true" : "false", g_runs[net], g_last_ms[net]);
        if (r > 0) n += (size_t)r < cap - n ? (size_t)r : cap - n - 1;
    }
    if (n + 3 < cap) n += (size_t)snprintf(out + n, cap - n, "}}");
    return n;
}

#else

bool oup_backend_available(oup_model m, int scale) {
    (void)m;
    (void)scale;
    return false;
}

oup_status oup_backend_run(const unsigned char *bytes, size_t len, const oup_req *req,
                           oup_result *out, char *err, size_t err_cap) {
    (void)bytes;
    (void)len;
    (void)req;
    memset(out, 0, sizeof *out);
    if (err && err_cap) snprintf(err, err_cap, "upscaler not built (WITH_SD=OFF)");
    return OUP_ERR_UNAVAILABLE;
}

int oup_backend_lease_mb(int w, int h, oup_model m, int scale) {
    (void)w;
    (void)h;
    (void)m;
    (void)scale;
    return 0;
}

void oup_backend_preload(void) {}

size_t oup_backend_status_json(char *out, size_t cap) {
    int r = snprintf(out, cap, "{\"lib\":false}");
    return r > 0 ? (size_t)r : 0;
}

#endif

void oup_result_free(oup_result *res) {
    if (!res) return;
    free(res->data);
    res->data = NULL;
}
