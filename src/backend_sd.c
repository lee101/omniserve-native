#include "obackend.h"

#ifdef USE_SD

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "stable-diffusion.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 4096
#include "stb_image.h"

static signed char b64_lut[256];
static pthread_once_t b64_once = PTHREAD_ONCE_INIT;

static void b64_lut_init(void) {
    memset(b64_lut, -1, sizeof b64_lut);
    static const char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) b64_lut[(unsigned char)alphabet[i]] = (signed char)i;
}

typedef void (*fn_webp_free)(void *);
typedef size_t (*fn_webp_encode_rgb)(const unsigned char *, int, int, int, float,
                                     unsigned char **);
typedef size_t (*fn_webp_encode_rgba)(const unsigned char *, int, int, int, float,
                                      unsigned char **);
typedef int (*fn_webp_get_info)(const unsigned char *, size_t, int *, int *);
typedef unsigned char *(*fn_webp_decode_rgb_into)(const unsigned char *, size_t,
                                                  unsigned char *, size_t, int);

static fn_webp_free p_webp_free;
static fn_webp_encode_rgb p_webp_encode_rgb;
static fn_webp_encode_rgba p_webp_encode_rgba; /* Qwen Image 2.1 decodes RGBA (layered/transparent output) */
static fn_webp_get_info p_webp_get_info;
static fn_webp_decode_rgb_into p_webp_decode_rgb_into; /* reference/init images may arrive as WebP; stb_image cannot decode it */
static pthread_once_t webp_once = PTHREAD_ONCE_INIT;

static void webp_lib_load_once(void) {
    static const char *const candidates[] = {
        "libwebp.so.7", "libwebp.so.6", "libwebp.so", NULL,
    };
    for (int i = 0; candidates[i]; ++i) {
        void *lib = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!lib) continue;
        fn_webp_encode_rgb encode_rgb = (fn_webp_encode_rgb)dlsym(lib, "WebPEncodeRGB");
        fn_webp_free free_fn = (fn_webp_free)dlsym(lib, "WebPFree");
        if (!encode_rgb || !free_fn) {
            dlclose(lib);
            continue;
        }
        p_webp_encode_rgb = encode_rgb;
        p_webp_free = free_fn;
        p_webp_encode_rgba = (fn_webp_encode_rgba)dlsym(lib, "WebPEncodeRGBA");
        p_webp_get_info = (fn_webp_get_info)dlsym(lib, "WebPGetInfo");
        p_webp_decode_rgb_into = (fn_webp_decode_rgb_into)dlsym(lib, "WebPDecodeRGBInto");
        return;
    }
}

static void webp_lib_load(void) { pthread_once(&webp_once, webp_lib_load_once); }

/* Keyed hash so a client cannot craft two reference images that collide in the
 * result cache. Per-process random key, never persisted. */
static uint64_t g_hash_key[2];
static pthread_once_t hash_key_once = PTHREAD_ONCE_INIT;

static void hash_key_init(void) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f || fread(g_hash_key, sizeof g_hash_key, 1, f) != 1) {
        g_hash_key[0] = (uint64_t)time(NULL) * 0x9e3779b97f4a7c15ULL;
        g_hash_key[1] = (uint64_t)(uintptr_t)&g_hash_key ^ 0xd1b54a32d192ed03ULL;
    }
    if (f) fclose(f);
}

#define ROTL64(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))
#define SIPROUND do { \
    v0 += v1; v1 = ROTL64(v1, 13); v1 ^= v0; v0 = ROTL64(v0, 32); \
    v2 += v3; v3 = ROTL64(v3, 16); v3 ^= v2; \
    v0 += v3; v3 = ROTL64(v3, 21); v3 ^= v0; \
    v2 += v1; v1 = ROTL64(v1, 17); v1 ^= v2; v2 = ROTL64(v2, 32); \
} while (0)

static uint64_t keyed_hash(const void *data, size_t len) {
    pthread_once(&hash_key_once, hash_key_init);
    const unsigned char *in = data;
    uint64_t v0 = 0x736f6d6570736575ULL ^ g_hash_key[0];
    uint64_t v1 = 0x646f72616e646f6dULL ^ g_hash_key[1];
    uint64_t v2 = 0x6c7967656e657261ULL ^ g_hash_key[0];
    uint64_t v3 = 0x7465646279746573ULL ^ g_hash_key[1];
    uint64_t tail = (uint64_t)len << 56;
    const unsigned char *end = in + (len & ~(size_t)7);
    for (; in != end; in += 8) {
        uint64_t m;
        memcpy(&m, in, 8);
        v3 ^= m; SIPROUND; SIPROUND; v0 ^= m;
    }
    for (size_t i = 0; i < (len & 7); ++i) tail |= (uint64_t)in[i] << (8 * i);
    v3 ^= tail; SIPROUND; SIPROUND; v0 ^= tail;
    v2 ^= 0xff; SIPROUND; SIPROUND; SIPROUND; SIPROUND;
    return v0 ^ v1 ^ v2 ^ v3;
}

bool osd_prepare_image(oimg_req *req) {
    if (!req->image_base64) return true;
    const char *src = req->image_base64;
    size_t len = strlen(src);
    if (!len || len > (8u << 20) || len % 4) return false;
    pthread_once(&b64_once, b64_lut_init);
    unsigned char *bytes = malloc(len / 4 * 3);
    if (!bytes) return false;
    size_t used = 0;
    bool valid = true;
    size_t last = len - 4;
    for (size_t i = 0; i < last; i += 4) {
        int a = b64_lut[(unsigned char)src[i]];
        int b = b64_lut[(unsigned char)src[i + 1]];
        int c = b64_lut[(unsigned char)src[i + 2]];
        int d = b64_lut[(unsigned char)src[i + 3]];
        if ((a | b | c | d) < 0) {
            valid = false;
            break;
        }
        bytes[used++] = (unsigned char)((a << 2) | (b >> 4));
        bytes[used++] = (unsigned char)((b << 4) | (c >> 2));
        bytes[used++] = (unsigned char)((c << 6) | d);
    }
    if (valid) {
        int a = b64_lut[(unsigned char)src[last]];
        int b = b64_lut[(unsigned char)src[last + 1]];
        bool pad_c = src[last + 2] == '=';
        bool pad_d = src[last + 3] == '=';
        int c = pad_c ? 0 : b64_lut[(unsigned char)src[last + 2]];
        int d = pad_d ? 0 : b64_lut[(unsigned char)src[last + 3]];
        if (a < 0 || b < 0 || c < 0 || d < 0 || (pad_c && !pad_d) ||
            (pad_c && (b & 15)) || (pad_d && !pad_c && (c & 3))) {
            valid = false;
        } else {
            bytes[used++] = (unsigned char)((a << 2) | (b >> 4));
            if (!pad_c) bytes[used++] = (unsigned char)((b << 4) | (c >> 2));
            if (!pad_d) bytes[used++] = (unsigned char)((c << 6) | d);
        }
    }
    int width = 0, height = 0, channels = 0;
    if (valid && stbi_info_from_memory(bytes, (int)used, &width, &height, &channels) &&
        width > 0 && height > 0 && width <= 4096 && height <= 4096) {
        req->image_pixels = stbi_load_from_memory(bytes, (int)used,
            &req->image_width, &req->image_height, &channels, 3);
    } else if (valid && used > 12 && memcmp(bytes, "RIFF", 4) == 0 && memcmp(bytes + 8, "WEBP", 4) == 0) {
        webp_lib_load();
        int w = 0, h = 0;
        /* Dimensions are checked before decoding: WebPDecodeRGB would allocate up to
         * 16383x16383x3 first. Decoding straight into our own malloc also avoids a copy
         * (stbi_image_free is free(); WebP-owned buffers must go through WebPFree). */
        if (p_webp_get_info && p_webp_decode_rgb_into &&
            p_webp_get_info(bytes, used, &w, &h) &&
            w > 0 && h > 0 && w <= 4096 && h <= 4096) {
            size_t n = (size_t)w * (size_t)h * 3;
            unsigned char *rgb = malloc(n);
            if (rgb && p_webp_decode_rgb_into(bytes, used, rgb, n, w * 3)) {
                req->image_pixels = rgb;
                req->image_width = w;
                req->image_height = h;
            } else {
                free(rgb);
            }
        }
    }
    free(bytes);
    if (req->image_pixels && req->cache) {
        req->image_hash = keyed_hash(src, len);
        req->image_len = len;
        req->image_hash_valid = true;
    }
    return req->image_pixels != NULL;
}

typedef void (*fn_ctx_params_init)(sd_ctx_params_t *);
typedef sd_ctx_t *(*fn_new_sd_ctx)(const sd_ctx_params_t *);
typedef void (*fn_img_params_init)(sd_img_gen_params_t *);
typedef bool (*fn_generate_image)(sd_ctx_t *, const sd_img_gen_params_t *, sd_image_t **, int *);
typedef void (*fn_free_images)(sd_image_t *, int);
#if !OMNISERVE_SD_LATENT_API
/* Compatibility shim for upstream stable-diffusion.cpp: the symbols are looked up
 * with dlsym and stay NULL, so latent_api_ready() reports false and every
 * teleport request takes the full_generation_fallback path. */
typedef struct sd_latent_t sd_latent_t;
typedef struct {
    const sd_latent_t *resume_latent;
    int capture_step;
    sd_latent_t **captured_latent_out;
    bool cache_hit;
    int resume_step;
} sd_latent_replay_params_t;
#endif
typedef void (*fn_latent_params_init)(sd_latent_replay_params_t *);
typedef bool (*fn_generate_image_with_latent)(sd_ctx_t *, const sd_img_gen_params_t *,
                                              sd_latent_replay_params_t *, sd_image_t **, int *);
typedef void (*fn_free_latent)(sd_latent_t *);

static fn_ctx_params_init p_ctx_params_init;
static fn_new_sd_ctx p_new_sd_ctx;
static fn_img_params_init p_img_params_init;
static fn_generate_image p_generate_image;
static int (*p_str_to_sample_method)(const char *);
static int (*p_str_to_scheduler)(const char *);
static fn_free_images p_free_images;
static fn_latent_params_init p_latent_params_init;
static fn_generate_image_with_latent p_generate_image_with_latent;
static fn_free_latent p_free_latent;
static bool g_webp_enabled = true;
static float g_webp_quality = 85.0f;

static int sd_env_int(const char *name, int fallback, int minimum, int maximum);
static int g_sd_log_min = SD_LOG_WARN;

static void sd_log_to_stderr(enum sd_log_level_t level, const char *text, void *data) {
    (void)data;
    if ((int)level >= g_sd_log_min && text) {
        fputs(text, stderr);
    }
}

static bool sd_lib_load(void) {
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
        fprintf(stderr, "sd dlopen failed: %s\n", dlerror());
        return false;
    }
    p_ctx_params_init = (fn_ctx_params_init)dlsym(lib, "sd_ctx_params_init");
    p_new_sd_ctx = (fn_new_sd_ctx)dlsym(lib, "new_sd_ctx");
    p_img_params_init = (fn_img_params_init)dlsym(lib, "sd_img_gen_params_init");
    p_generate_image = (fn_generate_image)dlsym(lib, "generate_image");
    *(void **)&p_str_to_sample_method = dlsym(lib, "str_to_sample_method");
    *(void **)&p_str_to_scheduler = dlsym(lib, "str_to_scheduler");
    p_free_images = (fn_free_images)dlsym(lib, "free_sd_images");
    p_latent_params_init = (fn_latent_params_init)dlsym(lib, "sd_latent_replay_params_init");
    p_generate_image_with_latent = (fn_generate_image_with_latent)dlsym(lib, "generate_image_with_latent");
    p_free_latent = (fn_free_latent)dlsym(lib, "free_sd_latent");
    /* Without a callback sd.cpp/ggml drop their logs, including the CUDA error
     * text printed right before GGML_ABORT. */
    typedef void (*fn_set_log_callback)(sd_log_cb_t, void *);
    fn_set_log_callback set_log = (fn_set_log_callback)dlsym(lib, "sd_set_log_callback");
    if (set_log) {
        g_sd_log_min = sd_env_int("OMNISERVE_NATIVE_SD_LOG_LEVEL", SD_LOG_WARN, SD_LOG_DEBUG, SD_LOG_ERROR);
        set_log(sd_log_to_stderr, NULL);
    }
    return p_ctx_params_init && p_new_sd_ctx && p_img_params_init && p_generate_image && p_free_images;
}

static sd_ctx_t *g_sd;
static pthread_mutex_t g_sd_lock = PTHREAD_MUTEX_INITIALIZER;
/* g_sd_lock serializes generation and every cache insert/evict; g_cache_lock only
 * guards lookups against those. A cache_entry pointer found under g_sd_lock stays
 * valid for as long as g_sd_lock is held. */
static pthread_mutex_t g_cache_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_sd_name[256];
static bool g_reference_edit;

bool osd_reference_edit_ready(void) { return g_sd && g_reference_edit; }

typedef struct {
    char *prompt;
    char *negative_prompt;
    uint64_t image_hash;
    size_t image_len;
    float strength;
    int width;
    int height;
    int steps;
    float guidance_scale;
    int64_t seed;
    int resume_step;
    char *lora_key;
    sd_latent_t *latent;
    unsigned char *encoded_image;
    size_t encoded_image_len;
    bool encoded_image_is_webp;
    unsigned long long tick;
} latent_cache_entry;

static latent_cache_entry *g_latent_cache;
static int g_latent_cache_size;
static unsigned long long g_latent_cache_tick;
static size_t g_cache_bytes;

/* Nyquist notch: (delta - b) along x then y with b = [-1 6 -15 20 -15 6 -1]/64,
 * which removes the 2 px lattice the VAE leaves in fine texture and keeps DC
 * gain 1. Separable, so two 7-tap passes; edges clamp. */
static void notch_rgb(unsigned char *px, int w, int h, int ch) {
    static const int k[7] = {1, -6, 15, 44, 15, -6, 1};
    if (w < 7 || h < 7 || ch < 3) return;
    int *tmp = malloc((size_t)w * h * 3 * sizeof *tmp);
    if (!tmp) return;
    for (int y = 0; y < h; ++y) {
        const unsigned char *row = px + (size_t)y * w * ch;
        int *out = tmp + (size_t)y * w * 3;
        for (int x = 0; x < w; ++x) {
            int a0 = 0, a1 = 0, a2 = 0;
            for (int t = -3; t <= 3; ++t) {
                int xx = x + t < 0 ? 0 : (x + t >= w ? w - 1 : x + t);
                const unsigned char *p = row + (size_t)xx * ch;
                a0 += k[t + 3] * p[0]; a1 += k[t + 3] * p[1]; a2 += k[t + 3] * p[2];
            }
            out[x * 3] = a0; out[x * 3 + 1] = a1; out[x * 3 + 2] = a2;
        }
    }
    for (int y = 0; y < h; ++y) {
        unsigned char *row = px + (size_t)y * w * ch;
        const int *r[7];
        for (int t = -3; t <= 3; ++t) {
            int yy = y + t < 0 ? 0 : (y + t >= h ? h - 1 : y + t);
            r[t + 3] = tmp + (size_t)yy * w * 3;
        }
        for (int i = 0; i < w * 3; ++i) {
            int v = 0;
            for (int t = 0; t < 7; ++t) v += k[t] * r[t][i];
            v = (v + 2048) >> 12;
            row[(i / 3) * ch + i % 3] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    free(tmp);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static bool sd_env_flag(const char *name, bool fallback) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    static const char *const yes[] = {"1", "t", "true", "y", "yes", "on", NULL};
    static const char *const no[] = {"0", "f", "false", "n", "no", "off", NULL};
    for (int i = 0; yes[i]; ++i) if (strcasecmp(value, yes[i]) == 0) return true;
    for (int i = 0; no[i]; ++i) if (strcasecmp(value, no[i]) == 0) return false;
    fprintf(stderr, "sd: ignoring %s=%s (not a boolean), using %s\n", name, value,
            fallback ? "true" : "false");
    return fallback;
}

static int sd_env_int(const char *name, int fallback, int minimum, int maximum) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (!end || end == value || *end || parsed < minimum || parsed > maximum) {
        fprintf(stderr, "sd: ignoring %s=%s (want integer %d..%d), using %d\n",
                name, value, minimum, maximum, fallback);
        return fallback;
    }
    return (int)parsed;
}

static float sd_env_float(const char *name, float fallback, float minimum, float maximum) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    float parsed = strtof(value, &end);
    if (!end || end == value || *end || !isfinite(parsed) ||
        parsed < minimum || parsed > maximum) {
        fprintf(stderr, "sd: ignoring %s=%s (want number %g..%g), using %g\n",
                name, value, (double)minimum, (double)maximum, (double)fallback);
        return fallback;
    }
    return parsed;
}

/* Per-request tunables, parsed once in osd_init so the hot path never touches the
 * environment and a bad value is reported once at startup. */
static struct {
    float zero_guidance;
    int vae_tiling; /* -1: keep the library default */
    int tile_x, tile_y;
    float tile_overlap;
    float flow_shift;
    const char *cache_mode; /* NULL: off; otherwise a static string */
    float easycache_threshold;
    float cache_start, cache_end;
    int taylor_derivatives, taylor_skip;
    int spectrum_warmup;
    float spectrum_stop, residual_diff;
    int teleport_start; /* -1: steps - 1 */
    size_t cache_bytes_max;
    int turbo_n; /* >0: distilled few-step schedule, nodes in turbo_nodes */
    float turbo_nodes[16];
    char default_lora[1024]; /* applied to requests that carry no LoRA of their own */
    float default_lora_scale;
    bool notch;
    float hq_threshold; /* EasyCache threshold for non-turbo text-to-image; 0 = use the global one */
} g_cfg;

/* Distilled few-step Qwen-Image (Viggle turbo) schedule: raw nodes pushed through the
 * pipeline's resolution-dependent exponential time shift, then a terminal 0. */
static void turbo_sigmas(int width, int height, float *out) {
    double seq = (double)(width / 16) * (double)(height / 16);
    double mu = 0.5 + (seq - 256.0) * (0.9 - 0.5) / (8192.0 - 256.0);
    double e = exp(mu);
    for (int i = 0; i < g_cfg.turbo_n; ++i) {
        double t = (double)g_cfg.turbo_nodes[i];
        out[i] = (float)(e / (e + (1.0 / t - 1.0)));
    }
    out[g_cfg.turbo_n] = 0.0f;
}

static void turbo_cfg_load(void) {
    const char *lora = getenv("OMNISERVE_NATIVE_SD_DEFAULT_LORA");
    g_cfg.default_lora[0] = 0;
    g_cfg.default_lora_scale = sd_env_float("OMNISERVE_NATIVE_SD_DEFAULT_LORA_SCALE", 1.0f, -4.0f, 4.0f);
    if (lora && lora[0] == '/' && strlen(lora) < sizeof g_cfg.default_lora) {
        strcpy(g_cfg.default_lora, lora);
        fprintf(stderr, "sd: default LoRA %s scale=%.2f\n", lora, (double)g_cfg.default_lora_scale);
    }
    const char *spec = getenv("OMNISERVE_NATIVE_SD_TURBO_NODES");
    g_cfg.turbo_n = 0;
    if (!spec || !spec[0]) return;
    float nodes[16];
    int n = 0;
    const char *p = spec;
    while (*p && n < 16) {
        char *end = NULL;
        float v = strtof(p, &end);
        if (end == p || !isfinite(v) || v <= 0.0f || v > 1.0f) {
            fprintf(stderr, "sd: ignoring OMNISERVE_NATIVE_SD_TURBO_NODES=%s\n", spec);
            return;
        }
        nodes[n++] = v;
        p = end;
        if (*p == ',') ++p;
    }
    memcpy(g_cfg.turbo_nodes, nodes, sizeof nodes);
    g_cfg.turbo_n = n;
    fprintf(stderr, "sd: turbo schedule enabled, %d steps\n", n);
}

static void sd_cfg_load(void) {
    turbo_cfg_load();
    g_cfg.zero_guidance = sd_env_float("OMNISERVE_NATIVE_SD_ZERO_GUIDANCE", 1.0f, 0.0f, 30.0f);
    const char *tiling = getenv("OMNISERVE_NATIVE_SD_VAE_TILING");
    g_cfg.vae_tiling = tiling && tiling[0] ? (int)sd_env_flag("OMNISERVE_NATIVE_SD_VAE_TILING", false) : -1;
    g_cfg.tile_x = sd_env_int("OMNISERVE_NATIVE_SD_VAE_TILE_X", 32, 32, 4096);
    g_cfg.tile_y = sd_env_int("OMNISERVE_NATIVE_SD_VAE_TILE_Y", 32, 32, 4096);
    g_cfg.tile_overlap = sd_env_float("OMNISERVE_NATIVE_SD_VAE_TILE_OVERLAP", 0.5f, 0.0f, 0.95f);
    g_cfg.flow_shift = sd_env_float("OMNISERVE_NATIVE_SD_FLOW_SHIFT", 3.0f, 0.1f, 20.0f);
    g_cfg.easycache_threshold = sd_env_float("OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD", 0.0f, 0.0f, 1.0f);
    g_cfg.cache_start = sd_env_float("OMNISERVE_NATIVE_SD_CACHE_START", 0.15f, 0.0f, 1.0f);
    g_cfg.cache_end = sd_env_float("OMNISERVE_NATIVE_SD_CACHE_END", 0.95f, 0.0f, 1.0f);
    g_cfg.taylor_derivatives = sd_env_int("OMNISERVE_NATIVE_SD_TAYLORSEER_DERIVATIVES", 1, 1, 4);
    g_cfg.taylor_skip = sd_env_int("OMNISERVE_NATIVE_SD_TAYLORSEER_SKIP_INTERVAL", 2, 1, 8);
    g_cfg.spectrum_warmup = sd_env_int("OMNISERVE_NATIVE_SD_SPECTRUM_WARMUP", 4, 1, 64);
    g_cfg.spectrum_stop = sd_env_float("OMNISERVE_NATIVE_SD_SPECTRUM_STOP", 0.9f, 0.0f, 1.0f);
    g_cfg.residual_diff = sd_env_float("OMNISERVE_NATIVE_SD_CACHE_RESIDUAL_DIFF", 0.08f, 0.0f, 10.0f);
    g_cfg.notch = sd_env_flag("OMNISERVE_NATIVE_SD_NOTCH", false);
    g_cfg.hq_threshold = sd_env_float("OMNISERVE_NATIVE_SD_HQ_EASYCACHE_THRESHOLD", 0.0f, 0.0f, 1.0f);
    g_cfg.teleport_start = sd_env_int("OMNISERVE_NATIVE_SD_TELEPORT_START_STEP", -1, 1, 99);
    g_cfg.cache_bytes_max = (size_t)sd_env_int("OMNISERVE_NATIVE_SD_CACHE_MAX_MB", 1024, 0, 1 << 20) << 20;
    /* OMNISERVE_NATIVE_SD_CACHE_MODE selects the stable-diffusion.cpp denoiser
     * cache: easycache | taylorseer | spectrum | cache-dit | dbcache | ucache.
     * The legacy OMNISERVE_NATIVE_SD_EASYCACHE_THRESHOLD alone still means easycache. */
    static const char *const modes[] = {"easycache", "ucache", "taylorseer", "spectrum",
                                        "cache-dit", "dbcache", NULL};
    const char *mode = getenv("OMNISERVE_NATIVE_SD_CACHE_MODE");
    g_cfg.cache_mode = NULL;
    if ((!mode || !mode[0]) && g_cfg.easycache_threshold > 0.0f) mode = "easycache";
    if (mode && mode[0] && strcmp(mode, "off") != 0 && strcmp(mode, "none") != 0) {
        for (int i = 0; modes[i]; ++i) if (strcmp(mode, modes[i]) == 0) g_cfg.cache_mode = modes[i];
        if (!g_cfg.cache_mode) fprintf(stderr, "sd: unknown OMNISERVE_NATIVE_SD_CACHE_MODE=%s, denoiser cache off\n", mode);
    }
}

static bool latent_api_ready(void) {
    return p_latent_params_init && p_generate_image_with_latent && p_free_latent &&
           g_latent_cache && g_latent_cache_size > 0;
}

/* Everything in a request that selects a cache entry and is not a plain scalar.
 * Built once per request: the LoRA key is a heap string and the image hash is
 * O(image size) when it was not memoized by osd_prepare_image. */
typedef struct {
    char *lora;
    uint64_t image_hash;
    size_t image_len;
} cache_key;

static char *lora_cache_key(const oimg_req *req) {
    size_t needed = 1;
    for (size_t i = 0; i < req->lora_count; i++) {
        const char *path = req->loras[i].path ? req->loras[i].path : "";
        needed += strlen(path) + 64;
    }
    char *key = malloc(needed);
    if (!key) return NULL;
    key[0] = '\0';
    size_t offset = 0;
    for (size_t i = 0; i < req->lora_count; i++) {
        const char *path = req->loras[i].path ? req->loras[i].path : "";
        int written = snprintf(key + offset, needed - offset, "%s|%.9g;",
                               path, (double)req->loras[i].scale);
        if (written < 0 || (size_t)written >= needed - offset) {
            free(key);
            return NULL;
        }
        offset += (size_t)written;
    }
    return key;
}

static bool cache_key_make(const oimg_req *req, cache_key *key) {
    key->lora = lora_cache_key(req);
    if (req->image_hash_valid) {
        key->image_len = req->image_len;
        key->image_hash = req->image_hash;
    } else {
        key->image_len = req->image_base64 ? strlen(req->image_base64) : 0;
        key->image_hash = key->image_len ? keyed_hash(req->image_base64, key->image_len) : 0;
    }
    return key->lora != NULL;
}

static void cache_key_free(cache_key *key) {
    free(key->lora);
    key->lora = NULL;
}

static bool cache_key_equal(const latent_cache_entry *entry, const oimg_req *req,
                            int resume_step, const cache_key *key) {
    const char *negative = req->negative_prompt ? req->negative_prompt : "";
    return entry->prompt && entry->width == req->width && entry->height == req->height &&
           entry->steps == req->steps && entry->guidance_scale == req->guidance_scale &&
           entry->seed == req->seed && entry->resume_step == resume_step &&
           entry->image_len == key->image_len && entry->image_hash == key->image_hash &&
           entry->strength == req->strength &&
           strcmp(entry->lora_key ? entry->lora_key : "", key->lora) == 0 &&
           strcmp(entry->prompt, req->prompt) == 0 &&
           strcmp(entry->negative_prompt, negative) == 0;
}

static latent_cache_entry *cache_find(const oimg_req *req, int resume_step, const cache_key *key) {
    for (int i = 0; i < g_latent_cache_size; i++) {
        if (cache_key_equal(&g_latent_cache[i], req, resume_step, key)) {
            g_latent_cache[i].tick = ++g_latent_cache_tick;
            return &g_latent_cache[i];
        }
    }
    return NULL;
}

static void cache_entry_clear(latent_cache_entry *entry) {
    if (entry->latent && p_free_latent) p_free_latent(entry->latent);
    g_cache_bytes -= entry->encoded_image_len;
    free(entry->prompt);
    free(entry->negative_prompt);
    free(entry->lora_key);
    free(entry->encoded_image);
    memset(entry, 0, sizeof *entry);
}

/* Callers hold g_sd_lock and g_cache_lock. An entry already holding this key is
 * reused, so two identical concurrent requests do not occupy two slots. */
static latent_cache_entry *cache_insert(const oimg_req *req, int resume_step,
                                        const cache_key *key, sd_latent_t *latent) {
    latent_cache_entry *slot = NULL;
    for (int i = 0; i < g_latent_cache_size; i++) {
        if (cache_key_equal(&g_latent_cache[i], req, resume_step, key)) {
            slot = &g_latent_cache[i];
            if (!latent) {
                slot->tick = ++g_latent_cache_tick;
                return slot;
            }
            break;
        }
    }
    if (!slot) {
        for (int i = 0; i < g_latent_cache_size; i++) {
            if (!g_latent_cache[i].prompt) {
                slot = &g_latent_cache[i];
                break;
            }
            if (!slot || g_latent_cache[i].tick < slot->tick) slot = &g_latent_cache[i];
        }
    }
    if (!slot) return NULL;
    char *prompt = strdup(req->prompt);
    char *negative = strdup(req->negative_prompt ? req->negative_prompt : "");
    char *lora_key = strdup(key->lora);
    if (!prompt || !negative || !lora_key) {
        free(prompt);
        free(negative);
        free(lora_key);
        return NULL;
    }
    cache_entry_clear(slot);
    slot->prompt = prompt;
    slot->negative_prompt = negative;
    slot->image_hash = key->image_hash;
    slot->image_len = key->image_len;
    slot->strength = req->strength;
    slot->width = req->width;
    slot->height = req->height;
    slot->steps = req->steps;
    slot->guidance_scale = req->guidance_scale;
    slot->seed = req->seed;
    slot->resume_step = resume_step;
    slot->lora_key = lora_key;
    slot->latent = latent;
    slot->tick = ++g_latent_cache_tick;
    return slot;
}

static bool cache_copy_encoded_result(const latent_cache_entry *entry, oimg_result *out) {
    if (!entry->encoded_image || !entry->encoded_image_len) return false;
    unsigned char **images = calloc(1, sizeof *images);
    size_t *lengths = calloc(1, sizeof *lengths);
    unsigned char *image = malloc(entry->encoded_image_len);
    if (!images || !lengths || !image) {
        free(images);
        free(lengths);
        free(image);
        return false;
    }
    memcpy(image, entry->encoded_image, entry->encoded_image_len);
    images[0] = image;
    lengths[0] = entry->encoded_image_len;
    out->images = images;
    out->image_lens = lengths;
    out->image_count = 1;
    out->png = image;
    out->png_len = entry->encoded_image_len;
    out->format = entry->encoded_image_is_webp ? "webp" : "png";
    out->images_malloc_owned = true;
    return true;
}

bool osd_try_cached_result(const oimg_req *req, oimg_result *out) {
    if (!g_sd || !req->cache || req->teleport || req->seed < 0 || req->batch_count > 1) return false;
    memset(out, 0, sizeof *out);
    double started = now_ms();
    cache_key key;
    if (!cache_key_make(req, &key)) return false;
    pthread_mutex_lock(&g_cache_lock);
    latent_cache_entry *entry = cache_find(req, 0, &key);
    bool found = entry && cache_copy_encoded_result(entry, out);
    pthread_mutex_unlock(&g_cache_lock);
    cache_key_free(&key);
    if (found) {
        out->cache_requested = out->cache_hit = true;
        out->denoiser_cache_threshold = g_cfg.easycache_threshold;
        out->elapsed_ms = now_ms() - started;
    }
    return found;
}

/* Callers hold g_sd_lock. Evicts least recently used encoded images, never
 * `keep`, until the new image fits the byte budget. */
static void cache_store_encoded_result(latent_cache_entry *entry, const oimg_result *out) {
    if (!entry || out->image_count != 1 || !out->images || !out->image_lens ||
        !out->images[0] || !out->image_lens[0] || out->image_lens[0] > (64u << 20) ||
        out->image_lens[0] > g_cfg.cache_bytes_max) return;
    size_t len = out->image_lens[0];
    unsigned char *copy = malloc(len);
    if (!copy) return;
    memcpy(copy, out->images[0], len);
    pthread_mutex_lock(&g_cache_lock);
    g_cache_bytes -= entry->encoded_image_len;
    free(entry->encoded_image);
    entry->encoded_image = NULL;
    entry->encoded_image_len = 0;
    while (g_cache_bytes + len > g_cfg.cache_bytes_max) {
        latent_cache_entry *victim = NULL;
        for (int i = 0; i < g_latent_cache_size; i++) {
            latent_cache_entry *e = &g_latent_cache[i];
            if (e == entry || !e->encoded_image) continue;
            if (!victim || e->tick < victim->tick) victim = e;
        }
        if (!victim) break;
        cache_entry_clear(victim);
    }
    entry->encoded_image = copy;
    entry->encoded_image_len = len;
    g_cache_bytes += len;
    entry->encoded_image_is_webp = out->format && strcmp(out->format, "webp") == 0;
    pthread_mutex_unlock(&g_cache_lock);
}

bool osd_init(const char *model_path) {
    if (g_sd) return true;
    if (!sd_lib_load()) return false;
    sd_cfg_load();
    sd_ctx_params_t params;
    p_ctx_params_init(&params);
    const char *diffusion = getenv("OMNISERVE_NATIVE_SD_DIFFUSION_MODEL");
    const char *vae = getenv("OMNISERVE_NATIVE_SD_VAE");
    const char *llm = getenv("OMNISERVE_NATIVE_SD_LLM");
    const char *llm_vision = getenv("OMNISERVE_NATIVE_SD_LLM_VISION");
    const char *taesd = getenv("OMNISERVE_NATIVE_SD_TAESD");
    const char *max_vram = getenv("OMNISERVE_NATIVE_SD_MAX_VRAM");
    const char *backend = getenv("OMNISERVE_NATIVE_SD_BACKEND");
    const char *params_backend = getenv("OMNISERVE_NATIVE_SD_PARAMS_BACKEND");
    if (diffusion && diffusion[0]) {
        params.diffusion_model_path = diffusion;
    } else {
        params.model_path = model_path;
    }
    if (vae && vae[0]) params.vae_path = vae;
    if (llm && llm[0]) params.llm_path = llm;
    if (llm_vision && llm_vision[0]) params.llm_vision_path = llm_vision;
    if (taesd && taesd[0]) params.taesd_path = taesd;
    if (max_vram && max_vram[0]) params.max_vram = max_vram;
    if (backend && backend[0]) params.backend = backend;
    if (params_backend && params_backend[0]) params.params_backend = params_backend;
    params.n_threads = sd_env_int("OMNISERVE_NATIVE_SD_THREADS", -1, 1, 256);
    params.flash_attn = sd_env_flag("OMNISERVE_NATIVE_SD_FLASH_ATTN", true);
    params.diffusion_flash_attn = sd_env_flag("OMNISERVE_NATIVE_SD_DIFFUSION_FLASH_ATTN", true);
#if OMNISERVE_SD_STREAM_LAYERS
    params.stream_layers = sd_env_flag("OMNISERVE_NATIVE_SD_STREAM_LAYERS", false);
#endif
    params.enable_mmap = sd_env_flag("OMNISERVE_NATIVE_SD_MMAP", true);
    params.eager_load = sd_env_flag("OMNISERVE_NATIVE_SD_EAGER_LOAD", false);
    params.auto_fit = sd_env_flag("OMNISERVE_NATIVE_SD_AUTO_FIT", false);
    const char *model_args = getenv("OMNISERVE_NATIVE_SD_MODEL_ARGS");
    if (model_args && model_args[0]) params.model_args = model_args;
    g_sd = p_new_sd_ctx(&params);
    if (!g_sd) return false;
    g_reference_edit = sd_env_flag("OMNISERVE_NATIVE_SD_REFERENCE_EDIT", false);
    const char *named_path = diffusion && diffusion[0] ? diffusion : model_path;
    if (!named_path) named_path = "";
    const char *slash = strrchr(named_path, '/');
    snprintf(g_sd_name, sizeof g_sd_name, "%s", slash ? slash + 1 : named_path);
    char *dot = strrchr(g_sd_name, '.');
    if (dot) *dot = 0;
    g_latent_cache_size = sd_env_int("OMNISERVE_NATIVE_SD_TELEPORT_CACHE_SIZE", 64, 0, 256);
    if (g_latent_cache_size > 0) {
        g_latent_cache = calloc((size_t)g_latent_cache_size, sizeof *g_latent_cache);
        if (!g_latent_cache) g_latent_cache_size = 0;
    }
    const char *format = getenv("OMNISERVE_NATIVE_SD_IMAGE_FORMAT");
    g_webp_enabled = !format || !format[0] || strcasecmp(format, "png") != 0;
    g_webp_quality = sd_env_float("OMNISERVE_NATIVE_SD_WEBP_QUALITY", 85.0f, 1.0f, 100.0f);
    if (g_webp_enabled) webp_lib_load();
    return true;
}

bool osd_ready(void) { return g_sd != NULL; }
const char *osd_model_name(void) { return g_sd_name[0] ? g_sd_name : "none"; }

typedef struct {
    unsigned char *data;
    size_t len;
    size_t cap;
    bool failed;
} png_sink;

static void png_write(void *ctx, void *data, int size) {
    png_sink *sink = ctx;
    if (sink->failed || size <= 0) return;
    size_t added = (size_t)size;
    if (added > SIZE_MAX - sink->len) {
        sink->failed = true;
        return;
    }
    size_t needed = sink->len + added;
    if (needed > sink->cap) {
        size_t next = sink->cap ? sink->cap : 64u << 10;
        while (next < needed) {
            if (next > SIZE_MAX / 2) {
                sink->failed = true;
                return;
            }
            next *= 2;
        }
        unsigned char *grown = realloc(sink->data, next);
        if (!grown) {
            sink->failed = true;
            return;
        }
        sink->data = grown;
        sink->cap = next;
    }
    memcpy(sink->data + sink->len, data, added);
    sink->len += added;
}

bool osd_generate(const oimg_req *req, oimg_result *out) {
    if (!g_sd) return false;
    memset(out, 0, sizeof *out);

    sd_img_gen_params_t params;
    p_img_params_init(&params);
    params.prompt = req->prompt;
    params.negative_prompt = req->negative_prompt ? req->negative_prompt : "";
    params.width = req->width > 0 ? req->width : 768;
    params.height = req->height > 0 ? req->height : 768;
    params.sample_params.sample_steps = req->steps > 0 ? req->steps : 4;
    /* stable-diffusion.cpp treats txt_cfg 0 as unconditioned mode (prompt
     * ignored). Distilled Flux/Z-Image pipelines want cfg 1.0, so an omitted or
     * zero guidance maps there unless OMNISERVE_NATIVE_SD_ZERO_GUIDANCE
     * overrides it. */
    params.sample_params.guidance.txt_cfg = req->guidance_scale == 0.0f
        ? g_cfg.zero_guidance
        : req->guidance_scale;
    if (g_cfg.vae_tiling >= 0) params.vae_tiling_params.enabled = g_cfg.vae_tiling;
    if (params.vae_tiling_params.enabled) {
        params.vae_tiling_params.tile_size_x = g_cfg.tile_x;
        params.vae_tiling_params.tile_size_y = g_cfg.tile_y;
        params.vae_tiling_params.target_overlap = g_cfg.tile_overlap;
    }
    /* Turbo only for plain text-to-image: reference edits and requests that bring their
     * own LoRA keep the base schedule, so edit quality is untouched. */
    bool turbo = g_cfg.turbo_n > 0 && !req->image_pixels && !req->lora_count && req->turbo != 2;
    float turbo_sig[17];
    if (turbo) {
        turbo_sigmas(params.width, params.height, turbo_sig);
        params.sample_params.sample_steps = g_cfg.turbo_n;
        params.sample_params.custom_sigmas = turbo_sig;
        params.sample_params.custom_sigmas_count = g_cfg.turbo_n + 1;
        params.sample_params.guidance.txt_cfg = 1.0f;
    }
    if (req->sampler[0] && p_str_to_sample_method) {
        int method = p_str_to_sample_method(req->sampler);
        if (method >= 0 && method < SAMPLE_METHOD_COUNT) params.sample_params.sample_method = (enum sample_method_t)method;
    }
    if (req->scheduler[0] && p_str_to_scheduler) {
        int scheduler = p_str_to_scheduler(req->scheduler);
        if (scheduler >= 0 && scheduler < SCHEDULER_COUNT) params.sample_params.scheduler = (enum scheduler_t)scheduler;
    }
    if (req->flow_shift > 0.0f) params.sample_params.flow_shift = req->flow_shift;
    if (req->extra_args[0]) params.sample_params.extra_sample_args = req->extra_args;
    params.seed = req->seed;
    sd_image_t source_image = {0};
    if (req->image_pixels) {
        source_image = (sd_image_t){
            .width = (uint32_t)req->image_width,
            .height = (uint32_t)req->image_height,
            .channel = 3, .data = req->image_pixels,
        };
        if (g_reference_edit) {
            params.ref_images = &source_image;
            params.ref_images_count = 1;
            params.sample_params.flow_shift = g_cfg.flow_shift;
        } else {
            params.init_image = source_image;
            params.strength = req->strength;
        }
    }
    params.batch_count = req->batch_count > 0 ? req->batch_count : 1;
    sd_lora_t *loras = NULL;
    if (req->lora_count) {
        loras = calloc(req->lora_count, sizeof *loras);
        if (!loras) return false;
        for (size_t i = 0; i < req->lora_count; ++i) {
            loras[i].path = req->loras[i].path;
            loras[i].multiplier = req->loras[i].scale;
        }
        params.loras = loras;
        params.lora_count = (uint32_t)req->lora_count;
    }
    sd_lora_t default_lora = {0};
    if (turbo && g_cfg.default_lora[0]) {
        default_lora.path = g_cfg.default_lora;
        default_lora.multiplier = g_cfg.default_lora_scale;
        params.loras = &default_lora;
        params.lora_count = 1;
    }

    out->teleport_requested = req->teleport;
    out->cache_requested = req->cache;
    /* Static per-process setting: result-cache entries never cross profiles.
     * Latent replay requires a dense trajectory, so it cannot use EasyCache. */
    bool overridden = req->cache_threshold > 0.0f || req->cache_end > 0.0f || req->cache_off || req->notch ||
                      req->flow_shift > 0.0f || req->sampler[0] || req->scheduler[0] || req->extra_args[0] ||
                      req->turbo;
    if (!req->teleport && g_cfg.cache_mode && !req->cache_off && !turbo) {
        const char *mode = g_cfg.cache_mode;
        float easycache_threshold = g_cfg.easycache_threshold;
        if (!req->image_pixels && g_cfg.hq_threshold > 0.0f) easycache_threshold = g_cfg.hq_threshold;
        if (req->cache_threshold > 0.0f) easycache_threshold = req->cache_threshold;
        params.cache.start_percent = g_cfg.cache_start;
        params.cache.end_percent = req->cache_end > 0.0f ? req->cache_end : g_cfg.cache_end;
        if (strcmp(mode, "easycache") == 0 || strcmp(mode, "ucache") == 0) {
            params.cache.mode = strcmp(mode, "ucache") == 0 ? SD_CACHE_UCACHE : SD_CACHE_EASYCACHE;
            params.cache.reuse_threshold = easycache_threshold > 0.0f ? easycache_threshold : 0.2f;
            out->denoiser_cache_threshold = params.cache.reuse_threshold;
        } else if (strcmp(mode, "taylorseer") == 0) {
            params.cache.mode = SD_CACHE_TAYLORSEER;
            params.cache.taylorseer_n_derivatives = g_cfg.taylor_derivatives;
            params.cache.taylorseer_skip_interval = g_cfg.taylor_skip;
            out->denoiser_cache_threshold = (float)params.cache.taylorseer_skip_interval;
        } else if (strcmp(mode, "spectrum") == 0) {
            params.cache.mode = SD_CACHE_SPECTRUM;
            params.cache.spectrum_warmup_steps = g_cfg.spectrum_warmup;
            params.cache.spectrum_stop_percent = g_cfg.spectrum_stop;
            out->denoiser_cache_threshold = params.cache.spectrum_stop_percent;
        } else {
            params.cache.mode = strcmp(mode, "dbcache") == 0 ? SD_CACHE_DBCACHE : SD_CACHE_CACHE_DIT;
            params.cache.residual_diff_threshold = g_cfg.residual_diff;
            out->denoiser_cache_threshold = params.cache.residual_diff_threshold;
        }
        out->denoiser_cache_mode = mode;
    }
    out->teleport_capture_step = -1;
    out->teleport_resume_step = 0;

    bool result_cache = req->cache && !req->teleport && !overridden && req->seed >= 0 && params.batch_count == 1;
    bool teleport_cache = req->teleport && !req->image_pixels && params.batch_count == 1 &&
                          req->steps > 1 && latent_api_ready();
    cache_key key = {0};
    if ((result_cache || teleport_cache) && !cache_key_make(req, &key)) {
        result_cache = teleport_cache = false;
    }
    pthread_mutex_lock(&g_sd_lock);
    double started = now_ms();
    sd_image_t *images = NULL;
    int image_count = 0;
    int cache_resume_step = 0;
    bool ok = false;
    if (result_cache) {
        pthread_mutex_lock(&g_cache_lock);
        latent_cache_entry *cached = cache_find(req, 0, &key);
        bool hit = cached && cache_copy_encoded_result(cached, out);
        pthread_mutex_unlock(&g_cache_lock);
        if (hit) {
            out->cache_hit = true;
            out->elapsed_ms = now_ms() - started;
            pthread_mutex_unlock(&g_sd_lock);
            cache_key_free(&key);
            free(loras);
            return true;
        }
    }
    if (teleport_cache) {
        int default_resume = g_cfg.teleport_start > 0 ? g_cfg.teleport_start : req->steps - 1;
        int resume_step = req->teleport_start_step > 0
            ? req->teleport_start_step : default_resume;
        if (resume_step >= req->steps) resume_step = req->steps - 1;
        cache_resume_step = resume_step;
        pthread_mutex_lock(&g_cache_lock);
        latent_cache_entry *cached = cache_find(req, resume_step, &key);
        bool hit = cached && cache_copy_encoded_result(cached, out);
        pthread_mutex_unlock(&g_cache_lock);
        if (hit) {
            out->teleport_used = true;
            out->teleport_cache_hit = true;
            out->teleport_result_cache_hit = true;
            out->teleport_capture_step = resume_step - 1;
            out->teleport_resume_step = resume_step;
            out->elapsed_ms = now_ms() - started;
            pthread_mutex_unlock(&g_sd_lock);
            cache_key_free(&key);
            free(loras);
            return true;
        }
        sd_latent_replay_params_t replay;
        p_latent_params_init(&replay);
        sd_latent_t *captured = NULL;
        if (cached) {
            replay.resume_latent = cached->latent;
        } else {
            replay.capture_step = resume_step - 1;
            replay.captured_latent_out = &captured;
        }
        ok = p_generate_image_with_latent(g_sd, &params, &replay, &images, &image_count);
        if (ok) {
            out->teleport_used = true;
            out->teleport_cache_hit = replay.cache_hit;
            out->teleport_capture_step = resume_step - 1;
            out->teleport_resume_step = replay.resume_step;
            if (captured) {
                pthread_mutex_lock(&g_cache_lock);
                bool inserted = cache_insert(req, resume_step, &key, captured) != NULL;
                pthread_mutex_unlock(&g_cache_lock);
                if (!inserted) {
                    fprintf(stderr, "diffusion teleport latent cache insert failed\n");
                    p_free_latent(captured);
                }
            } else if (!cached) {
                fprintf(stderr, "diffusion teleport completed without returning a captured latent\n");
            }
        } else {
            if (captured) p_free_latent(captured);
            if (cached) {
                pthread_mutex_lock(&g_cache_lock);
                cache_entry_clear(cached);
                pthread_mutex_unlock(&g_cache_lock);
            }
        }
    }
    if (!ok) {
        if (images) p_free_images(images, image_count);
        images = NULL;
        image_count = 0;
        ok = p_generate_image(g_sd, &params, &images, &image_count);
    }
    out->elapsed_ms = now_ms() - started;
    pthread_mutex_unlock(&g_sd_lock);
    free(loras);

    if (!ok || image_count <= 0 || !images) {
        if (images) p_free_images(images, image_count);
        cache_key_free(&key);
        return false;
    }
    out->images = calloc((size_t)image_count, sizeof *out->images);
    out->image_lens = calloc((size_t)image_count, sizeof *out->image_lens);
    if (!out->images || !out->image_lens) {
        p_free_images(images, image_count);
        free(out->images);
        free(out->image_lens);
        out->images = NULL;
        out->image_lens = NULL;
        cache_key_free(&key);
        return false;
    }
    if (req->notch ? req->notch == 1 : g_cfg.notch) {
        for (int i = 0; i < image_count; ++i)
            notch_rgb(images[i].data, (int)images[i].width, (int)images[i].height, (int)images[i].channel);
    }
    bool use_webp = g_webp_enabled && p_webp_encode_rgb && p_webp_free;
    for (int i = 0; i < image_count; ++i) {
        if (use_webp && (images[i].channel == 3 || (images[i].channel == 4 && p_webp_encode_rgba))) {
            fn_webp_encode_rgb encode = images[i].channel == 4 ? (fn_webp_encode_rgb)p_webp_encode_rgba : p_webp_encode_rgb;
            out->image_lens[i] = encode(
                images[i].data, (int)images[i].width, (int)images[i].height,
                (int)(images[i].width * images[i].channel), g_webp_quality,
                &out->images[i]);
            if (!out->image_lens[i] || !out->images[i]) use_webp = false;
        } else {
            use_webp = false;
        }
        if (!use_webp) break;
    }
    if (!use_webp) {
        for (int i = 0; i < image_count; ++i) {
            if (out->images[i]) p_webp_free(out->images[i]);
            out->images[i] = NULL;
            out->image_lens[i] = 0;
        }
        for (int i = 0; i < image_count; ++i) {
            png_sink sink = {0};
            int encoded = stbi_write_png_to_func(
                png_write, &sink, (int)images[i].width, (int)images[i].height,
                (int)images[i].channel, images[i].data,
                (int)(images[i].width * images[i].channel));
            if (!encoded || sink.failed || !sink.data) {
                free(sink.data);
                p_free_images(images, image_count);
                out->image_count = (size_t)i;
                osd_result_free(out);
                cache_key_free(&key);
                return false;
            }
            out->images[i] = sink.data;
            out->image_lens[i] = sink.len;
        }
    }
    p_free_images(images, image_count);
    out->image_count = (size_t)image_count;
    out->png = out->images[0];
    out->png_len = out->image_lens[0];
    out->format = use_webp ? "webp" : "png";
    if (result_cache) {
        pthread_mutex_lock(&g_sd_lock);
        pthread_mutex_lock(&g_cache_lock);
        latent_cache_entry *entry = cache_insert(req, 0, &key, NULL);
        pthread_mutex_unlock(&g_cache_lock);
        if (entry) cache_store_encoded_result(entry, out);
        pthread_mutex_unlock(&g_sd_lock);
    }
    if (out->teleport_used && cache_resume_step > 0 && out->image_count == 1) {
        pthread_mutex_lock(&g_sd_lock);
        pthread_mutex_lock(&g_cache_lock);
        latent_cache_entry *entry = cache_find(req, cache_resume_step, &key);
        pthread_mutex_unlock(&g_cache_lock);
        if (entry) cache_store_encoded_result(entry, out);
        pthread_mutex_unlock(&g_sd_lock);
    }
    cache_key_free(&key);
    return true;
}

void osd_result_free(oimg_result *result) {
    if (result->images) {
        for (size_t i = 0; i < result->image_count; ++i) {
            if (!result->images_malloc_owned && result->format &&
                strcmp(result->format, "webp") == 0 && p_webp_free) {
                p_webp_free(result->images[i]);
            } else {
                free(result->images[i]);
            }
        }
        free(result->images);
        free(result->image_lens);
    } else {
        free(result->png);
    }
    result->png = NULL;
    result->images = NULL;
    result->image_lens = NULL;
    result->image_count = 0;
}

#endif
