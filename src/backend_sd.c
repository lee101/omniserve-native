#include "obackend.h"

#ifdef USE_SD

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "stable-diffusion.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef void (*fn_ctx_params_init)(sd_ctx_params_t *);
typedef sd_ctx_t *(*fn_new_sd_ctx)(const sd_ctx_params_t *);
typedef void (*fn_img_params_init)(sd_img_gen_params_t *);
typedef bool (*fn_generate_image)(sd_ctx_t *, const sd_img_gen_params_t *, sd_image_t **, int *);
typedef void (*fn_free_images)(sd_image_t *, int);
typedef void (*fn_latent_params_init)(sd_latent_replay_params_t *);
typedef bool (*fn_generate_image_with_latent)(sd_ctx_t *, const sd_img_gen_params_t *,
                                              sd_latent_replay_params_t *, sd_image_t **, int *);
typedef void (*fn_free_latent)(sd_latent_t *);
typedef size_t (*fn_webp_encode_rgb)(const unsigned char *, int, int, int, float,
                                     unsigned char **);
typedef void (*fn_webp_free)(void *);

static fn_ctx_params_init p_ctx_params_init;
static fn_new_sd_ctx p_new_sd_ctx;
static fn_img_params_init p_img_params_init;
static fn_generate_image p_generate_image;
static fn_free_images p_free_images;
static fn_latent_params_init p_latent_params_init;
static fn_generate_image_with_latent p_generate_image_with_latent;
static fn_free_latent p_free_latent;
static fn_webp_encode_rgb p_webp_encode_rgb;
static fn_webp_free p_webp_free;
static bool g_webp_enabled = true;
static float g_webp_quality = 85.0f;

static void webp_lib_load(void) {
    static const char *const candidates[] = {
        "libwebp.so.7", "libwebp.so.6", "libwebp.so", NULL,
    };
    for (int i = 0; candidates[i]; ++i) {
        void *lib = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (!lib) continue;
        p_webp_encode_rgb = (fn_webp_encode_rgb)dlsym(lib, "WebPEncodeRGB");
        p_webp_free = (fn_webp_free)dlsym(lib, "WebPFree");
        if (p_webp_encode_rgb && p_webp_free) return;
        p_webp_encode_rgb = NULL;
        p_webp_free = NULL;
        dlclose(lib);
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
    p_free_images = (fn_free_images)dlsym(lib, "free_sd_images");
    p_latent_params_init = (fn_latent_params_init)dlsym(lib, "sd_latent_replay_params_init");
    p_generate_image_with_latent = (fn_generate_image_with_latent)dlsym(lib, "generate_image_with_latent");
    p_free_latent = (fn_free_latent)dlsym(lib, "free_sd_latent");
    return p_ctx_params_init && p_new_sd_ctx && p_img_params_init && p_generate_image && p_free_images;
}

static sd_ctx_t *g_sd;
static pthread_mutex_t g_sd_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_sd_name[256];

typedef struct {
    char *prompt;
    char *negative_prompt;
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

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static bool sd_env_flag(const char *name, bool fallback) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

static int sd_env_int(const char *name, int fallback, int minimum, int maximum) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (!end || *end || parsed < minimum || parsed > maximum) return fallback;
    return (int)parsed;
}

static float sd_env_float(const char *name, float fallback, float minimum, float maximum) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    float parsed = strtof(value, &end);
    if (!end || end == value || *end || parsed < minimum || parsed > maximum) return fallback;
    return parsed;
}

static bool latent_api_ready(void) {
    return p_latent_params_init && p_generate_image_with_latent && p_free_latent &&
           g_latent_cache && g_latent_cache_size > 0;
}

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

static bool cache_key_equal(const latent_cache_entry *entry, const oimg_req *req,
                            int resume_step, const char *lora_key) {
    const char *negative = req->negative_prompt ? req->negative_prompt : "";
    return entry->latent && entry->width == req->width && entry->height == req->height &&
           entry->steps == req->steps && entry->guidance_scale == req->guidance_scale &&
           entry->seed == req->seed && entry->resume_step == resume_step &&
           strcmp(entry->lora_key ? entry->lora_key : "", lora_key) == 0 &&
           strcmp(entry->prompt, req->prompt) == 0 &&
           strcmp(entry->negative_prompt, negative) == 0;
}

static latent_cache_entry *cache_find(const oimg_req *req, int resume_step) {
    char *lora_key = lora_cache_key(req);
    if (!lora_key) return NULL;
    for (int i = 0; i < g_latent_cache_size; i++) {
        if (cache_key_equal(&g_latent_cache[i], req, resume_step, lora_key)) {
            g_latent_cache[i].tick = ++g_latent_cache_tick;
            free(lora_key);
            return &g_latent_cache[i];
        }
    }
    free(lora_key);
    return NULL;
}

static void cache_entry_clear(latent_cache_entry *entry) {
    if (entry->latent && p_free_latent) p_free_latent(entry->latent);
    free(entry->prompt);
    free(entry->negative_prompt);
    free(entry->lora_key);
    free(entry->encoded_image);
    memset(entry, 0, sizeof *entry);
}

static bool cache_insert(const oimg_req *req, int resume_step, sd_latent_t *latent) {
    latent_cache_entry *slot = NULL;
    for (int i = 0; i < g_latent_cache_size; i++) {
        if (!g_latent_cache[i].latent) {
            slot = &g_latent_cache[i];
            break;
        }
        if (!slot || g_latent_cache[i].tick < slot->tick) slot = &g_latent_cache[i];
    }
    if (!slot) return false;
    char *prompt = strdup(req->prompt);
    char *negative = strdup(req->negative_prompt ? req->negative_prompt : "");
    char *lora_key = lora_cache_key(req);
    if (!prompt || !negative || !lora_key) {
        free(prompt);
        free(negative);
        free(lora_key);
        return false;
    }
    cache_entry_clear(slot);
    slot->prompt = prompt;
    slot->negative_prompt = negative;
    slot->width = req->width;
    slot->height = req->height;
    slot->steps = req->steps;
    slot->guidance_scale = req->guidance_scale;
    slot->seed = req->seed;
    slot->resume_step = resume_step;
    slot->lora_key = lora_key;
    slot->latent = latent;
    slot->tick = ++g_latent_cache_tick;
    return true;
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

static void cache_store_encoded_result(latent_cache_entry *entry, const oimg_result *out) {
    if (!entry || out->image_count != 1 || !out->images || !out->image_lens ||
        !out->images[0] || !out->image_lens[0] || out->image_lens[0] > (64u << 20)) return;
    unsigned char *copy = malloc(out->image_lens[0]);
    if (!copy) return;
    memcpy(copy, out->images[0], out->image_lens[0]);
    free(entry->encoded_image);
    entry->encoded_image = copy;
    entry->encoded_image_len = out->image_lens[0];
    entry->encoded_image_is_webp = out->format && strcmp(out->format, "webp") == 0;
}

bool osd_init(const char *model_path) {
    if (!sd_lib_load()) return false;
    sd_ctx_params_t params;
    p_ctx_params_init(&params);
    const char *diffusion = getenv("OMNISERVE_NATIVE_SD_DIFFUSION_MODEL");
    const char *vae = getenv("OMNISERVE_NATIVE_SD_VAE");
    const char *llm = getenv("OMNISERVE_NATIVE_SD_LLM");
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
    if (taesd && taesd[0]) params.taesd_path = taesd;
    if (max_vram && max_vram[0]) params.max_vram = max_vram;
    if (backend && backend[0]) params.backend = backend;
    if (params_backend && params_backend[0]) params.params_backend = params_backend;
    params.n_threads = -1;
    params.flash_attn = sd_env_flag("OMNISERVE_NATIVE_SD_FLASH_ATTN", true);
    params.diffusion_flash_attn = sd_env_flag("OMNISERVE_NATIVE_SD_DIFFUSION_FLASH_ATTN", true);
    params.stream_layers = sd_env_flag("OMNISERVE_NATIVE_SD_STREAM_LAYERS", false);
    params.enable_mmap = sd_env_flag("OMNISERVE_NATIVE_SD_MMAP", true);
    params.eager_load = sd_env_flag("OMNISERVE_NATIVE_SD_EAGER_LOAD", false);
    params.auto_fit = sd_env_flag("OMNISERVE_NATIVE_SD_AUTO_FIT", false);
    const char *model_args = getenv("OMNISERVE_NATIVE_SD_MODEL_ARGS");
    if (model_args && model_args[0]) params.model_args = model_args;
    g_sd = p_new_sd_ctx(&params);
    if (!g_sd) return false;
    const char *named_path = diffusion && diffusion[0] ? diffusion : model_path;
    const char *slash = strrchr(named_path, '/');
    snprintf(g_sd_name, sizeof g_sd_name, "%s", slash ? slash + 1 : named_path);
    char *dot = strrchr(g_sd_name, '.');
    if (dot) *dot = 0;
    g_latent_cache_size = sd_env_int("OMNISERVE_NATIVE_SD_TELEPORT_CACHE_SIZE", 64, 0, 256);
    if (g_latent_cache_size > 0 && p_latent_params_init &&
        p_generate_image_with_latent && p_free_latent) {
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
    /* Zero is intentional for distilled Flux/Z-Image pipelines. Treating it as
     * an unset sentinel changes both image quality and the latent replay key. */
    params.sample_params.guidance.txt_cfg = req->guidance_scale == 0.0f
        ? sd_env_float("OMNISERVE_NATIVE_SD_ZERO_GUIDANCE", 0.0f, 0.0f, 30.0f)
        : req->guidance_scale;
    params.vae_tiling_params.enabled = sd_env_flag(
        "OMNISERVE_NATIVE_SD_VAE_TILING", params.vae_tiling_params.enabled);
    if (params.vae_tiling_params.enabled) {
        params.vae_tiling_params.tile_size_x = sd_env_int(
            "OMNISERVE_NATIVE_SD_VAE_TILE_X", 32, 32, 4096);
        params.vae_tiling_params.tile_size_y = sd_env_int(
            "OMNISERVE_NATIVE_SD_VAE_TILE_Y", 32, 32, 4096);
        params.vae_tiling_params.target_overlap = sd_env_float(
            "OMNISERVE_NATIVE_SD_VAE_TILE_OVERLAP", 0.5f, 0.0f, 0.95f);
    }
    params.seed = req->seed;
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

    out->teleport_requested = req->teleport;
    out->teleport_capture_step = -1;
    out->teleport_resume_step = 0;

    pthread_mutex_lock(&g_sd_lock);
    double started = now_ms();
    sd_image_t *images = NULL;
    int image_count = 0;
    int cache_resume_step = 0;
    bool ok = false;
    if (req->teleport && params.batch_count == 1 &&
        req->steps > 1 && latent_api_ready()) {
        int default_resume = sd_env_int(
            "OMNISERVE_NATIVE_SD_TELEPORT_START_STEP", req->steps - 1, 1, 99);
        int resume_step = req->teleport_start_step > 0
            ? req->teleport_start_step : default_resume;
        if (resume_step >= req->steps) resume_step = req->steps - 1;
        cache_resume_step = resume_step;
        latent_cache_entry *cached = cache_find(req, resume_step);
        if (cached && cache_copy_encoded_result(cached, out)) {
            out->teleport_used = true;
            out->teleport_cache_hit = true;
            out->teleport_result_cache_hit = true;
            out->teleport_capture_step = resume_step - 1;
            out->teleport_resume_step = resume_step;
            out->elapsed_ms = now_ms() - started;
            pthread_mutex_unlock(&g_sd_lock);
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
                if (!cache_insert(req, resume_step, captured)) {
                    fprintf(stderr, "diffusion teleport latent cache insert failed\n");
                    p_free_latent(captured);
                }
            } else if (!cached) {
                fprintf(stderr, "diffusion teleport completed without returning a captured latent\n");
            }
        } else {
            if (captured) p_free_latent(captured);
            if (cached) cache_entry_clear(cached);
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
        return false;
    }
    bool use_webp = g_webp_enabled && p_webp_encode_rgb && p_webp_free;
    for (int i = 0; i < image_count; ++i) {
        if (use_webp && images[i].channel == 3) {
            out->image_lens[i] = p_webp_encode_rgb(
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
    if (out->teleport_used && cache_resume_step > 0 && out->image_count == 1) {
        pthread_mutex_lock(&g_sd_lock);
        latent_cache_entry *entry = cache_find(req, cache_resume_step);
        if (entry) cache_store_encoded_result(entry, out);
        pthread_mutex_unlock(&g_sd_lock);
    }
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
