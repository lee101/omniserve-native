#include "obackend.h"

#ifdef USE_SD

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "stable-diffusion.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef void (*fn_ctx_params_init)(sd_ctx_params_t *);
typedef sd_ctx_t *(*fn_new_sd_ctx)(const sd_ctx_params_t *);
typedef void (*fn_img_params_init)(sd_img_gen_params_t *);
typedef bool (*fn_generate_image)(sd_ctx_t *, const sd_img_gen_params_t *, sd_image_t **, int *);

static fn_ctx_params_init p_ctx_params_init;
static fn_new_sd_ctx p_new_sd_ctx;
static fn_img_params_init p_img_params_init;
static fn_generate_image p_generate_image;

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
    return p_ctx_params_init && p_new_sd_ctx && p_img_params_init && p_generate_image;
}

static sd_ctx_t *g_sd;
static pthread_mutex_t g_sd_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_sd_name[256];

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

bool osd_init(const char *model_path) {
    if (!sd_lib_load()) return false;
    sd_ctx_params_t params;
    p_ctx_params_init(&params);
    params.model_path = model_path;
    params.n_threads = -1;
    g_sd = p_new_sd_ctx(&params);
    if (!g_sd) return false;
    const char *slash = strrchr(model_path, '/');
    snprintf(g_sd_name, sizeof g_sd_name, "%s", slash ? slash + 1 : model_path);
    char *dot = strrchr(g_sd_name, '.');
    if (dot) *dot = 0;
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
    params.seed = req->seed;

    pthread_mutex_lock(&g_sd_lock);
    double started = now_ms();
    sd_image_t *images = NULL;
    int image_count = 0;
    bool ok = p_generate_image(g_sd, &params, &images, &image_count);
    out->elapsed_ms = now_ms() - started;
    pthread_mutex_unlock(&g_sd_lock);

    if (!ok || image_count <= 0 || !images) return false;
    png_sink sink = {0};
    int encoded = stbi_write_png_to_func(
        png_write, &sink, (int)images[0].width, (int)images[0].height,
        (int)images[0].channel, images[0].data,
        (int)(images[0].width * images[0].channel));
    for (int i = 0; i < image_count; i++) free(images[i].data);
    free(images);
    if (!encoded || sink.failed || !sink.data) {
        free(sink.data);
        return false;
    }
    out->png = sink.data;
    out->png_len = sink.len;
    return true;
}

void osd_result_free(oimg_result *result) {
    free(result->png);
    result->png = NULL;
}

#endif
