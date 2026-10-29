#include "stable-diffusion.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void sd_ctx_params_init(sd_ctx_params_t *params) { memset(params, 0, sizeof *params); }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *params) { (void)params; return (sd_ctx_t *)1; }
void sd_img_gen_params_init(sd_img_gen_params_t *params) { memset(params, 0, sizeof *params); }

#ifdef OMNISERVE_SD_RESULT_API_STUB
static bool stub_generate(sd_ctx_t *ctx, const sd_img_gen_params_t *params,
                          sd_image_t **images, int *count);
bool generate_image(sd_ctx_t *ctx, const sd_img_gen_params_t *params,
                    sd_image_generation_result_t **result_out) {
    sd_image_t *images = NULL;
    int count = 0;
    *result_out = NULL;
    if (!stub_generate(ctx, params, &images, &count)) return false;
    sd_image_generation_result_t *result = calloc(1, sizeof *result);
    result->items = calloc((size_t)count, sizeof *result->items);
    result->count = count;
    for (int i = 0; i < count; ++i) result->items[i].image = images[i];
    free(images);
    *result_out = result;
    return true;
}
void free_sd_image_generation_result(sd_image_generation_result_t *result) {
    if (!result) return;
    for (int i = 0; i < result->count; ++i) free(result->items[i].image.data);
    free(result->items);
    free(result);
}
static bool stub_generate(sd_ctx_t *ctx, const sd_img_gen_params_t *params,
                          sd_image_t **images, int *count) {
#else
bool generate_image(sd_ctx_t *ctx, const sd_img_gen_params_t *params,
                    sd_image_t **images, int *count) {
#endif
    (void)ctx;
    const char *capture = getenv("OMNISERVE_STUB_PARAMS");
    if (capture) {
        FILE *file = fopen(capture, "a");
        if (!file) return false;
        fprintf(file, "{\"steps\":%d,\"cfg\":%.9g,\"sampler\":%d,\"refs\":%d,\"loras\":%u,\"cache\":%d,\"sigmas\":[",
                params->sample_params.sample_steps, (double)params->sample_params.guidance.txt_cfg,
                params->sample_params.sample_method, params->ref_images_count, params->lora_count,
                params->cache.mode);
        for (int i = 0; i < params->sample_params.custom_sigmas_count; ++i)
            fprintf(file, "%s%.9g", i ? "," : "", (double)params->sample_params.custom_sigmas[i]);
        fputs("]}\n", file);
        fclose(file);
    }
    const char *delay_env = getenv("OMNISERVE_STUB_DELAY_MS");
    long delay_ms = delay_env ? atol(delay_env) : 2000;
    struct timespec delay = {delay_ms / 1000, (delay_ms % 1000) * 1000000L};
    if (delay_ms > 0) nanosleep(&delay, NULL);
    const char *size_env = getenv("OMNISERVE_STUB_SIZE");
    uint32_t size = size_env ? (uint32_t)atoi(size_env) : 64;
    *images = calloc(1, sizeof **images);
    if (!*images) return false;
    (*images)->width = (*images)->height = size;
    (*images)->channel = 3;
    size_t bytes = (size_t)size * size * 3;
    (*images)->data = malloc(bytes);
    if (!(*images)->data) { free(*images); *images = NULL; return false; }
    if (size == 64) {
        memset((*images)->data, (int)(params->seed & 255), bytes);
    } else {
        uint32_t x = (uint32_t)params->seed * 2654435761u + 1;
        for (size_t i = 0; i < bytes; ++i) {
            x = x * 1664525u + 1013904223u;
            (*images)->data[i] = (unsigned char)(((i / 3) % size) / 4 + ((x >> 24) & 15));
        }
    }
    *count = 1;
    return true;
}

void free_sd_images(sd_image_t *images, int count) {
    for (int i = 0; i < count; ++i) free(images[i].data);
    free(images);
}
