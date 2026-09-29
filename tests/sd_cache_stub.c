#include "stable-diffusion.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void sd_ctx_params_init(sd_ctx_params_t *params) { memset(params, 0, sizeof *params); }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *params) { (void)params; return (sd_ctx_t *)1; }
void sd_img_gen_params_init(sd_img_gen_params_t *params) { memset(params, 0, sizeof *params); }

bool generate_image(sd_ctx_t *ctx, const sd_img_gen_params_t *params,
                    sd_image_t **images, int *count) {
    (void)ctx;
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
