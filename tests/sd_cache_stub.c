#include "stable-diffusion.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

void sd_ctx_params_init(sd_ctx_params_t *params) { memset(params, 0, sizeof *params); }
sd_ctx_t *new_sd_ctx(const sd_ctx_params_t *params) { (void)params; return (sd_ctx_t *)1; }
void sd_img_gen_params_init(sd_img_gen_params_t *params) { memset(params, 0, sizeof *params); }

bool generate_image(sd_ctx_t *ctx, const sd_img_gen_params_t *params,
                    sd_image_t **images, int *count) {
    (void)ctx;
    struct timespec delay = {2, 0};
    nanosleep(&delay, NULL);
    *images = calloc(1, sizeof **images);
    if (!*images) return false;
    (*images)->width = (*images)->height = 64;
    (*images)->channel = 3;
    (*images)->data = malloc(64 * 64 * 3);
    if (!(*images)->data) { free(*images); *images = NULL; return false; }
    memset((*images)->data, (int)(params->seed & 255), 64 * 64 * 3);
    *count = 1;
    return true;
}

void free_sd_images(sd_image_t *images, int count) {
    for (int i = 0; i < count; ++i) free(images[i].data);
    free(images);
}
