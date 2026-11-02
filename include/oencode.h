#ifndef OENCODE_H
#define OENCODE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int oenc_png(const uint8_t *px, int w, int h, int channels, int level, int threads,
             uint8_t **out, size_t *n);

int oenc_webp(const uint8_t *rgba, int w, int h, int stride, float quality, int method,
              int alpha_filter, int blank_transparent, int threads, uint8_t **out, size_t *n);

void oenc_free(void *p);

#ifdef __cplusplus
}
#endif

#endif
