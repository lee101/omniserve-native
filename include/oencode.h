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

int oenc_png_mt(const uint8_t *px, int w, int h, int channels, int level, int threads,
                uint8_t **out, size_t *n);

int oenc_webp2(const uint8_t *rgba, int w, int h, int stride, float quality, int method, int alpha_method,
               int alpha_filter, int blank_transparent, int threads, uint8_t **out, size_t *n);

int oenc_png_info(const uint8_t *data, size_t n, int *w, int *h, int *channels);

int oenc_png_decode(const uint8_t *data, size_t n, uint8_t *out, size_t out_size, int stride, int channels);

int oenc_webp_info(const uint8_t *data, size_t n, int *w, int *h, int *has_alpha);

int oenc_webp_decode(const uint8_t *data, size_t n, uint8_t *out, size_t out_size, int stride,
                     int channels, int threads);

void oenc_free(void *p);

#ifdef __cplusplus
}
#endif

#endif
