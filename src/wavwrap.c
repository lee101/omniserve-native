#include "wavwrap.h"

#include <stdlib.h>
#include <string.h>

#define WAV_HEADER_LEN 44
#define WAV_MAX_DATA ((size_t)0xFFFFFFFFu - 36u)

static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8) & 0xffu);
}

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8) & 0xffu);
    p[2] = (uint8_t)((v >> 16) & 0xffu);
    p[3] = (uint8_t)((v >> 24) & 0xffu);
}

int wav_wrap_pcm16(const uint8_t *pcm, size_t pcm_len, uint32_t sample_rate,
                   uint8_t **out, size_t *out_len) {
    if (out == NULL || out_len == NULL) {
        return -1;
    }
    *out = NULL;
    *out_len = 0;
    if (pcm == NULL || pcm_len == 0 || (pcm_len & 1u) != 0 || sample_rate == 0 ||
        pcm_len > WAV_MAX_DATA) {
        return -1;
    }

    uint8_t *buf = malloc(WAV_HEADER_LEN + pcm_len);
    if (buf == NULL) {
        return -2;
    }

    memcpy(buf, "RIFF", 4);
    put_u32(buf + 4, (uint32_t)(36u + pcm_len));
    memcpy(buf + 8, "WAVE", 4);
    memcpy(buf + 12, "fmt ", 4);
    put_u32(buf + 16, 16);                      /* fmt chunk size */
    put_u16(buf + 20, 1);                       /* PCM format */
    put_u16(buf + 22, 1);                       /* mono */
    put_u32(buf + 24, sample_rate);
    put_u32(buf + 28, sample_rate * 2u);        /* byte rate: rate * channels * 2 */
    put_u16(buf + 32, 2);                       /* block align */
    put_u16(buf + 34, 16);                      /* bits per sample */
    memcpy(buf + 36, "data", 4);
    put_u32(buf + 40, (uint32_t)pcm_len);
    memcpy(buf + WAV_HEADER_LEN, pcm, pcm_len);

    *out = buf;
    *out_len = WAV_HEADER_LEN + pcm_len;
    return 0;
}
