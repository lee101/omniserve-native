#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "wavwrap.h"

static int fails = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            fails++;                                                       \
        }                                                                  \
    } while (0)

static void test_wraps_known_samples(void) {
    const uint8_t pcm[4] = {0x00, 0x00, 0x01, 0x00};
    uint8_t *wav = NULL;
    size_t len = 0;
    CHECK(wav_wrap_pcm16(pcm, sizeof pcm, 24000, &wav, &len) == 0);
    CHECK(len == 48);
    CHECK(memcmp(wav, "RIFF", 4) == 0);
    CHECK(memcmp(wav + 8, "WAVE", 4) == 0);
    CHECK(memcmp(wav + 12, "fmt ", 4) == 0);
    CHECK(memcmp(wav + 36, "data", 4) == 0);
    const uint32_t riff_len = (uint32_t)wav[4] | ((uint32_t)wav[5] << 8) |
                              ((uint32_t)wav[6] << 16) | ((uint32_t)wav[7] << 24);
    CHECK(riff_len == 40);
    const uint32_t rate = (uint32_t)wav[24] | ((uint32_t)wav[25] << 8) |
                          ((uint32_t)wav[26] << 16) | ((uint32_t)wav[27] << 24);
    CHECK(rate == 24000);
    const uint16_t bits = (uint16_t)(wav[34] | (wav[35] << 8));
    CHECK(bits == 16);
    CHECK(memcmp(wav + 44, pcm, 4) == 0);
    free(wav);
}

static void test_rejects_invalid_input(void) {
    uint8_t *wav = NULL;
    size_t len = 0;
    const uint8_t pcm[4] = {0, 0, 0, 0};
    CHECK(wav_wrap_pcm16(NULL, 4, 24000, &wav, &len) == -1);
    CHECK(wav_wrap_pcm16(pcm, 0, 24000, &wav, &len) == -1);
    CHECK(wav_wrap_pcm16(pcm, 3, 24000, &wav, &len) == -1); /* odd length */
    CHECK(wav_wrap_pcm16(pcm, 4, 0, &wav, &len) == -1);     /* zero rate */
    CHECK(wav == NULL && len == 0);
}

static void test_large_buffer_roundtrip(void) {
    const size_t n = 5u << 20; /* 5 MiB of samples like a long TTS reply */
    uint8_t *pcm = malloc(n);
    assert(pcm != NULL);
    for (size_t i = 0; i < n; i++) {
        pcm[i] = (uint8_t)(i * 31u);
    }
    uint8_t *wav = NULL;
    size_t len = 0;
    CHECK(wav_wrap_pcm16(pcm, n, 16000, &wav, &len) == 0);
    CHECK(len == 44 + n);
    CHECK(memcmp(wav + 44, pcm, n) == 0);
    free(wav);
    free(pcm);
}

int main(void) {
    test_wraps_known_samples();
    test_rejects_invalid_input();
    test_large_buffer_roundtrip();
    if (fails > 0) {
        return 1;
    }
    puts("test_wavwrap: all checks passed");
    return 0;
}
