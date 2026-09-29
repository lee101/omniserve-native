/* In-memory PCM16 -> WAV wrap benchmark: isolates the encode step from any
 * file I/O so the CPU ceiling is visible. Usage: bench_wavwrap [mib] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "wavwrap.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    const size_t mib = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 64u;
    const size_t pcm_len = mib << 20;

    uint8_t *pcm = malloc(pcm_len);
    if (pcm == NULL) {
        return 1;
    }
    memset(pcm, 0x5a, pcm_len);

    /* warm up allocator + pages */
    uint8_t *wav = NULL;
    size_t wav_len = 0;
    if (wav_wrap_pcm16(pcm, pcm_len, 24000, &wav, &wav_len) != 0) {
        return 1;
    }
    free(wav);

    const int rounds = 20;
    const double start = now_s();
    size_t sink = 0;
    for (int i = 0; i < rounds; i++) {
        wav = NULL;
        wav_len = 0;
        if (wav_wrap_pcm16(pcm, pcm_len, 24000, &wav, &wav_len) != 0) {
            return 1;
        }
        sink += wav[wav_len - 1];
        free(wav);
    }
    const double elapsed = now_s() - start;

    printf("C wav_wrap_pcm16: %.0f MiB/s (%zu MiB payload, %d rounds, sink=%zu)\n",
           ((double)mib * rounds) / elapsed, mib, rounds, sink);
    free(pcm);
    return 0;
}
