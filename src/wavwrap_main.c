/* wavwrap: wrap raw PCM16LE from stdin (or file arg) into a WAV on stdout.
 * Usage: wavwrap [--rate HZ] [input.pcm] > output.wav */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wavwrap.h"

static int read_all(FILE *f, uint8_t **buf, size_t *len) {
    size_t cap = 1u << 20;
    size_t n = 0;
    uint8_t *p = malloc(cap);
    if (p == NULL) {
        return -1;
    }
    for (;;) {
        if (n == cap) {
            if (cap > SIZE_MAX / 2) {
                free(p);
                return -1;
            }
            cap *= 2;
            uint8_t *grown = realloc(p, cap);
            if (grown == NULL) {
                free(p);
                return -1;
            }
            p = grown;
        }
        size_t got = fread(p + n, 1, cap - n, f);
        n += got;
        if (got == 0) {
            if (ferror(f)) {
                free(p);
                return -1;
            }
            break;
        }
    }
    *buf = p;
    *len = n;
    return 0;
}

int main(int argc, char **argv) {
    uint32_t rate = 24000;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rate") == 0 && i + 1 < argc) {
            rate = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else {
            path = argv[i];
        }
    }

    FILE *in = stdin;
    if (path != NULL) {
        in = fopen(path, "rb");
        if (in == NULL) {
            perror("open input");
            return 1;
        }
    }

    uint8_t *pcm = NULL;
    size_t pcm_len = 0;
    if (read_all(in, &pcm, &pcm_len) != 0) {
        fprintf(stderr, "wavwrap: read failed\n");
        return 1;
    }
    if (in != stdin) {
        fclose(in);
    }

    uint8_t *wav = NULL;
    size_t wav_len = 0;
    int rc = wav_wrap_pcm16(pcm, pcm_len, rate, &wav, &wav_len);
    free(pcm);
    if (rc != 0) {
        fprintf(stderr, "wavwrap: wrap failed (%d)\n", rc);
        return 1;
    }

    if (fwrite(wav, 1, wav_len, stdout) != wav_len) {
        fprintf(stderr, "wavwrap: write failed\n");
        free(wav);
        return 1;
    }
    free(wav);
    return 0;
}
