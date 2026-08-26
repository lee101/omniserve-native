#include "music3.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void write_wav(unsigned char *out, size_t *used, const int16_t *samples, unsigned frames, unsigned channels, unsigned rate) {
    unsigned data_bytes = frames * channels * 2;
    memcpy(out, "RIFF", 4);
    uint32_t riff_size = 36 + data_bytes;
    memcpy(out + 4, &riff_size, 4);
    memcpy(out + 8, "WAVEfmt ", 8);
    uint32_t fmt_size = 16;
    uint16_t format = 1, ch = (uint16_t)channels, width = 16;
    uint32_t sr = rate, byterate = rate * channels * 2;
    uint16_t block = (uint16_t)(channels * 2);
    memcpy(out + 16, &fmt_size, 4);
    memcpy(out + 20, &format, 2);
    memcpy(out + 22, &ch, 2);
    memcpy(out + 24, &sr, 4);
    memcpy(out + 28, &byterate, 4);
    memcpy(out + 32, &block, 2);
    memcpy(out + 34, &width, 2);
    memcpy(out + 36, "data", 4);
    memcpy(out + 40, &data_bytes, 4);
    memcpy(out + 44, samples, data_bytes);
    *used = 44 + data_bytes;
}

int main(void) {
    Music3Request request = {0};
    assert(music3_normalize_request("{\"input\":{\"prompt\":\"Warm synthwave at 105 BPM\",\"duration\":10,\"seed\":7}}", &request) == 0);
    assert(request.max_new_tokens == 250);
    assert(request.seed == 7);
    assert(strncmp(request.lyrics, "[Intro]\n", 8) == 0);
    assert(strstr(request.instructions, "instrumental") != NULL);

    assert(music3_normalize_request("{\"input\":{\"lyrics\":\"[Verse]\\nHello\",\"instructions\":\"Acoustic folk\",\"duration\":30}}", &request) == 0);
    assert(strcmp(request.lyrics, "[Verse]\nHello") == 0);
    assert(request.max_new_tokens == 750);
    assert(music3_normalize_request("{\"input\":{\"prompt\":\"x\",\"duration\":361}}", &request) != 0);
    assert(music3_normalize_request("[{\"id\":\"job-1\",\"input\":{\"prompt\":\"Array job\",\"duration\":12}}]", &request) == 0);
    assert(request.duration_seconds == 12);

    char digest[MUSIC3_SHA_SIZE];
    music3_sha256_hex((const unsigned char *)"abc", 3, digest);
    assert(strcmp(digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0);

    unsigned rate = 32000;
    int16_t stereo[64000];
    for (unsigned i = 0; i < rate; ++i) {
        int16_t sample = (int16_t)(0.25 * sin(2.0 * 3.141592653589793 * 440.0 * i / rate) * 32767.0);
        stereo[i * 2] = sample;
        stereo[i * 2 + 1] = sample;
    }
    unsigned char wav[44 + 128000];
    size_t used = 0;
    write_wav(wav, &used, stereo, rate, 2, rate);
    Music3WavStats stats = {0};
    assert(music3_wav_statistics(wav, used, &stats) == 0);
    assert(stats.sample_rate_hz == 32000);
    assert(stats.channels == 2);
    assert(stats.duration_seconds > 0.999 && stats.duration_seconds < 1.001);
    assert(stats.clipped_samples == 0);
    assert(stats.has_stereo_correlation);
    assert(stats.stereo_correlation > 0.999);
    assert(music3_name_included("dav.pth", "qwen_7B,flowmatching_vae.pth,dav.pth") == 1);
    assert(music3_name_included("dav.pth.tmp", "qwen_7B,flowmatching_vae.pth,dav.pth") == 0);
    assert(music3_name_included("q", "qwen_7B") == 0);
    assert(music3_name_included("", "qwen_7B") == 0);
    assert(music3_name_included("flowmatching_vae.pth", "flowmatching_vae.pth") == 1);
    puts("music3c tests passed");
    return 0;
}
