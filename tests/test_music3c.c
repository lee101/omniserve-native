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

    char cache_key[MUSIC3_SHA_SIZE], same_key[MUSIC3_SHA_SIZE], changed_key[MUSIC3_SHA_SIZE];
    music3_result_cache_key(&request, "MiniMaxAI/MiniMax-Music3", "release-a", cache_key);
    music3_result_cache_key(&request, "MiniMaxAI/MiniMax-Music3", "release-a", same_key);
    assert(strcmp(cache_key, same_key) == 0);
    snprintf(request.output_upload_url, sizeof(request.output_upload_url), "https://uploads.example/a");
    music3_result_cache_key(&request, "MiniMaxAI/MiniMax-Music3", "release-a", same_key);
    assert(strcmp(cache_key, same_key) == 0);
    request.output_upload_url[0] = '\0';
    music3_result_cache_key(&request, "MiniMaxAI/MiniMax-Music3", "release-b", changed_key);
    assert(strcmp(cache_key, changed_key) != 0);
    request.seed++;
    music3_result_cache_key(&request, "MiniMaxAI/MiniMax-Music3", "release-a", changed_key);
    assert(strcmp(cache_key, changed_key) != 0);

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
    assert(stats.continuity_score > 99.0);

    char result[4096];
    Music3Timings timings = {.generation_seconds = 0, .gpu_name = "test", .quality_attempts = 1,
                             .original_seed = request.seed, .exact_result_cache_hit = 1};
    assert(music3_write_result_json(result, sizeof(result), &request, &stats, NULL, NULL, &timings) == 0);
    assert(strstr(result, "\"exact_result_cache_hit\":true") != NULL);
    assert(strstr(result, "\"realtime_factor\":0.000") != NULL);

    /* A one-second internal collapse between healthy tone must trip the retry gate. */
    int16_t dropout[32000 * 6 * 2];
    for (unsigned i = 0; i < 32000 * 6; ++i) {
        double level = (i >= 32000 * 2 && i < 32000 * 3) ? 0.00001 : 0.25;
        int16_t sample = (int16_t)(level * sin(2.0 * 3.141592653589793 * 440.0 * i / 32000) * 32767.0);
        dropout[i * 2] = sample; dropout[i * 2 + 1] = sample;
    }
    unsigned char dropout_wav[44 + sizeof(dropout)];
    write_wav(dropout_wav, &used, dropout, 32000 * 6, 2, 32000);
    assert(music3_wav_statistics(dropout_wav, used, &stats) == 0);
    assert(stats.continuity_score < 80.0);
    assert(stats.longest_severe_drop_seconds >= 0.75);
    assert(music3_name_included("dav.pth", "qwen_7B,flowmatching_vae.pth,dav.pth") == 1);
    assert(music3_name_included("dav.pth.tmp", "qwen_7B,flowmatching_vae.pth,dav.pth") == 0);
    assert(music3_name_included("q", "qwen_7B") == 0);
    assert(music3_name_included("", "qwen_7B") == 0);
    assert(music3_name_included("flowmatching_vae.pth", "flowmatching_vae.pth") == 1);
    puts("music3c tests passed");
    return 0;
}
