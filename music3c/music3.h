#ifndef OMNISERVE_MUSIC3_H
#define OMNISERVE_MUSIC3_H

#include <stddef.h>
#include <stdint.h>

enum {
    MUSIC3_LYRICS_SIZE = 16384,
    MUSIC3_TEXT_SIZE = 4096,
    MUSIC3_URL_SIZE = 4096,
    MUSIC3_ERROR_SIZE = 512,
    MUSIC3_SHA_SIZE = 65,
    MUSIC3_MAX_DURATION_SECONDS = 360
};

typedef struct {
    char lyrics[MUSIC3_LYRICS_SIZE];
    char instructions[MUSIC3_TEXT_SIZE];
    char output_upload_url[MUSIC3_URL_SIZE];
    char output_public_url[MUSIC3_URL_SIZE];
    char error[MUSIC3_ERROR_SIZE];
    int duration_seconds;
    int max_new_tokens;
    long long seed;
} Music3Request;

typedef struct {
    unsigned sample_rate_hz;
    unsigned channels;
    unsigned bit_depth;
    unsigned frames;
    double duration_seconds;
    size_t bytes;
    char sha256[MUSIC3_SHA_SIZE];
    double peak_dbfs;
    double rms_dbfs;
    double crest_factor_db;
    double dc_offset;
    unsigned clipped_samples;
    double clipped_percent;
    double digital_silence_percent;
    int has_stereo_correlation;
    double stereo_correlation;
} Music3WavStats;

int music3_normalize_request(const char *json, Music3Request *request);
int music3_wav_statistics(const unsigned char *audio, size_t length, Music3WavStats *stats);
void music3_sha256_hex(const unsigned char *data, size_t length, char out[MUSIC3_SHA_SIZE]);
int music3_sha256_file(const char *path, char out[MUSIC3_SHA_SIZE]);
int music3_write_result_json(char *out, size_t size, const Music3Request *request,
                             const Music3WavStats *stats, const char *audio_url,
                             const char *inline_b64, double model_download_seconds,
                             double server_start_seconds, double generation_seconds,
                             double upload_seconds, double total_seconds,
                             double server_started_at);

#endif
