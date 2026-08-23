#include "music3.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t SHA_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

typedef struct {
    uint32_t h[8];
    unsigned char buffer[64];
    size_t buffer_used;
    uint64_t total;
} Sha256Ctx;

static void sha256_compress(Sha256Ctx *ctx, const unsigned char block[64]) {
    uint32_t w[64], a[8];
    size_t i;
    for (i = 0; i < 16; ++i)
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | block[i * 4 + 3];
    for (i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    memcpy(a, ctx->h, sizeof(a));
    for (i = 0; i < 64; ++i) {
        uint32_t s1 = rotr(a[4], 6) ^ rotr(a[4], 11) ^ rotr(a[4], 25);
        uint32_t ch = (a[4] & a[5]) ^ ((~a[4]) & a[6]);
        uint32_t t1 = a[7] + s1 + ch + SHA_K[i] + w[i];
        uint32_t s0 = rotr(a[0], 2) ^ rotr(a[0], 13) ^ rotr(a[0], 22);
        uint32_t maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
        uint32_t t2 = s0 + maj;
        a[7] = a[6]; a[6] = a[5]; a[5] = a[4]; a[4] = a[3] + t1;
        a[3] = a[2]; a[2] = a[1]; a[1] = a[0]; a[0] = t1 + t2;
    }
    for (i = 0; i < 8; ++i) ctx->h[i] += a[i];
}

static void sha256_init(Sha256Ctx *ctx) {
    static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memset(ctx, 0, sizeof(*ctx));
    memcpy(ctx->h, iv, sizeof(iv));
}

static void sha256_update(Sha256Ctx *ctx, const unsigned char *data, size_t length) {
    ctx->total += length;
    while (length > 0) {
        size_t take = 64 - ctx->buffer_used;
        if (take > length) take = length;
        memcpy(ctx->buffer + ctx->buffer_used, data, take);
        ctx->buffer_used += take;
        data += take;
        length -= take;
        if (ctx->buffer_used == 64) {
            sha256_compress(ctx, ctx->buffer);
            ctx->buffer_used = 0;
        }
    }
}

static void sha256_final(Sha256Ctx *ctx, char out[MUSIC3_SHA_SIZE]) {
    uint64_t bit_len = ctx->total * 8;
    unsigned char pad = 0x80;
    sha256_update(ctx, &pad, 1);
    pad = 0;
    while (ctx->buffer_used != 56) sha256_update(ctx, &pad, 1);
    unsigned char len[8];
    for (int i = 0; i < 8; ++i) len[7 - i] = (unsigned char)(bit_len >> (8 * i));
    sha256_update(ctx, len, 8);
    for (int i = 0; i < 8; ++i) snprintf(out + i * 8, 9, "%08x", ctx->h[i]);
}

void music3_sha256_hex(const unsigned char *data, size_t length, char out[MUSIC3_SHA_SIZE]) {
    Sha256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, length);
    sha256_final(&ctx, out);
}

int music3_sha256_file(const char *path, char out[MUSIC3_SHA_SIZE]) {
    FILE *file = fopen(path, "rb");
    unsigned char chunk[1 << 20];
    Sha256Ctx ctx;
    if (file == NULL) return -1;
    sha256_init(&ctx);
    for (;;) {
        size_t n = fread(chunk, 1, sizeof(chunk), file);
        if (n) sha256_update(&ctx, chunk, n);
        if (n < sizeof(chunk)) {
            int failed = ferror(file);
            fclose(file);
            if (failed) return -1;
            sha256_final(&ctx, out);
            return 0;
        }
    }
}

static const char *skip_ws(const char *p) {
    while (p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    return p;
}

static int json_unescape(const char *src, size_t length, char *out, size_t out_size) {
    size_t used = 0;
    for (size_t i = 0; i < length && used + 1 < out_size; ++i) {
        if (src[i] == '\\' && i + 1 < length) {
            char next = src[++i];
            if (next == 'n') out[used++] = '\n';
            else if (next == 'r') out[used++] = '\r';
            else if (next == 't') out[used++] = '\t';
            else out[used++] = next;
        } else out[used++] = src[i];
    }
    if (used + 1 >= out_size) return -1;
    out[used] = '\0';
    return 0;
}

static const char *json_key(const char *object, const char *end, const char *key) {
    size_t key_len = strlen(key);
    const char *p = object;
    int depth = 0;
    int in_string = 0;
    while (p < end && *p) {
        if (in_string) {
            if (*p == '\\' && p + 1 < end) { p += 2; continue; }
            if (*p == '"') in_string = 0;
            ++p;
            continue;
        }
        if (*p == '"') {
            if (depth == 1 && p + key_len + 2 < end && p[key_len + 1] == '"' &&
                strncmp(p + 1, key, key_len) == 0) {
                const char *colon = skip_ws(p + key_len + 2);
                if (*colon == ':') return skip_ws(colon + 1);
            }
            in_string = 1;
            ++p;
            continue;
        }
        if (*p == '{' || *p == '[') ++depth;
        else if (*p == '}' || *p == ']') { if (--depth <= 0) return NULL; }
        ++p;
    }
    return NULL;
}

static const char *json_object_span(const char *start, const char **end_out) {
    start = skip_ws(start);
    if (*start != '{') return NULL;
    int depth = 0, in_string = 0;
    const char *p = start;
    do {
        if (in_string) {
            if (*p == '\\' && p[1]) { p += 2; continue; }
            if (*p == '"') in_string = 0;
        } else {
            if (*p == '"') in_string = 1;
            else if (*p == '{') ++depth;
            else if (*p == '}') --depth;
        }
        ++p;
    } while (*p && depth > 0);
    if (depth != 0) return NULL;
    *end_out = p;
    return start;
}

static int json_string(const char *value, char *out, size_t out_size) {
    value = skip_ws(value);
    if (*value != '"') return -1;
    const char *start = value + 1;
    const char *p = start;
    while (*p) {
        if (*p == '\\' && p[1]) { p += 2; continue; }
        if (*p == '"') break;
        ++p;
    }
    if (*p != '"') return -1;
    return json_unescape(start, (size_t)(p - start), out, out_size);
}

static int json_long(const char *value, long long *out) {
    value = skip_ws(value);
    char *tail = NULL;
    long long parsed = strtoll(value, &tail, 10);
    if (tail == value) return -1;
    *out = parsed;
    return 0;
}

static int fail(Music3Request *request, const char *message) {
    snprintf(request->error, sizeof(request->error), "%s", message);
    return -1;
}

int music3_normalize_request(const char *json, Music3Request *request) {
    memset(request, 0, sizeof(*request));
    if (json == NULL) return fail(request, "input must be an object");
    const char *cursor = skip_ws(json);
    if (*cursor == '[') cursor = skip_ws(cursor + 1);
    const char *root_end = NULL;
    const char *root = json_object_span(cursor, &root_end);
    if (root == NULL) return fail(request, "input must be an object");
    const char *input = json_key(root, root_end, "input");
    const char *body = root;
    const char *body_end = root_end;
    if (input != NULL && *skip_ws(input) == '{') {
        body = json_object_span(input, &body_end);
        if (body == NULL) return fail(request, "input must be an object");
        const char *nested = json_key(body, body_end, "input");
        if (nested != NULL && *skip_ws(nested) == '{') {
            body = json_object_span(nested, &body_end);
            if (body == NULL) return fail(request, "input must be an object");
        }
    }
    char caption[MUSIC3_TEXT_SIZE] = {0}, lyrics[MUSIC3_LYRICS_SIZE] = {0};
    const char *field;
    if ((field = json_key(body, body_end, "instructions")) != NULL) json_string(field, caption, sizeof(caption));
    if (caption[0] == '\0' && (field = json_key(body, body_end, "caption")) != NULL) json_string(field, caption, sizeof(caption));
    if (caption[0] == '\0' && (field = json_key(body, body_end, "prompt")) != NULL) json_string(field, caption, sizeof(caption));
    if (caption[0] == '\0') return fail(request, "instructions or prompt is required");
    if ((field = json_key(body, body_end, "lyrics")) != NULL) json_string(field, lyrics, sizeof(lyrics));
    if (lyrics[0] == '\0' && (field = json_key(body, body_end, "input")) != NULL && *skip_ws(field) == '"')
        json_string(field, lyrics, sizeof(lyrics));
    long long duration = 30, frames = 0, seed = 0;
    int have_frames = 0;
    if ((field = json_key(body, body_end, "duration_seconds")) != NULL || (field = json_key(body, body_end, "duration")) != NULL) {
        if (json_long(field, &duration) != 0 || duration < 1 || duration > MUSIC3_MAX_DURATION_SECONDS)
            return fail(request, "duration_seconds must be between 1 and 360");
    }
    if ((field = json_key(body, body_end, "max_new_tokens")) != NULL) {
        if (json_long(field, &frames) != 0 || frames < 1 || frames > 9000)
            return fail(request, "max_new_tokens must be between 1 and 9000");
        have_frames = 1;
    }
    if (!have_frames) frames = duration * 25;
    if (frames < 1 || frames > 9000) return fail(request, "max_new_tokens must be between 1 and 9000");
    if ((field = json_key(body, body_end, "seed")) != NULL) {
        if (*skip_ws(field) == 't' || *skip_ws(field) == 'f') return fail(request, "seed must be a non-negative integer");
        if (json_long(field, &seed) != 0 || seed < 0) return fail(request, "seed must be a non-negative integer");
    }
    if ((field = json_key(body, body_end, "output_upload_url")) != NULL) json_string(field, request->output_upload_url, sizeof(request->output_upload_url));
    if ((field = json_key(body, body_end, "output_public_url")) != NULL) json_string(field, request->output_public_url, sizeof(request->output_public_url));
    if (request->output_upload_url[0] && strncmp(request->output_upload_url, "https://", 8) != 0)
        return fail(request, "output_upload_url must use https");
    if (request->output_public_url[0] && strncmp(request->output_public_url, "https://", 8) != 0)
        return fail(request, "output_public_url must use https");
    snprintf(request->instructions, sizeof(request->instructions), "%s", caption);
    if (lyrics[0] == '\0') {
        snprintf(request->lyrics, sizeof(request->lyrics), "[Intro]\n(instrumental)\n[Outro]\n(instrumental)");
        if (strstr(caption, "instrumental") == NULL && strstr(caption, "Instrumental") == NULL)
            snprintf(request->instructions, sizeof(request->instructions), "%s, instrumental, no vocals", caption);
    } else snprintf(request->lyrics, sizeof(request->lyrics), "%s", lyrics);
    request->duration_seconds = (int)duration;
    request->max_new_tokens = (int)frames;
    request->seed = seed;
    return 0;
}

static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t le16(const unsigned char *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }

int music3_wav_statistics(const unsigned char *audio, size_t length, Music3WavStats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (audio == NULL || length < 44 || memcmp(audio, "RIFF", 4) != 0 || memcmp(audio + 8, "WAVE", 4) != 0)
        return -1;
    size_t offset = 12;
    const unsigned char *data = NULL;
    size_t data_bytes = 0;
    unsigned channels = 0, sample_rate = 0, sample_width = 0;
    while (offset + 8 <= length) {
        unsigned chunk_size = le32(audio + offset + 4);
        if (offset + 8 + chunk_size > length) return -1;
        if (memcmp(audio + offset, "fmt ", 4) == 0) {
            if (chunk_size < 16) return -1;
            channels = le16(audio + offset + 10);
            sample_rate = le32(audio + offset + 12);
            sample_width = le16(audio + offset + 22) / 8;
        } else if (memcmp(audio + offset, "data", 4) == 0) {
            data = audio + offset + 8;
            data_bytes = chunk_size;
        }
        offset += 8 + chunk_size + (chunk_size & 1);
    }
    if (data == NULL || sample_width != 2 || channels < 1 || sample_rate == 0) return -1;
    size_t sample_count = data_bytes / 2;
    if (sample_count == 0 || sample_count % channels != 0) return -1;
    double sum = 0, sum_sq = 0, peak = 0;
    unsigned clipped = 0, silence = 0;
    double left_sum = 0, right_sum = 0, left_sq = 0, right_sq = 0, lr_sum = 0;
    unsigned frames = (unsigned)(sample_count / channels);
    for (size_t i = 0; i < sample_count; ++i) {
        int16_t raw = (int16_t)(data[i * 2] | (data[i * 2 + 1] << 8));
        double value = raw / 32768.0;
        sum += value;
        sum_sq += value * value;
        double abs_value = fabs(value);
        if (abs_value > peak) peak = abs_value;
        if (abs_value >= 32767.0 / 32768.0) ++clipped;
        if (abs_value < pow(10.0, -60.0 / 20.0)) ++silence;
        if (channels == 2) {
            if ((i % 2) == 0) { left_sum += value; left_sq += value * value; }
            else {
                right_sum += value; right_sq += value * value;
                double left = ((int16_t)(data[(i - 1) * 2] | (data[(i - 1) * 2 + 1] << 8))) / 32768.0;
                lr_sum += left * value;
            }
        }
    }
    double rms = sqrt(sum_sq / (double)sample_count);
    double peak_db = 20.0 * log10(peak > 1e-12 ? peak : 1e-12);
    double rms_db = 20.0 * log10(rms > 1e-12 ? rms : 1e-12);
    stats->sample_rate_hz = sample_rate;
    stats->channels = channels;
    stats->bit_depth = sample_width * 8;
    stats->frames = frames;
    stats->duration_seconds = (double)frames / (double)sample_rate;
    stats->bytes = length;
    music3_sha256_hex(audio, length, stats->sha256);
    stats->peak_dbfs = peak_db;
    stats->rms_dbfs = rms_db;
    stats->crest_factor_db = peak_db - rms_db;
    stats->dc_offset = sum / (double)sample_count;
    stats->clipped_samples = clipped;
    stats->clipped_percent = 100.0 * clipped / (double)sample_count;
    stats->digital_silence_percent = 100.0 * silence / (double)sample_count;
    if (channels == 2 && frames > 1) {
        double n = frames;
        double left_mean = left_sum / n, right_mean = right_sum / n;
        double left_var = left_sq / n - left_mean * left_mean;
        double right_var = right_sq / n - right_mean * right_mean;
        double cov = lr_sum / n - left_mean * right_mean;
        if (left_var > 0 && right_var > 0) {
            stats->has_stereo_correlation = 1;
            stats->stereo_correlation = cov / sqrt(left_var * right_var);
        }
    }
    return 0;
}

static int json_escape_into(char *out, size_t size, size_t *used, const char *text) {
    if (*used + 1 >= size) return -1;
    out[(*used)++] = '"';
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*used + 8 >= size) return -1;
        if (*p == '"' || *p == '\\') { out[(*used)++] = '\\'; out[(*used)++] = (char)*p; }
        else if (*p == '\n') { out[(*used)++] = '\\'; out[(*used)++] = 'n'; }
        else if (*p == '\r') { out[(*used)++] = '\\'; out[(*used)++] = 'r'; }
        else if (*p < 32) *used += (size_t)snprintf(out + *used, size - *used, "\\u%04x", *p);
        else out[(*used)++] = (char)*p;
    }
    if (*used + 1 >= size) return -1;
    out[(*used)++] = '"';
    return 0;
}

int music3_write_result_json(char *out, size_t size, const Music3Request *request,
                             const Music3WavStats *stats, const char *audio_url,
                             const char *inline_b64, const Music3Timings *timings) {
    double model_download_seconds = timings->model_download_seconds;
    double server_start_seconds = timings->server_start_seconds;
    double generation_seconds = timings->generation_seconds;
    double upload_seconds = timings->upload_seconds;
    double total_seconds = timings->total_seconds;
    double server_started_at = timings->server_started_at;
    /* GPU names are plain ASCII from the driver; keep the JSON simple. */
    const char *gpu_name = timings->gpu_name != NULL ? timings->gpu_name : "";
    size_t used = 0;
    double duration = stats->duration_seconds > 0.001 ? stats->duration_seconds : 0.001;
    int n = snprintf(out, size,
        "{\"route\":\"minimax-music3-local\",\"model\":\"MiniMaxAI/MiniMax-Music3\",\"seed\":%lld,"
        "\"requested_frames\":%d,\"metrics\":{\"sample_rate_hz\":%u,\"channels\":%u,\"bit_depth\":%u,"
        "\"frames\":%u,\"duration_seconds\":%.3f,\"bytes\":%zu,\"sha256\":",
        request->seed, request->max_new_tokens, stats->sample_rate_hz, stats->channels, stats->bit_depth,
        stats->frames, stats->duration_seconds, stats->bytes);
    if (n < 0 || (size_t)n >= size) return -1;
    used = (size_t)n;
    if (json_escape_into(out, size, &used, stats->sha256) != 0) return -1;
    n = snprintf(out + used, size - used,
        ",\"peak_dbfs\":%.3f,\"rms_dbfs\":%.3f,\"crest_factor_db\":%.3f,\"dc_offset\":%.7f,"
        "\"clipped_samples\":%u,\"clipped_percent\":%.6f,\"digital_silence_percent\":%.4f,"
        "\"stereo_correlation\":",
        stats->peak_dbfs, stats->rms_dbfs, stats->crest_factor_db, stats->dc_offset,
        stats->clipped_samples, stats->clipped_percent, stats->digital_silence_percent);
    if (n < 0 || used + (size_t)n >= size) return -1;
    used += (size_t)n;
    if (stats->has_stereo_correlation)
        n = snprintf(out + used, size - used, "%.4f", stats->stereo_correlation);
    else n = snprintf(out + used, size - used, "null");
    if (n < 0 || used + (size_t)n >= size) return -1;
    used += (size_t)n;
    n = snprintf(out + used, size - used,
        ",\"model_download_seconds\":%.3f,\"server_start_seconds\":%.3f,\"generation_seconds\":%.3f,"
        "\"realtime_factor\":%.3f,\"audio_seconds_per_compute_second\":%.3f,\"server_started_at\":%.3f,"
        "\"upload_seconds\":%.3f,\"total_seconds\":%.3f,\"prefetch_seconds\":%.3f,\"prefetch_gib\":%.3f,"
        "\"server_ready_before_job\":%s,\"gpu\":\"%s\",\"optimizations\":[\"backbone-cuda-graph\","
        "\"rvq-depth-cuda-graph\",\"compiled-dit-blocks\",\"compiled-dav-decoder\",\"batched-seeded-sampling\","
        "\"warm-start-thread\",\"parallel-weight-prefetch\"]}",
        model_download_seconds, server_start_seconds, generation_seconds,
        generation_seconds / duration, duration / (generation_seconds > 0.001 ? generation_seconds : 0.001),
        server_started_at, upload_seconds, total_seconds, timings->prefetch_seconds, timings->prefetch_gib,
        timings->server_ready_before_job ? "true" : "false", gpu_name);
    if (n < 0 || used + (size_t)n >= size) return -1;
    used += (size_t)n;
    if (audio_url && audio_url[0]) {
        n = snprintf(out + used, size - used, ",\"audio_url\":");
        if (n < 0 || used + (size_t)n >= size) return -1;
        used += (size_t)n;
        if (json_escape_into(out, size, &used, audio_url) != 0) return -1;
    } else if (inline_b64 && inline_b64[0]) {
        n = snprintf(out + used, size - used,
            ",\"outputs\":[{\"filename\":\"minimax-music3.wav\",\"content_type\":\"audio/wav\",\"data\":");
        if (n < 0 || used + (size_t)n >= size) return -1;
        used += (size_t)n;
        if (json_escape_into(out, size, &used, inline_b64) != 0) return -1;
        n = snprintf(out + used, size - used, "}]");
        if (n < 0 || used + (size_t)n >= size) return -1;
        used += (size_t)n;
    }
    if (used + 2 >= size) return -1;
    out[used++] = '}';
    out[used] = '\0';
    return 0;
}
