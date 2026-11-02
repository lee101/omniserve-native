#include "oencode.h"

#include <libdeflate.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <webp/encode.h>

#define NFILTERS 5

typedef struct {
    const uint8_t *px;
    uint8_t *filtered;
    int h, bpp;
    size_t stride;
    int y0, y1;
} filter_job;

static inline int paeth(int a, int b, int c)
{
    int p = a + b - c;
    int pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
}

static void apply_filter(int type, const uint8_t *cur, const uint8_t *prev, int bpp, size_t n, uint8_t *dst)
{
    size_t i = 0;
    switch (type) {
    case 0:
        memcpy(dst, cur, n);
        break;
    case 1:
        for (; i < (size_t)bpp && i < n; i++) dst[i] = cur[i];
        for (; i < n; i++) dst[i] = (uint8_t)(cur[i] - cur[i - bpp]);
        break;
    case 2:
        for (; i < n; i++) dst[i] = (uint8_t)(cur[i] - prev[i]);
        break;
    case 3:
        for (; i < (size_t)bpp && i < n; i++) dst[i] = (uint8_t)(cur[i] - (prev[i] >> 1));
        for (; i < n; i++) dst[i] = (uint8_t)(cur[i] - ((cur[i - bpp] + prev[i]) >> 1));
        break;
    default:
        for (; i < (size_t)bpp && i < n; i++) dst[i] = (uint8_t)(cur[i] - prev[i]);
        for (; i < n; i++) dst[i] = (uint8_t)(cur[i] - paeth(cur[i - bpp], prev[i], prev[i - bpp]));
        break;
    }
}

static uint64_t score(const uint8_t *row, size_t n)
{
    uint64_t s = 0;
    for (size_t i = 0; i < n; i++) {
        int v = (int8_t)row[i];
        s += (uint64_t)(v < 0 ? -v : v);
    }
    return s;
}

static void *filter_worker(void *arg)
{
    filter_job *j = arg;
    size_t n = j->stride;
    uint8_t *trial = malloc(n * 2);
    if (!trial) return (void *)1;
    uint8_t *zero = calloc(1, n);
    if (!zero) { free(trial); return (void *)1; }
    uint8_t *cand = trial, *best = trial + n;
    for (int y = j->y0; y < j->y1; y++) {
        const uint8_t *cur = j->px + (size_t)y * n;
        const uint8_t *prev = y ? cur - n : zero;
        uint64_t best_score = UINT64_MAX;
        int best_type = 0;
        for (int t = 0; t < NFILTERS; t++) {
            apply_filter(t, cur, prev, j->bpp, n, cand);
            uint64_t s = score(cand, n);
            if (s < best_score) {
                best_score = s;
                best_type = t;
                uint8_t *tmp = best; best = cand; cand = tmp;
            }
        }
        uint8_t *dst = j->filtered + (size_t)y * (n + 1);
        dst[0] = (uint8_t)best_type;
        memcpy(dst + 1, best, n);
    }
    free(trial);
    free(zero);
    return NULL;
}

static void put_be32(uint8_t *p, uint32_t v)
{
    p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v;
}

static uint8_t *put_chunk(uint8_t *p, const char *type, const uint8_t *data, size_t len)
{
    put_be32(p, (uint32_t)len);
    memcpy(p + 4, type, 4);
    if (len) memcpy(p + 8, data, len);
    uint32_t crc = libdeflate_crc32(0, p + 4, len + 4);
    put_be32(p + 8 + len, crc);
    return p + 12 + len;
}

int oenc_png(const uint8_t *px, int w, int h, int channels, int level, int threads,
             uint8_t **out, size_t *n)
{
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    int ctype;
    switch (channels) {
    case 1: ctype = 0; break;
    case 2: ctype = 4; break;
    case 3: ctype = 2; break;
    case 4: ctype = 6; break;
    default: return -1;
    }
    if (!px || w <= 0 || h <= 0 || !out || !n) return -1;
    if (level < 1) level = 1;
    if (level > 12) level = 12;
    if (threads < 1) threads = 1;
    if (threads > h) threads = h;
    if (threads > 64) threads = 64;

    size_t stride = (size_t)w * channels;
    uint8_t *filtered = malloc((stride + 1) * (size_t)h);
    if (!filtered) return -1;

    pthread_t tid[64];
    filter_job jobs[64];
    int started[64] = {0};
    int failed = 0;
    for (int t = 0; t < threads; t++) {
        jobs[t] = (filter_job){px, filtered, h, channels, stride, (int)((long)h * t / threads), (int)((long)h * (t + 1) / threads)};
        if (t == 0) continue;
        if (pthread_create(&tid[t], NULL, filter_worker, &jobs[t]) == 0) started[t] = 1;
        else if (filter_worker(&jobs[t])) failed = 1;
    }
    if (filter_worker(&jobs[0])) failed = 1;
    for (int t = 1; t < threads; t++) {
        void *r;
        if (started[t] && (pthread_join(tid[t], &r) || r)) failed = 1;
    }
    if (failed) { free(filtered); return -1; }

    struct libdeflate_compressor *c = libdeflate_alloc_compressor(level);
    if (!c) { free(filtered); return -1; }
    size_t raw = (stride + 1) * (size_t)h;
    size_t bound = libdeflate_zlib_compress_bound(c, raw);
    uint8_t *buf = malloc(8 + 25 + 12 + bound + 12);
    if (!buf) { libdeflate_free_compressor(c); free(filtered); return -1; }

    uint8_t *p = buf;
    memcpy(p, sig, 8);
    p += 8;
    uint8_t ihdr[13];
    put_be32(ihdr, (uint32_t)w);
    put_be32(ihdr + 4, (uint32_t)h);
    ihdr[8] = 8; ihdr[9] = (uint8_t)ctype; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    p = put_chunk(p, "IHDR", ihdr, 13);

    uint8_t *idat = p + 8;
    size_t z = libdeflate_zlib_compress(c, filtered, raw, idat, bound);
    libdeflate_free_compressor(c);
    free(filtered);
    if (!z) { free(buf); return -1; }
    put_be32(p, (uint32_t)z);
    memcpy(p + 4, "IDAT", 4);
    put_be32(idat + z, libdeflate_crc32(0, p + 4, z + 4));
    p = idat + z + 4;
    p = put_chunk(p, "IEND", NULL, 0);

    *out = buf;
    *n = (size_t)(p - buf);
    return 0;
}

typedef struct {
    uint8_t *buf;
    size_t len, cap;
} sink;

static int sink_write(const uint8_t *data, size_t size, const WebPPicture *pic)
{
    sink *s = pic->custom_ptr;
    if (s->len + size > s->cap) {
        size_t cap = s->cap ? s->cap : 65536;
        while (cap < s->len + size) cap *= 2;
        uint8_t *nb = realloc(s->buf, cap);
        if (!nb) return 0;
        s->buf = nb;
        s->cap = cap;
    }
    memcpy(s->buf + s->len, data, size);
    s->len += size;
    return 1;
}

int oenc_webp(const uint8_t *rgba, int w, int h, int stride, float quality, int method,
              int alpha_filter, int blank_transparent, int threads, uint8_t **out, size_t *n)
{
    if (!rgba || w <= 0 || h <= 0 || !out || !n) return -1;
    WebPConfig cfg;
    if (!WebPConfigInit(&cfg)) return -1;
    cfg.quality = quality;
    cfg.method = method;
    cfg.exact = 0;
    cfg.alpha_filtering = alpha_filter;
    cfg.thread_level = threads > 1;
    if (!WebPValidateConfig(&cfg)) return -1;

    WebPPicture pic;
    if (!WebPPictureInit(&pic)) return -1;
    pic.use_argb = 1;
    pic.width = w;
    pic.height = h;
    sink s = {0};
    pic.writer = sink_write;
    pic.custom_ptr = &s;
    if (!WebPPictureImportRGBA(&pic, rgba, stride ? stride : w * 4)) { WebPPictureFree(&pic); return -1; }
    if (blank_transparent) {
        for (int y = 0; y < h; y++) {
            uint32_t *row = pic.argb + (size_t)y * pic.argb_stride;
            for (int x = 0; x < w; x++)
                if (!(row[x] >> 24)) row[x] = 0;
        }
    }
    int ok = WebPEncode(&cfg, &pic);
    WebPPictureFree(&pic);
    if (!ok) { free(s.buf); return -1; }
    *out = s.buf;
    *n = s.len;
    return 0;
}

void oenc_free(void *p)
{
    free(p);
}
