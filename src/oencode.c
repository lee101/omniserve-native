#include "oencode.h"

#include <libdeflate.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <webp/decode.h>
#include <webp/encode.h>
#ifndef OENC_NO_ZLIB
#include <zlib.h>
#endif

#define NFILTERS 5
#define PNG_MT_MIN_STRIP (128u << 10)

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

#ifndef OENC_NO_ZLIB
typedef struct {
    filter_job f;
    int level, last, failed;
    uint8_t *out;
    size_t n;
    uint32_t adler;
} strip_job;

static void *strip_worker(void *arg)
{
    strip_job *j = arg;
    j->failed = 1;
    if (filter_worker(&j->f)) return NULL;
    const uint8_t *in = j->f.filtered + (size_t)j->f.y0 * (j->f.stride + 1);
    size_t len = (size_t)(j->f.y1 - j->f.y0) * (j->f.stride + 1);
    z_stream z = {0};
    if (deflateInit2(&z, j->level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK) return NULL;
    size_t cap = deflateBound(&z, len) + 16;
    j->out = malloc(cap);
    if (!j->out) { deflateEnd(&z); return NULL; }
    z.next_in = (Bytef *)in;
    z.avail_in = (uInt)len;
    z.next_out = j->out;
    z.avail_out = (uInt)cap;
    int r = deflate(&z, j->last ? Z_FINISH : Z_SYNC_FLUSH);
    j->n = z.total_out;
    int done = j->last ? r == Z_STREAM_END : (r == Z_OK && z.avail_in == 0 && z.avail_out > 0);
    deflateEnd(&z);
    if (!done) return NULL;
    j->adler = libdeflate_adler32(1, in, len);
    j->failed = 0;
    return NULL;
}
#endif

/* Rows are filtered and deflated in independent strips (one per thread) that are joined
   with sync flushes into a single zlib stream. */
int oenc_png_mt(const uint8_t *px, int w, int h, int channels, int level, int threads,
                uint8_t **out, size_t *n)
{
#ifdef OENC_NO_ZLIB
    (void)px; (void)w; (void)h; (void)channels; (void)level; (void)threads; (void)out; (void)n;
    return -2;
#else
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
    if (level > 9) level = 9;
    size_t stride = (size_t)w * channels;
    size_t raw = (stride + 1) * (size_t)h;
    if (threads > 64) threads = 64;
    if ((size_t)threads > raw / PNG_MT_MIN_STRIP) threads = (int)(raw / PNG_MT_MIN_STRIP);
    if (threads > h) threads = h;
    if (threads < 1) threads = 1;
    if (raw / (size_t)threads > 0x3fffffffu) return -1;

    uint8_t *filtered = malloc(raw);
    if (!filtered) return -1;
    pthread_t tid[64];
    strip_job jobs[64];
    int started[64] = {0};
    for (int t = 0; t < threads; t++) {
        int y0 = (int)((long)h * t / threads), y1 = (int)((long)h * (t + 1) / threads);
        jobs[t] = (strip_job){{px, filtered, h, channels, stride, y0, y1}, level, t == threads - 1, 0, NULL, 0, 1};
        if (t == 0) continue;
        if (pthread_create(&tid[t], NULL, strip_worker, &jobs[t]) == 0) started[t] = 1;
        else strip_worker(&jobs[t]);
    }
    strip_worker(&jobs[0]);
    int failed = 0;
    size_t z = 2 + 4;
    for (int t = 0; t < threads; t++) {
        if (started[t] && pthread_join(tid[t], NULL)) failed = 1;
        failed |= jobs[t].failed;
        z += jobs[t].n;
    }
    free(filtered);
    uint8_t *buf = failed || z > 0xffffffffu ? NULL : malloc(8 + 25 + 12 + z + 12);
    if (!buf) {
        for (int t = 0; t < threads; t++) free(jobs[t].out);
        return -1;
    }
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    uint8_t *p = buf;
    memcpy(p, sig, 8);
    p += 8;
    uint8_t ihdr[13];
    put_be32(ihdr, (uint32_t)w);
    put_be32(ihdr + 4, (uint32_t)h);
    ihdr[8] = 8; ihdr[9] = (uint8_t)ctype; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    p = put_chunk(p, "IHDR", ihdr, 13);
    put_be32(p, (uint32_t)z);
    memcpy(p + 4, "IDAT", 4);
    uint8_t *q = p + 8;
    *q++ = 0x78;
    *q++ = 0x01;
    uint32_t adler = 1;
    for (int t = 0; t < threads; t++) {
        memcpy(q, jobs[t].out, jobs[t].n);
        q += jobs[t].n;
        free(jobs[t].out);
        size_t len = (size_t)(jobs[t].f.y1 - jobs[t].f.y0) * (stride + 1);
        adler = (uint32_t)adler32_combine(adler, jobs[t].adler, (z_off_t)len);
    }
    put_be32(q, adler);
    put_be32(q + 4, libdeflate_crc32(0, p + 4, z + 4));
    p = put_chunk(q + 8, "IEND", NULL, 0);
    *out = buf;
    *n = (size_t)(p - buf);
    return 0;
#endif
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

typedef struct {
    const uint8_t *rgba;
    int w, stride, blank, y0, y1, has_alpha, failed;
    WebPPicture *dst;
} import_job;

static void *import_worker(void *arg)
{
    import_job *j = arg;
    int rows = j->y1 - j->y0, w = j->w;
    const uint8_t *src = j->rgba + (size_t)j->y0 * j->stride;
    int stride = j->stride;
    uint8_t *tmp = NULL;
    j->failed = 1;
    if (j->blank) {
        tmp = malloc((size_t)rows * w * 4);
        if (!tmp) return NULL;
        for (int y = 0; y < rows; y++) {
            const uint8_t *s = src + (size_t)y * j->stride;
            uint8_t *d = tmp + (size_t)y * w * 4;
            memcpy(d, s, (size_t)w * 4);
            for (int x = 0; x < w; x++)
                if (!d[4 * x + 3]) memset(d + 4 * x, 0, 4);
        }
        src = tmp;
        stride = w * 4;
    }
    WebPPicture sp;
    if (!WebPPictureInit(&sp)) { free(tmp); return NULL; }
    sp.use_argb = 0;
    sp.width = w;
    sp.height = rows;
    int ok = WebPPictureImportRGBA(&sp, src, stride);
    free(tmp);
    if (!ok) { WebPPictureFree(&sp); return NULL; }
    WebPPicture *d = j->dst;
    int cw = (w + 1) / 2, crows = (rows + 1) / 2;
    j->has_alpha = (sp.colorspace & WEBP_CSP_ALPHA_BIT) && sp.a;
    for (int y = 0; y < rows; y++) {
        memcpy(d->y + (size_t)(j->y0 + y) * d->y_stride, sp.y + (size_t)y * sp.y_stride, w);
        if (j->has_alpha) memcpy(d->a + (size_t)(j->y0 + y) * d->a_stride, sp.a + (size_t)y * sp.a_stride, w);
        else memset(d->a + (size_t)(j->y0 + y) * d->a_stride, 0xff, w);
    }
    for (int y = 0; y < crows; y++) {
        memcpy(d->u + (size_t)(j->y0 / 2 + y) * d->uv_stride, sp.u + (size_t)y * sp.uv_stride, cw);
        memcpy(d->v + (size_t)(j->y0 / 2 + y) * d->uv_stride, sp.v + (size_t)y * sp.uv_stride, cw);
    }
    WebPPictureFree(&sp);
    j->failed = 0;
    return NULL;
}

typedef struct {
    const WebPConfig *cfg;
    WebPPicture *pic;
    int ok;
} encode_job;

static void *encode_worker(void *arg)
{
    encode_job *j = arg;
    j->ok = WebPEncode(j->cfg, j->pic);
    return NULL;
}

static const uint8_t *find_chunk(const sink *s, const char *tag, size_t *len)
{
    size_t p = 12;
    while (s->buf && p + 8 <= s->len) {
        size_t size = (size_t)s->buf[p + 4] | (size_t)s->buf[p + 5] << 8 | (size_t)s->buf[p + 6] << 16 | (size_t)s->buf[p + 7] << 24;
        if (p + 8 + size > s->len) return NULL;
        if (!memcmp(s->buf + p, tag, 4)) { *len = size; return s->buf + p + 8; }
        p += 8 + size + (size & 1);
    }
    return NULL;
}

static void put_le32(uint8_t *p, uint32_t v)
{
    p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24;
}

static uint8_t *put_riff_chunk(uint8_t *p, const char *tag, const uint8_t *data, size_t len)
{
    memcpy(p, tag, 4);
    put_le32(p + 4, (uint32_t)len);
    memcpy(p + 8, data, len);
    p += 8 + len;
    if (len & 1) *p++ = 0;
    return p;
}

static int finish_single(const WebPConfig *cfg, WebPPicture *view, WebPPicture *owner, sink *s,
                         uint8_t **out, size_t *n)
{
    int ok = WebPEncode(cfg, view);
    WebPPictureFree(owner);
    if (!ok) { free(s->buf); return -1; }
    *out = s->buf;
    *n = s->len;
    return 0;
}

/* Same pixels as oenc_webp (same bytes when alpha_method == method), with the RGBA -> YUVA
   conversion done in parallel strips and the alpha plane encoded as its own job. The colour
   and alpha bitstreams are independent: the alpha is encoded beside a flat dummy luma, so
   only its ALPH chunk is real work, and the two are muxed by hand. alpha_method (0..6, or
   -1 for `method`) trades alpha bytes for time; alpha stays lossless at every setting. */
int oenc_webp2(const uint8_t *rgba, int w, int h, int stride, float quality, int method, int alpha_method,
               int alpha_filter, int blank_transparent, int threads, uint8_t **out, size_t *n)
{
    if (!rgba || w <= 0 || h <= 0 || !out || !n) return -1;
    if (!stride) stride = w * 4;
    if (alpha_method < 0 || alpha_method > 6) alpha_method = method;
    WebPConfig cfg, acfg;
    if (!WebPConfigInit(&cfg)) return -1;
    cfg.quality = quality;
    cfg.method = method;
    cfg.exact = 0;
    cfg.alpha_filtering = alpha_filter;
    cfg.thread_level = threads > 1;
    acfg = cfg;
    acfg.method = alpha_method;
    acfg.exact = 1;
    if (!WebPValidateConfig(&cfg) || !WebPValidateConfig(&acfg)) return -1;

    WebPPicture pic, apic;
    if (!WebPPictureInit(&pic) || !WebPPictureInit(&apic)) return -1;
    pic.use_argb = 0;
    pic.colorspace = WEBP_YUV420A;
    pic.width = w;
    pic.height = h;
    if (!WebPPictureAlloc(&pic)) return -1;

    if (threads < 1) threads = 1;
    if (threads > 64) threads = 64;
    int strips = threads;
    if (strips > h / 64) strips = h / 64;
    if (strips < 1) strips = 1;
    pthread_t tid[64];
    import_job jobs[64];
    int started[64] = {0};
    for (int t = 0; t < strips; t++) {
        int y0 = (int)((long)h * t / strips) & ~1;
        int y1 = t == strips - 1 ? h : (int)((long)h * (t + 1) / strips) & ~1;
        jobs[t] = (import_job){rgba, w, stride, blank_transparent, y0, y1, 0, 1, &pic};
        if (t == 0) continue;
        if (pthread_create(&tid[t], NULL, import_worker, &jobs[t]) == 0) started[t] = 1;
        else import_worker(&jobs[t]);
    }
    import_worker(&jobs[0]);
    int failed = 0, has_alpha = 0;
    for (int t = 0; t < strips; t++) {
        if (started[t] && pthread_join(tid[t], NULL)) failed = 1;
        failed |= jobs[t].failed;
        has_alpha |= jobs[t].has_alpha;
    }
    if (failed) { WebPPictureFree(&pic); return -1; }

    sink s = {0}, as = {0};
    WebPPicture main_pic = pic;
    main_pic.writer = sink_write;
    main_pic.custom_ptr = &s;
    if (!has_alpha) {
        main_pic.colorspace = WEBP_YUV420;
        main_pic.a = NULL;
        return finish_single(&cfg, &main_pic, &pic, &s, out, n);
    }
    WebPCleanupTransparentArea(&pic);
    if (threads < 2) return finish_single(&cfg, &main_pic, &pic, &s, out, n);

    main_pic.colorspace = WEBP_YUV420;
    main_pic.a = NULL;
    apic.use_argb = 0;
    apic.colorspace = WEBP_YUV420A;
    apic.width = w;
    apic.height = h;
    if (!WebPPictureAlloc(&apic)) { WebPPictureFree(&pic); return -1; }
    for (int y = 0; y < h; y++) {
        memset(apic.y + (size_t)y * apic.y_stride, 128, w);
        memcpy(apic.a + (size_t)y * apic.a_stride, pic.a + (size_t)y * pic.a_stride, w);
    }
    for (int y = 0; y < (h + 1) / 2; y++) {
        memset(apic.u + (size_t)y * apic.uv_stride, 128, (w + 1) / 2);
        memset(apic.v + (size_t)y * apic.uv_stride, 128, (w + 1) / 2);
    }
    apic.writer = sink_write;
    apic.custom_ptr = &as;
    encode_job aj = {&acfg, &apic, 0};
    pthread_t at;
    int threaded = pthread_create(&at, NULL, encode_worker, &aj) == 0;
    if (!threaded) encode_worker(&aj);
    int ok = WebPEncode(&cfg, &main_pic);
    if (threaded && pthread_join(at, NULL)) aj.ok = 0;
    WebPPictureFree(&pic);
    WebPPictureFree(&apic);

    size_t alen = 0, vlen = 0;
    const uint8_t *alph = ok && aj.ok ? find_chunk(&as, "ALPH", &alen) : NULL;
    const uint8_t *vp8 = alph ? find_chunk(&s, "VP8 ", &vlen) : NULL;
    size_t total = 12 + 18 + 8 + alen + (alen & 1) + 8 + vlen + (vlen & 1);
    uint8_t *buf = vp8 && total <= 0xfffffff6u ? malloc(total) : NULL;
    if (buf) {
        memcpy(buf, "RIFF", 4);
        put_le32(buf + 4, (uint32_t)(total - 8));
        memcpy(buf + 8, "WEBP", 4);
        uint8_t x[10] = {0x10, 0, 0, 0, (uint8_t)(w - 1), (uint8_t)((w - 1) >> 8), (uint8_t)((w - 1) >> 16),
                         (uint8_t)(h - 1), (uint8_t)((h - 1) >> 8), (uint8_t)((h - 1) >> 16)};
        uint8_t *p = put_riff_chunk(buf + 12, "VP8X", x, 10);
        p = put_riff_chunk(p, "ALPH", alph, alen);
        put_riff_chunk(p, "VP8 ", vp8, vlen);
    }
    free(s.buf);
    free(as.buf);
    if (!buf) return -1;
    *out = buf;
    *n = total;
    return 0;
}

static uint32_t get_be32(const uint8_t *p)
{
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}

/* 8-bit truecolour, non-interlaced, nothing that changes interpretation (tRNS, eXIf, APNG,
   unknown critical chunks): everything else returns -2 and is left to the caller's decoder. */
int oenc_png_info(const uint8_t *data, size_t n, int *w, int *h, int *channels)
{
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    if (!data || n < 8 + 25 + 12 || memcmp(data, sig, 8) || get_be32(data + 8) != 13 || memcmp(data + 12, "IHDR", 4))
        return -1;
    const uint8_t *ihdr = data + 16;
    uint32_t pw = get_be32(ihdr), ph = get_be32(ihdr + 4);
    if (!pw || !ph || pw > 0x7fffffffu || ph > 0x7fffffffu) return -1;
    if (ihdr[8] != 8 || (ihdr[9] != 2 && ihdr[9] != 6) || ihdr[10] || ihdr[11] || ihdr[12]) return -2;
    size_t p = 8 + 25;
    int idat = 0;
    while (p + 12 <= n) {
        size_t len = get_be32(data + p);
        const uint8_t *type = data + p + 4;
        if (len > n - p - 12) return -1;
        if (!memcmp(type, "IEND", 4)) break;
        if (!memcmp(type, "IDAT", 4)) idat = 1;
        else if (!memcmp(type, "tRNS", 4) || !memcmp(type, "eXIf", 4) || !memcmp(type, "acTL", 4)) return -2;
        else if (!(type[0] & 0x20) && memcmp(type, "PLTE", 4)) return -2;
        p += 12 + len;
    }
    if (!idat) return -1;
    if (w) *w = (int)pw;
    if (h) *h = (int)ph;
    if (channels) *channels = ihdr[9] == 6 ? 4 : 3;
    return 0;
}

#ifdef __SSE2__
#include <emmintrin.h>

static inline __m128i abs16(__m128i x)
{
    return _mm_max_epi16(x, _mm_sub_epi16(_mm_setzero_si128(), x));
}

static inline __m128i pick(__m128i mask, __m128i yes, __m128i no)
{
    return _mm_or_si128(_mm_and_si128(mask, yes), _mm_andnot_si128(mask, no));
}

/* One pixel per step, its channels in parallel. Each step loads 4 bytes, so a 3-byte row
   leaves its last pixel to the scalar loop. Returns the number of bytes done. */
static size_t unfilter_paeth_simd(uint8_t *row, const uint8_t *prev, size_t n, int bpp)
{
    const __m128i zero = _mm_setzero_si128();
    __m128i a = zero, c = zero;
    size_t i = 0;
    for (; i + 4 <= n; i += bpp) {
        int32_t pv, xv;
        memcpy(&pv, prev + i, 4);
        memcpy(&xv, row + i, 4);
        __m128i b = _mm_unpacklo_epi8(_mm_cvtsi32_si128(pv), zero);
        __m128i x = _mm_unpacklo_epi8(_mm_cvtsi32_si128(xv), zero);
        __m128i pa = _mm_sub_epi16(b, c), pb = _mm_sub_epi16(a, c);
        __m128i pc = abs16(_mm_add_epi16(pa, pb));
        pa = abs16(pa);
        pb = abs16(pb);
        __m128i smallest = _mm_min_epi16(pc, _mm_min_epi16(pa, pb));
        __m128i nearest = pick(_mm_cmpeq_epi16(smallest, pa), a, pick(_mm_cmpeq_epi16(smallest, pb), b, c));
        a = _mm_and_si128(_mm_add_epi16(x, nearest), _mm_set1_epi16(0xff));
        c = b;
        int32_t out = _mm_cvtsi128_si32(_mm_packus_epi16(a, a));
        memcpy(row + i, &out, bpp);
    }
    return i;
}
#endif

static void unfilter_row(int type, uint8_t *row, const uint8_t *prev, size_t n, int bpp)
{
    size_t i = 0;
    switch (type) {
    case 1:
        for (i = bpp; i < n; i++) row[i] = (uint8_t)(row[i] + row[i - bpp]);
        break;
    case 2:
        for (; i < n; i++) row[i] = (uint8_t)(row[i] + prev[i]);
        break;
    case 3:
        for (; i < (size_t)bpp && i < n; i++) row[i] = (uint8_t)(row[i] + (prev[i] >> 1));
        for (; i < n; i++) row[i] = (uint8_t)(row[i] + ((row[i - bpp] + prev[i]) >> 1));
        break;
    case 4:
#ifdef __SSE2__
        i = unfilter_paeth_simd(row, prev, n, bpp);
#endif
        for (; i < (size_t)bpp && i < n; i++) row[i] = (uint8_t)(row[i] + prev[i]);
        for (; i < n; i++) row[i] = (uint8_t)(row[i] + paeth(row[i - bpp], prev[i], prev[i - bpp]));
        break;
    }
}

int oenc_png_decode(const uint8_t *data, size_t n, uint8_t *out, size_t out_size, int stride, int channels)
{
    int w, h, src;
    if (!out || (channels != 3 && channels != 4) || oenc_png_info(data, n, &w, &h, &src)) return -1;
    if (w > (1 << 24) || h > (1 << 24)) return -1;
    if (stride < w * channels || out_size < (size_t)stride * (h - 1) + (size_t)w * channels) return -1;
    size_t row = (size_t)w * src, raw = (row + 1) * (size_t)h, zlen = 0, p = 8 + 25;
    uint8_t *z = malloc(n), *buf = malloc(raw + row);
    struct libdeflate_decompressor *d = NULL;
    int rc = -1;
    if (!z || !buf) goto done;
    while (p + 12 <= n) {
        size_t len = get_be32(data + p);
        if (len > n - p - 12) goto done;
        if (!memcmp(data + p + 4, "IEND", 4)) break;
        if (!memcmp(data + p + 4, "IDAT", 4)) {
            if (libdeflate_crc32(0, data + p + 4, len + 4) != get_be32(data + p + 8 + len)) goto done;
            memcpy(z + zlen, data + p + 8, len);
            zlen += len;
        }
        p += 12 + len;
    }
    d = libdeflate_alloc_decompressor();
    if (!d) goto done;
    size_t got = 0;
    if (libdeflate_zlib_decompress(d, z, zlen, buf + row, raw, &got) != LIBDEFLATE_SUCCESS || got != raw) goto done;
    /* Rows are unfiltered in place behind their filter byte; a zero row in front serves the first. */
    memset(buf, 0, row);
    const uint8_t *prev = buf;
    for (int y = 0; y < h; y++) {
        uint8_t *cur = buf + row + (size_t)y * (row + 1);
        int type = *cur++;
        if (type > 4) goto done;
        unfilter_row(type, cur, prev, row, src);
        prev = cur;
        uint8_t *dst = out + (size_t)y * stride;
        if (src == channels) {
            memcpy(dst, cur, row);
        } else if (src == 4) {
            for (int x = 0; x < w; x++) memcpy(dst + 3 * x, cur + 4 * x, 3);
        } else {
            for (int x = 0; x < w; x++) {
                memcpy(dst + 4 * x, cur + 3 * x, 3);
                dst[4 * x + 3] = 0xff;
            }
        }
    }
    rc = 0;
done:
    if (d) libdeflate_free_decompressor(d);
    free(z);
    free(buf);
    return rc;
}

/* Still images only: an animation, or EXIF that may carry an orientation, is left to the caller. */
int oenc_webp_info(const uint8_t *data, size_t n, int *w, int *h, int *has_alpha)
{
    WebPBitstreamFeatures f;
    if (!data || n < 30 || WebPGetFeatures(data, n, &f) != VP8_STATUS_OK) return -1;
    if (f.has_animation) return -2;
    if (!memcmp(data + 12, "VP8X", 4) && (data[20] & 0x08)) return -2;
    if (w) *w = f.width;
    if (h) *h = f.height;
    if (has_alpha) *has_alpha = f.has_alpha;
    return 0;
}

int oenc_webp_decode(const uint8_t *data, size_t n, uint8_t *out, size_t out_size, int stride,
                     int channels, int threads)
{
    WebPDecoderConfig cfg;
    if (!data || !out || (channels != 3 && channels != 4) || !WebPInitDecoderConfig(&cfg)) return -1;
    cfg.options.use_threads = threads > 1;
    cfg.output.colorspace = channels == 4 ? MODE_RGBA : MODE_RGB;
    cfg.output.is_external_memory = 1;
    cfg.output.u.RGBA.rgba = out;
    cfg.output.u.RGBA.stride = stride;
    cfg.output.u.RGBA.size = out_size;
    return WebPDecode(data, n, &cfg) == VP8_STATUS_OK ? 0 : -1;
}

void oenc_free(void *p)
{
    free(p);
}
