#define _GNU_SOURCE
#include "oupscale.h"
#include "ojson.h"

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>

static void set_err(char *err, size_t cap, const char *msg) {
    if (err && cap) snprintf(err, cap, "%s", msg);
}

void oup_req_free(oup_req *req) {
    if (!req) return;
    free(req->image_url);
    req->image_url = NULL;
}

const char *oup_model_name(oup_model m) { return m == OUP_ANIME ? "anime" : "general"; }

const char *oup_format_mime(oup_format f) {
    return f == OUP_WEBP ? "image/webp" : f == OUP_JPEG ? "image/jpeg" : "image/png";
}

bool oup_parse(const char *json, size_t len, oup_req *req, char *err, size_t err_cap) {
    memset(req, 0, sizeof *req);
    req->scale = 2;
    oj_tok toks[64];
    int n = json ? oj_parse(json, len, toks, 64) : 0;
    if (n <= 0 || toks[0].type != OJ_OBJECT) {
        set_err(err, err_cap, "invalid JSON body");
        return false;
    }
    int t = oj_obj_get(json, toks, n, 0, "image_url");
    if (t < 0 || toks[t].type != OJ_STRING || toks[t].end <= toks[t].start) {
        set_err(err, err_cap, "image_url required");
        return false;
    }
    req->image_url = oj_strdup(json, &toks[t]);
    if (!req->image_url) {
        set_err(err, err_cap, "allocation failed");
        return false;
    }
    if (strncmp(req->image_url, "data:image/", 11) != 0 && !oup_url_https(req->image_url)) {
        set_err(err, err_cap, "image_url must be an https URL or a data:image/...;base64 URL");
        goto fail;
    }
    t = oj_obj_get(json, toks, n, 0, "scale");
    if (t >= 0 && !(toks[t].type == OJ_PRIMITIVE && json[toks[t].start] == 'n')) {
        double s = toks[t].type == OJ_PRIMITIVE || toks[t].type == OJ_STRING ? oj_number(json, &toks[t], -1) : -1;
        if (s != 2 && s != 4) {
            set_err(err, err_cap, "scale must be 2 or 4");
            goto fail;
        }
        req->scale = (int)s;
    }
    t = oj_obj_get(json, toks, n, 0, "model");
    if (t >= 0 && toks[t].type == OJ_STRING) {
        if (oj_str_eq(json, &toks[t], "anime")) req->model = OUP_ANIME;
        else if (!oj_str_eq(json, &toks[t], "general")) {
            set_err(err, err_cap, "model must be general or anime");
            goto fail;
        }
    } else if (t >= 0 && !(toks[t].type == OJ_PRIMITIVE && json[toks[t].start] == 'n')) {
        set_err(err, err_cap, "model must be general or anime");
        goto fail;
    }
    t = oj_obj_get(json, toks, n, 0, "output_format");
    if (t >= 0 && toks[t].type == OJ_STRING) {
        if (oj_str_eq(json, &toks[t], "webp")) req->format = OUP_WEBP;
        else if (oj_str_eq(json, &toks[t], "jpeg") || oj_str_eq(json, &toks[t], "jpg")) req->format = OUP_JPEG;
        else if (!oj_str_eq(json, &toks[t], "png")) {
            set_err(err, err_cap, "output_format must be png, webp or jpeg");
            goto fail;
        }
    } else if (t >= 0 && !(toks[t].type == OJ_PRIMITIVE && json[toks[t].start] == 'n')) {
        set_err(err, err_cap, "output_format must be png, webp or jpeg");
        goto fail;
    }
    return true;
fail:
    oup_req_free(req);
    return false;
}

int oup_check_limits(int w, int h, int scale, char *err, size_t err_cap) {
    if (w <= 0 || h <= 0) {
        set_err(err, err_cap, "image has no pixels");
        return 400;
    }
    long long px = (long long)w * h;
    long long cap = scale == 4 ? OUP_MAX_IN_PX_X4 : OUP_MAX_IN_PX_X2;
    if (px > cap) {
        if (err && err_cap)
            snprintf(err, err_cap, "input %dx%d exceeds %lld pixels for scale %d", w, h, cap, scale);
        return 413;
    }
    if ((long long)w * scale > OUP_MAX_OUT_SIDE || (long long)h * scale > OUP_MAX_OUT_SIDE) {
        if (err && err_cap)
            snprintf(err, err_cap, "output side would exceed %d px", OUP_MAX_OUT_SIDE);
        return 413;
    }
    return 0;
}

bool oup_url_https(const char *url) {
    if (!url || strncasecmp(url, "https://", 8) != 0) return false;
    const char *host = url + 8;
    if (!*host || *host == '/' || *host == '?' || *host == '#') return false;
    for (const char *p = url; *p; ++p)
        if ((unsigned char)*p <= 0x20 || *p == '\\' || (unsigned char)*p == 0x7f) return false;
    const char *end = host + strcspn(host, "/?#");
    return !memchr(host, '@', (size_t)(end - host));
}

static bool v4_public(uint32_t a) {
    uint8_t b0 = a >> 24, b1 = (a >> 16) & 0xff, b2 = (a >> 8) & 0xff;
    if (b0 == 0 || b0 == 10 || b0 == 127 || b0 >= 224) return false;
    if (b0 == 100 && (b1 & 0xc0) == 64) return false;
    if (b0 == 169 && b1 == 254) return false;
    if (b0 == 172 && (b1 & 0xf0) == 16) return false;
    if (b0 == 192 && b1 == 168) return false;
    if (b0 == 192 && b1 == 0 && (b2 == 0 || b2 == 2)) return false;
    if (b0 == 198 && (b1 & 0xfe) == 18) return false;
    if (b0 == 198 && b1 == 51 && b2 == 100) return false;
    if (b0 == 203 && b1 == 0 && b2 == 113) return false;
    return true;
}

bool oup_ip_public(const struct sockaddr *sa) {
    if (!sa) return false;
    if (sa->sa_family == AF_INET)
        return v4_public(ntohl(((const struct sockaddr_in *)sa)->sin_addr.s_addr));
    if (sa->sa_family != AF_INET6) return false;
    const uint8_t *b = ((const struct sockaddr_in6 *)sa)->sin6_addr.s6_addr;
    static const uint8_t mapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    static const uint8_t nat64[12] = {0, 0x64, 0xff, 0x9b, 0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tail = ((uint32_t)b[12] << 24) | ((uint32_t)b[13] << 16) | ((uint32_t)b[14] << 8) | b[15];
    if (!memcmp(b, mapped, 12) || !memcmp(b, nat64, 12)) return v4_public(tail);
    if (b[0] == 0x20 && b[1] == 0x02)
        return v4_public(((uint32_t)b[2] << 24) | ((uint32_t)b[3] << 16) | ((uint32_t)b[4] << 8) | b[5]);
    if ((b[0] & 0xe0) != 0x20) return false;
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x0d && b[3] == 0xb8) return false;
    return true;
}

bool oup_probe_dims(const unsigned char *b, size_t n, int *w, int *h) {
    *w = *h = 0;
    if (n >= 24 && !memcmp(b, "\x89PNG\r\n\x1a\n", 8) && !memcmp(b + 12, "IHDR", 4)) {
        uint32_t pw = ((uint32_t)b[16] << 24) | ((uint32_t)b[17] << 16) | ((uint32_t)b[18] << 8) | b[19];
        uint32_t ph = ((uint32_t)b[20] << 24) | ((uint32_t)b[21] << 16) | ((uint32_t)b[22] << 8) | b[23];
        if (!pw || !ph || pw > 1u << 20 || ph > 1u << 20) return false;
        *w = (int)pw;
        *h = (int)ph;
        return true;
    }
    if (n >= 30 && !memcmp(b, "RIFF", 4) && !memcmp(b + 8, "WEBP", 4)) {
        if (!memcmp(b + 12, "VP8X", 4)) {
            *w = 1 + (b[24] | b[25] << 8 | b[26] << 16);
            *h = 1 + (b[27] | b[28] << 8 | b[29] << 16);
        } else if (!memcmp(b + 12, "VP8 ", 4) && b[23] == 0x9d && b[24] == 0x01 && b[25] == 0x2a) {
            *w = (b[26] | b[27] << 8) & 0x3fff;
            *h = (b[28] | b[29] << 8) & 0x3fff;
        } else if (!memcmp(b + 12, "VP8L", 4) && b[20] == 0x2f) {
            uint32_t v = b[21] | (uint32_t)b[22] << 8 | (uint32_t)b[23] << 16 | (uint32_t)b[24] << 24;
            *w = 1 + (int)(v & 0x3fff);
            *h = 1 + (int)((v >> 14) & 0x3fff);
        }
        return *w > 0 && *h > 0;
    }
    if (n >= 4 && b[0] == 0xff && b[1] == 0xd8) {
        size_t i = 2;
        while (i + 9 < n) {
            if (b[i] != 0xff) return false;
            uint8_t m = b[i + 1];
            if (m == 0xff) { i++; continue; }
            if (m == 0xd8 || m == 0x01 || (m >= 0xd0 && m <= 0xd7)) { i += 2; continue; }
            size_t seg = ((size_t)b[i + 2] << 8) | b[i + 3];
            if (seg < 2) return false;
            if (m >= 0xc0 && m <= 0xcf && m != 0xc4 && m != 0xc8 && m != 0xcc) {
                *h = (b[i + 5] << 8) | b[i + 6];
                *w = (b[i + 7] << 8) | b[i + 8];
                return *w > 0 && *h > 0;
            }
            i += 2 + seg;
        }
    }
    return false;
}

static signed char b64v(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return (signed char)(c - 'A');
    if (c >= 'a' && c <= 'z') return (signed char)(c - 'a' + 26);
    if (c >= '0' && c <= '9') return (signed char)(c - '0' + 52);
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

bool oup_data_url_decode(const char *url, unsigned char **out, size_t *out_len) {
    *out = NULL;
    *out_len = 0;
    if (strncmp(url, "data:image/", 11) != 0) return false;
    const char *comma = strchr(url, ',');
    if (!comma) return false;
    size_t hl = (size_t)(comma - url);
    if (hl < 7 || memcmp(comma - 7, ";base64", 7) != 0) return false;
    const char *s = comma + 1;
    size_t n = strlen(s);
    if (!n || n / 4 * 3 > OUP_MAX_FETCH_BYTES) return false;
    unsigned char *buf = malloc(n / 4 * 3 + 3);
    if (!buf) return false;
    size_t o = 0;
    uint32_t acc = 0;
    int bits = 0, pad = 0;
    for (size_t i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '=') { pad++; continue; }
        if (c == '\r' || c == '\n' || c == ' ') continue;
        signed char v = b64v(c);
        if (v < 0 || pad) { free(buf); return false; }
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf[o++] = (unsigned char)(acc >> bits);
        }
    }
    if (!o || pad > 2) { free(buf); return false; }
    *out = buf;
    *out_len = o;
    return true;
}

/* libcurl is dlopened like libwebp: the gateway has no TLS of its own and
 * builds without the -dev headers. ABI-stable option numbers from curl.h. */
enum {
    C_WRITEDATA = 10001, C_URL = 10002, C_PROXY = 10004, C_USERAGENT = 10018,
    C_WRITEFUNCTION = 20011, C_FAILONERROR = 45, C_FOLLOWLOCATION = 52,
    C_MAXREDIRS = 68, C_NOSIGNAL = 99, C_TIMEOUT_MS = 155, C_CONNECTTIMEOUT_MS = 156,
    C_PROTOCOLS = 181, C_REDIR_PROTOCOLS = 182, C_OPENSOCKETFUNCTION = 20163,
    C_OPENSOCKETDATA = 10164, C_MAXFILESIZE_LARGE = 30117,
    CI_RESPONSE_CODE = 0x200002,
    CPROTO_HTTPS = 2,
};

typedef void CURL;
static CURL *(*c_init)(void);
static int (*c_setopt)(CURL *, int, ...);
static int (*c_perform)(CURL *);
static void (*c_cleanup)(CURL *);
static int (*c_getinfo)(CURL *, int, ...);
static const char *(*c_strerror)(int);
static int (*c_global_init)(long);
static pthread_once_t c_once = PTHREAD_ONCE_INIT;
static bool c_ok;

static void curl_load(void) {
    static const char *const libs[] = {"libcurl.so.4", "libcurl-gnutls.so.4", "libcurl.so", NULL};
    for (int i = 0; libs[i]; ++i) {
        void *h = dlopen(libs[i], RTLD_NOW | RTLD_LOCAL);
        if (!h) continue;
        *(void **)&c_init = dlsym(h, "curl_easy_init");
        *(void **)&c_setopt = dlsym(h, "curl_easy_setopt");
        *(void **)&c_perform = dlsym(h, "curl_easy_perform");
        *(void **)&c_cleanup = dlsym(h, "curl_easy_cleanup");
        *(void **)&c_getinfo = dlsym(h, "curl_easy_getinfo");
        *(void **)&c_strerror = dlsym(h, "curl_easy_strerror");
        *(void **)&c_global_init = dlsym(h, "curl_global_init");
        if (c_init && c_setopt && c_perform && c_cleanup && c_getinfo && c_global_init) {
            c_global_init(3);
            c_ok = true;
            return;
        }
        dlclose(h);
    }
}

typedef struct {
    unsigned char *buf;
    size_t len, cap;
    bool overflow;
} fetch_sink;

static size_t fetch_write(char *p, size_t sz, size_t n, void *ud) {
    fetch_sink *s = ud;
    size_t add = sz * n;
    if (s->len + add > OUP_MAX_FETCH_BYTES) {
        s->overflow = true;
        return 0;
    }
    if (s->len + add > s->cap) {
        size_t nc = s->cap ? s->cap : 256u << 10;
        while (nc < s->len + add) nc *= 2;
        unsigned char *g = realloc(s->buf, nc);
        if (!g) return 0;
        s->buf = g;
        s->cap = nc;
    }
    memcpy(s->buf + s->len, p, add);
    s->len += add;
    return add;
}

struct oup_curl_sockaddr {
    int family, socktype, protocol;
    unsigned int addrlen;
    struct sockaddr_storage addr;
};

static int fetch_opensocket(void *ud, int purpose, struct oup_curl_sockaddr *a) {
    (void)purpose;
    int *blocked = ud;
    if (!oup_ip_public((const struct sockaddr *)&a->addr)) {
        *blocked = 1;
        return -1;
    }
    return socket(a->family, a->socktype, a->protocol);
}

int oup_fetch(const char *url, unsigned char **out, size_t *out_len, char *err, size_t err_cap) {
    *out = NULL;
    *out_len = 0;
    if (strncmp(url, "data:", 5) == 0) {
        if (oup_data_url_decode(url, out, out_len)) return 0;
        set_err(err, err_cap, "image_url data URL is not valid base64 image data");
        return 400;
    }
    if (!oup_url_https(url)) {
        set_err(err, err_cap, "image_url must be https");
        return 400;
    }
    pthread_once(&c_once, curl_load);
    if (!c_ok) {
        set_err(err, err_cap, "https fetch unavailable (libcurl missing); send a data URL");
        return 503;
    }
    CURL *c = c_init();
    if (!c) {
        set_err(err, err_cap, "fetch init failed");
        return 500;
    }
    fetch_sink sink = {0};
    int blocked = 0;
    c_setopt(c, C_URL, url);
    c_setopt(c, C_PROXY, "");
    c_setopt(c, C_PROTOCOLS, (long)CPROTO_HTTPS);
    c_setopt(c, C_REDIR_PROTOCOLS, (long)CPROTO_HTTPS);
    c_setopt(c, C_FOLLOWLOCATION, 1L);
    c_setopt(c, C_MAXREDIRS, 3L);
    c_setopt(c, C_NOSIGNAL, 1L);
    c_setopt(c, C_FAILONERROR, 1L);
    c_setopt(c, C_CONNECTTIMEOUT_MS, 5000L);
    c_setopt(c, C_TIMEOUT_MS, 20000L);
    c_setopt(c, C_MAXFILESIZE_LARGE, (long long)OUP_MAX_FETCH_BYTES);
    c_setopt(c, C_USERAGENT, "omniserve-native-upscale/1");
    c_setopt(c, C_WRITEFUNCTION, fetch_write);
    c_setopt(c, C_WRITEDATA, &sink);
    c_setopt(c, C_OPENSOCKETFUNCTION, fetch_opensocket);
    c_setopt(c, C_OPENSOCKETDATA, &blocked);
    int rc = c_perform(c);
    long code = 0;
    c_getinfo(c, CI_RESPONSE_CODE, &code);
    c_cleanup(c);
    if (rc != 0 || !sink.len) {
        free(sink.buf);
        if (blocked) set_err(err, err_cap, "image_url resolves to a non-public address");
        else if (sink.overflow || rc == 63) set_err(err, err_cap, "image_url body too large");
        else if (err && err_cap)
            snprintf(err, err_cap, "image_url fetch failed (%s, http %ld)",
                     c_strerror ? c_strerror(rc) : "error", code);
        return sink.overflow || rc == 63 ? 413 : 400;
    }
    *out = sink.buf;
    *out_len = sink.len;
    return 0;
}

void oup_box_down2(const unsigned char *src, int w, int h, int ch, unsigned char *dst) {
    int ow = w / 2, oh = h / 2;
    size_t rs = (size_t)w * ch;
    for (int y = 0; y < oh; ++y) {
        const unsigned char *r0 = src + (size_t)(2 * y) * rs, *r1 = r0 + rs;
        unsigned char *d = dst + (size_t)y * ow * ch;
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < ch; ++c) {
                size_t i = (size_t)(2 * x) * ch + c;
                d[(size_t)x * ch + c] = (unsigned char)((r0[i] + r0[i + ch] + r1[i] + r1[i + ch] + 2) >> 2);
            }
    }
}

void oup_resize_plane_bilinear(const unsigned char *src, int sw, int sh, int sstride,
                               unsigned char *dst, int dw, int dh, int dstride) {
    float fx = (float)sw / dw, fy = (float)sh / dh;
    for (int y = 0; y < dh; ++y) {
        float sy = (y + 0.5f) * fy - 0.5f;
        if (sy < 0) sy = 0;
        int y0 = (int)sy;
        if (y0 > sh - 1) y0 = sh - 1;
        int y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
        float wy = sy - y0;
        const unsigned char *r0 = src + (size_t)y0 * sw * sstride, *r1 = src + (size_t)y1 * sw * sstride;
        unsigned char *d = dst + (size_t)y * dw * dstride;
        for (int x = 0; x < dw; ++x) {
            float sx = (x + 0.5f) * fx - 0.5f;
            if (sx < 0) sx = 0;
            int x0 = (int)sx;
            if (x0 > sw - 1) x0 = sw - 1;
            int x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
            float wx = sx - x0;
            float top = r0[(size_t)x0 * sstride] + (r0[(size_t)x1 * sstride] - r0[(size_t)x0 * sstride]) * wx;
            float bot = r1[(size_t)x0 * sstride] + (r1[(size_t)x1 * sstride] - r1[(size_t)x0 * sstride]) * wx;
            float v = top + (bot - top) * wy + 0.5f;
            d[(size_t)x * dstride] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
}
