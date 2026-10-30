#define _POSIX_C_SOURCE 200809L
#include "cbed.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* small open-addressing string->int32 hash map (the WordPiece vocab)  */
/* ------------------------------------------------------------------ */

typedef struct {
    char **keys;      /* owned */
    int32_t *vals;
    size_t cap;       /* power of two */
    size_t len;
} vocab_map;

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void vm_init(vocab_map *v, size_t cap) {
    v->cap = 16;
    while (v->cap < cap * 2) v->cap <<= 1;
    v->keys = calloc(v->cap, sizeof(char *));
    v->vals = calloc(v->cap, sizeof(int32_t));
    v->len = 0;
}

static void vm_free(vocab_map *v) {
    for (size_t i = 0; i < v->cap; i++) free(v->keys[i]);
    free(v->keys); free(v->vals);
}

static void vm_put(vocab_map *v, const char *key, size_t klen, int32_t val) {
    size_t mask = v->cap - 1;
    size_t i = fnv1a(key, klen) & mask;
    while (v->keys[i]) {
        if (strncmp(v->keys[i], key, klen) == 0 && v->keys[i][klen] == 0) { v->vals[i] = val; return; }
        i = (i + 1) & mask;
    }
    v->keys[i] = malloc(klen + 1);
    memcpy(v->keys[i], key, klen); v->keys[i][klen] = 0;
    v->vals[i] = val; v->len++;
}

/* returns val or -1 */
static int32_t vm_get(const vocab_map *v, const char *key, size_t klen) {
    size_t mask = v->cap - 1;
    size_t i = fnv1a(key, klen) & mask;
    while (v->keys[i]) {
        if (strncmp(v->keys[i], key, klen) == 0 && v->keys[i][klen] == 0) return v->vals[i];
        i = (i + 1) & mask;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* model                                                               */
/* ------------------------------------------------------------------ */

struct cbed_model {
    int8_t *weights;   /* [vocab*dim] */
    float *scales;     /* [vocab] */
    int vocab_size;
    int dim;
    vocab_map vocab;
    int32_t cls_id, sep_id, unk_id;
};

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[n] = 0; fclose(f);
    if (len_out) *len_out = n;
    return buf;
}

/* minimal: find "name" then its "shape":[a,b] / "data_offsets":[s,e] */
static long find_i64_after(const char *h, const char *key) {
    const char *p = strstr(h, key);
    if (!p) return -1;
    p += strlen(key);
    while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
    return strtol(p, NULL, 10);
}

static int load_safetensors(cbed_model *m, const char *path) {
    size_t len; char *raw = read_file(path, &len);
    if (!raw) return -1;
    uint64_t hlen; memcpy(&hlen, raw, 8);
    char *header = malloc(hlen + 1);
    memcpy(header, raw + 8, hlen); header[hlen] = 0;
    size_t data_base = 8 + hlen;

    /* weight tensor entry */
    char *w = strstr(header, "embeddings.weight");
    char *s = strstr(header, "embeddings.scales");
    if (!w || !s) { free(raw); free(header); return -1; }

    long vocab = find_i64_after(w, "\"shape\":[");
    /* second shape element = dim */
    const char *shp = strstr(w, "\"shape\":[");
    long dim = strtol(strchr(shp + 9, ',') + 1, NULL, 10);
    long w_off0 = find_i64_after(w, "\"data_offsets\":[");
    const char *woff = strstr(w, "\"data_offsets\":[");
    long w_off1 = strtol(strchr(woff + 16, ',') + 1, NULL, 10);
    long s_off0 = find_i64_after(s, "\"data_offsets\":[");
    const char *soff = strstr(s, "\"data_offsets\":[");
    long s_off1 = strtol(strchr(soff + 16, ',') + 1, NULL, 10);

    m->vocab_size = (int)vocab; m->dim = (int)dim;
    size_t wbytes = (size_t)(w_off1 - w_off0);
    size_t sbytes = (size_t)(s_off1 - s_off0);
    m->weights = malloc(wbytes);
    m->scales = malloc(sbytes);
    memcpy(m->weights, raw + data_base + w_off0, wbytes);
    memcpy(m->scales, raw + data_base + s_off0, sbytes);

    free(raw); free(header);
    return 0;
}

/* Parse just "vocab":{ "tok":id, ... } from tokenizer.json without a full
 * JSON parser. Handles \" \\ \/ escapes in token strings (enough for BERT). */
static int load_vocab(cbed_model *m, const char *path) {
    char *raw = read_file(path, NULL);
    if (!raw) return -1;
    char *p = strstr(raw, "\"vocab\"");
    if (!p) { free(raw); return -1; }
    p = strchr(p, '{');
    if (!p) { free(raw); return -1; }
    p++;
    vm_init(&m->vocab, 32768);
    char tok[512];
    while (*p) {
        while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        if (*p == '}') break;
        if (*p != '"') break;
        p++;
        size_t tl = 0;
        while (*p && *p != '"') {
            if (*p == '\\') {
                p++;
                if (*p == 'u') { /* \uXXXX -> keep raw bytes, rare in BERT vocab */
                    tok[tl++] = '?'; p += 4; p++; continue;
                }
                if (*p == 'n') tok[tl++] = '\n';
                else tok[tl++] = *p;   /* \" \\ \/ */
                p++;
            } else {
                if (tl < sizeof(tok) - 1) tok[tl++] = *p;
                p++;
            }
        }
        tok[tl] = 0;
        if (*p == '"') p++;
        while (*p == ' ' || *p == ':') p++;
        int32_t id = (int32_t)strtol(p, &p, 10);
        if (id >= 0 && id < 32768) vm_put(&m->vocab, tok, tl, id);
    }
    free(raw);
    m->cls_id = vm_get(&m->vocab, "[CLS]", 5);
    m->sep_id = vm_get(&m->vocab, "[SEP]", 5);
    m->unk_id = vm_get(&m->vocab, "[UNK]", 5);
    return 0;
}

cbed_model *cbed_load(const char *model_dir) {
    cbed_model *m = calloc(1, sizeof(*m));
    char path[2048];
    snprintf(path, sizeof(path), "%s/modelint8_512dim.safetensors", model_dir);
    if (load_safetensors(m, path) != 0) { free(m); return NULL; }
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
    if (load_vocab(m, path) != 0) { cbed_free(m); return NULL; }
    return m;
}

void cbed_free(cbed_model *m) {
    if (!m) return;
    free(m->weights); free(m->scales);
    vm_free(&m->vocab);
    free(m);
}

int cbed_dim(const cbed_model *m) { return m->dim; }
int cbed_vocab_size(const cbed_model *m) { return m->vocab_size; }

/* ------------------------------------------------------------------ */
/* WordPiece tokenizer (faithful to pybed/zbed)                        */
/* ------------------------------------------------------------------ */

static int is_word_char(unsigned char c) { return isalnum(c) || c == '\'' || c == '_'; }

/* greedy longest-match WordPiece on lowercased word[0..n) */
static int tokenize_word(const cbed_model *m, const char *word, int n,
                         int16_t *out, int count, int max_len) {
    if (n == 0 || count >= max_len) return count;
    int32_t whole = vm_get(&m->vocab, word, n);
    if (whole >= 0) { out[count] = (int16_t)whole; return count + 1; }
    int pos = 0;
    while (pos < n && count < max_len) {
        int end = n, matched = -1;
        char piece[300];
        while (end > pos) {
            int32_t tid;
            if (pos == 0) {
                tid = vm_get(&m->vocab, word + pos, end - pos);
            } else {
                int pl = end - pos;
                if (pl > (int)sizeof(piece) - 3) { end--; continue; }
                piece[0] = '#'; piece[1] = '#';
                memcpy(piece + 2, word + pos, pl);
                tid = vm_get(&m->vocab, piece, pl + 2);
            }
            if (tid >= 0) { matched = tid; break; }
            end--;
        }
        if (matched < 0) {
            if (m->unk_id >= 0 && count < max_len) out[count++] = (int16_t)m->unk_id;
            return count;
        }
        out[count++] = (int16_t)matched;
        pos = end;
    }
    return count;
}

static int encode_into(const cbed_model *m, const char *text, int16_t *out, int max_len) {
    int count = 0;
    if (m->cls_id >= 0 && count < max_len) out[count++] = (int16_t)m->cls_id;
    size_t tn = strlen(text);
    char stack[2048];
    if (tn >= sizeof stack) tn = sizeof stack - 1;  /* prompts past 2KB are past CBED_MAX_TOKENS anyway */
    char *low = stack;
    for (size_t i = 0; i < tn; i++) low[i] = (char)tolower((unsigned char)text[i]);
    low[tn] = 0;

    int word_start = -1;
    for (size_t i = 0; i <= tn; i++) {
        unsigned char ch = (i < tn) ? (unsigned char)low[i] : 0;
        if (i < tn && is_word_char(ch)) {
            if (word_start < 0) word_start = (int)i;
        } else {
            if (word_start >= 0) {
                count = tokenize_word(m, low + word_start, (int)i - word_start, out, count, max_len);
                word_start = -1;
            }
            if (i < tn && ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r' && count < max_len) {
                char cc = (char)ch;
                int32_t tid = vm_get(&m->vocab, &cc, 1);
                if (tid >= 0) out[count++] = (int16_t)tid;
            }
        }
    }
    if (m->sep_id >= 0 && count < max_len) out[count++] = (int16_t)m->sep_id;
    return count;
}

/* ------------------------------------------------------------------ */
/* embed + quantize                                                    */
/* ------------------------------------------------------------------ */

void cbed_embed(const cbed_model *m, const char *text, float *out) {
    int16_t toks[CBED_MAX_TOKENS];
    int n = encode_into(m, text, toks, CBED_MAX_TOKENS);
    int dim = m->dim;
    for (int k = 0; k < dim; k++) out[k] = 0.0f;
    int valid = 0;
    for (int i = 0; i < n; i++) {
        int id = toks[i];
        if (id < 0 || id >= m->vocab_size) continue;
        const int8_t *row = m->weights + (size_t)id * dim;
        float sc = m->scales[id];
        for (int k = 0; k < dim; k++) out[k] += (float)row[k] * sc;
        valid++;
    }
    if (valid > 0) { float inv = 1.0f / valid; for (int k = 0; k < dim; k++) out[k] *= inv; }
}

static float quantize(const float *vec, int dim, int8_t *out) {
    float max_abs = 0.0f;
    for (int k = 0; k < dim; k++) { float a = fabsf(vec[k]); if (a > max_abs) max_abs = a; }
    if (max_abs == 0.0f) { for (int k = 0; k < dim; k++) out[k] = 0; return 1.0f; }
    float s = 127.0f / max_abs;
    for (int k = 0; k < dim; k++) {
        float q = roundf(vec[k] * s);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[k] = (int8_t)q;
    }
    return max_abs / 127.0f;
}

int32_t cbed_dot_i8(const int8_t *a, const int8_t *b, int n) {
    int32_t acc = 0;
    for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

static float qnorm(const int8_t *v, int n) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) s += (int64_t)v[i] * v[i];
    return sqrtf((float)s);
}

void cbed_embed_quantized(const cbed_model *m, const char *text,
                          int8_t *out, float *norm_out, float *scale_out) {
    float f[1024];
    cbed_embed(m, text, f);
    float sc = quantize(f, m->dim, out);
    if (scale_out) *scale_out = sc;
    if (norm_out) *norm_out = qnorm(out, m->dim);
}

/* ------------------------------------------------------------------ */
/* index + search                                                      */
/* ------------------------------------------------------------------ */

struct cbed_index {
    int dim;
    size_t len, cap;
    int8_t *vecs;      /* [cap*dim] */
    float *norms;      /* [cap] */
    char **ids;        /* [cap], owned */
};

cbed_index *cbed_index_new(int dim, size_t capacity_hint) {
    cbed_index *ix = calloc(1, sizeof(*ix));
    ix->dim = dim;
    ix->cap = capacity_hint < 8 ? 8 : capacity_hint;
    ix->vecs = malloc(ix->cap * dim);
    ix->norms = malloc(ix->cap * sizeof(float));
    ix->ids = malloc(ix->cap * sizeof(char *));
    return ix;
}

void cbed_index_free(cbed_index *ix) {
    if (!ix) return;
    for (size_t i = 0; i < ix->len; i++) free(ix->ids[i]);
    free(ix->vecs); free(ix->norms); free(ix->ids); free(ix);
}

static void ix_grow(cbed_index *ix) {
    ix->cap *= 2;
    ix->vecs = realloc(ix->vecs, ix->cap * ix->dim);
    ix->norms = realloc(ix->norms, ix->cap * sizeof(float));
    ix->ids = realloc(ix->ids, ix->cap * sizeof(char *));
}

int cbed_index_add(cbed_index *ix, const cbed_model *m, const char *id, const char *text) {
    if (m->dim != ix->dim) return -1;
    if (ix->len == ix->cap) ix_grow(ix);
    int8_t *slot = ix->vecs + ix->len * ix->dim;
    float norm;
    cbed_embed_quantized(m, text, slot, &norm, NULL);
    ix->norms[ix->len] = norm;
    ix->ids[ix->len] = strdup(id);
    ix->len++;
    return (int)ix->len;
}

/* insertion into a top-k array kept sorted descending by score */
static int topk_insert(cbed_hit *hits, int k, int filled, cbed_hit cand) {
    if (filled < k) {
        int i = filled - 1;
        while (i >= 0 && hits[i].score < cand.score) { hits[i + 1] = hits[i]; i--; }
        hits[i + 1] = cand;
        return filled + 1;
    }
    if (cand.score <= hits[k - 1].score) return filled;
    int i = k - 2;
    while (i >= 0 && hits[i].score < cand.score) { hits[i + 1] = hits[i]; i--; }
    hits[i + 1] = cand;
    return filled;
}

int cbed_search_vec(const cbed_index *ix, const int8_t *q, float q_norm, int k, cbed_hit *hits) {
    if (k <= 0 || ix->len == 0) return 0;
    int filled = 0;
    float inv_qn = (q_norm > 0) ? 1.0f / q_norm : 0.0f;
    for (size_t i = 0; i < ix->len; i++) {
        const int8_t *v = ix->vecs + i * ix->dim;
        int32_t dot = cbed_dot_i8(q, v, ix->dim);
        float score = ix->norms[i] > 0 ? (float)dot / ix->norms[i] * inv_qn : 0.0f;
        cbed_hit cand = { ix->ids[i], score };
        filled = topk_insert(hits, k, filled, cand);
    }
    return filled;
}

int cbed_search(const cbed_index *ix, const cbed_model *m, const char *query, int k, cbed_hit *hits) {
    int8_t q[1024]; float qn;
    cbed_embed_quantized(m, query, q, &qn, NULL);
    return cbed_search_vec(ix, q, qn, k, hits);
}
