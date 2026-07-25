#include "ojson.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int pos;
    int next_tok;
    int super;
} oj_parser;

static oj_tok *alloc_tok(oj_parser *p, oj_tok *toks, int max_toks) {
    if (p->next_tok >= max_toks) return NULL;
    oj_tok *t = &toks[p->next_tok++];
    t->start = t->end = -1;
    t->size = 0;
    t->parent = -1;
    return t;
}

static int parse_primitive(oj_parser *p, const char *js, size_t len, oj_tok *toks, int max_toks) {
    int start = p->pos;
    for (; p->pos < (int)len; p->pos++) {
        char c = js[p->pos];
        if (c == ',' || c == ']' || c == '}' || c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ':') break;
        if (c < 32 || c >= 127) return -1;
    }
    oj_tok *t = alloc_tok(p, toks, max_toks);
    if (!t) return -1;
    t->type = OJ_PRIMITIVE;
    t->start = start;
    t->end = p->pos;
    t->parent = p->super;
    p->pos--;
    return 0;
}

static int parse_string(oj_parser *p, const char *js, size_t len, oj_tok *toks, int max_toks) {
    int start = ++p->pos;
    for (; p->pos < (int)len; p->pos++) {
        char c = js[p->pos];
        if (c == '"') {
            oj_tok *t = alloc_tok(p, toks, max_toks);
            if (!t) return -1;
            t->type = OJ_STRING;
            t->start = start;
            t->end = p->pos;
            t->parent = p->super;
            return 0;
        }
        if (c == '\\' && p->pos + 1 < (int)len) p->pos++;
    }
    return -1;
}

int oj_parse(const char *js, size_t len, oj_tok *toks, int max_toks) {
    oj_parser p = { .pos = 0, .next_tok = 0, .super = -1 };
    for (; p.pos < (int)len; p.pos++) {
        char c = js[p.pos];
        switch (c) {
        case '{': case '[': {
            oj_tok *t = alloc_tok(&p, toks, max_toks);
            if (!t) return -1;
            if (p.super != -1) toks[p.super].size++;
            t->type = c == '{' ? OJ_OBJECT : OJ_ARRAY;
            t->start = p.pos;
            t->parent = p.super;
            p.super = p.next_tok - 1;
            break;
        }
        case '}': case ']': {
            oj_type want = c == '}' ? OJ_OBJECT : OJ_ARRAY;
            int i = p.super;
            for (; i != -1; i = toks[i].parent) {
                if (toks[i].end == -1) {
                    if (toks[i].type != want) return -1;
                    toks[i].end = p.pos + 1;
                    p.super = toks[i].parent;
                    break;
                }
            }
            if (i == -1) return -1;
            break;
        }
        case '"':
            if (parse_string(&p, js, len, toks, max_toks) != 0) return -1;
            if (p.super != -1) toks[p.super].size++;
            break;
        case ' ': case '\t': case '\r': case '\n': case ',': case ':':
            break;
        default:
            if (parse_primitive(&p, js, len, toks, max_toks) != 0) return -1;
            if (p.super != -1) toks[p.super].size++;
            break;
        }
    }
    for (int i = 0; i < p.next_tok; i++) {
        if (toks[i].end == -1) return -1;
    }
    return p.next_tok;
}

int oj_next_sibling(const oj_tok *toks, int ntoks, int tok) {
    int depth_parent = toks[tok].parent;
    for (int i = tok + 1; i < ntoks; i++) {
        if (toks[i].parent == depth_parent) return i;
        if (toks[i].parent < depth_parent && toks[i].parent != -1) break;
    }
    return -1;
}

int oj_obj_get(const char *js, const oj_tok *toks, int ntoks, int obj, const char *key) {
    if (obj < 0 || obj >= ntoks || toks[obj].type != OJ_OBJECT) return -1;
    size_t klen = strlen(key);
    int i = obj + 1;
    while (i < ntoks && toks[i].parent == obj) {
        int val = i + 1;
        if (val >= ntoks) return -1;
        if (toks[i].type == OJ_STRING &&
            (size_t)(toks[i].end - toks[i].start) == klen &&
            memcmp(js + toks[i].start, key, klen) == 0) {
            return val;
        }
        int sib = oj_next_sibling(toks, ntoks, val);
        if (sib == -1) {
            i = val + 1;
            while (i < ntoks && toks[i].parent != obj) i++;
        } else {
            i = sib;
        }
    }
    return -1;
}

int oj_arr_at(const oj_tok *toks, int ntoks, int arr, int index) {
    if (arr < 0 || arr >= ntoks || toks[arr].type != OJ_ARRAY) return -1;
    int count = 0;
    for (int i = arr + 1; i < ntoks; i++) {
        if (toks[i].parent == arr) {
            if (count == index) return i;
            count++;
        }
    }
    return -1;
}

bool oj_str_eq(const char *js, const oj_tok *t, const char *s) {
    size_t slen = strlen(s);
    return t->type == OJ_STRING && (size_t)(t->end - t->start) == slen &&
           memcmp(js + t->start, s, slen) == 0;
}

size_t oj_unescape(const char *js, const oj_tok *t, char *out, size_t out_cap) {
    size_t o = 0;
    for (int i = t->start; i < t->end && o + 4 < out_cap; i++) {
        char c = js[i];
        if (c == '\\' && i + 1 < t->end) {
            char e = js[++i];
            switch (e) {
            case 'n': out[o++] = '\n'; break;
            case 't': out[o++] = '\t'; break;
            case 'r': out[o++] = '\r'; break;
            case 'b': out[o++] = '\b'; break;
            case 'f': out[o++] = '\f'; break;
            case 'u': {
                if (i + 4 < t->end) {
                    unsigned code = 0;
                    for (int k = 1; k <= 4; k++) {
                        char h = js[i + k];
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') code |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= (unsigned)(h - 'A' + 10);
                    }
                    i += 4;
                    if (code < 0x80) out[o++] = (char)code;
                    else if (code < 0x800) {
                        out[o++] = (char)(0xC0 | (code >> 6));
                        out[o++] = (char)(0x80 | (code & 0x3F));
                    } else {
                        out[o++] = (char)(0xE0 | (code >> 12));
                        out[o++] = (char)(0x80 | ((code >> 6) & 0x3F));
                        out[o++] = (char)(0x80 | (code & 0x3F));
                    }
                }
                break;
            }
            default: out[o++] = e; break;
            }
        } else {
            out[o++] = c;
        }
    }
    out[o] = 0;
    return o;
}

char *oj_strdup(const char *js, const oj_tok *t) {
    size_t cap = (size_t)(t->end - t->start) * 3 + 4;
    char *out = malloc(cap);
    if (!out) return NULL;
    oj_unescape(js, t, out, cap);
    return out;
}

double oj_number(const char *js, const oj_tok *t, double fallback) {
    if (t->type != OJ_PRIMITIVE) return fallback;
    size_t len = (size_t)(t->end - t->start);
    if (len == 0 || len > 127) return fallback;
    char bounded[128];
    memcpy(bounded, js + t->start, len);
    bounded[len] = 0;
    char *end = NULL;
    double value = strtod(bounded, &end);
    return end == bounded + len ? value : fallback;
}

bool oj_bool(const char *js, const oj_tok *t, bool fallback) {
    if (t->type != OJ_PRIMITIVE) return fallback;
    if (js[t->start] == 't') return true;
    if (js[t->start] == 'f') return false;
    return fallback;
}

bool oj_escape_append(char **buf, size_t *len, size_t *cap, const char *s, size_t slen) {
    if (!buf || !len || !cap || (!s && slen)) return false;
    if (*len > SIZE_MAX - 8) return false;
    if (slen > (SIZE_MAX - *len - 8) / 6) return false;
    size_t needed = *len + slen * 6 + 8;
    if (needed > *cap) {
        size_t ncap = *cap ? *cap : 256;
        while (ncap < needed) {
            if (ncap > SIZE_MAX / 2) {
                ncap = needed;
                break;
            }
            ncap *= 2;
        }
        char *grown = realloc(*buf, ncap);
        if (!grown) return false;
        *buf = grown;
        *cap = ncap;
    }
    char *o = *buf + *len;
    for (size_t i = 0; i < slen; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': *o++ = '\\'; *o++ = '"'; break;
        case '\\': *o++ = '\\'; *o++ = '\\'; break;
        case '\n': *o++ = '\\'; *o++ = 'n'; break;
        case '\r': *o++ = '\\'; *o++ = 'r'; break;
        case '\t': *o++ = '\\'; *o++ = 't'; break;
        default:
            if (c < 0x20) {
                o += sprintf(o, "\\u%04x", c);
            } else {
                *o++ = (char)c;
            }
        }
    }
    *len = (size_t)(o - *buf);
    (*buf)[*len] = 0;
    return true;
}
