#ifndef OJSON_H
#define OJSON_H

#include <stdbool.h>
#include <stddef.h>

typedef enum { OJ_UNDEF, OJ_OBJECT, OJ_ARRAY, OJ_STRING, OJ_PRIMITIVE } oj_type;

typedef struct {
    oj_type type;
    int start;
    int end;
    int size;
    int parent;
} oj_tok;

int oj_parse(const char *js, size_t len, oj_tok *toks, int max_toks);

int oj_obj_get(const char *js, const oj_tok *toks, int ntoks, int obj, const char *key);
int oj_arr_at(const oj_tok *toks, int ntoks, int arr, int index);
int oj_next_sibling(const oj_tok *toks, int ntoks, int tok);

bool oj_str_eq(const char *js, const oj_tok *t, const char *s);
char *oj_strdup(const char *js, const oj_tok *t);
size_t oj_unescape(const char *js, const oj_tok *t, char *out, size_t out_cap);
double oj_number(const char *js, const oj_tok *t, double fallback);
bool oj_bool(const char *js, const oj_tok *t, bool fallback);

bool oj_escape_append(char **buf, size_t *len, size_t *cap, const char *s, size_t slen);

#endif
