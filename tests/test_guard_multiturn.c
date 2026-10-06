#include "oguard.h"
#include "ojson.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool stub_ready(void) {
    return true;
}

static double stub_pyes(const char *prompt) {
    if (prompt && strstr(prompt, "Sexual-intent check")) return 0.02;
    return 0.95;
}

static bool stub_score(const char *prompt, size_t len, double *pyes, double *ms) {
    (void)len;
    if (pyes) *pyes = stub_pyes(prompt);
    if (ms) *ms = 0.5;
    return true;
}

static bool stub_multi(const char **prompts, const size_t *lens, int n,
                       double *pyes, double *ms) {
    (void)lens;
    for (int i = 0; i < n; i++) pyes[i] = stub_pyes(prompts ? prompts[i] : NULL);
    if (ms) *ms = 0.5;
    return true;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || n > 4 * 1024 * 1024) {
        fclose(f);
        return NULL;
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = 0;
    if (len_out) *len_out = got;
    return buf;
}

static oguard_category cat_by_name(const char *name) {
    if (strcmp(name, "minor_sexual") == 0) return OGUARD_MINOR_SEXUAL;
    if (strcmp(name, "noncon_sexual") == 0) return OGUARD_NONCON_SEXUAL;
    if (strcmp(name, "realperson_sexual") == 0) return OGUARD_REALPERSON_SEXUAL;
    return OGUARD_NONE;
}

#define CTX_CAP 16384

static void run_arc(const char *js, const oj_tok *toks, int ntoks, int arc, bool allow,
                    oguard_category want, int block_at, int *fails, int *turns) {
    int sys_tok = oj_obj_get(js, toks, ntoks, arc, "system");
    int name_tok = oj_obj_get(js, toks, ntoks, arc, "name");
    int turns_tok = oj_obj_get(js, toks, ntoks, arc, "turns");
    char *sys = sys_tok >= 0 ? oj_strdup(js, &toks[sys_tok]) : NULL;
    char *name = name_tok >= 0 ? oj_strdup(js, &toks[name_tok]) : strdup("?");
    int nturns = toks[turns_tok].size;
    char **users = calloc((size_t)nturns, sizeof *users);
    char **assis = calloc((size_t)nturns, sizeof *assis);
    if (!users || !assis) {
        fprintf(stderr, "multiturn: out of memory\n");
        (*fails)++;
        free(sys);
        free(name);
        free(users);
        free(assis);
        return;
    }
    for (int i = 0; i < nturns; i++) {
        int item = oj_arr_at(toks, ntoks, turns_tok, i);
        int ut = oj_obj_get(js, toks, ntoks, item, "user");
        int at = oj_obj_get(js, toks, ntoks, item, "assistant");
        users[i] = ut >= 0 ? oj_strdup(js, &toks[ut]) : strdup("");
        assis[i] = at >= 0 ? oj_strdup(js, &toks[at]) : strdup("");
    }
    for (int i = 0; i < nturns; i++) {
        if (!allow && i + 1 > block_at) break;
        char ctx[CTX_CAP + 1];
        size_t ctx_len = 0;
        ctx[0] = 0;
        int prior = i * 2;
        int start = prior > 6 ? prior - 6 : 0;
        for (int m = start; m < prior; m++) {
            const char *s = (m % 2 == 0) ? users[m / 2] : assis[m / 2];
            if (!s || !s[0]) continue;
            if (ctx_len > 0 && ctx_len < CTX_CAP) ctx[ctx_len++] = '\n';
            size_t n = strlen(s);
            if (ctx_len + n > CTX_CAP) n = CTX_CAP - ctx_len;
            memcpy(ctx + ctx_len, s, n);
            ctx_len += n;
        }
        ctx[ctx_len] = 0;
        oguard_verdict v = oguard_classify(sys, users[i], ctx);
        (*turns)++;
        if (allow) {
            if (v.category != OGUARD_NONE) {
                fprintf(stderr, "FALSE-POSITIVE [%s#%d] got=%s rule=%s :: %.100s\n",
                        name, i + 1, oguard_category_name(v.category),
                        v.matched_rule, users[i]);
                (*fails)++;
            }
        } else if (i + 1 < block_at) {
            if (v.category != OGUARD_NONE) {
                fprintf(stderr, "EARLY-BLOCK [%s#%d] want=none got=%s rule=%s :: %.100s\n",
                        name, i + 1, oguard_category_name(v.category),
                        v.matched_rule, users[i]);
                (*fails)++;
            }
        } else {
            if (v.category != want) {
                fprintf(stderr, "MISS [%s#%d] want=%s got=%s rule=%s :: %.100s\n",
                        name, i + 1, oguard_category_name(want),
                        oguard_category_name(v.category), v.matched_rule, users[i]);
                (*fails)++;
            }
        }
    }
    printf("%s %s (%d turns)\n", *fails ? "FAIL" : "ok", name, allow ? nturns : block_at);
    for (int i = 0; i < nturns; i++) {
        free(users[i]);
        free(assis[i]);
    }
    free(users);
    free(assis);
    free(sys);
    free(name);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/data/guard-multiturn.json";
    const char *env_path = getenv("GUARD_MULTITURN");
    if (env_path && env_path[0]) path = env_path;
    size_t len = 0;
    char *js = read_file(path, &len);
    if (!js) {
        fprintf(stderr, "multiturn test: cannot read %s\n", path);
        return 1;
    }
    setenv("OMNISERVE_NATIVE_GUARD_JUDGE_CACHE_SIZE", "64", 1);
    oguard_init();
    if (oguard_names_loaded() <= 0) {
        fprintf(stderr, "multiturn test: names file not loaded\n");
        free(js);
        return 1;
    }
    oguard_set_judge_backend(stub_ready, stub_score);
    oguard_set_judge_multi_backend(stub_multi);
    oguard_judge_warmup();
    if (!oguard_judge_ready()) {
        fprintf(stderr, "multiturn test: judge stub not live\n");
        free(js);
        return 1;
    }
    oj_tok *toks = malloc(sizeof *toks * 16384);
    if (!toks) {
        free(js);
        return 1;
    }
    int n = oj_parse(js, len, toks, 16384);
    if (n <= 0 || toks[0].type != OJ_OBJECT) {
        fprintf(stderr, "multiturn test: invalid JSON\n");
        free(toks);
        free(js);
        return 1;
    }
    int fails = 0;
    int turns = 0;
    int allow_arcs = 0;
    int block_arcs = 0;
    int arr = oj_obj_get(js, toks, n, 0, "allow_arcs");
    if (arr < 0 || toks[arr].type != OJ_ARRAY) {
        fprintf(stderr, "multiturn test: missing allow_arcs\n");
        free(toks);
        free(js);
        return 1;
    }
    for (int i = 0; i < toks[arr].size; i++) {
        int arc = oj_arr_at(toks, n, arr, i);
        if (arc < 0) continue;
        run_arc(js, toks, n, arc, true, OGUARD_NONE, 0, &fails, &turns);
        allow_arcs++;
    }
    int barr = oj_obj_get(js, toks, n, 0, "block_arcs");
    if (barr < 0 || toks[barr].type != OJ_ARRAY) {
        fprintf(stderr, "multiturn test: missing block_arcs\n");
        free(toks);
        free(js);
        return 1;
    }
    for (int i = 0; i < toks[barr].size; i++) {
        int arc = oj_arr_at(toks, n, barr, i);
        if (arc < 0) continue;
        int ct = oj_obj_get(js, toks, n, arc, "category");
        int bt = oj_obj_get(js, toks, n, arc, "block_at");
        char *cat = ct >= 0 ? oj_strdup(js, &toks[ct]) : strdup("");
        int block_at = bt >= 0 ? (int)oj_number(js, &toks[bt], 0.0) : 0;
        oguard_category want = cat_by_name(cat);
        free(cat);
        if (want == OGUARD_NONE || block_at < 6) {
            fprintf(stderr, "multiturn test: block arc needs category + block_at>=6\n");
            fails++;
            continue;
        }
        run_arc(js, toks, n, arc, false, want, block_at, &fails, &turns);
        block_arcs++;
    }
    printf("multiturn: %d allow arcs, %d block arcs, %d turns, %d failures\n",
           allow_arcs, block_arcs, turns, fails);
    free(toks);
    free(js);
    if (allow_arcs < 6 || block_arcs < 2) {
        fprintf(stderr, "multiturn test: harness too small\n");
        return 1;
    }
    return fails ? 1 : 0;
}
