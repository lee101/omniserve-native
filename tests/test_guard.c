#include "oguard.h"
#include "ojson.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static long now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)(ts.tv_sec * 1000000L + ts.tv_nsec / 1000L);
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

typedef struct {
    const char *key;
    oguard_category expect;
} suite_t;

static const suite_t suites[] = {
    {"block_minor_sexual", OGUARD_MINOR_SEXUAL},
    {"block_noncon_sexual", OGUARD_NONCON_SEXUAL},
    {"block_realperson_sexual", OGUARD_REALPERSON_SEXUAL},
    {"allow_adult_roleplay", OGUARD_NONE},
    {"allow_dark_nonsexual", OGUARD_NONE},
    {"allow_ordinary", OGUARD_NONE},
};

static const suite_t holdout_suites[] = {
    {"block_minor_sexual", OGUARD_MINOR_SEXUAL},
    {"block_noncon_sexual", OGUARD_NONCON_SEXUAL},
    {"block_realperson_sexual", OGUARD_REALPERSON_SEXUAL},
    {"allow_roleplay_adult", OGUARD_NONE},
    {"allow_educational", OGUARD_NONE},
    {"allow_dark_nonsexual", OGUARD_NONE},
    {"allow_ordinary", OGUARD_NONE},
};

int main(int argc, char **argv) {
    const char *corpus = argc > 1 ? argv[1] : "tests/data/guard-corpus.json";
    const char *env_path = getenv("GUARD_CORPUS");
    if (env_path && env_path[0]) corpus = env_path;
    bool holdout = (argc > 2 && strcmp(argv[2], "--holdout") == 0) ||
        (argc > 1 && strcmp(argv[1], "--holdout") == 0);
    const suite_t *suites_run = holdout ? holdout_suites : suites;
    size_t suites_n = holdout ? sizeof holdout_suites / sizeof holdout_suites[0]
                              : sizeof suites / sizeof suites[0];
    size_t len = 0;
    char *js = read_file(corpus, &len);
    if (!js) {
        fprintf(stderr, "guard test: cannot read %s\n", corpus);
        return 1;
    }
    oguard_init();
    if (oguard_names_loaded() <= 0) {
        fprintf(stderr, "guard test: names file not loaded\n");
        free(js);
        return 1;
    }
    oj_tok *toks = malloc(sizeof *toks * 8192);
    if (!toks) {
        free(js);
        return 1;
    }
    int n = oj_parse(js, len, toks, 8192);
    if (n <= 0 || toks[0].type != OJ_OBJECT) {
        fprintf(stderr, "guard test: invalid corpus JSON\n");
        free(toks);
        free(js);
        return 1;
    }
    int block_total = 0, block_hit = 0, allow_total = 0, allow_pass = 0;
    long total_us = 0;
    int cases = 0;
    for (size_t s = 0; s < suites_n; s++) {
        int arr = oj_obj_get(js, toks, n, 0, suites_run[s].key);
        if (arr < 0 || toks[arr].type != OJ_ARRAY) {
            fprintf(stderr, "guard test: missing array %s\n", suites_run[s].key);
            free(toks);
            free(js);
            return 1;
        }
        for (int i = 0; i < toks[arr].size; i++) {
            int item = oj_arr_at(toks, n, arr, i);
            if (item < 0) continue;
            char *system = NULL;
            char *user = NULL;
            char *ctx = NULL;
            if (toks[item].type == OJ_STRING) {
                user = oj_strdup(js, &toks[item]);
            } else if (toks[item].type == OJ_OBJECT) {
                int st = oj_obj_get(js, toks, n, item, "system");
                int ut = oj_obj_get(js, toks, n, item, "user");
                int ct = oj_obj_get(js, toks, n, item, "context");
                if (st >= 0) system = oj_strdup(js, &toks[st]);
                if (ut >= 0) user = oj_strdup(js, &toks[ut]);
                if (ct >= 0) ctx = oj_strdup(js, &toks[ct]);
            }
            if (!user) user = strdup("");
            long t0 = now_us();
            oguard_verdict v = oguard_classify(system, user, ctx);
            total_us += now_us() - t0;
            cases++;
            bool want_block = suites_run[s].expect != OGUARD_NONE;
            if (want_block) {
                block_total++;
                if (v.category == suites_run[s].expect) {
                    block_hit++;
                } else {
                    fprintf(stderr, "MISS [%s#%d] want=%s got=%s rule=%s :: %.120s\n",
                            suites_run[s].key, i, oguard_category_name(suites_run[s].expect),
                            oguard_category_name(v.category), v.matched_rule, user ? user : "");
                }
            } else {
                allow_total++;
                if (v.category == OGUARD_NONE) {
                    allow_pass++;
                } else {
                    fprintf(stderr, "FALSE-POSITIVE [%s#%d] got=%s rule=%s :: %.120s\n",
                            suites_run[s].key, i, oguard_category_name(v.category),
                            v.matched_rule, user ? user : "");
                }
            }
            free(system);
            free(user);
            free(ctx);
        }
    }
    double block_rate = block_total ? (double)block_hit / (double)block_total : 1.0;
    double allow_rate = allow_total ? (double)allow_pass / (double)allow_total : 1.0;
    printf("guard corpus: block %d/%d (%.1f%%) allow %d/%d (%.1f%%) names=%d avg_classify_us=%.1f\n",
           block_hit, block_total, block_rate * 100.0, allow_pass, allow_total,
           allow_rate * 100.0, oguard_names_loaded(),
           cases ? (double)total_us / (double)cases : 0.0);
    free(toks);
    free(js);
    if (allow_pass != allow_total) return 1;
    if (!holdout && block_rate < 0.95) return 1;
    if (!holdout && (block_total < 40 || allow_total < 40)) {
        fprintf(stderr, "guard test: corpus too small\n");
        return 1;
    }
    if (holdout && (block_total < 30 || allow_total < 30)) {
        fprintf(stderr, "guard test: holdout too small\n");
        return 1;
    }
    return 0;
}
