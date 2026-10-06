#include "obackend.h"
#include "oguard.h"
#include "ojson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bool sweep_embed_ready(void) {
    return oembed_ready();
}

static bool sweep_embed_text(const char *text, size_t len, float **values, int *dim) {
    oembed_result r;
    if (!oembed_text(text, len, 0, &r)) return false;
    *values = r.values;
    *dim = r.dimensions;
    return true;
}

static void sweep_embed_free(float *values) {
    oembed_result r;
    memset(&r, 0, sizeof r);
    r.values = values;
    oembed_result_free(&r);
}

static bool sweep_judge_ready(void) {
    return ojudge_ready();
}

static bool sweep_judge_score(const char *prompt, size_t len, double *pyes, double *ms) {
    return ojudge_score(prompt, len, pyes, ms);
}

static bool sweep_judge_multi(const char **prompts, const size_t *lens, int n,
                              double *pyes, double *ms) {
    return ojudge_score_multi(prompts, lens, n, pyes, ms);
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

static const suite_t tune_suites[] = {
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

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *corpus = argc > 1 ? argv[1] : "tests/data/guard-corpus.json";
    bool holdout = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--holdout") == 0) holdout = true;
        if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc)
            setenv("OMNISERVE_NATIVE_GUARD_EMBED_THRESHOLD", argv[i + 1], 1);
        if (strcmp(argv[i], "--margin") == 0 && i + 1 < argc)
            setenv("OMNISERVE_NATIVE_GUARD_EMBED_MARGIN", argv[i + 1], 1);
    }
    const suite_t *suites = holdout ? holdout_suites : tune_suites;
    size_t suites_n = holdout ? sizeof holdout_suites / sizeof holdout_suites[0]
                              : sizeof tune_suites / sizeof tune_suites[0];
    const char *gguf = getenv("OMNISERVE_NATIVE_EMBEDDING_GGUF");
    const char *ngl = getenv("OMNISERVE_NATIVE_EMBEDDING_NGL");
    const char *ctx = getenv("OMNISERVE_NATIVE_EMBEDDING_CTX");
    const char *threads = getenv("OMNISERVE_NATIVE_EMBEDDING_THREADS");
    if (!gguf || !gguf[0]) {
        fprintf(stderr, "sweep: OMNISERVE_NATIVE_EMBEDDING_GGUF required\n");
        return 2;
    }
    if (!oembed_init(gguf, ngl ? atoi(ngl) : 0, ctx ? atoi(ctx) : 512,
                     threads ? atoi(threads) : 8)) {
        fprintf(stderr, "sweep: oembed_init failed\n");
        return 2;
    }
    const char *jgguf = getenv("OMNISERVE_NATIVE_GUARD_JUDGE_GGUF");
    if (!jgguf || !jgguf[0])
        jgguf = "/nvme0n1-disk/models/omniserve-native/shieldgemma-2b-q4_k_m.gguf";
    const char *jthreads = getenv("OMNISERVE_NATIVE_GUARD_JUDGE_THREADS");
    bool no_judge = false;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--no-judge") == 0) no_judge = true;
    if (!no_judge) {
        if (!ojudge_init(jgguf, 4096, jthreads ? atoi(jthreads) : 24))
            fprintf(stderr, "sweep: ojudge_init failed, stages 1+2 only\n");
    }
    oguard_init();
    oguard_set_embed_backend(sweep_embed_ready, sweep_embed_text, sweep_embed_free);
    oguard_embed_warmup();
    oguard_set_judge_backend(sweep_judge_ready, sweep_judge_score);
    oguard_set_judge_multi_backend(sweep_judge_multi);
    oguard_judge_warmup();
    for (int i = 0; i < 900 && !oguard_embed_ready(); i++) sleep(1);
    if (!oguard_embed_ready()) {
        fprintf(stderr, "sweep: embedding stage never went live\n");
        return 2;
    }
    size_t len = 0;
    char *js = read_file(corpus, &len);
    if (!js) {
        fprintf(stderr, "sweep: cannot read %s\n", corpus);
        return 1;
    }
    oj_tok *toks = malloc(sizeof *toks * 16384);
    if (!toks) {
        free(js);
        return 1;
    }
    int n = oj_parse(js, len, toks, 16384);
    if (n <= 0 || toks[0].type != OJ_OBJECT) {
        fprintf(stderr, "sweep: invalid corpus JSON\n");
        free(toks);
        free(js);
        return 1;
    }
    printf("suite\tidx\twant\tlex\temb\tjudge\tfinal\tprohib\tbenign\tdiff\tpyes\tjinv\tveto\tems\tjms\tcat0\tcat1\tcat2\ts0\ts1\ts2\tgpyes\tgopen\tgms\n");
    int lex_only = 0, emb_only = 0, both = 0, missed = 0;
    int pipe_lex = 0, pipe_emb = 0, pipe_judge = 0;
    int allow_pass = 0, allow_total = 0, block_total = 0;
    int judge_inv = 0, judge_corr = 0, vetoes = 0, gate_open = 0;
    double *lat = NULL;
    size_t lat_n = 0, lat_cap = 0;
    double lat_sum = 0.0, lat_max = 0.0;
    double *jlat = NULL;
    size_t jlat_n = 0, jlat_cap = 0;
    double jlat_sum = 0.0, jlat_max = 0.0;
    for (size_t s = 0; s < suites_n; s++) {
        int arr = oj_obj_get(js, toks, n, 0, suites[s].key);
        if (arr < 0 || toks[arr].type != OJ_ARRAY) {
            fprintf(stderr, "sweep: missing array %s\n", suites[s].key);
            free(lat);
            free(toks);
            free(js);
            return 1;
        }
        for (int i = 0; i < toks[arr].size; i++) {
            int item = oj_arr_at(toks, n, arr, i);
            if (item < 0) continue;
            char *system = NULL;
            char *user = NULL;
            char *ctxs = NULL;
            if (toks[item].type == OJ_STRING) {
                user = oj_strdup(js, &toks[item]);
            } else if (toks[item].type == OJ_OBJECT) {
                int st = oj_obj_get(js, toks, n, item, "system");
                int ut = oj_obj_get(js, toks, n, item, "user");
                int ct = oj_obj_get(js, toks, n, item, "context");
                if (st >= 0) system = oj_strdup(js, &toks[st]);
                if (ut >= 0) user = oj_strdup(js, &toks[ut]);
                if (ct >= 0) ctxs = oj_strdup(js, &toks[ct]);
            }
            if (!user) user = strdup("");
            oguard_full f = oguard_classify_diagnose(system, user, ctxs);
            double epro = f.emb_prohib, eben = f.emb_benign, ems = f.embed_ms;
            bool want_block = suites[s].expect != OGUARD_NONE;
            bool lex_hit = f.lexical.category == suites[s].expect && want_block;
            bool emb_hit = f.embed.category == suites[s].expect && want_block;
            bool judge_hit = f.judge.category == suites[s].expect && want_block;
            if (f.judge_invoked) {
                judge_inv++;
                if (jlat_n == jlat_cap) {
                    size_t next = jlat_cap ? jlat_cap * 2 : 256;
                    double *grown = realloc(jlat, next * sizeof *grown);
                    if (grown) {
                        jlat = grown;
                        jlat_cap = next;
                    }
                }
                if (jlat_n < jlat_cap) {
                    jlat[jlat_n++] = f.judge_ms;
                    jlat_sum += f.judge_ms;
                    if (f.judge_ms > jlat_max) jlat_max = f.judge_ms;
                }
            }
            if (want_block) {
                block_total++;
                if (lex_hit && emb_hit) both++;
                else if (lex_hit) lex_only++;
                else if (emb_hit) emb_only++;
                else if (judge_hit) judge_corr++;
                else if (f.final.category != OGUARD_NONE) judge_corr++;
                else missed++;
                if (f.lexical.category != OGUARD_NONE) pipe_lex++;
                else if (f.embed.category != OGUARD_NONE) pipe_emb++;
                else if (f.final.category != OGUARD_NONE) pipe_judge++;
            } else {
                allow_total++;
                if (f.final.category == OGUARD_NONE) allow_pass++;
            }
            if (lat_n == lat_cap) {
                size_t next = lat_cap ? lat_cap * 2 : 256;
                double *grown = realloc(lat, next * sizeof *grown);
                if (!grown) break;
                lat = grown;
                lat_cap = next;
            }
            lat[lat_n++] = ems;
            lat_sum += ems;
            if (ems > lat_max) lat_max = ems;
            if (f.judge_veto) vetoes++;
            if (f.gate_open) gate_open++;
            printf("%s\t%d\t%s\t%s\t%s\t%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%.2f\t%.1f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%.1f\n",
                   suites[s].key, i, oguard_category_name(suites[s].expect),
                   oguard_category_name(f.lexical.category),
                   oguard_category_name(f.embed.category),
                   oguard_category_name(f.judge.category),
                   oguard_category_name(f.final.category), epro, eben, epro - eben,
                   f.judge_pyes, f.judge_invoked ? 1 : 0, f.judge_veto ? 1 : 0, ems,
                   f.judge_ms, f.emb_cat[0], f.emb_cat[1], f.emb_cat[2],
                   f.judge_single[0], f.judge_single[1], f.judge_single[2],
                   f.gate_pyes, f.gate_open ? 1 : 0, f.gate_ms);
            free(system);
            free(user);
            free(ctxs);
        }
    }
    qsort(lat, lat_n, sizeof *lat, cmp_double);
    double p50 = lat_n ? lat[lat_n / 2] : 0.0;
    double p95 = lat_n ? lat[(lat_n * 95) / 100] : 0.0;
    if (jlat_n) qsort(jlat, jlat_n, sizeof *jlat, cmp_double);
    double jp50 = jlat_n ? jlat[jlat_n / 2] : 0.0;
    double jp95 = jlat_n ? jlat[(jlat_n * 95) / 100] : 0.0;
    printf("summary blocks=%d lex_only=%d emb_only=%d both=%d judge_caught=%d missed=%d allow=%d/%d pipe_lex=%d pipe_emb=%d pipe_judge=%d vetoes=%d judge_inv=%d/%zu gate_open=%d embed_ms_n=%zu avg=%.2f p50=%.2f p95=%.2f max=%.2f judge_ms_n=%zu avg=%.1f p50=%.1f p95=%.1f max=%.1f\n",
           block_total, lex_only, emb_only, both, judge_corr, missed, allow_pass, allow_total,
           pipe_lex, pipe_emb, pipe_judge, vetoes, judge_inv, lat_n, gate_open, lat_n,
           lat_n ? lat_sum / (double)lat_n : 0.0, p50, p95, lat_max, jlat_n,
           jlat_n ? jlat_sum / (double)jlat_n : 0.0, jp50, jp95, jlat_max);
    free(lat);
    free(jlat);
    free(toks);
    free(js);
    if (allow_pass != allow_total) return 1;
    return 0;
}
