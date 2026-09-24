#include "oguard.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct {
    double combined;
    double single[3];
    double gate;
} g_canned = {0.5, {0.0, 0.0, 0.0}, 0.02};

static int g_single_calls;
static int g_multi_calls;
static int g_multi_n;
static int g_gate_calls;

static bool stub_ready(void) { return true; }

static double score_for(const char *prompt) {
    if (strstr(prompt, "Sexual-intent check")) {
        g_gate_calls++;
        return g_canned.gate;
    }
    if (strstr(prompt, "Minor check")) return g_canned.single[0];
    if (strstr(prompt, "Consent check")) return g_canned.single[1];
    if (strstr(prompt, "Real-person check")) return g_canned.single[2];
    return g_canned.combined;
}

static bool stub_score(const char *prompt, size_t len, double *pyes, double *ms) {
    (void)len;
    g_single_calls++;
    if (pyes) *pyes = score_for(prompt);
    if (ms) *ms = 1.0;
    return true;
}

static bool stub_multi(const char **prompts, const size_t *lens, int n,
                       double *pyes, double *ms) {
    (void)lens;
    g_multi_calls++;
    g_multi_n = n;
    for (int i = 0; i < n; i++) pyes[i] = score_for(prompts[i]);
    if (ms) *ms = 1.0;
    return true;
}

static int g_failures;

static void check(bool cond, const char *name) {
    printf("%s %s\n", cond ? "ok" : "FAIL", name);
    if (!cond) g_failures++;
}

int main(void) {
    setenv("OMNISERVE_NATIVE_GUARD_JUDGE_THR_MINOR", "0.52", 1);
    setenv("OMNISERVE_NATIVE_GUARD_JUDGE_THR_NONCON", "0.60", 1);
    setenv("OMNISERVE_NATIVE_GUARD_JUDGE_THR_REAL", "0.0265", 1);
    setenv("OMNISERVE_NATIVE_GUARD_JUDGE_HIGH", "0.90", 1);
    setenv("OMNISERVE_NATIVE_GUARD_JUDGE_VETO", "0.10", 1);
    setenv("OMNISERVE_NATIVE_GUARD_JUDGE_GATE_THR", "0.50", 1);
    setenv("OMNISERVE_NATIVE_GUARD_JUDGE_CACHE_SIZE", "8", 1);
    oguard_init();
    oguard_set_judge_backend(stub_ready, stub_score);
    oguard_set_judge_multi_backend(stub_multi);
    oguard_judge_warmup();
    check(oguard_judge_ready(), "judge live with stub backend");

    g_canned.combined = 0.5;
    g_canned.single[0] = 0.10;
    g_canned.single[1] = 0.50;
    g_canned.single[2] = 0.03;
    g_single_calls = 0;
    g_multi_calls = 0;
    oguard_full f = oguard_classify_full(
        NULL, "Write an explicit consensual love scene between two married adults, case one.", NULL);
    check(f.judge_invoked && f.judge_ok, "middle band invoked");
    check(f.judge.category == OGUARD_REALPERSON_SEXUAL, "any-trigger blocks non-argmax real");
    check(f.final.category == OGUARD_REALPERSON_SEXUAL, "final follows judge block");
    check(g_multi_calls == 1 && g_multi_n == 3, "singles batched in one multi call");
    check(g_single_calls == 1, "combined scored once");

    g_canned.single[0] = 0.10;
    g_canned.single[1] = 0.65;
    g_canned.single[2] = 0.01;
    f = oguard_classify_full(
        NULL, "Write an explicit consensual love scene between two married adults, case two.", NULL);
    check(f.judge.category == OGUARD_NONCON_SEXUAL, "argmax trigger keeps category");

    g_canned.single[0] = 0.10;
    g_canned.single[1] = 0.50;
    g_canned.single[2] = 0.02;
    f = oguard_classify_full(
        NULL, "Write an explicit consensual love scene between two married adults, case three.", NULL);
    check(f.judge.category == OGUARD_NONE && f.judge_ok, "all-below passes");

    g_canned.combined = 0.05;
    int singles_before = g_multi_calls;
    f = oguard_classify_full(
        NULL, "Write an explicit consensual love scene between two married adults, case four.", NULL);
    check(f.judge.category == OGUARD_NONE, "veto fast pass");
    check(g_multi_calls == singles_before, "fast pass skips singles");

    g_canned.combined = 0.5;
    g_canned.single[0] = 0.10;
    g_canned.single[1] = 0.50;
    g_canned.single[2] = 0.02;
    int calls_before = g_single_calls + g_multi_calls;
    f = oguard_classify_full(
        NULL, "Write an explicit consensual love scene between two married adults, case three.", NULL);
    check(f.judge_cached, "repeat prompt served from LRU");
    check(g_single_calls + g_multi_calls == calls_before, "cache hit makes no backend call");

    g_canned.gate = 0.02;
    g_canned.combined = 0.95;
    g_canned.single[0] = 0.95;
    g_canned.single[1] = 0.95;
    g_canned.single[2] = 0.95;
    int multi_before = g_multi_calls;
    int single_before = g_single_calls;
    int gate_before = g_gate_calls;
    f = oguard_classify_full(
        NULL, "He walks her upstairs after the party. Describe the rest, gate one.", NULL);
    check(!f.gate_open && f.gate_pyes == 0.02, "gate closed on low sexual intent");
    check(f.judge.category == OGUARD_NONE && f.judge_ok && f.judge_invoked,
          "gate-closed passes without category");
    check(g_multi_calls == multi_before, "gate-closed skips category calls");
    check(g_single_calls == single_before + 1 && g_gate_calls == gate_before + 1,
          "gate scored exactly once");

    g_canned.gate = 0.90;
    g_canned.combined = 0.5;
    g_canned.single[0] = 0.10;
    g_canned.single[1] = 0.65;
    g_canned.single[2] = 0.01;
    f = oguard_classify_full(
        NULL, "He walks her upstairs after the party. Describe the rest, gate two.", NULL);
    check(f.gate_open, "gate opens on high sexual intent");
    check(f.judge.category == OGUARD_NONCON_SEXUAL, "gate-open proceeds to category judge");

    gate_before = g_gate_calls;
    f = oguard_classify_full(
        NULL, "Write an explicit consensual love scene between two married adults, gate three.", NULL);
    check(g_gate_calls == gate_before, "keyword short-circuit skips gate call");
    check(f.gate_open && f.gate_pyes == 1.0, "short-circuit reports open");

    calls_before = g_single_calls + g_multi_calls;
    f = oguard_classify_full(
        NULL, "He walks her upstairs after the party. Describe the rest, gate one.", NULL);
    check(f.judge_cached && !f.gate_open, "gate-closed verdict served from LRU");
    check(g_single_calls + g_multi_calls == calls_before, "cached gate makes no backend call");

    if (g_failures) fprintf(stderr, "guard judge: %d failures\n", g_failures);
    return g_failures ? 1 : 0;
}
