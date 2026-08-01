#include "ospec.h"

#include <string.h>

#define OSPEC_DEFAULT_MIN_NGRAM 3
#define OSPEC_DEFAULT_MAX_NGRAM 8

void ospec_config_default(ospec_config *cfg) {
    if (!cfg) return;
    cfg->min_ngram = OSPEC_DEFAULT_MIN_NGRAM;
    cfg->max_ngram = OSPEC_DEFAULT_MAX_NGRAM;
}

int ospec_draft(const ospec_config *cfg, const int32_t *context, int context_len,
                int32_t *draft_out, int draft_cap) {
    if (!cfg || !context || !draft_out || context_len <= 0 || draft_cap <= 0) return 0;

    int min_ngram = cfg->min_ngram > 0 ? cfg->min_ngram : OSPEC_DEFAULT_MIN_NGRAM;
    int max_ngram = cfg->max_ngram > 0 ? cfg->max_ngram : OSPEC_DEFAULT_MAX_NGRAM;
    if (max_ngram < min_ngram) max_ngram = min_ngram;

    /* Longest suffix first: a longer match is a stronger claim about what comes
     * next, and taking the first length that hits means a strong match is never
     * passed over for a weaker one that happens to sit closer to the end. */
    for (int n = max_ngram; n >= min_ngram; n--) {
        if (context_len < n + 1) continue;
        const int32_t *needle = context + context_len - n;

        /* Backwards from the most recent occurrence. In a conversation the
         * nearest repeat is usually the relevant one — the current paragraph
         * rather than something the same words did four turns ago. The upper
         * bound stops the suffix from matching itself. */
        for (int start = context_len - n - 1; start >= 0; start--) {
            if (memcmp(context + start, needle, (size_t)n * sizeof *needle) != 0) continue;

            int available = context_len - (start + n);
            if (available <= 0) continue;
            int take = available < draft_cap ? available : draft_cap;
            memcpy(draft_out, context + start + n, (size_t)take * sizeof *draft_out);
            return take;
        }
    }
    return 0;
}

void ospec_governor_init(ospec_governor *g, int max_length, int probe_interval,
                         int patience) {
    if (!g) return;
    memset(g, 0, sizeof *g);
    g->max_length = max_length > 0 ? max_length : 4;
    g->probe_interval = probe_interval > 0 ? probe_interval : 32;
    g->patience = patience > 0 ? patience : 1;
    /* Starts optimistic rather than off. The first rounds of a request are the
     * cheapest place to find out whether it is the copying kind, and a request
     * that is not pays for that answer once. */
    g->length = g->max_length;
}

int ospec_governor_next(ospec_governor *g) {
    if (!g) return 0;
    if (g->length > 0) return g->length;
    /* Disabled: probe on the interval so a request that starts inventing and
     * later starts quoting is not left decoding one token at a time forever.
     * The probe clock lives here rather than in observe() because only this
     * function can tell a round that declined to speculate from a round that
     * tried and found no matching n-gram — both report zero drafted tokens. */
    if (g->idle_rounds >= g->probe_interval) {
        g->idle_rounds = 0;
        return 1;
    }
    g->idle_rounds++;
    return 0;
}

void ospec_governor_observe(ospec_governor *g, int drafted, int accepted) {
    if (!g) return;
    if (drafted < 0) drafted = 0;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;

    g->rounds++;
    g->drafted += (unsigned long long)drafted;
    g->accepted += (unsigned long long)accepted;
    /* Each accepted draft token is a model call that did not happen: it was
     * produced by a batch slot the unspeculated decode would have paid for
     * anyway. */
    g->saved_calls += (unsigned long long)accepted;

    /* No draft is no evidence either way, so the length is left alone: a probe
     * that found no matching n-gram says nothing about whether this text is
     * the copying kind. */
    if (drafted == 0) return;

    if (accepted == 0) {
        /* A rejection usually means the model is composing rather than echoing,
         * and that state persists for many tokens — but "usually" is not
         * "always", and one miss inside a passage that is otherwise being
         * quoted is not worth abandoning the whole request over. How many
         * misses to sit through before giving up is the patience setting,
         * because the answer depends on what a miss costs on this device. */
        g->misses++;
        if (g->misses >= g->patience) {
            g->length = 0;
            g->misses = 0;
        } else if (g->length > 1) {
            /* Halve rather than hold: a draft that missed entirely was probably
             * reaching too far, and a shorter one is likelier to land. */
            g->length /= 2;
        }
        g->idle_rounds = 0;
        return;
    }

    g->misses = 0;
    g->idle_rounds = 0;
    if (accepted == drafted) {
        /* Every guess landed, so the draft was probably too short to be worth
         * the round trip it saved. */
        if (g->length < g->max_length) g->length++;
        else g->length = g->max_length;
    } else {
        /* Partially right: the draft ran past where the text stopped matching.
         * Settle on the length that was actually accepted. */
        g->length = accepted;
    }
    if (g->length > g->max_length) g->length = g->max_length;
}

double ospec_acceptance_rate(const ospec_governor *g) {
    if (!g || g->drafted == 0) return 0.0;
    return (double)g->accepted / (double)g->drafted;
}
