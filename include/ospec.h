#ifndef OSPEC_H
#define OSPEC_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Speculative decoding: draft several tokens cheaply, then confirm them all in
 * one model call.
 *
 * Decoding one token at a time is not compute-bound, it is bandwidth-bound: the
 * whole weight matrix is read to produce a single token, and reading it for a
 * batch of five costs barely more than reading it for one. So a decode step has
 * four spare token-slots in it that are already paid for. Speculation spends
 * them on guesses.
 *
 * The guesses here come from the context itself, not from a second model. A
 * suffix of what has been written so far is looked up in what came before, and
 * whatever followed it last time becomes the draft. That is free in both VRAM
 * and compute, which matters on a device where the LLM shares memory with an
 * image model — a draft model would need a lease from the VRAM broker before it
 * could hold anything. The verify loop this feeds is the same one a draft model
 * would use, so adding one later is a change of draft source, not of design.
 *
 * Correctness does not depend on the draft being good. The caller samples each
 * token from the real model's distribution and merely *checks* whether it
 * matched the guess; a hit lets it keep the already-computed logits for the next
 * position instead of running the model again. Nothing is ever accepted because
 * the drafter proposed it, so the distribution the caller draws from is the
 * model's own. Speculation is not a quality setting.
 *
 * It is not, however, bit-reproducible, and the reason is worth stating because
 * the obvious claim — "greedy output is unchanged" — is false on real hardware.
 * Verifying k drafts means one decode of width k+1 where there would have been
 * k+1 decodes of width 1, and a wider matmul sums in a different order. The
 * logits differ in the last bits, so a near-tied argmax can land on the other
 * token. Measured on this tree: a prompt whose answer copies its input came out
 * byte-identical, while a creative continuation diverged partway through and
 * stayed fluent. This is the same effect performance/quality.md already records
 * for prefix reuse, and it is why that file tracks greedy agreement as a score
 * rather than asserting determinism.
 *
 * What a bad draft costs is a wider batch that produced one token instead of
 * several. That is nearly free only where decode is bandwidth-bound and the
 * weights are read once per batch rather than once per token — which is to say
 * on a GPU. On CPU the wider batch costs proportionally more arithmetic and
 * there is nothing to reclaim; measured on this host, speculation on CPU is a
 * wash at best. The caller is expected to enable it accordingly.
 *
 * The governor below keeps the cost from accumulating even where it is cheap: a
 * workload that never copies from its context turns speculation off by itself,
 * and probes rarely enough to notice if that stops being true.
 */

typedef struct {
    /* Shortest suffix worth trusting as a match. Below about three tokens a
     * match is coincidence — common words recur everywhere — and the draft it
     * yields is rejected often enough to cost more than it saves. */
    int min_ngram;
    /* Longest suffix tried first. A longer match is a stronger signal, so the
     * search walks down from here and takes the first length that hits. */
    int max_ngram;
} ospec_config;

void ospec_config_default(ospec_config *cfg);

/*
 * Proposes up to draft_cap continuation tokens for context, returning how many
 * were written. Zero means no suffix of the context recurred earlier in it,
 * which is a normal answer: the caller decodes one token the ordinary way.
 *
 * Pure, and deliberately takes plain token ids rather than a llama context, so
 * the draft policy is testable without a GPU or a model.
 */
int ospec_draft(const ospec_config *cfg, const int32_t *context, int context_len,
                int32_t *draft_out, int draft_cap);

/*
 * Adaptive draft length.
 *
 * Speculation pays off in proportion to how repetitive the text is, and that
 * varies by request, not by deployment: a summarization request copies most of
 * its answer out of the prompt while a roleplay continuation invents nearly all
 * of it. A fixed draft length has to be tuned for one of those and is wrong for
 * the other, so the length follows the observed acceptance rate instead.
 *
 * At length zero speculation is off and costs nothing at all. It cannot stay
 * off forever, because a request can change regime halfway through — a model
 * that starts quoting the prompt after ten tokens of preamble should be
 * noticed — so a probe is issued every probe_interval rounds to re-measure.
 */
typedef struct {
    int length;          /* tokens to draft next round; 0 disables speculation */
    int max_length;
    int probe_interval;  /* rounds between probes while disabled */
    int idle_rounds;     /* rounds spent at length 0 since the last probe */
    /*
     * Consecutive missed rounds tolerated before speculation switches off.
     *
     * This is the one hardware-dependent number in the module, and it is the
     * difference between a few percent and a multiple. Giving up after a single
     * miss makes speculation nearly free to be wrong about, which is what a CPU
     * needs; on a GPU a miss costs almost nothing and quitting early forfeits
     * most of the win. Measured on CPU with patience 1, drafting stopped so
     * readily that only 40 of 640 tokens came free despite a 68% acceptance
     * rate on the drafts that were issued.
     */
    int patience;
    int misses;          /* consecutive rounds where no draft token landed */
    unsigned long long rounds;
    unsigned long long drafted;
    unsigned long long accepted;   /* drafted tokens that matched the sample */
    unsigned long long saved_calls; /* model calls speculation avoided */
} ospec_governor;

/* patience of 1 gives up after a single missed round; higher values keep
 * drafting through misses, which is what a device where a miss is nearly free
 * should do. Values below 1 are treated as 1. */
void ospec_governor_init(ospec_governor *g, int max_length, int probe_interval,
                         int patience);

/* How many tokens to draft for the next round, 0 to skip speculating. Advances
 * the probe clock, so it must be called once per decode round even when the
 * caller ends up not speculating. */
int ospec_governor_next(ospec_governor *g);

/* Records the outcome of a verify round: how many tokens were drafted and how
 * many of them the sampler independently agreed with. */
void ospec_governor_observe(ospec_governor *g, int drafted, int accepted);

/* Accepted over drafted, or 0 when nothing has been drafted yet. */
double ospec_acceptance_rate(const ospec_governor *g);

#endif
