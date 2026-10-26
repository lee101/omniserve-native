#define _GNU_SOURCE
#include "obackend.h"
#include "osched.h"
#include "ospec.h"
#include "otext.h"
#include "otune.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

int ollm_parse_tensor_overrides(const char *spec, ollm_tensor_override *out, int cap) {
    if (!spec || !spec[0]) return 0;
    if (!out || cap <= 0) return -1;
    int count = 0;
    const char *p = spec;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        const char *comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);
        while (len > 0 && (p[len - 1] == ' ' || p[len - 1] == '\t')) len--;
        const char *eq = memchr(p, '=', len);
        if (!eq) return -1;
        const char *pend = eq;
        while (pend > p && (pend[-1] == ' ' || pend[-1] == '\t')) pend--;
        const char *bstart = eq + 1;
        while (bstart < p + len && (*bstart == ' ' || *bstart == '\t')) bstart++;
        size_t plen = (size_t)(pend - p);
        size_t blen = (size_t)(p + len - bstart);
        if (plen == 0 || plen >= sizeof out[0].pattern) return -1;
        if (blen == 0 || blen >= 16) return -1;
        char buft[16];
        memcpy(buft, bstart, blen);
        buft[blen] = 0;
        bool cpu;
        if (strcasecmp(buft, "CPU") == 0) cpu = true;
        else if (strcasecmp(buft, "CUDA0") == 0) cpu = false;
        else return -1;
        if (count >= cap) return -1;
        memcpy(out[count].pattern, p, plen);
        out[count].pattern[plen] = 0;
        out[count].cpu = cpu;
        count++;
        if (!comma) break;
        p = comma + 1;
    }
    return count;
}

int ollm_parse_moe_cpu_experts(const char *spec, ollm_moe_mode *mode_out, int *n_out) {
    ollm_moe_mode mode = OLLM_MOE_OFF;
    int n = 0;
    if (spec && spec[0]) {
        while (*spec == ' ' || *spec == '\t') spec++;
        if (strcasecmp(spec, "all") == 0) mode = OLLM_MOE_ALL;
        else if (strcasecmp(spec, "auto") == 0) mode = OLLM_MOE_AUTO;
        else {
            char *end = NULL;
            long v = strtol(spec, &end, 10);
            if (end == spec || v < 0 || v > 100000) return -1;
            while (*end == ' ' || *end == '\t') end++;
            if (*end) return -1;
            if (v > 0) {
                mode = OLLM_MOE_N;
                n = (int)v;
            }
        }
    }
    if (mode_out) *mode_out = mode;
    if (n_out) *n_out = n;
    return 0;
}

void ollm_spec_mtp_default(ollm_spec_mtp_config *cfg) {
    if (!cfg) return;
    cfg->draft_max = 1;
    /* Zero: the head decode is sunk cost before confidence is known, so the
     * filter only saves the verify widening while forfeiting unsure-but-right
     * drafts. Measured on CPU: p_min 0.5 accepts more (61% vs 58%) but runs
     * 7% slower than no filter. */
    cfg->p_min = 0.0f;
}

int ollm_spec_mtp_parse(ollm_spec_mtp_config *cfg, const char *draft_env,
                        const char *pmin_env) {
    if (!cfg) return -1;
    ollm_spec_mtp_default(cfg);
    if (draft_env && draft_env[0]) {
        char *end = NULL;
        long v = strtol(draft_env, &end, 10);
        if (end == draft_env || v < 0 || v > OLLM_SPEC_MTP_DRAFT_MAX) return -1;
        while (*end == ' ' || *end == '\t') end++;
        if (*end) return -1;
        cfg->draft_max = (int)v;
    }
    if (pmin_env && pmin_env[0]) {
        char *end = NULL;
        float v = strtof(pmin_env, &end);
        if (end == pmin_env || !(v >= 0.0f) || !(v <= 1.0f)) return -1;
        while (*end == ' ' || *end == '\t') end++;
        if (*end) return -1;
        cfg->p_min = v;
    }
    return 0;
}

bool ollm_tensor_is_expert(const char *name) {
    if (!name) return false;
    if (!strstr(name, "exps")) return false;
    return strstr(name, "ffn_") != NULL;
}

int ollm_tensor_block_index(const char *name) {
    if (!name) return -1;
    const char *blk = strstr(name, "blk.");
    if (!blk) return -1;
    char *end = NULL;
    long v = strtol(blk + 4, &end, 10);
    if (end == blk + 4 || v < 0 || v > 100000 || *end != '.') return -1;
    return (int)v;
}

int ollm_gpu_expert_layers(const unsigned long long *expert_bytes, int n_layers,
                           unsigned long long nonexp_bytes, unsigned long long kv_bytes,
                           unsigned long long compute_reserve, unsigned long long budget) {
    if (!expert_bytes || n_layers <= 0) return 0;
    if (nonexp_bytes > ULLONG_MAX - kv_bytes) return 0;
    unsigned long long base = nonexp_bytes + kv_bytes;
    if (base > ULLONG_MAX - compute_reserve) return 0;
    base += compute_reserve;
    int best = 0;
    unsigned long long acc = 0;
    for (int k = 1; k <= n_layers; k++) {
        if (expert_bytes[k - 1] > ULLONG_MAX - acc) break;
        acc += expert_bytes[k - 1];
        if (acc > ULLONG_MAX - base) break;
        if (base + acc <= budget) best = k;
        else break;
    }
    return best;
}

double ollm_kv_type_size(const char *kv_type) {
    if (!kv_type) return 2.0;
    if (strcasecmp(kv_type, "q8_0") == 0) return 34.0 / 32.0;
    if (strcasecmp(kv_type, "q4_0") == 0) return 18.0 / 32.0;
    if (strcasecmp(kv_type, "q5_1") == 0) return 24.0 / 32.0;
    return 2.0;
}

void ollm_strip_channels(const char *src, char *dst, size_t cap) {
    if (!dst || cap == 0) return;
    static const char open[] = "<|channel>";
    static const char close[] = "<channel|>";
    size_t n = 0;
    if (src) {
        const char *p = src;
        for (;;) {
            const char *cut = strstr(p, open);
            size_t keep = cut ? (size_t)(cut - p) : strlen(p);
            size_t room = n + 1 < cap ? cap - n - 1 : 0;
            if (room > 0) {
                if (keep > room) keep = room;
                memmove(dst + n, p, keep);
                n += keep;
            }
            if (!cut) break;
            const char *end = strstr(cut + sizeof open - 1, close);
            if (!end) break;
            p = end + sizeof close - 1;
        }
    }
    size_t r = 0, w = 0;
    while (r < n) {
        if (n - r >= sizeof close - 1 && memcmp(dst + r, close, sizeof close - 1) == 0) {
            r += sizeof close - 1;
        } else {
            dst[w++] = dst[r++];
        }
    }
    dst[w] = 0;
}

void ollm_strip_channels_final(char *text, size_t cap) {
    if (!text || cap == 0) return;
    ollm_strip_channels(text, text, cap);
    static const char *markers[] = {"<|channel>", "<channel|>"};
    size_t len = strlen(text);
    for (size_t k = 9; k >= 2; k--) {
        if (len < k) continue;
        for (size_t m = 0; m < 2; m++) {
            if (memcmp(text + len - k, markers[m], k) == 0) {
                text[len - k] = 0;
                return;
            }
        }
    }
}

bool ollm_thinking_default(void) {
    const char *v = getenv("OMNISERVE_NATIVE_LLM_THINKING_DEFAULT");
    if (!v || !v[0]) return false;
    return strcasecmp(v, "1") == 0 || strcasecmp(v, "true") == 0 ||
        strcasecmp(v, "yes") == 0 || strcasecmp(v, "on") == 0;
}

bool ollm_regex_balanced(const char *pattern) {
    if (!pattern || !pattern[0]) return false;
    int depth = 0;
    bool atom = false;
    for (size_t i = 0; pattern[i]; i++) {
        char c = pattern[i];
        if (c == '\\') {
            if (!pattern[i + 1]) return false;
            i++;
            atom = true;
            continue;
        }
        if (c == '[') {
            i++;
            if (pattern[i] == '^') i++;
            if (pattern[i] == ']') i++;
            bool closed = false;
            for (; pattern[i]; i++) {
                if (pattern[i] == '\\' && pattern[i + 1]) {
                    i++;
                    continue;
                }
                if (pattern[i] == ']') {
                    closed = true;
                    break;
                }
            }
            if (!closed) return false;
            atom = true;
            continue;
        }
        if (c == '(') {
            depth++;
            atom = false;
            continue;
        }
        if (c == ')') {
            if (depth == 0) return false;
            depth--;
            atom = true;
            continue;
        }
        if (c == '*' || c == '+' || c == '?') {
            if (!atom) return false;
            continue;
        }
        if (c == '{') {
            size_t j = i + 1;
            unsigned long lo = 0, hi = 0;
            bool range = false;
            while (pattern[j] >= '0' && pattern[j] <= '9') {
                lo = lo * 10 + (unsigned long)(pattern[j++] - '0');
                range = true;
            }
            if (pattern[j] == ',') {
                j++;
                range = false;
                while (pattern[j] >= '0' && pattern[j] <= '9') {
                    hi = hi * 10 + (unsigned long)(pattern[j++] - '0');
                    range = true;
                }
                if (!range) hi = lo;
            } else {
                hi = lo;
            }
            if (range && pattern[j] == '}' && atom && hi >= lo) {
                i = j;
                continue;
            }
            atom = true;
            continue;
        }
        if (c == '|' || c == '^' || c == '$' || c == '.') {
            if (c == '|') atom = false;
            else if (c == '.') atom = true;
            continue;
        }
        atom = true;
    }
    return depth == 0;
}

int ollm_moe_cpu_pattern(int first_layer, int n_layers, char *out, size_t cap) {
    static const char tail[] = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    if (!out || cap == 0 || first_layer < 0 || n_layers <= first_layer) return -1;
    size_t n = 0;
    if (n + 5 >= cap) return -1;
    memcpy(out + n, "blk\\.(", 6);
    n += 6;
    for (int layer = first_layer; layer < n_layers; layer++) {
        char num[16];
        int len = snprintf(num, sizeof num, "%s%d", layer > first_layer ? "|" : "", layer);
        if (len < 0 || (size_t)len + n + sizeof tail >= cap) return -1;
        memcpy(out + n, num, (size_t)len);
        n += (size_t)len;
    }
    if (n + 1 + sizeof tail > cap) return -1;
    out[n++] = ')';
    memcpy(out + n, tail, sizeof tail);
    return 0;
}

typedef struct {
    bool valid;
    bool active;
    char path[1024];
    int ctx_len;
    int contexts;
    char kv_type[16];
    ollm_moe_mode mode;
    int n_layers;
    int cpu_layers;
    unsigned long long free_bytes;
    unsigned long long budget_bytes;
    unsigned long long nonexp_bytes;
    unsigned long long kv_bytes;
    unsigned long long cpu_bytes;
    unsigned long long est_bytes;
} moe_decision;

static moe_decision g_moe_decision;

bool ollm_moe_last_active(void) { return g_moe_decision.valid && g_moe_decision.active; }

unsigned long long ollm_moe_last_free_bytes(void) {
    return g_moe_decision.valid ? g_moe_decision.free_bytes : 0;
}

unsigned long long ollm_moe_last_est_bytes(void) {
    return g_moe_decision.valid ? g_moe_decision.est_bytes : 0;
}

#ifdef USE_LLAMA
#include "llama.h"

#include <dlfcn.h>

#define PENALTY_LAST_N_DEFAULT 256

static int penalty_last_n_default(void) {
    const char *v = getenv("OMNISERVE_NATIVE_PENALTY_LAST_N");
    return v && v[0] ? atoi(v) : PENALTY_LAST_N_DEFAULT;
}

/* ggml's backend-device registry lives in libggml-base, which reaches this
 * binary only as a transitive dependency of libllama. Resolving through the
 * already-loaded image keeps the link line unchanged and degrades to
 * "unknown placement" if the ABI ever moves. */
enum { OGGML_DEVICE_TYPE_GPU = 1 };
typedef void *oggml_dev;
typedef oggml_dev (*oggml_dev_by_type_fn)(int type);
typedef const char *(*oggml_dev_description_fn)(oggml_dev dev);

static oggml_dev_by_type_fn oggml_dev_by_type;
static oggml_dev_description_fn oggml_dev_description;
static bool g_gpu_device_present;
static bool g_gpu_requested;
static bool g_on_gpu;
static char g_device_desc[96];
static char g_kv_type_name[16] = "f16";
static char g_tune_class[16] = "unknown";
static bool g_flash_attn;

static struct llama_model *g_model;
static const struct llama_vocab *g_vocab;
/* Next-token-prediction staging API from llama.cpp's src/llama-ext.h. That
 * header is C++-only, and the symbols it declares have C++ linkage, so they
 * are resolved by mangled name out of the already-loaded libllama exactly
 * like the ggml device probes above. Pinned to the same checkout as the
 * rest of this backend; a missing symbol disables MTP, nothing else. */
typedef void (*onextn_set_fn)(struct llama_context *ctx, bool value, bool masked);
typedef float *(*onextn_get_ith_fn)(struct llama_context *ctx, int32_t i);
typedef struct llama_context *(*oget_ctx_other_fn)(struct llama_context *ctx);
static onextn_set_fn ollama_set_nextn;
static onextn_get_ith_fn ollama_get_nextn_ith;
static oget_ctx_other_fn ollama_get_ctx_other;

static bool probe_nextn_fns(void) {
    if (!ollama_set_nextn) {
        ollama_set_nextn = (onextn_set_fn)dlsym(
            RTLD_DEFAULT, "_Z26llama_set_embeddings_nextnP13llama_contextbb");
        ollama_get_nextn_ith = (onextn_get_ith_fn)dlsym(
            RTLD_DEFAULT, "_Z30llama_get_embeddings_nextn_ithP13llama_contexti");
        ollama_get_ctx_other = (oget_ctx_other_fn)dlsym(
            RTLD_DEFAULT, "_Z19llama_get_ctx_otherP13llama_context");
    }
    return ollama_set_nextn && ollama_get_nextn_ith && ollama_get_ctx_other;
}

typedef struct {
    float probability;
} probability_capture;

enum { OLLM_VERIFY_BATCH_CAP = 17 };
enum { OLLM_RING = 64 };
enum { OLLM_SHARE_MIN = 32 };
enum { OLLM_MTP_DRAFT_CAP = 2 };
enum { OLLM_MTP_BATCH_DRAFT_MAX = 8 };
typedef struct ollm_slot_s {
    struct llama_context *ctx;
    /* MTP draft context sharing this slot's target context: one per parallel
     * target context, travelling with the slot so prefix reuse keeps working.
     * NULL unless an MTP head is loaded and this slot initialized one. */
    struct llama_context *mtp_ctx;
    struct llama_sampler *mtp_smpl; /* chain holding only top_k(10) */
    float *mtp_pending; /* h row of the last committed token, [n_embd] */
    float *mtp_verify;  /* h rows of the last verify batch, [rows][n_embd] */
    float *mtp_embd;    /* staging row for the single-row draft batch */
    llama_token_data *mtp_cand; /* draft sampling candidates, [n_vocab] */
    llama_token mtp_token;
    llama_pos mtp_pos;
    int32_t mtp_n_seq;
    llama_seq_id mtp_seq;
    llama_seq_id *mtp_seq_ptr;
    int8_t mtp_logits;
    bool busy;
    llama_token *cached_tokens;
    int cached_count;
    int cached_cap;
    /* Speculative verification is bounded to sixteen drafts plus the token
     * that precedes them. Keep that tiny batch in the slot: llama_batch_init
     * allocates every member separately, and doing that once per decode round
     * puts allocator traffic on the hottest CPU path. A busy slot has exactly
     * one owner, so these arrays need no additional lock. */
    llama_token verify_tokens[OLLM_VERIFY_BATCH_CAP];
    llama_pos verify_positions[OLLM_VERIFY_BATCH_CAP];
    int32_t verify_n_seq_ids[OLLM_VERIFY_BATCH_CAP];
    llama_seq_id verify_seq_ids[OLLM_VERIFY_BATCH_CAP];
    llama_seq_id *verify_seq_id_ptrs[OLLM_VERIFY_BATCH_CAP];
    int8_t verify_logits[OLLM_VERIFY_BATCH_CAP];
    /* Continuous batching: the slot is a sequence id in the shared context.
     * Everything below is guarded by g_engine_lock except while in_step, when
     * the engine thread alone touches the sampler and the step fields. */
    llama_seq_id seq;
    int state;
    bool in_step;
    bool stop;
    bool failed;
    bool has_next;
    int rm_from;
    const llama_token *prompt;
    int prompt_len;
    int prompt_done;
    int step_chunk;
    int row;
    llama_token next;
    int n_past;
    int produced;
    int max_new;
    int prefilter_k;
    unsigned long long order;
    /* Prefix fork: the engine copies [0, cp_len) of cp_src's cache into this
     * sequence instead of prefilling it. cp_src is pinned (not acquirable)
     * until the copy runs, so its prefix cannot be trimmed underneath it. */
    struct ollm_slot_s *cp_src;
    int cp_len;
    int pins;
    unsigned long long last_used;
    struct llama_sampler *smpl;
    const probability_capture *capture;
    llama_token_data *cand;
    const float *logits_row;
    llama_token ring[OLLM_RING];
    float ring_p[OLLM_RING];
    int ring_head;
    int ring_count;
    pthread_cond_t cond;
    /* MTP in the batched engine: drafts riding behind `next` in this step,
     * their verify rows' logits, and the run the sampler produced from them.
     * h_row is the row of the last target decode (h_dec) whose hidden state
     * pairs with `next`. */
    int draft_cap;
    int n_draft;
    int n_pad; /* verify rows past the drafts that only hold the step's shape */
    bool draft_dummy;
    int draft_row;
    int draft_row_prev;
    int matched;
    int n_out;
    int h_row;
    unsigned long long h_dec;
    int req_drafted;
    int req_accepted;
    llama_token draft[OLLM_MTP_BATCH_DRAFT_MAX];
    float draft_conf[OLLM_MTP_BATCH_DRAFT_MAX]; /* head's own softmax max per draft */
    float draft_cum;
    llama_token out_tok[OLLM_MTP_BATCH_DRAFT_MAX + 1];
    float out_p[OLLM_MTP_BATCH_DRAFT_MAX + 1];
    const float *vrows[OLLM_MTP_BATCH_DRAFT_MAX + 1];
    llama_token_data *vcand[OLLM_MTP_BATCH_DRAFT_MAX];
    long gn[OLLM_MTP_BATCH_DRAFT_MAX + 1]; /* pre-gathered count per verify row, -1 none */
} ollm_slot;
static ollm_slot *g_slots;
static int g_slot_count;
static pthread_mutex_t g_slot_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_slot_cond = PTHREAD_COND_INITIALIZER;
static char g_model_name[256];
static int g_ctx_len;
static int g_batch_size;
static int g_ubatch_size;

/* Speculation settings and fleet-wide counters. The counters are the only way
 * to tell "speculation is off" from "speculation is running and never landing",
 * which look identical from latency alone. */
static int g_spec_draft_max;
static int g_spec_probe_interval;
static int g_spec_patience;
static ospec_config g_spec_cfg;
static pthread_mutex_t g_spec_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long g_spec_drafted;
static unsigned long long g_spec_accepted;
static unsigned long long g_spec_rounds;
static unsigned long long g_spec_saved_calls;

/* MTP-head speculation. Empty GGUF path means off; when a head is loaded it
 * replaces prompt-lookup as the draft source while reusing the same verify
 * loop, governor, and counters. */
static struct llama_model *g_spec_mtp_model;
static const struct llama_vocab *g_spec_mtp_vocab;
static ollm_spec_mtp_config g_spec_mtp_cfg;
static int32_t g_spec_mtp_embd;
static int32_t g_spec_mtp_vocab_size;
static char g_spec_mtp_path[1024];

static bool spec_mtp_active(void) {
    return g_spec_mtp_model != NULL && g_spec_mtp_cfg.draft_max > 0;
}

static struct llama_context *g_mtp_ctx;
static int g_mtp_draft;

static const char *spec_source_name(void) {
    if (spec_mtp_active() || g_mtp_ctx) return "mtp";
    if (g_spec_draft_max > 0) return "prompt-lookup";
    return "off";
}

static int spec_effective_max(void) {
    if (spec_mtp_active()) return g_spec_mtp_cfg.draft_max;
    if (g_mtp_ctx) return g_mtp_draft;
    return g_spec_draft_max;
}

static void spec_record(const ospec_governor *g) {
    if (!g || g->rounds == 0) return;
    pthread_mutex_lock(&g_spec_lock);
    g_spec_rounds += g->rounds;
    g_spec_drafted += g->drafted;
    g_spec_accepted += g->accepted;
    g_spec_saved_calls += g->saved_calls;
    pthread_mutex_unlock(&g_spec_lock);
}

static struct llama_model *g_embed_model;
static const struct llama_vocab *g_embed_vocab;
static struct llama_context *g_embed_ctx;
static pthread_mutex_t g_embed_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_embed_model_name[256];
static int g_embed_ctx_len;

static pthread_mutex_t g_runtime_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_runtime_refs;
/* Model replacement is an exclusive operation. Readers cover the complete
 * chat request, including prompt formatting and decode, so an admin unload
 * cannot free a llama model while a request still holds its vocab/context. */
static pthread_rwlock_t g_model_lifecycle_lock = PTHREAD_RWLOCK_INITIALIZER;

static char g_ovr_tensor[512];
static char g_ovr_moe[32];
static char g_ovr_mtp[4096];
static bool g_ovr_set;

void ollm_set_load_overrides(const char *tensor_override, const char *moe_cpu_experts,
                             const char *spec_mtp_gguf) {
    snprintf(g_ovr_tensor, sizeof g_ovr_tensor, "%s", tensor_override ? tensor_override : "");
    snprintf(g_ovr_moe, sizeof g_ovr_moe, "%s", moe_cpu_experts ? moe_cpu_experts : "");
    snprintf(g_ovr_mtp, sizeof g_ovr_mtp, "%s", spec_mtp_gguf ? spec_mtp_gguf : "");
    g_ovr_set = true;
}

static const char *ollm_cfg(const char *name, const char *ovr) {
    return g_ovr_set ? ovr : getenv(name);
}

static void llama_runtime_acquire(void) {
    pthread_mutex_lock(&g_runtime_lock);
    if (g_runtime_refs++ == 0) llama_backend_init();
    pthread_mutex_unlock(&g_runtime_lock);
}

static void llama_runtime_release(void) {
    pthread_mutex_lock(&g_runtime_lock);
    if (g_runtime_refs > 0 && --g_runtime_refs == 0) llama_backend_free();
    pthread_mutex_unlock(&g_runtime_lock);
}

/* Captures the greedy (max) raw-softmax probability of each step. Placed at
 * the HEAD of the sampler chain so it sees unfiltered logits — matching
 * text-generator.io's min_probability rule: cumulative product of per-step
 * max softmax probabilities over the full vocab. */

static const char *probability_capture_name(const struct llama_sampler *sampler) {
    (void)sampler;
    return "probability-capture";
}

static void probability_capture_apply(struct llama_sampler *sampler,
                                      llama_token_data_array *candidates) {
    probability_capture *capture = sampler->ctx;
    capture->probability = 1.0f;
    if (candidates->size == 0) return;
    float max_logit = candidates->data[0].logit;
    for (size_t i = 1; i < candidates->size; i++) {
        if (candidates->data[i].logit > max_logit) max_logit = candidates->data[i].logit;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < candidates->size; i++) {
        sum += expf(candidates->data[i].logit - max_logit);
    }
    if (sum > 0.0f) capture->probability = 1.0f / sum;
}

static struct llama_sampler_i probability_capture_i = {
    .name = probability_capture_name,
    .apply = probability_capture_apply,
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Must run after llama_backend_init(), which is what registers the devices. */
static void probe_gpu_device(void) {
    if (!oggml_dev_by_type) {
        oggml_dev_by_type = (oggml_dev_by_type_fn)dlsym(RTLD_DEFAULT, "ggml_backend_dev_by_type");
        oggml_dev_description =
            (oggml_dev_description_fn)dlsym(RTLD_DEFAULT, "ggml_backend_dev_description");
    }
    if (!oggml_dev_by_type) {
        snprintf(g_device_desc, sizeof g_device_desc, "unknown");
        return;
    }
    oggml_dev dev = oggml_dev_by_type(OGGML_DEVICE_TYPE_GPU);
    g_gpu_device_present = dev != NULL;
    const char *desc = dev && oggml_dev_description ? oggml_dev_description(dev) : NULL;
    snprintf(g_device_desc, sizeof g_device_desc, "%s", desc ? desc : (dev ? "gpu" : "cpu"));
}

typedef void *oggml_buft;
typedef oggml_dev (*odev_by_name_fn)(const char *name);
typedef oggml_buft (*odev_buft_fn)(oggml_dev dev);
typedef oggml_buft (*ocpu_buft_fn)(void);

static odev_by_name_fn oggml_dev_by_name;
static odev_buft_fn oggml_dev_buft;
static ocpu_buft_fn oggml_cpu_buft;

static void probe_buft_fns(void) {
    if (!oggml_dev_by_name) {
        oggml_dev_by_name = (odev_by_name_fn)dlsym(RTLD_DEFAULT, "ggml_backend_dev_by_name");
        oggml_dev_buft =
            (odev_buft_fn)dlsym(RTLD_DEFAULT, "ggml_backend_dev_buffer_type");
        oggml_cpu_buft =
            (ocpu_buft_fn)dlsym(RTLD_DEFAULT, "ggml_backend_cpu_buffer_type");
    }
}

enum { MOE_MAX_LAYERS = 1024 };

typedef struct {
    unsigned long long *expert_bytes;
    int n_layers;
    unsigned long long nonexp_bytes;
    int n_head_kv;
    int key_len;
    int n_layer_meta;
} moe_scan;

static moe_scan g_scan;
static char g_scan_path[1024];
static struct stat g_scan_st;
static bool g_scan_valid;

static void moe_scan_reset(moe_scan *scan) {
    free(scan->expert_bytes);
    memset(scan, 0, sizeof *scan);
}

static int64_t moe_meta_key(const struct gguf_context *gguf, const char *arch, const char *suffix) {
    char key[96];
    snprintf(key, sizeof key, "%.70s.%s", arch ? arch : "unknown", suffix);
    int64_t id = gguf_find_key(gguf, key);
    if (id >= 0) return id;
    snprintf(key, sizeof key, "general.%s", suffix);
    return gguf_find_key(gguf, key);
}

static int moe_meta_u32(const struct gguf_context *gguf, int64_t id) {
    enum gguf_type type = gguf_get_kv_type(gguf, id);
    if (type == GGUF_TYPE_UINT32) {
        uint32_t v = gguf_get_val_u32(gguf, id);
        return v <= INT_MAX ? (int)v : 0;
    }
    if (type == GGUF_TYPE_INT32) {
        int32_t v = gguf_get_val_i32(gguf, id);
        return v > 0 && v <= INT_MAX ? v : 0;
    }
    if (type == GGUF_TYPE_UINT64) {
        uint64_t v = gguf_get_val_u64(gguf, id);
        return v <= INT_MAX ? (int)v : 0;
    }
    if (type == GGUF_TYPE_INT64) {
        int64_t v = gguf_get_val_i64(gguf, id);
        return v > 0 && v <= INT_MAX ? (int)v : 0;
    }
    return 0;
}

static bool moe_scan_file(const char *path, moe_scan *out) {
    moe_scan_reset(out);
    struct gguf_init_params params;
    memset(&params, 0, sizeof params);
    params.no_alloc = true;
    struct gguf_context *gguf = gguf_init_from_file(path, params);
    if (!gguf) return false;
    bool ok = false;
    int64_t n_tensors = gguf_get_n_tensors(gguf);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(gguf, i);
        if (!name) continue;
        size_t size = gguf_get_tensor_size(gguf, i);
        if (!ollm_tensor_is_expert(name)) {
            out->nonexp_bytes += (unsigned long long)size;
            continue;
        }
        int layer = ollm_tensor_block_index(name);
        if (layer < 0 || layer >= MOE_MAX_LAYERS) {
            if (layer >= MOE_MAX_LAYERS)
                fprintf(stderr, "MoE scan: tensor %s beyond layer cap, counting as dense\n", name);
            out->nonexp_bytes += (unsigned long long)size;
            continue;
        }
        if (layer >= out->n_layers) {
            int grown_n = layer + 1;
            unsigned long long *grown =
                realloc(out->expert_bytes, (size_t)grown_n * sizeof *grown);
            if (!grown) goto done;
            memset(grown + out->n_layers, 0,
                   (size_t)(grown_n - out->n_layers) * sizeof *grown);
            out->expert_bytes = grown;
            out->n_layers = grown_n;
        }
        out->expert_bytes[layer] += (unsigned long long)size;
    }
    const char *arch = "unknown";
    int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(gguf, arch_id) == GGUF_TYPE_STRING) {
        arch = gguf_get_val_str(gguf, arch_id);
    }
    int64_t id = moe_meta_key(gguf, arch, "block_count");
    if (id >= 0) out->n_layer_meta = moe_meta_u32(gguf, id);
    id = moe_meta_key(gguf, arch, "attention.head_count_kv");
    if (id >= 0) {
        if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_n(gguf, id) > 0) {
            enum gguf_type elem = gguf_get_arr_type(gguf, id);
            size_t n = gguf_get_arr_n(gguf, id);
            const void *data = gguf_get_arr_data(gguf, id);
            int peak = 0;
            if (data && elem == GGUF_TYPE_UINT32) {
                const uint32_t *heads = data;
                for (size_t k = 0; k < n; k++) {
                    if (heads[k] <= INT_MAX && (int)heads[k] > peak) peak = (int)heads[k];
                }
            } else if (data && elem == GGUF_TYPE_INT32) {
                const int32_t *heads = data;
                for (size_t k = 0; k < n; k++) {
                    if (heads[k] > peak && heads[k] <= INT_MAX) peak = heads[k];
                }
            } else if (data && elem == GGUF_TYPE_UINT64) {
                const uint64_t *heads = data;
                for (size_t k = 0; k < n; k++) {
                    if (heads[k] <= INT_MAX && (int)heads[k] > peak) peak = (int)heads[k];
                }
            } else if (data && elem == GGUF_TYPE_INT64) {
                const int64_t *heads = data;
                for (size_t k = 0; k < n; k++) {
                    if (heads[k] > peak && heads[k] <= INT_MAX) peak = (int)heads[k];
                }
            }
            out->n_head_kv = peak;
        } else {
            out->n_head_kv = moe_meta_u32(gguf, id);
        }
    }
    id = moe_meta_key(gguf, arch, "attention.key_length");
    if (id >= 0 && moe_meta_u32(gguf, id) > 0) {
        out->key_len = moe_meta_u32(gguf, id);
    } else {
        int n_embd = 0, n_head = 0;
        id = moe_meta_key(gguf, arch, "embedding_length");
        if (id >= 0) n_embd = moe_meta_u32(gguf, id);
        id = moe_meta_key(gguf, arch, "attention.head_count");
        if (id >= 0) n_head = moe_meta_u32(gguf, id);
        if (n_embd > 0 && n_head > 0) out->key_len = n_embd / n_head;
    }
    ok = true;
done:
    gguf_free(gguf);
    if (!ok) moe_scan_reset(out);
    return ok;
}

static const moe_scan *moe_cached_scan(const char *path) {
    if (!path || !path[0]) return NULL;
    struct stat st;
    if (stat(path, &st) != 0) return NULL;
    if (g_scan_valid && strcmp(g_scan_path, path) == 0 &&
        g_scan_st.st_ino == st.st_ino && g_scan_st.st_mtime == st.st_mtime &&
        g_scan_st.st_mtim.tv_nsec == st.st_mtim.tv_nsec &&
        g_scan_st.st_size == st.st_size) {
        return &g_scan;
    }
    if (!moe_scan_file(path, &g_scan)) {
        g_scan_valid = false;
        return NULL;
    }
    snprintf(g_scan_path, sizeof g_scan_path, "%s", path);
    g_scan_st = st;
    g_scan_valid = true;
    return &g_scan;
}

static unsigned long long moe_kv_bytes(const moe_scan *scan, int ctx_len,
                                       int contexts, const char *kv_type) {
    if (!scan || scan->n_head_kv <= 0 || scan->key_len <= 0) return 0;
    int n_layer = scan->n_layer_meta > 0 ? scan->n_layer_meta : scan->n_layers;
    if (n_layer <= 0 || ctx_len <= 0 || contexts <= 0) return 0;
    double bytes = 2.0 * (double)n_layer * (double)scan->n_head_kv *
        (double)scan->key_len * (double)ctx_len *
        ollm_kv_type_size(kv_type) * (double)contexts;
    if (bytes > (double)ULLONG_MAX) return ULLONG_MAX;
    return (unsigned long long)bytes;
}

static bool moe_sample_free(unsigned long long *free_out, unsigned long long *budget_out) {
    if (free_out) *free_out = 0;
    if (budget_out) *budget_out = 0;
    double free_gib = -1.0;
    if (!ogpu_memory_gib(&free_gib, NULL) || free_gib <= 0.0) return false;
    long long keep_mb = 2048;
    const char *keep_env = getenv("OMNISERVE_NATIVE_NGL_AUTO_KEEP_FREE_MB");
    if (keep_env && keep_env[0]) keep_mb = atoll(keep_env);
    if (keep_mb < 0) keep_mb = 0;
    double free_b = free_gib * 1024.0 * 1024.0 * 1024.0;
    double budget = free_b - (double)keep_mb * 1024.0 * 1024.0;
    if (free_b <= 0.0 || free_b > (double)ULLONG_MAX) return false;
    if (free_out) *free_out = (unsigned long long)free_b;
    if (budget > 0.0 && budget <= (double)ULLONG_MAX) {
        if (budget_out) *budget_out = (unsigned long long)budget;
    }
    return true;
}

static const char *moe_mode_name(ollm_moe_mode mode) {
    switch (mode) {
    case OLLM_MOE_N: return "N";
    case OLLM_MOE_ALL: return "all";
    case OLLM_MOE_AUTO: return "auto";
    default: return "off";
    }
}

static int moe_decide_fresh(const char *model_path, int ctx_len, int contexts,
                            const char *kv_type, unsigned long long *est_gpu_out,
                            unsigned long long *cpu_bytes_out) {
    if (est_gpu_out) *est_gpu_out = 0;
    if (cpu_bytes_out) *cpu_bytes_out = 0;
    memset(&g_moe_decision, 0, sizeof g_moe_decision);
    ollm_moe_mode mode = OLLM_MOE_OFF;
    int n = 0;
    if (ollm_parse_moe_cpu_experts(ollm_cfg("OMNISERVE_NATIVE_MOE_CPU_EXPERTS", g_ovr_moe),
                                   &mode, &n) != 0) {
        fprintf(stderr, "OMNISERVE_NATIVE_MOE_CPU_EXPERTS: invalid value, ignoring\n");
        return 0;
    }
    if (mode == OLLM_MOE_OFF) return 0;
    const moe_scan *scan = moe_cached_scan(model_path);
    if (!scan || scan->n_layers <= 0) {
        fprintf(stderr, "MOE_CPU_EXPERTS set but no expert tensors found in %s\n",
                model_path ? model_path : "(null)");
        return 0;
    }
    int cpu_layers = 0;
    unsigned long long kv = moe_kv_bytes(scan, ctx_len, contexts, kv_type);
    unsigned long long free_bytes = 0, budget = 0;
    if (mode == OLLM_MOE_ALL) {
        cpu_layers = scan->n_layers;
    } else if (mode == OLLM_MOE_N) {
        cpu_layers = n < scan->n_layers ? n : scan->n_layers;
    } else if (scan->n_head_kv <= 0 || scan->key_len <= 0) {
        fprintf(stderr, "llm moe experts: mode=auto KV dims unknown for %s, placing all experts on CPU\n",
                model_path ? model_path : "(null)");
        cpu_layers = scan->n_layers;
    } else {
        if (!moe_sample_free(&free_bytes, &budget)) {
            fprintf(stderr, "llm moe experts: mode=auto VRAM sample failed, placing all experts on CPU\n");
            cpu_layers = scan->n_layers;
        } else {
            unsigned long long judge_reserve = ojudge_vram_reserve_bytes();
            if (judge_reserve > budget) judge_reserve = budget;
            budget -= judge_reserve;
            int gpu = ollm_gpu_expert_layers(scan->expert_bytes, scan->n_layers,
                                             scan->nonexp_bytes, kv,
                                             1024ULL * 1024ULL * 1024ULL, budget);
            cpu_layers = scan->n_layers - gpu;
        }
    }
    unsigned long long cpu_bytes = 0, gpu_experts = 0;
    for (int i = 0; i < scan->n_layers; i++) {
        if (i >= scan->n_layers - cpu_layers) cpu_bytes += scan->expert_bytes[i];
        else gpu_experts += scan->expert_bytes[i];
    }
    unsigned long long est =
        scan->nonexp_bytes + gpu_experts + kv + 1024ULL * 1024ULL * 1024ULL;
    g_moe_decision.valid = true;
    g_moe_decision.active = true;
    snprintf(g_moe_decision.path, sizeof g_moe_decision.path, "%s",
             model_path ? model_path : "");
    g_moe_decision.ctx_len = ctx_len;
    g_moe_decision.contexts = contexts;
    snprintf(g_moe_decision.kv_type, sizeof g_moe_decision.kv_type, "%s",
             kv_type ? kv_type : "f16");
    g_moe_decision.mode = mode;
    g_moe_decision.n_layers = scan->n_layers;
    g_moe_decision.cpu_layers = cpu_layers;
    g_moe_decision.free_bytes = free_bytes;
    g_moe_decision.budget_bytes = budget;
    g_moe_decision.nonexp_bytes = scan->nonexp_bytes;
    g_moe_decision.kv_bytes = kv;
    g_moe_decision.cpu_bytes = cpu_bytes;
    g_moe_decision.est_bytes = est;
    if (mode == OLLM_MOE_AUTO && free_bytes > 0) {
        fprintf(stderr,
                "llm moe experts: mode=auto free=%llu MiB budget=%llu MiB nonexp=%llu MiB kv=%llu MiB reserve=1024 MiB judge=%llu MiB -> gpu=%d cpu=%d est=%llu MiB\n",
                free_bytes / (1024ULL * 1024ULL), budget / (1024ULL * 1024ULL),
                scan->nonexp_bytes / (1024ULL * 1024ULL), kv / (1024ULL * 1024ULL),
                ojudge_vram_reserve_bytes() / (1024ULL * 1024ULL),
                scan->n_layers - cpu_layers, cpu_layers, est / (1024ULL * 1024ULL));
    } else {
        fprintf(stderr, "llm moe experts: mode=%s -> gpu=%d cpu=%d est=%llu MiB\n",
                moe_mode_name(mode), scan->n_layers - cpu_layers, cpu_layers,
                est / (1024ULL * 1024ULL));
    }
    if (cpu_bytes_out) *cpu_bytes_out = cpu_bytes;
    if (est_gpu_out) *est_gpu_out = est;
    return cpu_layers;
}

static bool moe_decision_matches(const char *model_path, int ctx_len, int contexts,
                                 const char *kv_type) {
    if (!g_moe_decision.valid || !g_moe_decision.active) return false;
    const char *want_path = model_path ? model_path : "";
    const char *want_kv = kv_type ? kv_type : "f16";
    return strcmp(g_moe_decision.path, want_path) == 0 &&
        g_moe_decision.ctx_len == ctx_len && g_moe_decision.contexts == contexts &&
        strcmp(g_moe_decision.kv_type, want_kv) == 0;
}

static int moe_cached_or_fresh(const char *model_path, int ctx_len, int contexts,
                               const char *kv_type, unsigned long long *est_gpu_out,
                               unsigned long long *cpu_bytes_out) {
    if (moe_decision_matches(model_path, ctx_len, contexts, kv_type)) {
        int cpu = g_moe_decision.cpu_layers;
        if (cpu_bytes_out) *cpu_bytes_out = g_moe_decision.cpu_bytes;
        if (est_gpu_out) *est_gpu_out = g_moe_decision.est_bytes;
        return cpu;
    }
    return moe_decide_fresh(model_path, ctx_len, contexts, kv_type,
                            est_gpu_out, cpu_bytes_out);
}

unsigned long long ollm_cpu_offloaded_bytes(const char *model_path, int ctx_len, int contexts) {
    unsigned long long cpu_bytes = 0;
    int ctx = ctx_len;
    if (ctx < 1) {
        const char *ctx_env = getenv("OMNISERVE_NATIVE_CTX");
        ctx = ctx_env ? atoi(ctx_env) : 8192;
        if (ctx < 1) ctx = 8192;
    }
    if (contexts < 1) {
        const char *contexts_env = getenv("OMNISERVE_NATIVE_LLM_CONTEXTS");
        contexts = contexts_env ? atoi(contexts_env) : 1;
        if (contexts < 1) contexts = 1;
    }
    (void)moe_decide_fresh(model_path, ctx, contexts,
                            getenv("OMNISERVE_NATIVE_KV_TYPE"), NULL, &cpu_bytes);
    return cpu_bytes;
}

static struct llama_model_tensor_buft_override *g_buft_overrides;
static char *g_buft_pattern_store;
static int g_gpu_expert_layers;
static int g_cpu_expert_layers;
static char g_override_display[512];
static unsigned long long g_est_gpu_bytes;

static void buft_overrides_clear(void) {
    free(g_buft_overrides);
    g_buft_overrides = NULL;
    free(g_buft_pattern_store);
    g_buft_pattern_store = NULL;
    g_gpu_expert_layers = 0;
    g_cpu_expert_layers = 0;
    g_override_display[0] = 0;
    g_est_gpu_bytes = 0;
}

static const struct llama_model_tensor_buft_override *buft_overrides_build(
    const char *model_path, int ctx_len, int contexts, const char *kv_type) {
    buft_overrides_clear();
    probe_buft_fns();
    ollm_tensor_override user[128];
    int user_count =
        ollm_parse_tensor_overrides(ollm_cfg("OMNISERVE_NATIVE_TENSOR_OVERRIDE", g_ovr_tensor),
                                    user, (int)(sizeof user / sizeof *user));
    if (user_count < 0) {
        fprintf(stderr, "OMNISERVE_NATIVE_TENSOR_OVERRIDE: invalid syntax, ignoring\n");
        user_count = 0;
    }
    int valid_users = 0;
    for (int i = 0; i < user_count; i++) {
        if (!ollm_regex_balanced(user[i].pattern)) {
            fprintf(stderr, "tensor override '%s': invalid regex, skipping\n",
                    user[i].pattern);
            continue;
        }
        user[valid_users++] = user[i];
    }
    user_count = valid_users;
    g_est_gpu_bytes = 0;
    int cpu_layers =
        moe_cached_or_fresh(model_path, ctx_len, contexts, kv_type,
                            &g_est_gpu_bytes, NULL);
    bool decided = g_moe_decision.valid && g_moe_decision.active;
    int snap_n = decided ? g_moe_decision.n_layers : 0;
    g_moe_decision.valid = false;
    if (decided) {
        g_cpu_expert_layers = cpu_layers;
        g_gpu_expert_layers = snap_n - cpu_layers;
    }
    int moe_patterns = (decided && cpu_layers > 0) ? 1 : 0;
    int total = user_count + moe_patterns;
    if (total <= 0) return NULL;
    size_t omax = llama_max_tensor_buft_overrides();
    if (omax < 2) omax = 2;
    if ((size_t)total + 1 > omax) {
        int keep_users = (int)(omax - 1) - moe_patterns;
        if (keep_users < 0) keep_users = 0;
        if (user_count > keep_users) {
            fprintf(stderr, "tensor overrides: %d exceed llama limit %zu, keeping %d user + %d MoE\n",
                    total, omax, keep_users, moe_patterns);
            user_count = keep_users;
            total = user_count + moe_patterns;
        }
    }
    enum { MOE_PATTERN_CAP = 4608 };
    g_buft_overrides = calloc((size_t)total + 1, sizeof *g_buft_overrides);
    size_t store_cap = (size_t)user_count * 256 + (moe_patterns ? MOE_PATTERN_CAP : 0) + 1;
    g_buft_pattern_store = malloc(store_cap);
    if (!g_buft_overrides || !g_buft_pattern_store) {
        buft_overrides_clear();
        return NULL;
    }
    char *store = g_buft_pattern_store;
    char *store_end = store + store_cap;
    int filled = 0;
    oggml_buft cuda_buft = NULL;
    if (oggml_dev_by_name && oggml_dev_buft) {
        oggml_dev dev = oggml_dev_by_name("CUDA0");
        if (dev) cuda_buft = oggml_dev_buft(dev);
    }
    oggml_buft cpu_buft = oggml_cpu_buft ? oggml_cpu_buft() : NULL;
    for (int i = 0; i < user_count; i++) {
        oggml_buft buft = user[i].cpu ? cpu_buft : cuda_buft;
        if (!buft) {
            fprintf(stderr, "tensor override '%s': backend unavailable, skipping\n",
                    user[i].pattern);
            continue;
        }
        size_t plen = strlen(user[i].pattern) + 1;
        if (store + plen > store_end) break;
        memcpy(store, user[i].pattern, plen);
        g_buft_overrides[filled].pattern = store;
        g_buft_overrides[filled].buft =
            (ggml_backend_buffer_type_t)(void *)buft;
        store += plen;
        filled++;
    }
    int user_filled = filled;
    if (moe_patterns && cpu_buft) {
        if (ollm_moe_cpu_pattern(snap_n - cpu_layers, snap_n, store,
                                 (size_t)(store_end - store)) == 0) {
            g_buft_overrides[filled].pattern = store;
            g_buft_overrides[filled].buft =
                (ggml_backend_buffer_type_t)(void *)cpu_buft;
            store += strlen(store) + 1;
            filled++;
        } else {
            fprintf(stderr, "MoE override pattern exceeds %d bytes, skipping\n",
                    MOE_PATTERN_CAP);
        }
    } else if (moe_patterns) {
        fprintf(stderr, "MoE CPU experts requested but no CPU buffer type, skipping\n");
    }
    g_buft_overrides[filled].pattern = NULL;
    g_buft_overrides[filled].buft = NULL;
    if (filled == 0) {
        buft_overrides_clear();
        return NULL;
    }
    char *d = g_override_display;
    size_t left = sizeof g_override_display;
    if (decided) {
        int w = cpu_layers > 0
            ? snprintf(d, left, "moe:blk.%d-%d->CPU", snap_n - cpu_layers, snap_n - 1)
            : snprintf(d, left, "moe:all-gpu");
        if (w < 0) w = 0;
        if ((size_t)w >= left) w = (int)(left - 1);
        d += w;
        left -= (size_t)w;
    }
    if (user_filled > 0 && left > 1) {
        int w = snprintf(d, left, "%suser:", d == g_override_display ? "" : ";");
        if (w < 0) w = 0;
        if ((size_t)w >= left) w = (int)(left - 1);
        d += w;
        left -= (size_t)w;
        for (int i = 0; i < user_filled; i++) {
            const char *pat = g_buft_overrides[i].pattern;
            size_t need = strlen(pat) + (size_t)(i ? 1 : 0);
            if (need + 1 > left) {
                snprintf(d, left, "%s+%d more", i ? "," : "", user_filled - i);
                break;
            }
            if (i) {
                *d++ = ',';
                left--;
            }
            size_t len = strlen(pat);
            memcpy(d, pat, len + 1);
            d += len;
            left -= len;
        }
    }
    fprintf(stderr, "llm tensor overrides: %d active (%s)\n", filled, g_override_display);
    return g_buft_overrides;
}

/* Quantized KV halves (q8_0) or quarters (q4_0) the per-context cache, which
 * is what caps how many parallel contexts fit beside the weights. llama.cpp
 * requires flash attention for a quantized V cache. */
static enum ggml_type kv_type_named(const char *value, const char **name_out) {
    if (!value || !value[0]) { *name_out = "f16"; return GGML_TYPE_F16; }
    if (strcasecmp(value, "q8_0") == 0) { *name_out = "q8_0"; return GGML_TYPE_Q8_0; }
    if (strcasecmp(value, "q4_0") == 0) { *name_out = "q4_0"; return GGML_TYPE_Q4_0; }
    if (strcasecmp(value, "q5_1") == 0) { *name_out = "q5_1"; return GGML_TYPE_Q5_1; }
    if (strcasecmp(value, "bf16") == 0) { *name_out = "bf16"; return GGML_TYPE_BF16; }
    *name_out = "f16";
    return GGML_TYPE_F16;
}

/* Explicit configuration always wins over the per-device profile. */
static enum ggml_type kv_type_resolved(const otune_profile *profile, const char **name_out) {
    const char *value = getenv("OMNISERVE_NATIVE_KV_TYPE");
    if (value && value[0]) return kv_type_named(value, name_out);
    return kv_type_named(profile->kv_type, name_out);
}

static enum llama_flash_attn_type flash_attn_resolved(const otune_profile *profile,
                                                      bool quantized_kv) {
    const char *value = getenv("OMNISERVE_NATIVE_FLASH_ATTN");
    if (value && value[0]) {
        if (strcmp(value, "0") == 0 || strcasecmp(value, "off") == 0) {
            return LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    /* llama.cpp requires flash attention for a quantized V cache, so a
     * quantized KV type forces it on regardless of the device profile. */
    if (quantized_kv) return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    return profile->flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
}

static bool tune_value_auto(const char *value) {
    return value && strcasecmp(value, "auto") == 0;
}

static void auto_batch_geometry(bool on_gpu, int *batch, int *ubatch) {
    if (!on_gpu) {
        *batch = 512;
        *ubatch = 128;
        return;
    }
    double free_gib = -1.0;
    if (!ogpu_memory_gib(&free_gib, NULL) || free_gib < 0.0) {
        *batch = 512;
        *ubatch = 128;
        return;
    }
    /* A full Blackwell profile is excellent when the device is dedicated, but
     * it reserves more scratch than a shared card can afford. Narrow the
     * prefill in steps so a swap can keep the model resident without making
     * decode pay for an oversized prompt buffer. */
    if (free_gib < 8.0) {
        *batch = 256;
        *ubatch = 64;
    } else if (free_gib < 16.0) {
        *batch = 512;
        *ubatch = 128;
    } else if (free_gib < 24.0) {
        *batch = 1024;
        *ubatch = 256;
    } else {
        *batch = 4096;
        *ubatch = 1024;
    }
    fprintf(stderr, "llm batch=auto: free=%.2f GiB -> batch=%d ubatch=%d\n",
            free_gib, *batch, *ubatch);
}

static int g_engine_live;
int ollm_engine_live(void) { return __atomic_load_n(&g_engine_live, __ATOMIC_RELAXED); }

int ollm_suggested_contexts(void) {
    otune_profile profile;
    if (!g_device_desc[0]) probe_gpu_device();
    otune_profile_for(g_gpu_device_present ? g_device_desc : "cpu", &profile);
    return profile.parallel_contexts;
}

static void spec_mtp_free_locked(void) {
    if (g_slots) {
        for (int i = 0; i < g_slot_count; i++) {
            if (g_slots[i].mtp_smpl) llama_sampler_free(g_slots[i].mtp_smpl);
            g_slots[i].mtp_smpl = NULL;
            if (g_slots[i].mtp_ctx) llama_free(g_slots[i].mtp_ctx);
            g_slots[i].mtp_ctx = NULL;
            free(g_slots[i].mtp_pending);
            g_slots[i].mtp_pending = NULL;
            free(g_slots[i].mtp_verify);
            g_slots[i].mtp_verify = NULL;
            free(g_slots[i].mtp_embd);
            g_slots[i].mtp_embd = NULL;
            free(g_slots[i].mtp_cand);
            g_slots[i].mtp_cand = NULL;
        }
    }
    if (g_spec_mtp_model) llama_model_free(g_spec_mtp_model);
    g_spec_mtp_model = NULL;
    g_spec_mtp_vocab = NULL;
    g_spec_mtp_embd = 0;
    g_spec_mtp_vocab_size = 0;
    g_spec_mtp_path[0] = 0;
}

/* Loads the MTP head and pairs every target context with a draft context.
 * Any failure disables MTP and leaves the target usable on its own: a draft
 * source must never take down the model it accelerates. */
static void spec_mtp_init_locked(int n_gpu_layers, const struct llama_context_params *tcp,
                                 enum ggml_type kv_type) {
    spec_mtp_free_locked();
    ollm_spec_mtp_default(&g_spec_mtp_cfg);
    const char *path = ollm_cfg("OMNISERVE_NATIVE_SPEC_MTP_GGUF", g_ovr_mtp);
    if (!path || !path[0]) return;
    if (ollm_spec_mtp_parse(&g_spec_mtp_cfg, getenv("OMNISERVE_NATIVE_SPEC_MTP_DRAFT"),
                            getenv("OMNISERVE_NATIVE_SPEC_MTP_P_MIN")) != 0) {
        fprintf(stderr, "spec MTP: invalid SPEC_MTP_DRAFT/P_MIN, MTP off\n");
        ollm_spec_mtp_default(&g_spec_mtp_cfg);
        g_spec_mtp_cfg.draft_max = 0;
        return;
    }
    if (g_spec_mtp_cfg.draft_max <= 0) return;
    if (!probe_nextn_fns()) {
        fprintf(stderr, "spec MTP: nextn staging API missing from libllama, MTP off\n");
        g_spec_mtp_cfg.draft_max = 0;
        return;
    }
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    struct llama_model *head = llama_model_load_from_file(path, mp);
    if (!head) {
        fprintf(stderr, "spec MTP: failed to load head '%s', MTP off\n", path);
        g_spec_mtp_cfg.draft_max = 0;
        return;
    }
    int32_t head_embd = llama_model_n_embd_out(head);
    int32_t tgt_embd = llama_model_n_embd(g_model);
    if (head_embd != tgt_embd) {
        fprintf(stderr, "spec MTP: head n_embd %d != target %d, MTP off\n",
                head_embd, tgt_embd);
        llama_model_free(head);
        g_spec_mtp_cfg.draft_max = 0;
        return;
    }
    int32_t vocab_size = 0;
    const struct llama_vocab *hv = llama_model_get_vocab(head);
    if (hv) vocab_size = llama_vocab_n_tokens(hv);
    if (vocab_size <= 0) {
        fprintf(stderr, "spec MTP: head has no vocab, MTP off\n");
        llama_model_free(head);
        g_spec_mtp_cfg.draft_max = 0;
        return;
    }
    bool fail = false;
    for (int i = 0; i < g_slot_count && !fail; i++) {
        struct llama_context_params dcp = llama_context_default_params();
        dcp.n_ctx = (unsigned)g_ctx_len;
        dcp.n_batch = 8;
        dcp.n_ubatch = 8;
        dcp.n_seq_max = 1;
        dcp.type_k = kv_type;
        dcp.type_v = kv_type;
        dcp.flash_attn_type = tcp->flash_attn_type;
        dcp.no_perf = true;
        dcp.n_threads = tcp->n_threads;
        dcp.n_threads_batch = tcp->n_threads_batch;
        dcp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        dcp.ctx_other = g_slots[i].ctx;
        dcp.n_rs_seq = 0;
        struct llama_context *dctx = llama_init_from_model(head, dcp);
        if (!dctx) {
            fprintf(stderr, "spec MTP: draft context %d failed to init, MTP off\n", i);
            fail = true;
            break;
        }
        g_slots[i].mtp_ctx = dctx;
        /* Unmasked on the target: every decoded row stages its h vector for
         * the snapshot below. Masked on the draft: only logits rows. */
        ollama_set_nextn(g_slots[i].ctx, true, false);
        ollama_set_nextn(dctx, true, true);
        /* Only the shared-memory (Gemma-4) layout is implemented: other
         * arches need a catch-up decode per target batch to fill the draft
         * KV, and drafting against an empty cache would only burn time. */
        if (ollama_get_ctx_other(dctx) != g_slots[i].ctx) {
            fprintf(stderr, "spec MTP: head does not share target memory, MTP off\n");
            fail = true;
            break;
        }
        g_slots[i].mtp_pending = malloc((size_t)head_embd * sizeof(float));
        g_slots[i].mtp_verify =
            malloc((size_t)(OLLM_MTP_DRAFT_CAP + 1) * (size_t)head_embd * sizeof(float));
        g_slots[i].mtp_embd = malloc((size_t)head_embd * sizeof(float));
        g_slots[i].mtp_cand = malloc((size_t)vocab_size * sizeof *g_slots[i].mtp_cand);
        struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        g_slots[i].mtp_smpl = llama_sampler_chain_init(sp);
        if (!g_slots[i].mtp_pending || !g_slots[i].mtp_verify || !g_slots[i].mtp_embd ||
            !g_slots[i].mtp_cand || !g_slots[i].mtp_smpl) {
            fprintf(stderr, "spec MTP: out of memory for slot %d, MTP off\n", i);
            fail = true;
            break;
        }
        llama_sampler_chain_add(g_slots[i].mtp_smpl, llama_sampler_init_top_k(10));
        memset(g_slots[i].mtp_pending, 0, (size_t)head_embd * sizeof(float));
        g_slots[i].mtp_n_seq = 1;
        g_slots[i].mtp_seq = 0;
        g_slots[i].mtp_seq_ptr = &g_slots[i].mtp_seq;
        g_slots[i].mtp_logits = 1;
    }
    if (fail) {
        spec_mtp_free_locked();
        for (int i = 0; i < g_slot_count; i++) ollama_set_nextn(g_slots[i].ctx, false, false);
        llama_model_free(head);
        g_spec_mtp_cfg.draft_max = 0;
        return;
    }
    g_spec_mtp_model = head;
    g_spec_mtp_vocab = hv;
    g_spec_mtp_embd = head_embd;
    g_spec_mtp_vocab_size = vocab_size;
    snprintf(g_spec_mtp_path, sizeof g_spec_mtp_path, "%s", path);
    fprintf(stderr, "spec MTP: head '%s' draft_max=%d p_min=%.2f n_embd=%d layers_nextn=%d\n",
            path, g_spec_mtp_cfg.draft_max, (double)g_spec_mtp_cfg.p_min,
            head_embd, llama_model_n_layer_nextn(head));
}

/* Continuous batching.
 *
 * One context holds every slot as a sequence id, and one engine thread runs
 * the model for all of them: each step packs one token per decoding sequence
 * plus prefill chunks up to the step budget, decodes once, samples every
 * output row on a small pool, and hands tokens to the request threads through
 * per-slot rings. Decode is bandwidth-bound, so a step of sixteen tokens costs
 * little more than a step of one; separate contexts each re-read the weights.
 *
 * Request threads never touch the context. Text assembly, stop strings and
 * streaming writes run on them while the next step is already on the GPU, and
 * a stalled client only stalls its own ring (a full ring drops the sequence
 * out of the batch, it does not block the step).
 *
 * The KV cache is unified because llama.cpp's per-sequence streams split a
 * mixed batch into one ubatch per non-consecutive sequence id.
 *
 * Sampling is the request's own sampler chain over the row's logits, the same
 * thing llama_sampler_sample does. When the chain can only lower logits
 * (penalties >= neutral, no DRY, no full-vocab probability capture), the
 * candidates are first cut to the top (top_k + penalty window) by logit: every
 * token the chain could keep is in that set, so the distribution is unchanged
 * while the per-row work drops from a 262k-entry chain to a linear select. */
enum { OSEQ_IDLE = 0, OSEQ_PREFILL, OSEQ_DECODE, OSEQ_DONE };

static bool g_batched;
static bool g_share_prefix;
/* MTP drafting in the engine (OMNISERVE_NATIVE_LLM_MTP): one draft context
 * over the shared target KV drafts for every eligible sequence at once, and
 * the drafts ride in the next target step as extra verify rows. Only while at
 * most g_mtp_max_active sequences are live: a wide batch is compute-bound and
 * verify rows would cost what they save. */
static struct llama_model *g_mtp_model;
static int g_mtp_max_active;
static int32_t g_mtp_embd;
static struct llama_batch g_mtp_batch;
static struct llama_sampler **g_mtp_greedy;
/* Confidence-scheduled drafting: a sequence keeps drafting while the product of
 * the head's own probabilities for its drafts stays at or above g_mtp_conf
 * (0 = always draft to the cap). Verification is unchanged, so this only moves
 * where the MTP decodes and verify rows are spent. */
static float g_mtp_conf;
static bool g_mtp_has_conf;
/* With gating, a sequence that stops early still takes its full draft_cap of
 * verify rows (pads) and a dummy row in later MTP decodes, so neither graph
 * changes shape from step to step: a rebuilt llama.cpp graph plus a CUDA graph
 * capture costs more than the rows. */
static bool g_mtp_pad;
enum { OLLM_CONF_BINS = 10, OLLM_CONF_POS = 4 };
static unsigned long long g_cal_n[OLLM_CONF_POS][OLLM_CONF_BINS], g_cal_ok[OLLM_CONF_POS][OLLM_CONF_BINS];
static ollm_slot **g_mtp_rows;
static bool g_mtp_go;
static int g_step_ndec;
static unsigned long long g_bdecodes = 1;
static double g_trace_draft, g_trace_draft_call;
static int g_trace_extra, g_trace_draft_steps;
static unsigned long long g_shared_tokens;
static int g_kv_seqs;
static unsigned long long g_kv_evictions, g_kv_preemptions, g_kv_requeues;
static bool g_prefill_hold; /* pool full: decode only until a sequence ends */
static struct llama_context *g_bctx;
static int32_t g_n_vocab;
static int g_step_budget;
static pthread_t g_engine_thread;
static bool g_engine_started;
static bool g_engine_quit;
static pthread_mutex_t g_engine_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_engine_cond = PTHREAD_COND_INITIALIZER;
static unsigned long long g_engine_order;
static struct llama_batch g_step;
static ollm_slot **g_step_slots;
static ollm_slot **g_sample_jobs;
static unsigned long long g_engine_steps;
static unsigned long long g_engine_rows;
static unsigned long long g_engine_tokens;
static bool g_engine_trace;
/* Per 512 steps, by step kind: decode-only with the previous step's shape
 * (CUDA graph replays), decode-only reshaped, and mixed with prefill. */
static int g_trace_n, g_trace_prev_n;
static double g_trace_samp, g_trace_gap, g_trace_last;
static double g_trace_cnt[3], g_trace_dec[3], g_trace_rows[3], g_trace_tok[3];

static pthread_t *g_samp_threads;
static int g_samp_count;
static pthread_mutex_t g_samp_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_samp_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_samp_done = PTHREAD_COND_INITIALIZER;
static unsigned g_samp_gen;
static int g_samp_total;
static int g_samp_next;
static int g_samp_finished;
static bool g_samp_quit;

static bool g_engine_trace_flag;
static unsigned long long g_trace_rows_sampled;
static pthread_mutex_t g_trace_samp_lock = PTHREAD_MUTEX_INITIALIZER;
/* Moves the k largest logits to the front, unordered: three-way quickselect,
 * linear on average and safe on the long runs of equal logits a softcap makes. */
static void select_top(llama_token_data *d, size_t n, size_t k) {
    size_t lo = 0, hi = n;
    while (hi - lo > 1) {
        float a = d[lo].logit, b = d[lo + (hi - lo) / 2].logit, c = d[hi - 1].logit;
        float pivot = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
        size_t lt = lo, i = lo, gt = hi;
        while (i < gt) {
            float v = d[i].logit;
            if (v > pivot) {
                llama_token_data t = d[lt]; d[lt++] = d[i]; d[i++] = t;
            } else if (v < pivot) {
                llama_token_data t = d[--gt]; d[gt] = d[i]; d[i] = t;
            } else {
                i++;
            }
        }
        if (k <= lt) hi = lt;
        else if (k <= gt) return;
        else lo = gt;
    }
}

/* Exact top-k superset without sorting the row: the threshold is the r-th
 * largest of a strided sample, sized so ~2k tokens clear it, and one compare
 * pass (eight lanes at a time, almost all rejected) collects every token at or
 * above it. At least k collected means the set holds the true top k, ties
 * included; fewer means the sample guessed high, so it retries lower. Tokens
 * come out in id order, unsorted: penalties and DRY reorder them anyway. */
enum { OLLM_TOPK_SAMPLE = 2048 };
/* Every logit >= thr, in id order, at most cap. Thirty-two lanes per test:
 * nearly every block is rejected by one movemask. */
static size_t collect_ge(const float *logits, int32_t n_vocab, float thr,
                         llama_token_data *out, size_t cap) {
    size_t n = 0;
    int32_t i = 0;
#if defined(__AVX2__)
    const __m256 t = _mm256_set1_ps(thr);
    for (; i + 32 <= n_vocab; i += 32) {
        __m256 a = _mm256_cmp_ps(_mm256_loadu_ps(logits + i), t, _CMP_GE_OQ);
        __m256 b = _mm256_cmp_ps(_mm256_loadu_ps(logits + i + 8), t, _CMP_GE_OQ);
        __m256 c = _mm256_cmp_ps(_mm256_loadu_ps(logits + i + 16), t, _CMP_GE_OQ);
        __m256 d = _mm256_cmp_ps(_mm256_loadu_ps(logits + i + 24), t, _CMP_GE_OQ);
        __m256 any = _mm256_or_ps(_mm256_or_ps(a, b), _mm256_or_ps(c, d));
        if (_mm256_testz_ps(any, any)) continue;
        uint32_t m = (uint32_t)_mm256_movemask_ps(a) | ((uint32_t)_mm256_movemask_ps(b) << 8) |
                     ((uint32_t)_mm256_movemask_ps(c) << 16) | ((uint32_t)_mm256_movemask_ps(d) << 24);
        while (m && n < cap) {
            int j = __builtin_ctz(m);
            m &= m - 1;
            out[n].id = i + j;
            out[n].logit = logits[i + j];
            out[n].p = 0.0f;
            n++;
        }
    }
#endif
    for (; i < n_vocab && n < cap; i++) {
        if (logits[i] >= thr) {
            out[n].id = i;
            out[n].logit = logits[i];
            out[n].p = 0.0f;
            n++;
        }
    }
    return n;
}
static size_t gather_top(const float *logits, int32_t n_vocab, size_t k,
                         llama_token_data *out, size_t cap) {
    float sample[OLLM_TOPK_SAMPLE];
    llama_token_data sd[OLLM_TOPK_SAMPLE];
    const int32_t stride = n_vocab / OLLM_TOPK_SAMPLE > 0 ? n_vocab / OLLM_TOPK_SAMPLE : 1;
    int ns = 0;
    for (int32_t i = 0; i < n_vocab && ns < OLLM_TOPK_SAMPLE; i += stride) sample[ns++] = logits[i];
    size_t rank = (size_t)((double)k * ns / n_vocab * 2.0) + 8;
    for (;;) {
        float thr = -INFINITY;
        if (rank < (size_t)ns) {
            for (int i = 0; i < ns; i++) sd[i].logit = sample[i];
            select_top(sd, (size_t)ns, rank);
            thr = sd[0].logit;
            for (size_t i = 1; i < rank; i++) thr = sd[i].logit < thr ? sd[i].logit : thr;
        }
        size_t n = collect_ge(logits, n_vocab, thr, out, cap);
        if (n >= k || thr == -INFINITY) return n;
        rank *= 4;
    }
}

/* llama_sampler_sample over a raw row, with the exact top-(k+window) cut when
 * prefilter_k is set. cand holds n_vocab entries. */
static double g_trace_gather_us, g_trace_chain_us;
static llama_token sample_row_pre(struct llama_sampler *smpl, const float *logits, int prefilter_k,
                                  llama_token_data *cand, long pre_n) {
    struct timespec ta, tb, tc;
    if (g_engine_trace_flag) clock_gettime(CLOCK_MONOTONIC, &ta);
    size_t n = 0;
    if (pre_n >= 0) {
        n = (size_t)pre_n;
    } else if (prefilter_k > 0 && prefilter_k < g_n_vocab) {
        n = gather_top(logits, g_n_vocab, (size_t)prefilter_k, cand, (size_t)g_n_vocab);
    } else {
        for (int32_t i = 0; i < g_n_vocab; i++) {
            cand[i].id = i;
            cand[i].logit = logits[i];
            cand[i].p = 0.0f;
        }
        n = (size_t)g_n_vocab;
    }
    if (g_engine_trace_flag) clock_gettime(CLOCK_MONOTONIC, &tb);
    llama_token_data_array cur = { cand, n, -1, false };
    llama_sampler_apply(smpl, &cur);
    llama_token tok = cur.selected >= 0 && cur.selected < (int64_t)cur.size
        ? cur.data[cur.selected].id : cand[0].id;
    llama_sampler_accept(smpl, tok);
    if (g_engine_trace_flag) {
        clock_gettime(CLOCK_MONOTONIC, &tc);
        double g = (tb.tv_sec - ta.tv_sec) * 1e6 + (tb.tv_nsec - ta.tv_nsec) / 1e3;
        double c = (tc.tv_sec - tb.tv_sec) * 1e6 + (tc.tv_nsec - tb.tv_nsec) / 1e3;
        __atomic_fetch_add(&g_trace_rows_sampled, 1, __ATOMIC_RELAXED);
        pthread_mutex_lock(&g_trace_samp_lock);
        g_trace_gather_us += g;
        g_trace_chain_us += c;
        pthread_mutex_unlock(&g_trace_samp_lock);
    }
    return tok;
}

static llama_token sample_row(struct llama_sampler *smpl, const float *logits, int prefilter_k,
                              llama_token_data *cand) {
    return sample_row_pre(smpl, logits, prefilter_k, cand, -1);
}

/* Verify rows' candidate cuts do not depend on the sampler state, so the
 * engine gathers them for every row of every sequence on the pool first and
 * the per-sequence chains then run over ready candidates. vcand holds rows
 * 1..n_draft (row 0 uses cand); a cut that hits the cap is redone serially. */
enum { OLLM_VCAND_CAP = 32768 };
typedef struct { struct ollm_slot_s *s; int k; } gather_item;
static gather_item *g_gather_items;

/* With drafts the rows are sampled in order exactly as sequential decoding
 * would, each token accepted into the chain; a draft only lets the next row's
 * logits stand because the sampler chose the drafted token itself. */
static void engine_sample_one(ollm_slot *s) {
    if (s->n_draft <= 0) {
        s->next = sample_row(s->smpl, s->logits_row, s->prefilter_k, s->cand);
        return;
    }
    int n = 0, m = 0;
    for (int i = 0; i <= s->n_draft; i++) {
        llama_token_data *buf = i == 0 ? s->cand : s->vcand[i - 1];
        llama_token tok = sample_row_pre(s->smpl, s->vrows[i], s->prefilter_k,
                                         buf ? buf : s->cand, buf ? s->gn[i] : -1);
        s->out_tok[n] = tok;
        s->out_p[n++] = s->capture ? s->capture->probability : 1.0f;
        if (i == s->n_draft || tok != s->draft[i] || llama_vocab_is_eog(g_vocab, tok)) break;
        m++;
    }
    s->n_out = n;
    s->matched = m;
    s->next = s->out_tok[n - 1];
}

/* The per-slot (speculative) path's sampler: same cut, same chain. */
static llama_token slot_sample(ollm_slot *slot, struct llama_sampler *smpl,
                               struct llama_context *ctx, int32_t idx) {
    if (slot->cand && slot->prefilter_k > 0) {
        const float *logits = llama_get_logits_ith(ctx, idx);
        if (logits) return sample_row(smpl, logits, slot->prefilter_k, slot->cand);
    }
    return llama_sampler_sample(smpl, ctx, idx);
}

static int g_samp_phase; /* 0: gather items, 1: sample jobs */
static void gather_one(const gather_item *it) {
    ollm_slot *s = it->s;
    llama_token_data *buf = it->k == 0 ? s->cand : s->vcand[it->k - 1];
    size_t cap = it->k == 0 ? (size_t)g_n_vocab : OLLM_VCAND_CAP;
    size_t n = gather_top(s->vrows[it->k], g_n_vocab, (size_t)s->prefilter_k, buf, cap);
    s->gn[it->k] = n < cap ? (long)n : -1;
}

static void samp_drain(void) {
    for (;;) {
        int i = __atomic_fetch_add(&g_samp_next, 1, __ATOMIC_ACQ_REL);
        if (i >= g_samp_total) return;
        if (g_samp_phase == 0) gather_one(&g_gather_items[i]);
        else engine_sample_one(g_sample_jobs[i]);
        pthread_mutex_lock(&g_samp_lock);
        if (++g_samp_finished == g_samp_total) pthread_cond_signal(&g_samp_done);
        pthread_mutex_unlock(&g_samp_lock);
    }
}

static void *samp_main(void *arg) {
    (void)arg;
    unsigned seen = 0;
    pthread_mutex_lock(&g_samp_lock);
    for (;;) {
        while (!g_samp_quit && g_samp_gen == seen) pthread_cond_wait(&g_samp_cond, &g_samp_lock);
        if (g_samp_quit) break;
        seen = g_samp_gen;
        pthread_mutex_unlock(&g_samp_lock);
        samp_drain();
        pthread_mutex_lock(&g_samp_lock);
    }
    pthread_mutex_unlock(&g_samp_lock);
    return NULL;
}

static void samp_run(int total, int phase) {
    pthread_mutex_lock(&g_samp_lock);
    g_samp_phase = phase;
    g_samp_total = total;
    g_samp_finished = 0;
    __atomic_store_n(&g_samp_next, 0, __ATOMIC_RELEASE);
    g_samp_gen++;
    pthread_cond_broadcast(&g_samp_cond);
    pthread_mutex_unlock(&g_samp_lock);
    samp_drain();
    pthread_mutex_lock(&g_samp_lock);
    while (g_samp_finished < g_samp_total) pthread_cond_wait(&g_samp_done, &g_samp_lock);
    g_samp_total = 0;
    pthread_mutex_unlock(&g_samp_lock);
}

static void engine_sample_all(int jobs) {
    if (jobs <= 0) return;
    int items = 0;
    for (int i = 0; i < jobs; i++) {
        ollm_slot *s = g_sample_jobs[i];
        for (int k = 0; k <= OLLM_MTP_BATCH_DRAFT_MAX; k++) s->gn[k] = -1;
        if (!g_gather_items || s->n_draft <= 0 || s->prefilter_k <= 0 || s->prefilter_k >= g_n_vocab ||
            !s->vcand[0])
            continue;
        for (int k = 0; k <= s->n_draft; k++) g_gather_items[items++] = (gather_item){ s, k };
    }
    if (items > 1 && g_samp_count > 0) samp_run(items, 0);
    else for (int i = 0; i < items; i++) gather_one(&g_gather_items[i]);
    if (jobs == 1 || g_samp_count == 0) {
        for (int i = 0; i < jobs; i++) engine_sample_one(g_sample_jobs[i]);
        return;
    }
    samp_run(jobs, 1);
}

static void step_add(int *n, llama_token tok, llama_pos pos, llama_seq_id seq, bool logits) {
    int i = (*n)++;
    g_step.token[i] = tok;
    g_step.pos[i] = pos;
    g_step.n_seq_id[i] = 1;
    g_step.seq_id[i][0] = seq;
    g_step.logits[i] = logits ? 1 : 0;
}

static void engine_finish_locked(ollm_slot *s, bool failed) {
    g_prefill_hold = false;
    s->state = OSEQ_DONE;
    s->has_next = false;
    if (failed) s->failed = true;
    pthread_cond_broadcast(&s->cond);
}

/* Builds the next step under g_engine_lock: decode tokens first (they are what
 * a waiting user sees), then prefill in arrival order. Only the last prompt
 * admitted is chunked, so at most one prompt is ever half-done per step. */
static int engine_build_locked(int *rows_out) {
    llama_memory_t mem = llama_get_memory(g_bctx);
    int n = 0, rows = 0, active = 0;
    for (int i = 0; i < g_slot_count; i++) {
        ollm_slot *s = &g_slots[i];
        s->step_chunk = 0;
        s->row = -1;
        s->n_draft = 0;
        s->n_pad = 0;
        s->draft_cap = 0;
        if (s->state == OSEQ_IDLE || s->state == OSEQ_DONE) continue;
        active++;
        if (s->stop) {
            engine_finish_locked(s, false);
            continue;
        }
        if (s->cp_src) {
            llama_memory_seq_rm(mem, s->seq, -1, -1);
            llama_memory_seq_cp(mem, s->cp_src->seq, s->seq, 0, s->cp_len);
            s->prompt_done = s->cp_len;
            g_shared_tokens += (unsigned long long)s->cp_len;
            pthread_mutex_lock(&g_slot_lock);
            s->cached_count = s->cp_len;
            s->cp_src->pins--;
            pthread_cond_broadcast(&g_slot_cond);
            pthread_mutex_unlock(&g_slot_lock);
            s->cp_src = NULL;
            s->rm_from = -1;
        }
        if (s->rm_from >= 0) {
            if (s->rm_from == 0 || !llama_memory_seq_rm(mem, s->seq, s->rm_from, -1)) {
                llama_memory_seq_rm(mem, s->seq, -1, -1);
                s->prompt_done = 0;
                pthread_mutex_lock(&g_slot_lock);
                s->cached_count = 0;
                pthread_mutex_unlock(&g_slot_lock);
            }
            s->rm_from = -1;
        }
        if (s->state == OSEQ_DECODE && s->has_next && s->ring_count < OLLM_RING) {
            s->row = n;
            step_add(&n, s->next, (llama_pos)s->n_past, s->seq, true);
            s->in_step = true;
            g_step_slots[rows++] = s;
            if (g_mtp_ctx && s->h_dec == g_bdecodes && s->h_row >= 0) {
                int cap = g_mtp_draft;
                if (cap > OLLM_RING - s->ring_count - 1) cap = OLLM_RING - s->ring_count - 1;
                if (cap > s->max_new - s->produced - 1) cap = s->max_new - s->produced - 1;
                s->draft_cap = cap > 0 ? cap : 0;
            }
        }
    }
    g_step_ndec = rows;
    g_mtp_go = g_mtp_ctx && rows > 0 && active <= g_mtp_max_active;
    /* A quarter of the budget is always left for prefill so a full decode
     * batch cannot starve new arrivals of their first token. */
    int budget = g_step_budget - n;
    if (budget < g_step_budget / 4) budget = g_step_budget / 4;
    if (g_prefill_hold && n > 0) budget = 0;
    while (budget > 0) {
        ollm_slot *pick = NULL;
        for (int i = 0; i < g_slot_count; i++) {
            ollm_slot *s = &g_slots[i];
            if (s->state != OSEQ_PREFILL || s->in_step) continue;
            if (!pick || s->order < pick->order) pick = s;
        }
        if (!pick) break;
        int left = pick->prompt_len - pick->prompt_done;
        int chunk = left < budget ? left : budget;
        for (int j = 0; j < chunk; j++) {
            int p = pick->prompt_done + j;
            bool last = p == pick->prompt_len - 1;
            if (last) pick->row = n;
            step_add(&n, pick->prompt[p], (llama_pos)p, pick->seq, last);
        }
        pick->step_chunk = chunk;
        pick->in_step = true;
        g_step_slots[rows++] = pick;
        budget -= chunk;
    }
    g_step.n_tokens = n;
    __atomic_store_n(&g_engine_live, active, __ATOMIC_RELAXED);
    *rows_out = rows;
    return n;
}

static bool engine_has_work_locked(void) {
    for (int i = 0; i < g_slot_count; i++) {
        const ollm_slot *s = &g_slots[i];
        if (s->state == OSEQ_PREFILL) return true;
        if (s->state == OSEQ_DECODE && (s->stop || (s->has_next && s->ring_count < OLLM_RING)))
            return true;
    }
    return false;
}

/* Frees the least recently used idle prefix (not busy, not a pending fork
 * source) when the pool cannot place a step. */
static bool engine_evict_idle(void) {
    pthread_mutex_lock(&g_slot_lock);
    ollm_slot *victim = NULL;
    for (int i = 0; i < g_slot_count; i++) {
        ollm_slot *s = &g_slots[i];
        if (s->busy || s->pins > 0 || s->cached_count <= 0) continue;
        if (!victim || s->last_used < victim->last_used) victim = s;
    }
    if (victim) victim->cached_count = 0;
    pthread_mutex_unlock(&g_slot_lock);
    if (!victim) return false;
    llama_memory_seq_rm(llama_get_memory(g_bctx), victim->seq, -1, -1);
    g_kv_evictions++;
    return true;
}

static llama_token argmax_row(const float *logits, int32_t n) {
    int32_t best = 0;
    float bv = logits[0];
    for (int32_t i = 1; i < n; i++) {
        if (logits[i] > bv) {
            bv = logits[i];
            best = i;
        }
    }
    return best;
}

/* Device greedy that also returns the argmax's softmax probability, and drops
 * the logits view so llama.cpp does not copy the 1 MB row back per draft. */
static const char *confg_name(const struct llama_sampler *smpl) {
    (void)smpl;
    return "omni-conf-greedy";
}
static void confg_apply(struct llama_sampler *smpl, llama_token_data_array *cur) {
    (void)smpl;
    if (cur->size == 0) return;
    size_t best = 0;
    for (size_t i = 1; i < cur->size; i++)
        if (cur->data[i].logit > cur->data[best].logit) best = i;
    cur->selected = (int64_t)best;
}
static bool confg_backend_init(struct llama_sampler *smpl, ggml_backend_buffer_type_t buft) {
    (void)smpl;
    (void)buft;
    return true;
}
static void confg_backend_apply(struct llama_sampler *smpl, struct ggml_context *ctx,
                                struct ggml_cgraph *gf, struct llama_sampler_data *data) {
    (void)smpl;
    (void)gf;
    struct ggml_tensor *am = ggml_argmax(ctx, data->logits);
    struct ggml_tensor *p = ggml_soft_max(ctx, data->logits);
    struct ggml_tensor *pr = ggml_get_rows(ctx, ggml_reshape_2d(ctx, p, 1, p->ne[0]), am);
    data->sampled = am;
    data->probs = ggml_reshape_1d(ctx, pr, 1);
    data->logits = NULL;
}
static struct llama_sampler_i g_confg_iface = {
    .name = confg_name,
    .apply = confg_apply,
    .backend_init = confg_backend_init,
    .backend_apply = confg_backend_apply,
};

/* Drafts greedily for every eligible decode row of the step just built, one
 * MTP decode per draft position across all of them (Gemma-4's head reads the
 * target KV and keeps every draft at the position of `next`), then splices the
 * drafts in behind their rows so each sequence's verify rows stay contiguous
 * and ahead of the prefill. Returns the new token count. */
static int engine_mtp_draft(int n, int rows) {
    if (!g_mtp_go) return n;
    int nd = 0;
    for (int r = 0; r < g_step_ndec; r++) {
        ollm_slot *s = g_step_slots[r];
        if (s->draft_cap > 0) g_mtp_rows[nd++] = s;
    }
    if (nd == 0) return n;
    const size_t rb = (size_t)g_mtp_embd * sizeof(float);
    for (int i = 0; i < nd; i++) g_mtp_rows[i]->draft_cum = 1.0f;
    for (int j = 0; j < g_mtp_draft; j++) {
        int b = 0, live = 0;
        for (int i = 0; i < nd; i++) {
            ollm_slot *s = g_mtp_rows[i];
            s->draft_row = -1;
            s->draft_dummy = false;
            if (s->n_draft != j || j >= s->draft_cap) continue;
            if (j > 0 && g_mtp_conf > 0.0f && s->draft_cum < g_mtp_conf) continue;
            live++;
        }
        if (live == 0) break;
        for (int i = 0; i < nd; i++) {
            ollm_slot *s = g_mtp_rows[i];
            if (j >= s->draft_cap) continue;
            bool go = s->n_draft == j && !(j > 0 && g_mtp_conf > 0.0f && s->draft_cum < g_mtp_conf);
            if (!go && !g_mtp_pad) continue;
            const float *h = go && j > 0 ? ollama_get_nextn_ith(g_mtp_ctx, s->draft_row_prev)
                                         : ollama_get_nextn_ith(g_bctx, s->h_row);
            if (!h) continue;
            s->draft_dummy = !go;
            g_mtp_batch.token[b] = go && j > 0 ? s->draft[j - 1] : s->next;
            memcpy(g_mtp_batch.embd + (size_t)b * g_mtp_embd, h, rb);
            g_mtp_batch.pos[b] = (llama_pos)s->n_past;
            g_mtp_batch.n_seq_id[b] = 1;
            g_mtp_batch.seq_id[b][0] = s->seq;
            g_mtp_batch.logits[b] = 1;
            s->draft_row = b++;
        }
        if (b == 0) break;
        g_mtp_batch.n_tokens = b;
        double t0 = g_engine_trace ? now_ms() : 0.0;
        if (llama_decode(g_mtp_ctx, g_mtp_batch) != 0) break;
        if (g_engine_trace) {
            g_trace_draft_call += now_ms() - t0;
            g_trace_draft_steps++;
        }
        for (int i = 0; i < nd; i++) {
            ollm_slot *s = g_mtp_rows[i];
            if (s->draft_row < 0 || s->draft_dummy) continue;
            llama_token t = LLAMA_TOKEN_NULL;
            float conf = 1.0f;
            if (g_mtp_greedy) {
                t = llama_get_sampled_token_ith(g_mtp_ctx, s->draft_row);
                if (g_mtp_has_conf) {
                    const float *pp = llama_get_sampled_probs_ith(g_mtp_ctx, s->draft_row);
                    if (pp) conf = pp[0];
                }
            } else {
                const float *lg = llama_get_logits_ith(g_mtp_ctx, s->draft_row);
                if (lg) t = argmax_row(lg, g_n_vocab);
            }
            if (t < 0 || t >= g_n_vocab) continue;
            s->draft[j] = t;
            s->draft_conf[j] = conf;
            s->draft_cum *= conf;
            s->n_draft = j + 1;
            s->draft_row_prev = s->draft_row;
        }
    }
    int extra = 0;
    for (int i = 0; i < nd; i++) {
        ollm_slot *s = g_mtp_rows[i];
        s->n_pad = g_mtp_pad && s->n_draft > 0 && s->draft_cap > s->n_draft ? s->draft_cap - s->n_draft : 0;
        extra += s->n_draft + s->n_pad;
    }
    if (extra == 0) return n;
    for (int i = n - 1; i >= g_step_ndec; i--) {
        g_step.token[i + extra] = g_step.token[i];
        g_step.pos[i + extra] = g_step.pos[i];
        g_step.n_seq_id[i + extra] = 1;
        g_step.seq_id[i + extra][0] = g_step.seq_id[i][0];
        g_step.logits[i + extra] = g_step.logits[i];
    }
    int end = g_step_ndec + extra;
    for (int r = g_step_ndec - 1; r >= 0; r--) {
        ollm_slot *s = g_step_slots[r];
        end -= 1 + s->n_draft + s->n_pad;
        s->row = end;
        for (int k = 0; k <= s->n_draft + s->n_pad; k++) {
            int i = end + k;
            int kd = k <= s->n_draft ? k : s->n_draft;
            g_step.token[i] = kd == 0 ? s->next : s->draft[kd - 1];
            g_step.pos[i] = (llama_pos)(s->n_past + k);
            g_step.n_seq_id[i] = 1;
            g_step.seq_id[i][0] = s->seq;
            g_step.logits[i] = 1;
        }
    }
    for (int r = g_step_ndec; r < rows; r++) {
        if (g_step_slots[r]->row >= 0) g_step_slots[r]->row += extra;
    }
    g_step.n_tokens = n + extra;
    return n + extra;
}

static void *engine_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&g_engine_lock);
    for (;;) {
        while (!g_engine_quit && !engine_has_work_locked()) {
            /* Nothing to step (idle, or every live ring is full): the GPU is free. */
            __atomic_store_n(&g_engine_live, 0, __ATOMIC_RELAXED);
            pthread_cond_wait(&g_engine_cond, &g_engine_lock);
        }
        if (g_engine_quit) break;
        int rows = 0;
        int n = engine_build_locked(&rows);
        if (n == 0) continue;
        pthread_mutex_unlock(&g_engine_lock);

        double t_draft = now_ms();
        int n_base = n;
        n = engine_mtp_draft(n, rows);
        double t_dec = now_ms();
        g_bdecodes++;
        int rc = llama_decode(g_bctx, g_step);
        while (rc == 1 && engine_evict_idle()) rc = llama_decode(g_bctx, g_step);
        /* Still full with every idle prefix gone. A prompt still prefilling
         * has streamed nothing, so the youngest one goes back to the queue
         * with its cells freed; only a decode-only step fails a request. */
        ollm_slot *preempt = NULL;
        bool requeue = false;
        if (rc == 1) {
            for (int r = 0; r < rows; r++) {
                ollm_slot *s = g_step_slots[r];
                bool pf = s->state == OSEQ_PREFILL;
                if (!preempt || (pf && !requeue) || (pf == requeue && s->order > preempt->order)) {
                    preempt = s;
                    requeue = pf;
                }
            }
            llama_memory_seq_rm(llama_get_memory(g_bctx), preempt->seq, -1, -1);
            pthread_mutex_lock(&g_slot_lock);
            preempt->cached_count = 0;
            pthread_mutex_unlock(&g_slot_lock);
            if (!requeue) fprintf(stderr, "engine: KV pool full, failing the youngest request\n");
        }
        if (g_engine_trace) llama_synchronize(g_bctx);
        double t_samp = now_ms();
        int jobs = 0;
        if (rc == 0) {
            for (int r = 0; r < rows; r++) {
                ollm_slot *s = g_step_slots[r];
                if (s->row < 0) continue;
                s->logits_row = llama_get_logits_ith(g_bctx, s->row);
                for (int k = 0; k <= s->n_draft && s->n_draft > 0; k++) {
                    s->vrows[k] = llama_get_logits_ith(g_bctx, s->row + k);
                    if (!s->vrows[k]) s->logits_row = NULL;
                }
                if (s->logits_row) g_sample_jobs[jobs++] = s;
                else rc = -1;
            }
            engine_sample_all(jobs);
        }
        if (g_engine_trace) {
            double t_end = now_ms();
            int kind = n_base > jobs ? 2 : n == g_trace_prev_n ? 0 : 1;
            g_trace_prev_n = n;
            g_trace_draft += t_dec - t_draft;
            g_trace_extra += n - n_base;
            g_trace_cnt[kind] += 1;
            g_trace_dec[kind] += t_samp - t_dec;
            g_trace_rows[kind] += jobs;
            g_trace_tok[kind] += n;
            g_trace_samp += t_end - t_samp;
            g_trace_gap += g_trace_last > 0 ? t_dec - g_trace_last : 0;
            g_trace_last = t_end;
            if (++g_trace_n == 512) {
                double c[3];
                for (int k = 0; k < 3; k++) c[k] = g_trace_cnt[k] > 0 ? g_trace_cnt[k] : 1;
                fprintf(stderr, "engine: steady %.0f rows=%.1f dec=%.2fms | reshaped %.0f rows=%.1f "
                        "dec=%.2fms | mixed %.0f tok=%.0f dec=%.2fms | sample=%.2fms gap=%.3fms "
                        "draft=%.2fms drafts=%.2f draft_steps=%.2f draft_call=%.2fms\n",
                        g_trace_cnt[0], g_trace_rows[0] / c[0], g_trace_dec[0] / c[0],
                        g_trace_cnt[1], g_trace_rows[1] / c[1], g_trace_dec[1] / c[1],
                        g_trace_cnt[2], g_trace_tok[2] / c[2], g_trace_dec[2] / c[2],
                        g_trace_samp / 512, g_trace_gap / 512, g_trace_draft / 512,
                        g_trace_extra / 512.0, g_trace_draft_steps / 512.0, g_trace_draft_call / 512);
                if (g_trace_rows_sampled) {
                    fprintf(stderr, "engine: sample rows=%llu gather=%.1fus chain=%.1fus per row\n",
                            g_trace_rows_sampled, g_trace_gather_us / g_trace_rows_sampled,
                            g_trace_chain_us / g_trace_rows_sampled);
                    g_trace_rows_sampled = 0;
                    g_trace_gather_us = g_trace_chain_us = 0;
                }
                if (g_mtp_ctx && g_mtp_has_conf) {
                    char cal[1024];
                    size_t o = 0;
                    for (int k = 0; k < OLLM_CONF_POS; k++) {
                        o += (size_t)snprintf(cal + o, sizeof cal - o, " d%d", k + 1);
                        for (int b = 0; b < OLLM_CONF_BINS && o < sizeof cal - 32; b++) {
                            if (!g_cal_n[k][b]) continue;
                            o += (size_t)snprintf(cal + o, sizeof cal - o, " %d:%llu/%llu", b,
                                                  g_cal_ok[k][b], g_cal_n[k][b]);
                        }
                    }
                    fprintf(stderr, "engine: mtp conf bins (accepted/drafted, cumulative):%s\n", cal);
                }
                g_trace_draft_call = 0;
                g_trace_draft_steps = 0;
                g_trace_n = 0;
                g_trace_samp = g_trace_gap = g_trace_draft = 0;
                g_trace_extra = 0;
                memset(g_trace_cnt, 0, sizeof g_trace_cnt);
                memset(g_trace_dec, 0, sizeof g_trace_dec);
                memset(g_trace_rows, 0, sizeof g_trace_rows);
                memset(g_trace_tok, 0, sizeof g_trace_tok);
            }
        }

        pthread_mutex_lock(&g_engine_lock);
        if (preempt) {
            for (int r = 0; r < rows; r++) g_step_slots[r]->in_step = false;
            if (requeue) {
                g_kv_requeues++;
                preempt->prompt_done = 0;
                preempt->n_past = 0;
                g_prefill_hold = true;
            } else {
                g_kv_preemptions++;
                engine_finish_locked(preempt, true);
            }
            continue;
        }
        g_engine_steps++;
        g_engine_rows += (unsigned long long)jobs;
        g_engine_tokens += (unsigned long long)n;
        unsigned long long sp_rounds = 0, sp_drafted = 0, sp_accepted = 0;
        for (int r = 0; r < rows; r++) {
            ollm_slot *s = g_step_slots[r];
            s->in_step = false;
            if (rc != 0) {
                engine_finish_locked(s, true);
                continue;
            }
            int matched = s->n_draft > 0 ? s->matched : 0;
            if (s->state == OSEQ_PREFILL) {
                s->prompt_done += s->step_chunk;
                if (g_share_prefix) {
                    pthread_mutex_lock(&g_slot_lock);
                    if (!s->cp_src) s->cached_count = s->prompt_done;
                    pthread_mutex_unlock(&g_slot_lock);
                }
                if (s->prompt_done < s->prompt_len) continue;
                s->n_past = s->prompt_len;
                s->state = OSEQ_DECODE;
            } else {
                s->n_past += 1 + matched;
                if (s->n_draft + s->n_pad > matched)
                    llama_memory_seq_rm(llama_get_memory(g_bctx), s->seq, s->n_past, -1);
            }
            if (s->row < 0) continue;
            s->h_row = s->row + matched;
            s->h_dec = g_bdecodes;
            if (s->n_draft > 0) {
                sp_rounds++;
                sp_drafted += (unsigned long long)s->n_draft;
                sp_accepted += (unsigned long long)matched;
                s->req_drafted += s->n_draft;
                s->req_accepted += matched;
                if (g_engine_trace) {
                    for (int k = 0; k < s->n_draft && k < OLLM_CONF_POS; k++) {
                        int bin = (int)(s->draft_conf[k] * OLLM_CONF_BINS);
                        if (bin >= OLLM_CONF_BINS) bin = OLLM_CONF_BINS - 1;
                        if (bin < 0) bin = 0;
                        g_cal_n[k][bin]++;
                        if (k < matched) g_cal_ok[k][bin]++;
                        if (k >= matched) break;
                    }
                }
                for (int k = 0; k < s->n_out; k++) {
                    int slot_i = (s->ring_head + s->ring_count) % OLLM_RING;
                    s->ring[slot_i] = s->out_tok[k];
                    s->ring_p[slot_i] = s->out_p[k];
                    s->ring_count++;
                }
                s->produced += s->n_out;
            } else {
                int slot_i = (s->ring_head + s->ring_count) % OLLM_RING;
                s->ring[slot_i] = s->next;
                s->ring_p[slot_i] = s->capture ? s->capture->probability : 1.0f;
                s->ring_count++;
                s->produced++;
            }
            if (llama_vocab_is_eog(g_vocab, s->next) || s->produced >= s->max_new) {
                engine_finish_locked(s, false);
            } else {
                s->has_next = true;
                pthread_cond_broadcast(&s->cond);
            }
        }
        if (sp_rounds) {
            pthread_mutex_lock(&g_spec_lock);
            g_spec_rounds += sp_rounds;
            g_spec_drafted += sp_drafted;
            g_spec_accepted += sp_accepted;
            g_spec_saved_calls += sp_accepted;
            pthread_mutex_unlock(&g_spec_lock);
        }
    }
    pthread_mutex_unlock(&g_engine_lock);
    return NULL;
}

/* Hands a prompt to the engine. rm_from is the cached prefix to keep (0 clears
 * the sequence); the engine applies it, since only it may touch the cache. */
static void engine_submit(ollm_slot *s, const llama_token *prompt, int prompt_len,
                          int keep, int max_new, struct llama_sampler *smpl,
                          const probability_capture *capture, int prefilter_k) {
    pthread_mutex_lock(&g_engine_lock);
    s->rm_from = keep;
    s->prompt = prompt;
    s->prompt_len = prompt_len;
    s->prompt_done = keep;
    s->n_past = keep;
    s->produced = 0;
    s->max_new = max_new;
    s->smpl = smpl;
    s->capture = capture;
    s->prefilter_k = prefilter_k;
    s->stop = false;
    s->failed = false;
    s->has_next = false;
    s->ring_head = 0;
    s->ring_count = 0;
    s->h_row = -1;
    s->h_dec = 0;
    s->n_draft = 0;
    s->n_pad = 0;
    s->req_drafted = 0;
    s->req_accepted = 0;
    s->order = ++g_engine_order;
    s->state = OSEQ_PREFILL;
    pthread_cond_signal(&g_engine_cond);
    pthread_mutex_unlock(&g_engine_lock);
}

/* Blocks until at least one token is ready; 0 means the sequence ended
 * (failed, or finished with nothing left to hand over). */
static int engine_pull(ollm_slot *s, llama_token *out, float *probs, int cap, bool *failed) {
    pthread_mutex_lock(&g_engine_lock);
    while (s->ring_count == 0 && s->state != OSEQ_DONE)
        pthread_cond_wait(&s->cond, &g_engine_lock);
    bool was_full = s->ring_count >= OLLM_RING;
    int n = 0;
    while (n < cap && s->ring_count > 0) {
        out[n] = s->ring[s->ring_head];
        probs[n] = s->ring_p[s->ring_head];
        s->ring_head = (s->ring_head + 1) % OLLM_RING;
        s->ring_count--;
        n++;
    }
    *failed = s->failed;
    if (was_full && n > 0) pthread_cond_signal(&g_engine_cond);
    pthread_mutex_unlock(&g_engine_lock);
    return n;
}

/* Detaches the request from the engine and returns how many leading tokens of
 * prompt + reply are in the cache, so a follow-up turn that quotes the reply
 * reuses it instead of prefilling it again. */
static int engine_release(ollm_slot *s) {
    pthread_mutex_lock(&g_engine_lock);
    if (s->state != OSEQ_IDLE && s->state != OSEQ_DONE) {
        s->stop = true;
        pthread_cond_signal(&g_engine_cond);
        while (s->in_step || (s->state != OSEQ_DONE && s->state != OSEQ_IDLE))
            pthread_cond_wait(&s->cond, &g_engine_lock);
    }
    int cached = s->prompt_done < s->prompt_len ? s->prompt_done : s->n_past;
    if (s->failed || s->rm_from >= 0 || s->cp_src) cached = 0;
    if (s->cp_src) {
        pthread_mutex_lock(&g_slot_lock);
        s->cp_src->pins--;
        pthread_cond_broadcast(&g_slot_cond);
        pthread_mutex_unlock(&g_slot_lock);
        s->cp_src = NULL;
    }
    s->state = OSEQ_IDLE;
    s->smpl = NULL;
    s->capture = NULL;
    s->prompt = NULL;
    pthread_mutex_unlock(&g_engine_lock);
    return cached;
}

/* The sampler chain is penalties -> top_k -> top_p/min_p -> temp/greedy. A cut
 * to the top (top_k + window) logits is exact whenever every step before top_k
 * can only lower a logit: at most `window` distinct tokens are penalized, so
 * at least top_k untouched tokens in the cut outrank anything outside it. */
/* Penalties and DRY only lower logits, and only of tokens already in their
 * history, so the post-penalty top_k lies within the pre-penalty top
 * (top_k + distinct history tokens). History grows by at most max_new. */
/* llama_sampler_init_dry detokenizes the whole vocabulary once per sequence
 * breaker (~1M calls for Gemma's 262k), so it is built once per parameter set
 * and cloned; a clone shares the processed breakers and starts empty. */
#define DRY_CACHE 8
static struct {
    float mult, base;
    int allowed;
    struct llama_sampler *s;
} g_dry_cache[DRY_CACHE];
static pthread_mutex_t g_dry_lock = PTHREAD_MUTEX_INITIALIZER;

static void dry_cache_clear(void) {
    pthread_mutex_lock(&g_dry_lock);
    for (int i = 0; i < DRY_CACHE; i++) {
        if (g_dry_cache[i].s) llama_sampler_free(g_dry_cache[i].s);
        g_dry_cache[i].s = NULL;
    }
    pthread_mutex_unlock(&g_dry_lock);
}

static struct llama_sampler *dry_sampler(float mult, float base, int allowed) {
    static const char *breakers[] = { "\n", ":", "\"", "*" };
    pthread_mutex_lock(&g_dry_lock);
    int slot = -1;
    for (int i = 0; i < DRY_CACHE; i++) {
        if (g_dry_cache[i].s && g_dry_cache[i].mult == mult && g_dry_cache[i].base == base &&
            g_dry_cache[i].allowed == allowed) {
            struct llama_sampler *c = llama_sampler_clone(g_dry_cache[i].s);
            pthread_mutex_unlock(&g_dry_lock);
            return c;
        }
        if (slot < 0 && !g_dry_cache[i].s) slot = i;
    }
    struct llama_sampler *t = llama_sampler_init_dry(g_vocab, llama_model_n_ctx_train(g_model),
                                                     mult, base, allowed, -1, breakers,
                                                     sizeof breakers / sizeof *breakers);
    if (slot < 0) {
        pthread_mutex_unlock(&g_dry_lock);
        return t;
    }
    g_dry_cache[slot].mult = mult;
    g_dry_cache[slot].base = base;
    g_dry_cache[slot].allowed = allowed;
    g_dry_cache[slot].s = t;
    struct llama_sampler *c = llama_sampler_clone(t);
    pthread_mutex_unlock(&g_dry_lock);
    return c;
}

static int engine_prefilter_k(const ochat_req *req, bool penalized, bool dry, int last_n,
                              int top_k, const llama_token *tokens, int n_tokens, int max_new) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("OMNISERVE_NATIVE_LLM_PREFILTER");
        enabled = !(env && strcmp(env, "0") == 0);
    }
    if (!enabled || req->min_probability > 0.0f || top_k <= 0) return 0;
    long window = 0;
    if (penalized) {
        if (req->repetition_penalty > 0.0f && req->repetition_penalty < 1.0f) return 0;
        if (req->frequency_penalty < 0.0f || req->presence_penalty < 0.0f) return 0;
        int hist = dry || last_n > n_tokens ? n_tokens : last_n;
        uint8_t *seen = calloc((size_t)g_n_vocab / 8 + 1, 1);
        if (!seen) return 0;
        for (int i = n_tokens - hist; i < n_tokens; i++) {
            llama_token t = tokens[i];
            if (t < 0 || t >= g_n_vocab || (seen[t >> 3] & (1u << (t & 7)))) continue;
            seen[t >> 3] |= (uint8_t)(1u << (t & 7));
            window++;
        }
        free(seen);
        window += max_new;
    }
    long k = (long)top_k + window;
    if (k > 8192 || k >= g_n_vocab) return 0;
    return (int)k;
}

static void engine_stop_locked(void) {
    if (g_engine_started) {
        pthread_mutex_lock(&g_engine_lock);
        g_engine_quit = true;
        pthread_cond_broadcast(&g_engine_cond);
        pthread_mutex_unlock(&g_engine_lock);
        pthread_join(g_engine_thread, NULL);
        g_engine_started = false;
    }
    if (g_samp_threads) {
        pthread_mutex_lock(&g_samp_lock);
        g_samp_quit = true;
        pthread_cond_broadcast(&g_samp_cond);
        pthread_mutex_unlock(&g_samp_lock);
        for (int i = 0; i < g_samp_count; i++) pthread_join(g_samp_threads[i], NULL);
        free(g_samp_threads);
        g_samp_threads = NULL;
    }
    g_samp_count = 0;
    g_samp_quit = false;
    if (g_step.token) llama_batch_free(g_step);
    memset(&g_step, 0, sizeof g_step);
    free(g_step_slots);
    g_step_slots = NULL;
    free(g_sample_jobs);
    g_sample_jobs = NULL;
    free(g_gather_items);
    g_gather_items = NULL;
    g_engine_quit = false;
}

static double engine_probe_free_mb(struct llama_context_params cp, int kv_seqs) {
    cp.n_ctx = (unsigned)g_ctx_len * (unsigned)kv_seqs;
    struct llama_context *c = llama_init_from_model(g_model, cp);
    if (!c) return -1.0;
    double free_gib = -1.0;
    bool ok = ogpu_memory_gib(&free_gib, NULL);
    llama_free(c);
    return ok ? free_gib * 1024.0 : -1.0;
}

/* LLM_KV_SEQS=auto: the pool takes what the device has left after a reserve
 * (LLM_KV_RESERVE_MB, default 1024) for neighbours. Two probe contexts give
 * the per-context KV cost and the fixed compute buffers. */
static int engine_auto_kv_seqs(struct llama_context_params cp) {
    double before_gib = -1.0;
    if (!ogpu_memory_gib(&before_gib, NULL) || before_gib <= 0.0) return g_slot_count;
    double f1 = engine_probe_free_mb(cp, 1), f2 = engine_probe_free_mb(cp, 2);
    double before = before_gib * 1024.0, per = f1 - f2, fixed = before - f1 - per;
    const char *res_env = getenv("OMNISERVE_NATIVE_LLM_KV_RESERVE_MB");
    double reserve = res_env && res_env[0] ? atof(res_env) : 1024.0;
    if (f1 < 0.0 || f2 < 0.0 || per <= 1.0) return g_slot_count;
    int n = (int)((before - (fixed > 0.0 ? fixed : 0.0) - reserve) / per);
    if (n < 1) n = 1;
    if (n > g_slot_count) n = g_slot_count;
    fprintf(stderr, "llm: KV pool auto: %d contexts (%.0f MiB each, %.0f MiB fixed, %.0f MiB free, "
            "%.0f MiB reserved)\n", n, per, fixed, before, reserve);
    return n;
}

static void engine_mtp_free(void) {
    if (g_mtp_ctx) llama_free(g_mtp_ctx);
    g_mtp_ctx = NULL;
    for (int i = 0; g_mtp_greedy && i < g_slot_count; i++) llama_sampler_free(g_mtp_greedy[i]);
    free(g_mtp_greedy);
    g_mtp_greedy = NULL;
    if (g_mtp_model) llama_model_free(g_mtp_model);
    g_mtp_model = NULL;
    if (g_mtp_batch.pos) llama_batch_free(g_mtp_batch);
    memset(&g_mtp_batch, 0, sizeof g_mtp_batch);
    free(g_mtp_rows);
    g_mtp_rows = NULL;
    g_mtp_go = false;
}

/* Draft size from OMNISERVE_NATIVE_LLM_MTP_DRAFT, 0 when no head is set; read
 * before the target context exists because its n_batch has to hold the
 * verify rows. */
static int engine_mtp_draft_cfg(void) {
    const char *path = getenv("OMNISERVE_NATIVE_LLM_MTP");
    if (!path || !path[0]) return 0;
    const char *d = getenv("OMNISERVE_NATIVE_LLM_MTP_DRAFT");
    int n = d && d[0] ? atoi(d) : 2;
    if (n < 0) n = 0;
    if (n > OLLM_MTP_BATCH_DRAFT_MAX) n = OLLM_MTP_BATCH_DRAFT_MAX;
    return n;
}

/* Pairs the batched target with one MTP draft context. Any failure leaves
 * MTP off and the engine as it was. */
static void engine_mtp_init(int n_gpu_layers, struct llama_context_params cp) {
    engine_mtp_free();
    if (g_mtp_draft <= 0) return;
    const char *path = getenv("OMNISERVE_NATIVE_LLM_MTP");
    const char *ma = getenv("OMNISERVE_NATIVE_LLM_MTP_MAX_ACTIVE");
    g_mtp_max_active = ma && ma[0] ? atoi(ma) : 4;
    if (g_mtp_max_active > g_slot_count) g_mtp_max_active = g_slot_count;
    if (g_mtp_max_active <= 0 || !probe_nextn_fns()) {
        if (g_mtp_max_active > 0) fprintf(stderr, "llm MTP: nextn API missing from libllama, off\n");
        return;
    }
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    g_mtp_model = llama_model_load_from_file(path, mp);
    const struct llama_vocab *hv = g_mtp_model ? llama_model_get_vocab(g_mtp_model) : NULL;
    if (!g_mtp_model || llama_model_n_embd_out(g_mtp_model) != llama_model_n_embd(g_model) ||
        !hv || llama_vocab_n_tokens(hv) != g_n_vocab) {
        fprintf(stderr, "llm MTP: head '%s' failed to load or does not match the target, off\n",
                path);
        engine_mtp_free();
        return;
    }
    g_mtp_embd = llama_model_n_embd_out(g_mtp_model);
    struct llama_context_params dcp = cp;
    dcp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    dcp.ctx_other = g_bctx;
    dcp.n_rs_seq = 0;
    dcp.n_seq_max = (unsigned)g_slot_count;
    dcp.n_batch = (unsigned)g_slot_count;
    dcp.n_ubatch = (unsigned)g_slot_count;
    dcp.kv_unified = true;
    /* Greedy drafting on the device: no 1 MB logits row per draft over PCIe
     * and no host argmax. */
    const char *bs = getenv("OMNISERVE_NATIVE_LLM_MTP_BACKEND_SAMPLE");
    const char *cg = getenv("OMNISERVE_NATIVE_LLM_MTP_CONF_SAMPLER");
    g_mtp_has_conf = !(cg && strcmp(cg, "0") == 0);
    const char *ct = getenv("OMNISERVE_NATIVE_LLM_MTP_CONF");
    g_mtp_conf = ct && ct[0] && g_mtp_has_conf ? (float)atof(ct) : 0.0f;
    const char *pd = getenv("OMNISERVE_NATIVE_LLM_MTP_PAD");
    g_mtp_pad = g_mtp_conf > 0.0f && !(pd && strcmp(pd, "0") == 0);
    struct llama_sampler_seq_config *sc = NULL;
    if (!(bs && strcmp(bs, "0") == 0)) {
        g_mtp_greedy = calloc((size_t)g_slot_count, sizeof *g_mtp_greedy);
        sc = calloc((size_t)g_slot_count, sizeof *sc);
        for (int i = 0; g_mtp_greedy && sc && i < g_slot_count; i++) {
            g_mtp_greedy[i] = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(g_mtp_greedy[i], g_mtp_has_conf
                ? llama_sampler_init(&g_confg_iface, NULL) : llama_sampler_init_greedy());
            sc[i].seq_id = i;
            sc[i].sampler = g_mtp_greedy[i];
        }
        if (sc && g_mtp_greedy) {
            dcp.samplers = sc;
            dcp.n_samplers = (size_t)g_slot_count;
        }
    }
    g_mtp_ctx = llama_init_from_model(g_mtp_model, dcp);
    free(sc);
    if (!g_mtp_ctx || ollama_get_ctx_other(g_mtp_ctx) != g_bctx) {
        fprintf(stderr, "llm MTP: draft context failed or does not share the target KV, off\n");
        engine_mtp_free();
        return;
    }
    g_mtp_batch = llama_batch_init(g_slot_count, g_mtp_embd, 1);
    g_mtp_batch.token = malloc((size_t)g_slot_count * sizeof *g_mtp_batch.token);
    g_mtp_rows = calloc((size_t)g_slot_count, sizeof *g_mtp_rows);
    if (!g_mtp_batch.token || !g_mtp_batch.embd || !g_mtp_rows) {
        engine_mtp_free();
        return;
    }
    ollama_set_nextn(g_bctx, true, true);
    ollama_set_nextn(g_mtp_ctx, true, true);
    fprintf(stderr, "llm MTP: head '%s' draft=%d max_active=%d n_embd=%d device_sample=%d conf=%d min_cum=%.2f\n", path,
            g_mtp_draft, g_mtp_max_active, g_mtp_embd, g_mtp_greedy ? 1 : 0, g_mtp_has_conf ? 1 : 0,
            (double)g_mtp_conf);
}

static bool engine_start_locked(void) {
    g_step = llama_batch_init(g_step_budget + g_slot_count * (1 + g_mtp_draft), 0, 1);
    g_step_slots = calloc((size_t)g_slot_count + 1, sizeof *g_step_slots);
    g_sample_jobs = calloc((size_t)g_slot_count + 1, sizeof *g_sample_jobs);
    if (!g_step.token || !g_step_slots || !g_sample_jobs) return false;
    const char *pg = getenv("OMNISERVE_NATIVE_LLM_PAR_GATHER");
    bool par_gather = g_mtp_draft > 0 && pg && strcmp(pg, "1") == 0;
    for (int i = 0; i < g_slot_count; i++) {
        ollm_slot *s = &g_slots[i];
        s->seq = i;
        s->rm_from = -1;
        s->row = -1;
        s->cand = malloc((size_t)g_n_vocab * sizeof *s->cand);
        if (!s->cand) return false;
        for (int k = 0; par_gather && k < g_mtp_draft && k < OLLM_MTP_BATCH_DRAFT_MAX; k++) {
            free(s->vcand[k]);
            s->vcand[k] = malloc((size_t)OLLM_VCAND_CAP * sizeof *s->vcand[k]);
            if (!s->vcand[k]) return false;
        }
        pthread_cond_init(&s->cond, NULL);
    }
    if (par_gather) {
        g_gather_items = calloc((size_t)g_slot_count * (OLLM_MTP_BATCH_DRAFT_MAX + 1), sizeof *g_gather_items);
        if (!g_gather_items) return false;
    }
    const char *st = getenv("OMNISERVE_NATIVE_LLM_SAMPLE_THREADS");
    int want = st && st[0] ? atoi(st) : (g_slot_count < 16 ? g_slot_count : 16) - 1;
    if (want < 0) want = 0;
    if (want > 64) want = 64;
    if (want > 0) {
        g_samp_threads = calloc((size_t)want, sizeof *g_samp_threads);
        if (!g_samp_threads) return false;
        for (int i = 0; i < want; i++) {
            if (pthread_create(&g_samp_threads[i], NULL, samp_main, NULL) != 0) break;
            g_samp_count++;
        }
    }
    const char *trace = getenv("OMNISERVE_NATIVE_LLM_ENGINE_TRACE");
    g_engine_trace = trace && trace[0] == '1';
    g_engine_trace_flag = g_engine_trace;
    if (pthread_create(&g_engine_thread, NULL, engine_main, NULL) != 0) return false;
    g_engine_started = true;
    return true;
}

static bool ollm_init_locked(const char *model_path, int n_gpu_layers, int ctx_len,
                             int parallel_contexts) {
    llama_runtime_acquire();
    probe_gpu_device();
    g_gpu_requested = n_gpu_layers > 0;
    g_on_gpu = g_gpu_requested && g_gpu_device_present;
    otune_profile early_profile;
    otune_profile_for(g_on_gpu ? g_device_desc : "cpu", &early_profile);
    const char *early_kv = "f16";
    (void)kv_type_resolved(&early_profile, &early_kv);
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.tensor_buft_overrides = buft_overrides_build(
        model_path, ctx_len > 0 ? ctx_len : 8192,
        parallel_contexts > 0 ? parallel_contexts : 1, early_kv);
    g_model = llama_model_load_from_file(model_path, mp);
    if (!g_model) {
        buft_overrides_clear();
        llama_runtime_release();
        return false;
    }
    g_vocab = llama_model_get_vocab(g_model);
    g_n_vocab = llama_vocab_n_tokens(g_vocab);

    g_ctx_len = ctx_len > 0 ? ctx_len : 8192;
    /* Batch geometry that saturates a 5090 stalls a T4, so the defaults follow
     * the device the weights actually landed on. */
    otune_profile profile;
    otune_profile_for(g_on_gpu ? g_device_desc : "cpu", &profile);
    snprintf(g_tune_class, sizeof g_tune_class, "%s", profile.class_name);
    const char *batch_env = getenv("OMNISERVE_NATIVE_BATCH");
    const char *ubatch_env = getenv("OMNISERVE_NATIVE_UBATCH");
    if (tune_value_auto(batch_env) || tune_value_auto(ubatch_env)) {
        auto_batch_geometry(g_on_gpu, &g_batch_size, &g_ubatch_size);
    } else {
        g_batch_size = batch_env ? atoi(batch_env) : profile.n_batch;
        g_ubatch_size = ubatch_env ? atoi(ubatch_env) : profile.n_ubatch;
    }
    if (g_batch_size < 1) g_batch_size = 1;
    if (g_batch_size > g_ctx_len) g_batch_size = g_ctx_len;
    if (g_ubatch_size < 1) g_ubatch_size = 1;
    if (g_ubatch_size > g_batch_size) g_ubatch_size = g_batch_size;

    /* Off unless asked for. Speculation trades arithmetic for model calls, and
     * that is only a trade worth making where decode is bandwidth-bound: on CPU
     * the wider verify batch costs proportionally more work and buys nothing,
     * which is what measuring it on this host showed. The GPU case is the one
     * the technique exists for, but this host's GPU is fully committed to
     * co-tenants, so it is unmeasured here and therefore not a default.
     * scripts/spec_bench.sh flips it on and reports whether it paid.
     *
     * A drafted token only rides along cheaply while the verify batch still
     * fits in one micro-batch; past that the round splits and the saving is
     * gone, so the ubatch is a hard ceiling. */
    const char *spec_env = getenv("OMNISERVE_NATIVE_SPEC_DRAFT");
    g_spec_draft_max = spec_env ? atoi(spec_env) : 0;
    if (g_spec_draft_max < 0) g_spec_draft_max = 0;
    if (g_spec_draft_max > g_ubatch_size - 1) g_spec_draft_max = g_ubatch_size - 1;
    if (g_spec_draft_max > 16) g_spec_draft_max = 16;
    const char *probe_env = getenv("OMNISERVE_NATIVE_SPEC_PROBE");
    g_spec_probe_interval = probe_env ? atoi(probe_env) : 32;
    if (g_spec_probe_interval < 1) g_spec_probe_interval = 1;
    /* Defaults to a device where a wrong guess is nearly free, because that is
     * the only kind of device worth turning speculation on for at all. */
    const char *patience_env = getenv("OMNISERVE_NATIVE_SPEC_PATIENCE");
    g_spec_patience = patience_env ? atoi(patience_env) : 4;
    if (g_spec_patience < 1) g_spec_patience = 1;
    ospec_config_default(&g_spec_cfg);
    const char *min_ngram_env = getenv("OMNISERVE_NATIVE_SPEC_MIN_NGRAM");
    const char *max_ngram_env = getenv("OMNISERVE_NATIVE_SPEC_MAX_NGRAM");
    if (min_ngram_env) g_spec_cfg.min_ngram = atoi(min_ngram_env);
    if (max_ngram_env) g_spec_cfg.max_ngram = atoi(max_ngram_env);
    if (g_spec_cfg.min_ngram < 1) g_spec_cfg.min_ngram = 1;
    if (g_spec_cfg.max_ngram < g_spec_cfg.min_ngram) g_spec_cfg.max_ngram = g_spec_cfg.min_ngram;

    g_slot_count = parallel_contexts > 0 ? parallel_contexts : 1;
    g_slots = calloc((size_t)g_slot_count, sizeof *g_slots);
    if (!g_slots) {
        llama_model_free(g_model);
        g_model = NULL;
        llama_runtime_release();
        return false;
    }
    const char *kv_name = "f16";
    enum ggml_type kv_type = kv_type_resolved(&profile, &kv_name);
    bool quantized_kv = kv_type != GGML_TYPE_F16 && kv_type != GGML_TYPE_BF16;
    enum llama_flash_attn_type flash_attn = flash_attn_resolved(&profile, quantized_kv);
    snprintf(g_kv_type_name, sizeof g_kv_type_name, "%s", kv_name);
    g_flash_attn = flash_attn != LLAMA_FLASH_ATTN_TYPE_DISABLED;

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (unsigned)g_ctx_len;
    cp.n_batch = (unsigned)g_batch_size;
    cp.n_ubatch = (unsigned)g_ubatch_size;
    cp.n_seq_max = 1;
    cp.type_k = kv_type;
    cp.type_v = kv_type;
    cp.flash_attn_type = flash_attn;
    cp.no_perf = true;
    /* Decode and prefill have different bandwidth/compute profiles on a
     * shared CPU host. llama.cpp defaults to 4 threads, which starves
     * CPU-side MoE experts on a many-core box; 16 keeps a loaded host
     * responsive without oversubscribing it. Explicit settings win. */
    const char *thread_names[] = {"OMNISERVE_NATIVE_LLM_THREADS",
                                  "OMNISERVE_NATIVE_LLM_THREADS_BATCH"};
    for (int i = 0; i < 2; ++i) {
        const char *value = getenv(thread_names[i]);
        if (!value || !value[0]) {
            if (i == 0) cp.n_threads = 16;
            else cp.n_threads_batch = 16;
            continue;
        }
        char *end = NULL;
        long threads = strtol(value, &end, 10);
        if (end == value || *end || threads < 1 || threads > 256) {
            fprintf(stderr, "%s must be an integer in [1, 256]; using 16\n", thread_names[i]);
            if (i == 0) cp.n_threads = 16;
            else cp.n_threads_batch = 16;
            continue;
        }
        if (i == 0) cp.n_threads = (int)threads;
        else cp.n_threads_batch = (int)threads;
    }
    fprintf(stderr, "llm CPU threads: decode=%d prefill=%d\n", cp.n_threads, cp.n_threads_batch);
    const char *batched_env = getenv("OMNISERVE_NATIVE_LLM_BATCHED");
    const char *mtp_path = ollm_cfg("OMNISERVE_NATIVE_SPEC_MTP_GGUF", g_ovr_mtp);
    g_batched = !(batched_env && strcmp(batched_env, "0") == 0) && g_spec_draft_max == 0 &&
                !(mtp_path && mtp_path[0]);
    if (g_batched) {
        const char *step_env = getenv("OMNISERVE_NATIVE_LLM_STEP_TOKENS");
        g_step_budget = step_env && step_env[0] ? atoi(step_env)
                                                : (g_batch_size < 512 ? g_batch_size : 512);
        if (g_step_budget < 16) g_step_budget = 16;
        const char *swa_env = getenv("OMNISERVE_NATIVE_LLM_SWA_FULL");
        /* The unified cache is a pool: sized for LLM_KV_SEQS full contexts,
         * shared by all sequences, with idle prefixes evicted when it fills. */
        const char *kv_env = getenv("OMNISERVE_NATIVE_LLM_KV_SEQS");
        bool kv_auto = kv_env && strcmp(kv_env, "auto") == 0;
        g_kv_seqs = kv_env && kv_env[0] && !kv_auto ? atoi(kv_env) : g_slot_count;
        if (g_kv_seqs < 1) g_kv_seqs = 1;
        if (g_kv_seqs > g_slot_count) g_kv_seqs = g_slot_count;
        cp.n_seq_max = (unsigned)g_slot_count;
        g_mtp_draft = engine_mtp_draft_cfg();
        cp.n_batch = (unsigned)(g_step_budget + g_slot_count * (1 + g_mtp_draft));
        if (cp.n_ubatch > cp.n_batch) cp.n_ubatch = cp.n_batch;
        cp.kv_unified = true;
        cp.swa_full = !(swa_env && strcmp(swa_env, "0") == 0);
        /* A fork copies cells; with a pruned sliding window the early SWA
         * cells are gone, so sharing needs the full-size window cache. */
        const char *share_env = getenv("OMNISERVE_NATIVE_LLM_SHARE_PREFIX");
        g_share_prefix = cp.swa_full && !(share_env && strcmp(share_env, "0") == 0);
        if (kv_auto) g_kv_seqs = engine_auto_kv_seqs(cp);
        cp.n_ctx = (unsigned)g_ctx_len * (unsigned)g_kv_seqs;
        g_bctx = llama_init_from_model(g_model, cp);
        bool started = g_bctx != NULL;
        for (int i = 0; started && i < g_slot_count; i++) g_slots[i].ctx = g_bctx;
        if (started) engine_mtp_init(n_gpu_layers, cp);
        if (!g_mtp_ctx) g_mtp_draft = 0;
        if (started) started = engine_start_locked();
        if (!started) {
            engine_stop_locked();
            engine_mtp_free();
            for (int i = 0; i < g_slot_count; i++) free(g_slots[i].cand);
            if (g_bctx) llama_free(g_bctx);
            g_bctx = NULL;
            free(g_slots);
            g_slots = NULL;
            g_slot_count = 0;
            llama_model_free(g_model);
            g_model = NULL;
            llama_runtime_release();
            return false;
        }
        fprintf(stderr, "llm batched: seqs=%d step=%d n_ctx=%u ubatch=%u swa_full=%d share=%d sample_threads=%d\n",
                g_slot_count, g_step_budget, cp.n_ctx, cp.n_ubatch, cp.swa_full ? 1 : 0,
                g_share_prefix ? 1 : 0, g_samp_count + 1);
    }
    for (int i = 0; !g_batched && i < g_slot_count; i++) {
        g_slots[i].ctx = llama_init_from_model(g_model, cp);
        g_slots[i].cand = malloc((size_t)g_n_vocab * sizeof *g_slots[i].cand);
        if (!g_slots[i].ctx || !g_slots[i].cand) {
            for (int j = 0; j <= i; j++) {
                if (g_slots[j].ctx) llama_free(g_slots[j].ctx);
                free(g_slots[j].cand);
            }
            free(g_slots);
            g_slots = NULL;
            g_slot_count = 0;
            llama_model_free(g_model);
            g_model = NULL;
            llama_runtime_release();
            return false;
        }
    }

    if (!g_batched) spec_mtp_init_locked(n_gpu_layers, &cp, kv_type);

    const char *slash = strrchr(model_path, '/');
    snprintf(g_model_name, sizeof g_model_name, "%s", slash ? slash + 1 : model_path);
    char *dot = strstr(g_model_name, ".gguf");
    if (dot) *dot = 0;
    return true;
}

bool ollm_ready(void) { return g_slots != NULL && g_slot_count > 0; }
const char *ollm_model_name(void) { return g_model_name[0] ? g_model_name : "none"; }

void ollm_placement_snapshot(ollm_placement *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!oggml_dev_by_type && !g_device_desc[0]) probe_gpu_device();
    out->gpu_device_present = g_gpu_device_present;
    out->gpu_requested = g_gpu_requested;
    out->on_gpu = g_on_gpu;
    out->flash_attn = g_flash_attn;
    snprintf(out->device, sizeof out->device, "%s",
             g_device_desc[0] ? g_device_desc : "unknown");
    snprintf(out->kv_type, sizeof out->kv_type, "%s", g_kv_type_name);
    snprintf(out->tune_class, sizeof out->tune_class, "%s", g_tune_class);
    out->n_batch = g_batch_size;
    out->n_ubatch = g_ubatch_size;
    out->gpu_expert_layers = g_gpu_expert_layers;
    out->cpu_expert_layers = g_cpu_expert_layers;
    snprintf(out->tensor_override, sizeof out->tensor_override, "%s", g_override_display);
    out->est_gpu_bytes = g_est_gpu_bytes;
    out->spec_draft_max = spec_effective_max();
    snprintf(out->spec_source, sizeof out->spec_source, "%s", spec_source_name());
    pthread_mutex_lock(&g_spec_lock);
    out->spec_rounds = g_spec_rounds;
    out->spec_drafted = g_spec_drafted;
    out->spec_accepted = g_spec_accepted;
    out->spec_saved_calls = g_spec_saved_calls;
    pthread_mutex_unlock(&g_spec_lock);
    out->batched = g_batched;
    out->batch_seqs = g_batched ? g_slot_count : 0;
    out->step_tokens = g_batched ? g_step_budget : 0;
    pthread_mutex_lock(&g_engine_lock);
    out->engine_steps = g_engine_steps;
    out->engine_rows = g_engine_rows;
    out->engine_tokens = g_engine_tokens;
    out->shared_prefix_tokens = g_shared_tokens;
    out->kv_seqs = g_batched ? g_kv_seqs : 0;
    out->kv_evictions = g_kv_evictions;
    out->kv_preemptions = g_kv_preemptions;
    out->kv_requeues = g_kv_requeues;
    pthread_mutex_unlock(&g_engine_lock);
}

static int common_prefix(const ollm_slot *slot, const llama_token *tokens, int count) {
    int common = slot->cached_count < count ? slot->cached_count : count;
    int i = 0;
    while (i < common && slot->cached_tokens[i] == tokens[i]) i++;
    /* Re-evaluate the final shared token so logits correspond to this prompt,
     * not to the previous request's generated suffix. */
    return i > 0 ? i - 1 : 0;
}

/* Picks the free slot whose cache shares the longest prefix with the prompt,
 * least recently used on a tie so one hot cache is not evicted while colder
 * ones sit idle. With prefix sharing on, a longer prefix held by any other
 * slot (busy ones included: their kept prefix is fixed until they release) is
 * forked into the chosen sequence by the engine instead of prefilled. */
static ollm_slot *acquire_slot(const llama_token *tokens, int count, int *cached) {
    static unsigned long long use_clock;
    pthread_mutex_lock(&g_slot_lock);
    for (;;) {
        ollm_slot *best = NULL;
        int best_prefix = -1;
        for (int i = 0; i < g_slot_count; i++) {
            ollm_slot *s = &g_slots[i];
            if (s->busy || s->pins > 0) continue;
            int prefix = common_prefix(s, tokens, count);
            if (prefix > best_prefix ||
                (prefix == best_prefix && s->last_used < best->last_used)) {
                best = s;
                best_prefix = prefix;
            }
        }
        if (best) {
            best->busy = true;
            best->last_used = ++use_clock;
            best->cached_count = best_prefix;
            *cached = best_prefix;
            if (g_share_prefix) {
                ollm_slot *src = NULL;
                int src_prefix = best_prefix + OLLM_SHARE_MIN;
                for (int i = 0; i < g_slot_count; i++) {
                    ollm_slot *s = &g_slots[i];
                    if (s == best) continue;
                    int prefix = common_prefix(s, tokens, count);
                    if (prefix > src_prefix) {
                        src = s;
                        src_prefix = prefix;
                    }
                }
                if (src) {
                    src->pins++;
                    best->cp_src = src;
                    best->cp_len = src_prefix;
                    best->cached_count = 0;
                    *cached = src_prefix;
                }
            }
            pthread_mutex_unlock(&g_slot_lock);
            return best;
        }
        pthread_cond_wait(&g_slot_cond, &g_slot_lock);
    }
}

static void cache_prompt_locked(ollm_slot *slot, const llama_token *tokens, int count) {
    if (count > slot->cached_cap) {
        llama_token *grown = realloc(slot->cached_tokens, (size_t)count * sizeof *grown);
        if (!grown) {
            slot->cached_count = 0;
            return;
        }
        slot->cached_tokens = grown;
        slot->cached_cap = count;
    }
    memcpy(slot->cached_tokens, tokens, (size_t)count * sizeof *tokens);
    slot->cached_count = count;
}

static void cache_prompt(ollm_slot *slot, const llama_token *tokens, int count) {
    pthread_mutex_lock(&g_slot_lock);
    cache_prompt_locked(slot, tokens, count);
    pthread_mutex_unlock(&g_slot_lock);
}

/* Snapshots the h rows staged by the target decode that just ran: pending
 * becomes the last row, and short batches (verify rounds and single decodes,
 * never prefill chunks) are kept whole for the accept step. Must run after
 * EVERY target decode while MTP is active; a skipped decode leaves pending
 * paired with the wrong token and the drafts silently degrade. */
static void spec_mtp_process(ollm_slot *slot, int n_rows) {
    if (!slot->mtp_ctx || n_rows <= 0) return;
    size_t row_bytes = (size_t)g_spec_mtp_embd * sizeof(float);
    float *last = ollama_get_nextn_ith(slot->ctx, n_rows - 1);
    if (last) memcpy(slot->mtp_pending, last, row_bytes);
    if (n_rows > OLLM_MTP_DRAFT_CAP + 1) return;
    for (int i = 0; i < n_rows; i++) {
        float *row = ollama_get_nextn_ith(slot->ctx, i);
        if (row) memcpy(slot->mtp_verify + (size_t)i * g_spec_mtp_embd, row, row_bytes);
    }
}

/* The last committed token's h row is verify row `matched`: row 0 is the
 * sampled token and row i the i-th draft, so the next draft round pairs its
 * first row with the state left behind by the last token that landed. */
static void spec_mtp_accept(ollm_slot *slot, int matched) {
    if (!slot->mtp_ctx) return;
    if (matched < 0) matched = 0;
    if (matched > OLLM_MTP_DRAFT_CAP) matched = OLLM_MTP_DRAFT_CAP;
    size_t row_bytes = (size_t)g_spec_mtp_embd * sizeof(float);
    memcpy(slot->mtp_pending,
           slot->mtp_verify + (size_t)matched * g_spec_mtp_embd, row_bytes);
}

/* Drafts up to cap tokens with the MTP head: one single-row decode per token,
 * each pairing the previous token with its h row at the same position (the
 * shared-memory layout reads the target KV, so positions never advance).
 * Stops early on a decode failure or when the head is unsure of itself. */
static int spec_mtp_draft(ollm_slot *slot, llama_token id_last, int n_past,
                          llama_token *draft_out, int cap) {
    if (!slot->mtp_ctx || !draft_out || cap <= 0) return 0;
    if (cap > OLLM_MTP_DRAFT_CAP) cap = OLLM_MTP_DRAFT_CAP;
    size_t row_bytes = (size_t)g_spec_mtp_embd * sizeof(float);
    memcpy(slot->mtp_embd, slot->mtp_pending, row_bytes);
    slot->mtp_token = id_last;
    slot->mtp_pos = (llama_pos)n_past;
    int drafted = 0;
    for (int step = 0; step < cap; step++) {
        struct llama_batch batch = {
            .n_tokens = 1,
            .token = &slot->mtp_token,
            .embd = slot->mtp_embd,
            .pos = &slot->mtp_pos,
            .n_seq_id = &slot->mtp_n_seq,
            .seq_id = &slot->mtp_seq_ptr,
            .logits = &slot->mtp_logits,
        };
        if (llama_decode(slot->mtp_ctx, batch) != 0) break;
        float *logits = llama_get_logits_ith(slot->mtp_ctx, 0);
        float *h_row = ollama_get_nextn_ith(slot->mtp_ctx, 0);
        if (!logits || !h_row) break;
        for (int32_t i = 0; i < g_spec_mtp_vocab_size; i++) {
            slot->mtp_cand[i].id = i;
            slot->mtp_cand[i].logit = logits[i];
            slot->mtp_cand[i].p = 0.0f;
        }
        llama_token_data_array cur = {
            slot->mtp_cand, (size_t)g_spec_mtp_vocab_size, -1, false
        };
        llama_sampler_apply(slot->mtp_smpl, &cur);
        if (cur.size == 0) break;
        float max_logit = cur.data[0].logit;
        for (size_t i = 1; i < cur.size; i++) {
            if (cur.data[i].logit > max_logit) max_logit = cur.data[i].logit;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < cur.size; i++) {
            float p = expf(cur.data[i].logit - max_logit);
            cur.data[i].p = p;
            sum += p;
        }
        if (!(sum > 0.0f)) break;
        float top_p = cur.data[0].p / sum;
        if (top_p < g_spec_mtp_cfg.p_min) break;
        draft_out[drafted++] = cur.data[0].id;
        slot->mtp_token = cur.data[0].id;
        memcpy(slot->mtp_embd, h_row, row_bytes);
    }
    return drafted;
}

/* One decode round, speculative or not, returning how many tokens it produced
 * into out_tokens/out_probs (0 on a decode failure).
 *
 * The exactness argument lives here. Every token returned was drawn by the
 * sampler from the real model's distribution at its own position; a drafted
 * token is never emitted because the drafter proposed it, only kept when the
 * sampler independently chose the same id. Sampling position j+1 is legitimate
 * precisely because position j's draft was confirmed, which is what makes the
 * logits at j+1 the ones sequential decoding would have produced. At
 * temperature 0 that yields the identical token sequence; above it, the
 * identical distribution.
 *
 * n_past is the number of tokens already in the KV cache and is advanced by the
 * tokens this round commits. `previous` is the last token sampled but not yet
 * in the cache.
 */
static int decode_round(ollm_slot *slot, struct llama_sampler *smpl,
                        const probability_capture *sampled,
                        llama_token previous, int *n_past,
                        const llama_token *context, int context_len,
                        ospec_governor *gov,
                        llama_token *out_tokens, float *out_probs, int out_cap) {
    if (out_cap <= 0) return 0;
    struct llama_context *ctx = slot->ctx;

    /* Straight after the prefill there is nothing to decode: the logits for the
     * next token are already the ones the prompt left behind. */
    if (previous == LLAMA_TOKEN_NULL) {
        out_tokens[0] = slot_sample(slot, smpl, ctx, -1);
        out_probs[0] = sampled->probability;
        return 1;
    }

    llama_token draft[16];
    int draft_len = 0;
    int want = gov ? ospec_governor_next(gov) : 0;
    int eff_max = spec_effective_max();
    if (want > 0 && eff_max > 0) {
        int cap = want < eff_max ? want : eff_max;
        if (cap > (int)(sizeof draft / sizeof *draft)) cap = (int)(sizeof draft / sizeof *draft);
        if (cap > out_cap - 1) cap = out_cap - 1;
        if (cap > 0) {
            if (slot->mtp_ctx) {
                draft_len = spec_mtp_draft(slot, previous, *n_past, draft, cap);
            } else {
                draft_len = ospec_draft(&g_spec_cfg, context, context_len, draft, cap);
            }
        }
    }

    if (draft_len <= 0) {
        struct llama_batch one = llama_batch_get_one(&previous, 1);
        if (llama_decode(ctx, one) != 0) return 0;
        spec_mtp_process(slot, 1);
        (*n_past)++;
        out_tokens[0] = slot_sample(slot, smpl, ctx, -1);
        out_probs[0] = sampled->probability;
        if (gov) ospec_governor_observe(gov, 0, 0);
        return 1;
    }

    /* The verify batch is `previous` followed by the draft, every position
     * asked for logits: a draft is only useful if the token after it can be
     * sampled without running the model again. */
    const int verify_count = draft_len + 1;
    if (verify_count > OLLM_VERIFY_BATCH_CAP) return 0;
    for (int i = 0; i < draft_len + 1; i++) {
        slot->verify_tokens[i] = i == 0 ? previous : draft[i - 1];
        slot->verify_positions[i] = (llama_pos)(*n_past + i);
        slot->verify_n_seq_ids[i] = 1;
        slot->verify_seq_ids[i] = 0;
        slot->verify_seq_id_ptrs[i] = &slot->verify_seq_ids[i];
        slot->verify_logits[i] = 1;
    }
    struct llama_batch batch = {
        .n_tokens = verify_count,
        .token = slot->verify_tokens,
        .embd = NULL,
        .pos = slot->verify_positions,
        .n_seq_id = slot->verify_n_seq_ids,
        .seq_id = slot->verify_seq_id_ptrs,
        .logits = slot->verify_logits,
    };
    int rc = llama_decode(ctx, batch);
    if (rc != 0) return 0;
    spec_mtp_process(slot, verify_count);

    int produced = 0;
    int matched = 0;
    for (int i = 0; i <= draft_len && produced < out_cap; i++) {
        llama_token tok = slot_sample(slot, smpl, ctx, i);
        out_tokens[produced] = tok;
        out_probs[produced] = sampled->probability;
        produced++;
        /* Stop at end-of-generation rather than sampling past it: the caller is
         * about to discard everything after it, and sampling also advances the
         * penalty sampler's history. */
        if (llama_vocab_is_eog(g_vocab, tok)) break;
        if (i == draft_len || tok != draft[i]) break;
        matched++;
    }

    /* Committed: `previous` plus the drafts the sampler agreed with. The
     * remaining drafted positions were written to the cache by the decode and
     * have to come back out, or the next round would attend to tokens that were
     * never generated. */
    *n_past += matched + 1;
    llama_memory_t memory = llama_get_memory(ctx);
    if (matched < draft_len) llama_memory_seq_rm(memory, 0, *n_past, -1);
    if (slot->mtp_ctx) {
        spec_mtp_accept(slot, matched);
        if (matched < draft_len) {
            llama_memory_t dmem = llama_get_memory(slot->mtp_ctx);
            llama_memory_seq_rm(dmem, 0, *n_past, -1);
        }
    }
    if (gov) ospec_governor_observe(gov, draft_len, matched);
    return produced;
}

static void release_slot(ollm_slot *slot) {
    pthread_mutex_lock(&g_slot_lock);
    slot->busy = false;
    pthread_cond_signal(&g_slot_cond);
    pthread_mutex_unlock(&g_slot_lock);
}

static size_t matched_stop_suffix(const ochat_req *req, const char *text, size_t text_len) {
    size_t matched = 0;
    for (int i = 0; i < req->stop_count; i++) {
        const char *stop = req->stop_sequences[i];
        if (!stop) continue;
        size_t stop_len = strlen(stop);
        if (stop_len && stop_len <= text_len && stop_len > matched &&
            memcmp(text + text_len - stop_len, stop, stop_len) == 0) {
            matched = stop_len;
        }
    }
    return matched;
}

static size_t possible_stop_prefix(const ochat_req *req, const char *text, size_t text_len) {
    size_t held = 0;
    for (int i = 0; i < req->stop_count; i++) {
        const char *stop = req->stop_sequences[i];
        if (!stop || !stop[0]) continue;
        size_t stop_len = strlen(stop);
        size_t try_len = stop_len - 1;
        if (try_len > text_len) try_len = text_len;
        while (try_len > held) {
            if (memcmp(text + text_len - try_len, stop, try_len) == 0) {
                held = try_len;
                break;
            }
            try_len--;
        }
    }
    return held;
}

static size_t think_marker_hold(const char *text, size_t text_len,
                                  const char *marker, size_t marker_len) {
    size_t most = marker_len - 1;
    if (most > text_len) most = text_len;
    for (size_t k = most; k >= 1; k--) {
        if (memcmp(text + text_len - k, marker, k) == 0) return k;
        if (k == 1) break;
    }
    return 0;
}

static bool prompt_append(char **prompt, size_t *length, size_t *capacity,
                          const char *text, size_t text_len) {
    if (text_len > SIZE_MAX - *length - 1) return false;
    size_t needed = *length + text_len + 1;
    if (needed > *capacity) {
        size_t grown_capacity = *capacity ? *capacity : 1024;
        while (grown_capacity < needed) {
            if (grown_capacity > SIZE_MAX / 2) {
                grown_capacity = needed;
                break;
            }
            grown_capacity *= 2;
        }
        char *grown = realloc(*prompt, grown_capacity);
        if (!grown) return false;
        *prompt = grown;
        *capacity = grown_capacity;
    }
    memcpy(*prompt + *length, text, text_len);
    *length += text_len;
    (*prompt)[*length] = 0;
    return true;
}

static bool prompt_append_string(char **prompt, size_t *length, size_t *capacity,
                                 const char *text) {
    return prompt_append(prompt, length, capacity, text, strlen(text));
}

static bool prompt_append_trimmed(char **prompt, size_t *length, size_t *capacity,
                                  const char *text) {
    const unsigned char *start = (const unsigned char *)text;
    const unsigned char *end = start + strlen(text);
    while (start < end && isspace(*start)) start++;
    while (end > start && isspace(end[-1])) end--;
    return prompt_append(prompt, length, capacity, (const char *)start,
                         (size_t)(end - start));
}

/* llama_chat_apply_template() intentionally implements only a fixed set of
 * legacy templates. Gemma 4 uses a newer Jinja template. Our public message
 * contract is string-only and has no tool-call objects, so its exact basic
 * form can be rendered without adding a C++/Jinja runtime to the data plane. */
static char *format_gemma4_prompt(const struct llama_chat_message *messages,
                                  int message_count, bool enable_thinking,
                                  int32_t *formatted_len) {
    size_t capacity = 1024;
    size_t length = 0;
    char *prompt = malloc(capacity);
    if (!prompt) return NULL;
    prompt[0] = 0;

#define APPEND_LITERAL(value) \
    do { \
        if (!prompt_append_string(&prompt, &length, &capacity, (value))) goto fail; \
    } while (0)

    /* ollm_chat tokenizes with add_special=true, which supplies Gemma's BOS. */
    int first_message = 0;
    bool starts_with_system = message_count > 0 &&
        (strcmp(messages[0].role, "system") == 0 ||
         strcmp(messages[0].role, "developer") == 0);
    if (enable_thinking || starts_with_system) {
        APPEND_LITERAL("<|turn>system\n");
        if (enable_thinking) APPEND_LITERAL("<|think|>\n");
        if (starts_with_system) {
            if (!prompt_append_trimmed(&prompt, &length, &capacity,
                                       messages[0].content)) goto fail;
            first_message = 1;
        }
        APPEND_LITERAL("<turn|>\n");
    }

    const char *prev_role = NULL;
    for (int i = first_message; i < message_count; i++) {
        const char *source_role = messages[i].role;
        if (strcmp(source_role, "tool") == 0) continue;
        const char *role = strcmp(source_role, "assistant") == 0 ? "model" : source_role;
        if (!(strcmp(role, "model") == 0 && prev_role &&
              strcmp(prev_role, "assistant") == 0)) {
            APPEND_LITERAL("<|turn>");
            APPEND_LITERAL(role);
            APPEND_LITERAL("\n");
        }
        prev_role = source_role;
        const char *content = messages[i].content;
        char *stripped = NULL;
        if (strcmp(role, "model") == 0 && strstr(content, "<|channel>") != NULL) {
            stripped = malloc(strlen(content) + 1);
            if (!stripped) goto fail;
            ollm_strip_channels(content, stripped, strlen(content) + 1);
            content = stripped;
        }
        bool appended = prompt_append_trimmed(&prompt, &length, &capacity, content);
        free(stripped);
        if (!appended) goto fail;
        bool continues = strcmp(role, "model") == 0;
        if (continues) {
            continues = false;
            for (int j = i + 1; j < message_count; j++) {
                if (strcmp(messages[j].role, "tool") == 0) continue;
                continues = strcmp(messages[j].role, "assistant") == 0;
                break;
            }
        }
        if (!continues) APPEND_LITERAL("<turn|>\n");
    }
    APPEND_LITERAL("<|turn>model\n");
    if (!enable_thinking) APPEND_LITERAL("<|channel>thought\n<channel|>");
    if (length > INT32_MAX) goto fail;
    *formatted_len = (int32_t)length;
#undef APPEND_LITERAL
    return prompt;

fail:
#undef APPEND_LITERAL
    free(prompt);
    return NULL;
}

static bool ollm_chat_locked(const ochat_req *req, otoken_cb on_token, void *user,
                             ochat_result *out) {
    if (!ollm_ready()) return false;
    memset(out, 0, sizeof *out);
    out->finish_reason = "stop";

    char *prompt = NULL;
    int32_t plen = -1;
    if (req->raw_prompt) {
        size_t raw_len = strlen(req->raw_prompt);
        if (raw_len > INT32_MAX) return false;
        prompt = malloc(raw_len + 1);
        if (!prompt) return false;
        memcpy(prompt, req->raw_prompt, raw_len + 1);
        plen = (int32_t)raw_len;
    } else {
        struct llama_chat_message *msgs = calloc((size_t)req->message_count, sizeof *msgs);
        if (!msgs) return false;
        for (int i = 0; i < req->message_count; i++) {
            msgs[i].role = req->messages[i].role;
            msgs[i].content = req->messages[i].content;
        }
        bool qwen_no_think = !req->enable_thinking && strcasestr(g_model_name, "qwen") != NULL;
        char *no_think_content = NULL;
        if (qwen_no_think) {
            for (int i = req->message_count - 1; i >= 0; i--) {
                if (strcmp(msgs[i].role, "user") != 0) continue;
                size_t content_len = strlen(msgs[i].content);
                static const char suffix[] = "\n/no_think";
                no_think_content = malloc(content_len + sizeof suffix);
                if (!no_think_content) {
                    free(msgs);
                    return false;
                }
                memcpy(no_think_content, msgs[i].content, content_len);
                memcpy(no_think_content + content_len, suffix, sizeof suffix);
                msgs[i].content = no_think_content;
                break;
            }
        }
        const char *tmpl = llama_model_chat_template(g_model, NULL);
        int32_t prompt_cap = 32768;
        prompt = malloc((size_t)prompt_cap);
        if (!prompt) {
            free(no_think_content);
            free(msgs);
            return false;
        }
        plen = llama_chat_apply_template(tmpl, msgs, (size_t)req->message_count, true,
                                         prompt, prompt_cap);
        if (plen > prompt_cap) {
            prompt_cap = plen + 1;
            char *grown = realloc(prompt, (size_t)prompt_cap);
            if (!grown) {
                free(prompt);
                free(no_think_content);
                free(msgs);
                return false;
            }
            prompt = grown;
            plen = llama_chat_apply_template(tmpl, msgs, (size_t)req->message_count, true,
                                             prompt, prompt_cap);
        }
        if (plen < 0 && tmpl && strstr(tmpl, "<|turn>") != NULL) {
            free(prompt);
            prompt = format_gemma4_prompt(msgs, req->message_count,
                                          req->enable_thinking, &plen);
            if (!prompt) {
                free(no_think_content);
                free(msgs);
                return false;
            }
            prompt_cap = plen + 1;
        }
        if (plen >= 0 && qwen_no_think) {
            static const char closed_thought[] = "<think>\n\n</think>\n\n";
            size_t needed = (size_t)plen + sizeof closed_thought;
            if (needed > (size_t)prompt_cap) {
                char *grown = realloc(prompt, needed);
                if (!grown) {
                    free(prompt);
                    free(no_think_content);
                    free(msgs);
                    return false;
                }
                prompt = grown;
                prompt_cap = (int32_t)needed;
            }
            memcpy(prompt + plen, closed_thought, sizeof closed_thought);
            plen += (int32_t)(sizeof closed_thought - 1);
        }
        free(no_think_content);
        free(msgs);
    }
    if (plen < 0) { free(prompt); return false; }

    double t0 = now_ms();

    int n_tokens_max = plen + 64;
    llama_token *tokens = malloc((size_t)n_tokens_max * sizeof(llama_token));
    if (!tokens) {
        free(prompt);
        return false;
    }
    int n_tokens = llama_tokenize(g_vocab, prompt, plen, tokens, n_tokens_max, true, true);
    if (n_tokens < 0) {
        n_tokens_max = -n_tokens;
        llama_token *grown = realloc(tokens, (size_t)n_tokens_max * sizeof(llama_token));
        if (!grown) {
            free(tokens);
            free(prompt);
            return false;
        }
        tokens = grown;
        n_tokens = llama_tokenize(g_vocab, prompt, plen, tokens, n_tokens_max, true, true);
    }
    free(prompt);
    if (n_tokens < 0) {
        free(tokens);
        return false;
    }
    out->prompt_tokens = n_tokens;

    int max_new = req->max_tokens > 0 ? req->max_tokens : 512;
    if (n_tokens >= g_ctx_len) {
        free(tokens);
        return false;
    }
    if (max_new > g_ctx_len - n_tokens) max_new = g_ctx_len - n_tokens;

    int cached_tokens = 0;
    ollm_slot *slot = acquire_slot(tokens, n_tokens, &cached_tokens);
    struct llama_context *ctx = slot->ctx;
    llama_memory_t memory = llama_get_memory(ctx);

    if (g_batched) {
        /* the engine trims the sequence to the kept prefix on its own thread */
    } else if (cached_tokens > 0) {
        if (!llama_memory_seq_rm(memory, -1, cached_tokens, -1)) {
            llama_memory_clear(memory, false);
            cached_tokens = 0;
        }
    } else {
        llama_memory_clear(memory, false);
    }
    if (slot->mtp_ctx) {
        llama_memory_t dmem = llama_get_memory(slot->mtp_ctx);
        if (dmem != memory) {
            if (cached_tokens > 0) {
                if (!llama_memory_seq_rm(dmem, -1, cached_tokens, -1)) {
                    llama_memory_clear(dmem, false);
                }
            } else {
                llama_memory_clear(dmem, false);
            }
        }
    }
    if (!g_batched) slot->cached_count = cached_tokens;
    out->cached_prompt_tokens = cached_tokens;

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    probability_capture sampled = { .probability = 1.0f };
    if (req->min_probability > 0.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init(&probability_capture_i, &sampled));
    float repetition = req->repetition_penalty > 0 ? req->repetition_penalty : 1.0f;
    bool dry = req->dry_multiplier > 0.0f;
    bool penalized = repetition != 1.0f || req->frequency_penalty != 0.0f ||
                     req->presence_penalty != 0.0f || dry;
    int last_n = 0;
    if (penalized) {
        /* llama.cpp clamps a negative window to 0, which silently disables the
         * sampler, so "whole context" has to be spelled out as n_ctx. */
        last_n = req->repetition_last_n ? req->repetition_last_n : penalty_last_n_default();
        if (last_n < 0) last_n = g_ctx_len;
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            last_n, repetition, req->frequency_penalty, req->presence_penalty));
    }
    if (dry) {
        llama_sampler_chain_add(smpl, dry_sampler(req->dry_multiplier,
            req->dry_base > 0.0f ? req->dry_base : 1.75f,
            req->dry_allowed_length > 0 ? req->dry_allowed_length : 2));
    }
    int top_k = req->top_k > 0 ? req->top_k : 40;
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(req->top_p > 0 ? req->top_p : 0.92f, 1));
    if (req->min_p > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(req->min_p, 1));
    }
    if (req->temperature <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(req->temperature));
        uint32_t seed = req->seed >= 0 ? (uint32_t)req->seed : LLAMA_DEFAULT_SEED;
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }
    /* Only the penalty and DRY samplers keep history; DRY spans the context. */
    if (penalized) {
        int history = dry ? n_tokens : last_n;
        for (int i = n_tokens > history ? n_tokens - history : 0; i < n_tokens; i++)
            llama_sampler_accept(smpl, tokens[i]);
    }

    size_t text_cap = 4096, text_len = 0;
    char *text = malloc(text_cap);
    if (!text) {
        llama_sampler_free(smpl);
        free(tokens);
        release_slot(slot);
        return false;
    }
    text[0] = 0;
    bool ok = true;
    bool cancelled = false;
    bool sentence_end = false;
    int sentences = 0;
    size_t streamed_len = 0;
    size_t think_hold = (size_t)-1;

    int decoded = g_batched ? n_tokens : cached_tokens;
    while (decoded < n_tokens) {
        int chunk = n_tokens - decoded;
        if (chunk > g_batch_size) chunk = g_batch_size;
        struct llama_batch prompt_batch = llama_batch_get_one(tokens + decoded, chunk);
        if (llama_decode(ctx, prompt_batch) != 0) {
            ok = false;
            out->finish_reason = "error";
            break;
        }
        spec_mtp_process(slot, chunk);
        decoded += chunk;
    }
    if (g_batched) {
        /* prefill happens in the engine; the cache is recorded at release */
    } else if (ok) {
        cache_prompt(slot, tokens, n_tokens);
    } else {
        slot->cached_count = 0;
    }

    llama_token previous = LLAMA_TOKEN_NULL;
    float cumulative_probability = 1.0f;

    /* The n-gram drafter looks the generated text up in the whole context, so
     * the prompt and the reply so far have to live in one array. Sized once for
     * the worst case: a realloc here would invalidate the pointer mid-round. */
    int context_len = n_tokens;
    llama_token *context = malloc((size_t)(n_tokens + max_new + 1) * sizeof *context);
    if (!context) {
        llama_sampler_free(smpl);
        free(text);
        free(tokens);
        release_slot(slot);
        return false;
    }
    memcpy(context, tokens, (size_t)n_tokens * sizeof *context);

    ospec_governor gov;
    ospec_governor_init(&gov, spec_effective_max(), g_spec_probe_interval, g_spec_patience);
    int n_past = n_tokens;
    llama_token round_tokens[17];
    float round_probs[17];
    int round_count = 0, round_pos = 0;
    int prefilter_k = engine_prefilter_k(req, penalized, dry, last_n, top_k, tokens, n_tokens,
                                         max_new);
    if (!g_batched) slot->prefilter_k = prefilter_k;
    if (g_batched) {
        /* Publish the prompt now; the engine advances cached_count as prefill
         * lands, so concurrent requests can fork the part already computed. */
        pthread_mutex_lock(&g_slot_lock);
        int visible = slot->cp_src ? 0 : cached_tokens;
        cache_prompt_locked(slot, tokens, n_tokens);
        slot->cached_count = visible < slot->cached_count ? visible : slot->cached_count;
        pthread_mutex_unlock(&g_slot_lock);
        engine_submit(slot, tokens, n_tokens, cached_tokens, max_new, smpl, &sampled,
                      prefilter_k);
    }

    for (int i = 0; i < max_new; i++) {
        if (!ok) break;
        if (round_pos >= round_count) {
            if (g_batched) {
                bool failed = false;
                round_count = engine_pull(slot, round_tokens, round_probs,
                                          (int)(sizeof round_tokens / sizeof *round_tokens),
                                          &failed);
                if (failed && round_count == 0) round_count = -1;
            } else {
                round_count = decode_round(slot, smpl, &sampled, previous, &n_past,
                                           context, context_len,
                                           spec_effective_max() > 0 ? &gov : NULL,
                                           round_tokens, round_probs,
                                           (int)(sizeof round_tokens / sizeof *round_tokens));
            }
            round_pos = 0;
            if (round_count <= 0) {
                ok = i > 0;
                out->finish_reason = "error";
                break;
            }
        }
        llama_token tok = round_tokens[round_pos];
        float token_probability = round_probs[round_pos];
        round_pos++;
        previous = tok;
        if (llama_vocab_is_eog(g_vocab, tok)) break;
        /* Recorded before the stop checks so the drafter sees exactly the token
         * sequence the model produced, whether or not the caller keeps it. */
        context[context_len++] = tok;
        cumulative_probability *= token_probability;
        /* textgen semantics: the token that crosses the threshold is kept */
        bool min_prob_stop = req->min_probability > 0.0f &&
                             cumulative_probability < req->min_probability;
        char piece[256];
        int n = llama_token_to_piece(g_vocab, tok, piece, sizeof piece, 0, true);
        if (n > 0) {
            bool add_boundary_space = text_len == 0 && req->completion_prefix &&
                otext_completion_needs_space(req->completion_prefix,
                                             strlen(req->completion_prefix),
                                             piece, (size_t)n,
                                             req->raw_prompt != NULL);
            size_t append_len = (size_t)n + (add_boundary_space ? 1u : 0u);
            if (text_len + append_len + 1 > text_cap) {
                while (text_len + append_len + 1 > text_cap) text_cap *= 2;
                char *grown = realloc(text, text_cap);
                if (!grown) { ok = false; out->finish_reason = "error"; break; }
                text = grown;
            }
            if (add_boundary_space) text[text_len++] = ' ';
            memcpy(text + text_len, piece, (size_t)n);
            size_t piece_start = text_len;
            text_len += (size_t)n;
            text[text_len] = 0;

            /* Gemma 4's native Jinja template emits an empty thought-channel
             * opener when thinking is disabled. It is a prompt control marker,
             * not user-visible assistant content. Hold a possible partial
             * marker during streaming, then remove it once complete. */
            static const char no_think_marker[] =
                "<|channel>thought\n<channel|>";
            const size_t no_think_marker_len = sizeof no_think_marker - 1;
            if (!req->enable_thinking && text_len >= no_think_marker_len &&
                memcmp(text, no_think_marker, no_think_marker_len) == 0) {
                memmove(text, text + no_think_marker_len,
                        text_len - no_think_marker_len + 1);
                text_len -= no_think_marker_len;
                piece_start = piece_start >= no_think_marker_len
                    ? piece_start - no_think_marker_len : 0;
            }

            /* A reasoning model may still open a thought channel with thinking
             * disabled. Excise closed blocks and hold an unterminated one out
             * of the streamed prefix; the tail below drops a dangling block.
             * Token pieces split markers arbitrarily, so the search reaches
             * back over the previous piece tail. */
            static const char think_open[] = "<|channel>";
            static const char think_close[] = "<channel|>";
            static const size_t think_open_len = sizeof think_open - 1;
            static const size_t think_close_len = sizeof think_close - 1;
            if (!req->enable_thinking) {
                size_t search_from = piece_start >= think_open_len - 1
                    ? piece_start - (think_open_len - 1) : 0;
                if (think_hold == (size_t)-1) {
                    const char *open_at = strstr(text + search_from, think_open);
                    if (open_at) {
                        think_hold = (size_t)(open_at - text);
                    } else {
                        size_t close_from = piece_start >= think_close_len - 1
                            ? piece_start - (think_close_len - 1) : 0;
                        char *bare = strstr(text + close_from, think_close);
                        if (bare) {
                            size_t off = (size_t)(bare - text);
                            memmove(bare, bare + think_close_len,
                                    text_len - off - think_close_len + 1);
                            text_len -= think_close_len;
                            if (piece_start > off) {
                                piece_start = piece_start >= off + think_close_len
                                    ? piece_start - think_close_len : off;
                            }
                        }
                    }
                }
                if (think_hold != (size_t)-1) {
                    const char *close_at = strstr(text + think_hold + think_open_len,
                                                  think_close);
                    if (close_at) {
                        size_t drop = (size_t)(close_at - text) - think_hold +
                            think_close_len;
                        memmove(text + think_hold, text + think_hold + drop,
                                text_len - think_hold - drop + 1);
                        text_len -= drop;
                        if (piece_start > think_hold) {
                            piece_start = piece_start >= think_hold + drop
                                ? piece_start - drop : think_hold;
                        }
                        think_hold = (size_t)-1;
                    }
                }
            }

            size_t stop_len = matched_stop_suffix(req, text, text_len);
            bool should_stop = stop_len > 0;
            if (should_stop) {
                text_len -= stop_len;
                text[text_len] = 0;
                out->finish_reason = "stop";
            }
            if (!should_stop && req->max_sentences > 0) {
                for (size_t j = piece_start; j < text_len; j++) {
                    bool punctuation = text[j] == '.' || text[j] == '!' || text[j] == '?';
                    if (punctuation && !sentence_end) sentences++;
                    if (!punctuation && text[j] != ' ' && text[j] != '\t' &&
                        text[j] != '\r' && text[j] != '\n' && text[j] != '"' &&
                        text[j] != '\'' && text[j] != ')' && text[j] != ']') {
                        sentence_end = false;
                    } else if (punctuation) {
                        sentence_end = true;
                    }
                }
                if (sentences >= req->max_sentences) {
                    should_stop = true;
                    out->finish_reason = "max_sentences";
                }
            }
            if (on_token) {
                bool hold_no_think_marker = !req->enable_thinking &&
                    text_len > 0 && text_len < no_think_marker_len &&
                    memcmp(text, no_think_marker, text_len) == 0;
                size_t held = should_stop ? 0 : possible_stop_prefix(req, text, text_len);
                size_t safe_len = text_len - held;
                if (think_hold != (size_t)-1 && safe_len > think_hold) safe_len = think_hold;
                if (!req->enable_thinking) {
                    size_t open_hold = think_marker_hold(text, text_len, think_open,
                                                         think_open_len);
                    size_t close_hold = think_marker_hold(text, text_len, think_close,
                                                          think_close_len);
                    size_t marker_hold =
                        open_hold > close_hold ? open_hold : close_hold;
                    if (marker_hold > 0 && safe_len > text_len - marker_hold)
                        safe_len = text_len - marker_hold;
                }
                if (safe_len < streamed_len) safe_len = streamed_len;
                if (!hold_no_think_marker && safe_len > streamed_len &&
                    !on_token(text + streamed_len, safe_len - streamed_len, user)) {
                    out->finish_reason = "cancelled";
                    cancelled = true;
                    break;
                }
                streamed_len = safe_len;
            }
            if (should_stop) break;
        }
        out->completion_tokens++;
        if (out->completion_tokens >= max_new) out->finish_reason = "length";
        if (min_prob_stop) {
            out->finish_reason = "min_probability";
            break;
        }
    }

    if (!req->enable_thinking) {
        if (think_hold != (size_t)-1 && think_hold < text_len) {
            text_len = think_hold;
            text[text_len] = 0;
        }
        ollm_strip_channels_final(text, text_cap);
        text_len = strlen(text);
        if (streamed_len > text_len) streamed_len = text_len;
    }

    if (on_token && !cancelled && text_len > streamed_len &&
        !on_token(text + streamed_len, text_len - streamed_len, user)) {
        out->finish_reason = "cancelled";
    }

    spec_record(&gov);
    out->drafted_tokens = (int)gov.drafted;
    out->accepted_drafts = (int)gov.accepted;
    if (g_batched) {
        int kept = engine_release(slot);
        cache_prompt(slot, context, kept < context_len ? kept : context_len);
        out->drafted_tokens = slot->req_drafted;
        out->accepted_drafts = slot->req_accepted;
    }
    free(context);
    llama_sampler_free(smpl);
    free(tokens);
    out->elapsed_ms = now_ms() - t0;
    release_slot(slot);
    if (!ok) {
        free(text);
        out->text = NULL;
        return false;
    }
    out->text = text;
    return true;
}

void ollm_result_free(ochat_result *r) {
    free(r->text);
    r->text = NULL;
}

static void ollm_shutdown_locked(void) {
    bool had_model = g_model != NULL;
    dry_cache_clear();
    spec_mtp_free_locked();
    if (g_batched) {
        engine_stop_locked();
        engine_mtp_free();
        g_mtp_draft = 0;
        for (int i = 0; i < g_slot_count; i++) {
            g_slots[i].ctx = NULL;
            pthread_cond_destroy(&g_slots[i].cond);
        }
        if (g_bctx) llama_free(g_bctx);
        g_bctx = NULL;
        g_batched = false;
        g_share_prefix = false;
    }
    for (int i = 0; i < g_slot_count; i++) {
        if (g_slots[i].ctx) llama_free(g_slots[i].ctx);
        free(g_slots[i].cached_tokens);
        free(g_slots[i].cand);
        for (int k = 0; k < OLLM_MTP_BATCH_DRAFT_MAX; k++) free(g_slots[i].vcand[k]);
    }
    free(g_slots);
    g_slots = NULL;
    g_slot_count = 0;
    if (g_model) llama_model_free(g_model);
    g_model = NULL;
    g_vocab = NULL;
    buft_overrides_clear();
    g_model_name[0] = 0;
    g_gpu_requested = false;
    g_on_gpu = false;
    g_batch_size = 0;
    g_ubatch_size = 0;
    g_kv_type_name[0] = 0;
    g_tune_class[0] = 0;
    g_flash_attn = false;
    if (had_model) llama_runtime_release();
}

bool ollm_init(const char *model_path, int n_gpu_layers, int ctx_len, int parallel_contexts) {
    pthread_rwlock_wrlock(&g_model_lifecycle_lock);
    bool ok = ollm_init_locked(model_path, n_gpu_layers, ctx_len, parallel_contexts);
    pthread_rwlock_unlock(&g_model_lifecycle_lock);
    return ok;
}

bool ollm_chat(const ochat_req *req, otoken_cb on_token, void *user, ochat_result *out) {
    pthread_rwlock_rdlock(&g_model_lifecycle_lock);
    bool ok = ollm_chat_locked(req, on_token, user, out);
    pthread_rwlock_unlock(&g_model_lifecycle_lock);
    return ok;
}

void ollm_shutdown(void) {
    pthread_rwlock_wrlock(&g_model_lifecycle_lock);
    ollm_shutdown_locked();
    pthread_rwlock_unlock(&g_model_lifecycle_lock);
}

bool oembed_init(const char *model_path, int n_gpu_layers, int ctx_len, int threads) {
    if (!model_path || !model_path[0] || g_embed_model) return false;
    llama_runtime_acquire();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    g_embed_model = llama_model_load_from_file(model_path, mp);
    if (!g_embed_model) {
        llama_runtime_release();
        return false;
    }
    g_embed_vocab = llama_model_get_vocab(g_embed_model);
    g_embed_ctx_len = ctx_len > 0 ? ctx_len : 512;
    int trained_ctx = llama_model_n_ctx_train(g_embed_model);
    if (trained_ctx > 0 && g_embed_ctx_len > trained_ctx) g_embed_ctx_len = trained_ctx;
    if (g_embed_ctx_len < 8) g_embed_ctx_len = 8;

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)g_embed_ctx_len;
    cp.n_batch = (uint32_t)g_embed_ctx_len;
    cp.n_ubatch = (uint32_t)g_embed_ctx_len;
    cp.n_seq_max = 1;
    cp.n_threads = threads > 0 ? threads : 8;
    cp.n_threads_batch = cp.n_threads;
    cp.embeddings = true;
    /* ModernBERT-base is an MLM trunk with no sentence objective, so mean
     * pooling is the closest match to the existing Python contract. Retrieval
     * finetunes of the same architecture (gte-modernbert, nomic-embed) are
     * trained against the CLS token instead and lose most of their separation
     * under mean pooling, so the pooling mode has to follow the artifact. */
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    const char *pooling = getenv("OMNISERVE_NATIVE_EMBEDDING_POOLING");
    if (pooling && pooling[0]) {
        if (strcasecmp(pooling, "cls") == 0) cp.pooling_type = LLAMA_POOLING_TYPE_CLS;
        else if (strcasecmp(pooling, "last") == 0) cp.pooling_type = LLAMA_POOLING_TYPE_LAST;
        else if (strcasecmp(pooling, "none") == 0) cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }
    cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    if (n_gpu_layers <= 0) {
        cp.offload_kqv = false;
        cp.op_offload = false;
    }
    g_embed_ctx = llama_init_from_model(g_embed_model, cp);
    if (!g_embed_ctx) {
        llama_model_free(g_embed_model);
        g_embed_model = NULL;
        g_embed_vocab = NULL;
        llama_runtime_release();
        return false;
    }

    const char *slash = strrchr(model_path, '/');
    snprintf(g_embed_model_name, sizeof g_embed_model_name, "%s",
             slash ? slash + 1 : model_path);
    char *dot = strstr(g_embed_model_name, ".gguf");
    if (dot) *dot = 0;
    return true;
}

bool oembed_ready(void) { return g_embed_ctx != NULL; }

const char *oembed_model_name(void) {
    return g_embed_model_name[0] ? g_embed_model_name : "none";
}

static size_t utf8_prefix(const char *text, size_t len, size_t max_codepoints) {
    size_t i = 0;
    size_t codepoints = 0;
    while (i < len && codepoints < max_codepoints) {
        unsigned char c = (unsigned char)text[i];
        size_t width = 1;
        if ((c & 0xe0) == 0xc0) width = 2;
        else if ((c & 0xf0) == 0xe0) width = 3;
        else if ((c & 0xf8) == 0xf0) width = 4;
        if (width > len - i) width = 1;
        i += width;
        codepoints++;
    }
    return i;
}

bool oembed_text(const char *text, size_t text_len, int max_dimensions,
                 oembed_result *out) {
    if (!oembed_ready() || !text || !out) return false;
    memset(out, 0, sizeof *out);
    /* The existing Python API slices to 511 code points before tokenization. */
    text_len = utf8_prefix(text, text_len, 511);
    if (text_len > INT32_MAX) return false;

    double t0 = now_ms();
    int token_cap = g_embed_ctx_len;
    llama_token *tokens = malloc((size_t)token_cap * sizeof *tokens);
    if (!tokens) return false;
    int n_tokens = llama_tokenize(g_embed_vocab, text, (int32_t)text_len,
                                  tokens, token_cap, true, true);
    if (n_tokens < 0) {
        token_cap = -n_tokens;
        llama_token *grown = realloc(tokens, (size_t)token_cap * sizeof *grown);
        if (!grown) {
            free(tokens);
            return false;
        }
        tokens = grown;
        n_tokens = llama_tokenize(g_embed_vocab, text, (int32_t)text_len,
                                  tokens, token_cap, true, true);
    }
    if (n_tokens <= 0) {
        free(tokens);
        return false;
    }
    if (n_tokens > g_embed_ctx_len) n_tokens = g_embed_ctx_len;

    pthread_mutex_lock(&g_embed_lock);
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    int rc = llama_encode(g_embed_ctx, batch);
    const float *embedding = rc == 0 ? llama_get_embeddings_seq(g_embed_ctx, 0) : NULL;
    int dimensions = llama_model_n_embd_out(g_embed_model);
    if (dimensions <= 0) dimensions = llama_model_n_embd(g_embed_model);
    if (max_dimensions > 0 && max_dimensions < dimensions) dimensions = max_dimensions;
    float *values = embedding && dimensions > 0
        ? malloc((size_t)dimensions * sizeof *values) : NULL;
    if (values) memcpy(values, embedding, (size_t)dimensions * sizeof *values);
    pthread_mutex_unlock(&g_embed_lock);
    free(tokens);
    if (!values) return false;

    out->values = values;
    out->dimensions = dimensions;
    out->prompt_tokens = n_tokens;
    out->elapsed_ms = now_ms() - t0;
    return true;
}

void oembed_result_free(oembed_result *r) {
    if (!r) return;
    free(r->values);
    memset(r, 0, sizeof *r);
}

void oembed_shutdown(void) {
    bool had_model = g_embed_model != NULL;
    if (g_embed_ctx) llama_free(g_embed_ctx);
    g_embed_ctx = NULL;
    if (g_embed_model) llama_model_free(g_embed_model);
    g_embed_model = NULL;
    g_embed_vocab = NULL;
    g_embed_model_name[0] = 0;
    if (had_model) llama_runtime_release();
}

static struct llama_model *g_judge_model;
static struct llama_context *g_judge_ctx;
static const struct llama_vocab *g_judge_vocab;
static pthread_mutex_t g_judge_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_judge_model_name[256];
static int g_judge_ctx_len;
static int g_judge_batch;
static int g_judge_ngl;
static llama_token g_judge_yes[4];
static int g_judge_yes_n;
static llama_token g_judge_no[4];
static int g_judge_no_n;

#define OLLM_JUDGE_DEFAULT_GGUF \
    "/nvme0n1-disk/models/omniserve-native/shieldgemma-2b-q4_k_m.gguf"
#define OLLM_JUDGE_MAX_SEQ 4
#define OLLM_JUDGE_SEQ_TOKENS 1024

static const char *ojudge_gguf_path(void) {
    const char *p = getenv("OMNISERVE_NATIVE_GUARD_JUDGE_GGUF");
    if (p && p[0]) return p;
    return OLLM_JUDGE_DEFAULT_GGUF;
}

static bool ojudge_enabled_env(void) {
    const char *v = getenv("OMNISERVE_NATIVE_GUARD_JUDGE");
    if (!v || !v[0]) return true;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
}

static unsigned long long ojudge_file_bytes(void) {
    struct stat st;
    if (stat(ojudge_gguf_path(), &st) != 0 || st.st_size <= 0) return 0;
    return (unsigned long long)st.st_size;
}

/* NGL env: integer layer count, or "auto" (default): full GPU offload when
 * the weights plus KV/compute headroom fit in current free VRAM. Returns
 * 999 for full offload, 0 for CPU, -1 when undecided (auto at estimate
 * time, before the LLM has claimed its share). */
static int ojudge_parse_ngl_env(void) {
    const char *v = getenv("OMNISERVE_NATIVE_GUARD_JUDGE_NGL");
    if (!v || !v[0] || strcasecmp(v, "auto") == 0) return -1;
    long n = atol(v);
    if (n >= 999) return 999;
    if (n > 0) return (int)n;
    return 0;
}

unsigned long long ojudge_vram_reserve_bytes(void) {
    if (!ojudge_enabled_env()) return 0;
    int ngl = ojudge_parse_ngl_env();
    if (ngl == 0) return 0;
    unsigned long long file = ojudge_file_bytes();
    if (file == 0) return 0;
    unsigned long long kv =
        256ULL * 1024ULL * 1024ULL + 1024ULL * 1024ULL * 1024ULL / 4;
    if (file > ULLONG_MAX - kv) return ULLONG_MAX;
    return file + kv;
}

static int ojudge_decide_ngl(void) {
    int ngl = ojudge_parse_ngl_env();
    if (ngl >= 0) return ngl;
    unsigned long long need = ojudge_vram_reserve_bytes();
    if (need == 0) return 0;
    double free_gib = -1.0;
    if (!ogpu_memory_gib(&free_gib, NULL) || free_gib <= 0.0) return 0;
    long long keep_mb = 2048;
    const char *keep_env = getenv("OMNISERVE_NATIVE_NGL_AUTO_KEEP_FREE_MB");
    if (keep_env && keep_env[0]) keep_mb = atoll(keep_env);
    if (keep_mb < 0) keep_mb = 0;
    double free_b = free_gib * 1024.0 * 1024.0 * 1024.0;
    double avail = free_b - (double)keep_mb * 1024.0 * 1024.0;
    bool fits = avail > 0.0 && (double)need <= avail;
    fprintf(stderr,
            "guard judge NGL=auto: need=%llu MiB free=%.0f MiB keep_free=%lld MiB -> %s\n",
            need / (1024ULL * 1024ULL), free_b / (1024.0 * 1024.0),
            keep_mb, fits ? "full GPU offload" : "CPU");
    return fits ? 999 : 0;
}

static void ojudge_resolve_verdict_tokens(void) {
    static const char *const yes_v[] = {"Yes", " Yes"};
    static const char *const no_v[] = {"No", " No"};
    g_judge_yes_n = 0;
    g_judge_no_n = 0;
    for (int i = 0; i < 2; i++) {
        llama_token tok = 0;
        if (llama_tokenize(g_judge_vocab, yes_v[i], (int32_t)strlen(yes_v[i]),
                           &tok, 1, false, false) == 1)
            g_judge_yes[g_judge_yes_n++] = tok;
        if (llama_tokenize(g_judge_vocab, no_v[i], (int32_t)strlen(no_v[i]),
                           &tok, 1, false, false) == 1)
            g_judge_no[g_judge_no_n++] = tok;
    }
}

static bool ojudge_try_init(const char *model_path, int ctx_len, int threads,
                               int ngl) {
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    g_judge_model = llama_model_load_from_file(model_path, mp);
    if (!g_judge_model) return false;
    g_judge_vocab = llama_model_get_vocab(g_judge_model);
    g_judge_ctx_len = ctx_len > 0 ? ctx_len : 4096;
    int trained_ctx = llama_model_n_ctx_train(g_judge_model);
    if (trained_ctx > 0 && g_judge_ctx_len > trained_ctx) g_judge_ctx_len = trained_ctx;
    if (g_judge_ctx_len < OLLM_JUDGE_MAX_SEQ * 256) g_judge_ctx_len = OLLM_JUDGE_MAX_SEQ * 256;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)g_judge_ctx_len;
    cp.n_batch = (uint32_t)g_judge_ctx_len;
    cp.n_ubatch = (uint32_t)(g_judge_ctx_len >= 512 ? 512 : g_judge_ctx_len);
    cp.n_seq_max = OLLM_JUDGE_MAX_SEQ;
    cp.n_threads = threads > 0 ? threads : 8;
    cp.n_threads_batch = cp.n_threads;
    cp.offload_kqv = ngl > 0;
    cp.op_offload = ngl > 0;
    g_judge_ngl = ngl;
    g_judge_ctx = llama_init_from_model(g_judge_model, cp);
    if (!g_judge_ctx) {
        llama_model_free(g_judge_model);
        g_judge_model = NULL;
        g_judge_vocab = NULL;
        return false;
    }
    g_judge_batch = (int)cp.n_batch;
    ojudge_resolve_verdict_tokens();
    if (g_judge_yes_n == 0 || g_judge_no_n == 0) {
        llama_free(g_judge_ctx);
        g_judge_ctx = NULL;
        llama_model_free(g_judge_model);
        g_judge_model = NULL;
        g_judge_vocab = NULL;
        return false;
    }
    const char *slash = strrchr(model_path, '/');
    snprintf(g_judge_model_name, sizeof g_judge_model_name, "%s",
             slash ? slash + 1 : model_path);
    char *dot = strstr(g_judge_model_name, ".gguf");
    if (dot) *dot = 0;
    fprintf(stderr, "guard judge live: %s ngl=%d ctx=%d seq=%d\n",
            g_judge_model_name, g_judge_ngl, g_judge_ctx_len, OLLM_JUDGE_MAX_SEQ);
    return true;
}

bool ojudge_init(const char *model_path, int ctx_len, int threads) {
    if (!model_path || !model_path[0] || g_judge_model) return false;
    llama_runtime_acquire();
    int ngl = ojudge_decide_ngl();
    if (ojudge_try_init(model_path, ctx_len, threads, ngl)) return true;
    if (ngl > 0) {
        fprintf(stderr, "guard judge: GPU load failed, retrying CPU-only\n");
        if (ojudge_try_init(model_path, ctx_len, threads, 0)) return true;
    }
    llama_runtime_release();
    return false;
}

bool ojudge_ready(void) { return g_judge_ctx != NULL; }

int ojudge_ngl(void) { return g_judge_ngl; }

const char *ojudge_model_name(void) {
    return g_judge_model_name[0] ? g_judge_model_name : "none";
}

static double ojudge_pyes_from_row(const float *logits) {
    if (!logits) return -1.0;
    int32_t n_vocab = llama_vocab_n_tokens(g_judge_vocab);
    float yes = -INFINITY, no = -INFINITY;
    for (int i = 0; i < g_judge_yes_n; i++)
        if (g_judge_yes[i] >= 0 && g_judge_yes[i] < n_vocab &&
            logits[g_judge_yes[i]] > yes)
            yes = logits[g_judge_yes[i]];
    for (int i = 0; i < g_judge_no_n; i++)
        if (g_judge_no[i] >= 0 && g_judge_no[i] < n_vocab &&
            logits[g_judge_no[i]] > no)
            no = logits[g_judge_no[i]];
    if (!isfinite(yes) || !isfinite(no)) return -1.0;
    return 1.0 / (1.0 + exp((double)no - (double)yes));
}

static int ojudge_tokenize_one(const char *prompt, size_t prompt_len,
                               llama_token *dst, int cap,
                               llama_token *scratch, int scratch_cap) {
    if (prompt_len > INT32_MAX || cap < 1 || !scratch || scratch_cap < 1) return -1;
    int n = llama_tokenize(g_judge_vocab, prompt, (int32_t)prompt_len,
                           scratch, scratch_cap, true, false);
    if (n < 0) {
        if (-n > scratch_cap) return -1;
        n = llama_tokenize(g_judge_vocab, prompt, (int32_t)prompt_len,
                           scratch, -n, true, false);
    }
    if (n <= 0) return -1;
    if (n > cap) n = cap;
    memcpy(dst, scratch, (size_t)n * sizeof *dst);
    return n;
}

bool ojudge_score_multi(const char **prompts, const size_t *lens, int n,
                        double *pyes_out, double *ms_out) {
    if (!ojudge_ready() || !prompts || !lens || n <= 0 || n > OLLM_JUDGE_MAX_SEQ ||
        !pyes_out) return false;
    double t0 = now_ms();
    int per_seq = g_judge_ctx_len / OLLM_JUDGE_MAX_SEQ;
    if (per_seq > OLLM_JUDGE_SEQ_TOKENS) per_seq = OLLM_JUDGE_SEQ_TOKENS;
    if (per_seq < 1) return false;
    llama_token *tokens = malloc((size_t)n * (size_t)per_seq * sizeof *tokens);
    if (!tokens) return false;
    size_t scratch_cap = 8192;
    for (int s = 0; s < n; s++)
        if (lens[s] + 8 > scratch_cap) scratch_cap = lens[s] + 8;
    if (scratch_cap > (size_t)INT32_MAX) {
        free(tokens);
        return false;
    }
    llama_token *scratch = malloc(scratch_cap * sizeof *scratch);
    if (!scratch) {
        free(tokens);
        return false;
    }
    int counts[OLLM_JUDGE_MAX_SEQ] = {0, 0, 0, 0};
    int total = 0;
    for (int s = 0; s < n; s++) {
        if (!prompts[s] || lens[s] == 0) {
            free(scratch);
            free(tokens);
            return false;
        }
        int got = ojudge_tokenize_one(prompts[s], lens[s],
                                      tokens + (size_t)s * (size_t)per_seq, per_seq,
                                      scratch, (int)scratch_cap);
        if (got <= 0) {
            free(scratch);
            free(tokens);
            return false;
        }
        counts[s] = got;
        total += got;
    }
    free(scratch);
    if (total > g_judge_batch) {
        free(tokens);
        return false;
    }
    struct llama_batch batch = llama_batch_init(total, 0, n);
    if (!batch.token) {
        llama_batch_free(batch);
        free(tokens);
        return false;
    }
    llama_seq_id seq_ids[OLLM_JUDGE_MAX_SEQ];
    for (int s = 0; s < n; s++) seq_ids[s] = s;
    int pos = 0;
    for (int s = 0; s < n; s++) {
        for (int i = 0; i < counts[s]; i++) {
            int k = pos++;
            batch.token[k] = tokens[(size_t)s * (size_t)per_seq + (size_t)i];
            batch.pos[k] = i;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = seq_ids[s];
            batch.logits[k] = (i == counts[s] - 1) ? 1 : 0;
        }
    }
    batch.n_tokens = total;
    pthread_mutex_lock(&g_judge_lock);
    llama_memory_t memory = llama_get_memory(g_judge_ctx);
    llama_memory_clear(memory, false);
    int rc = llama_decode(g_judge_ctx, batch);
    double pyes[OLLM_JUDGE_MAX_SEQ] = {0.0, 0.0, 0.0, 0.0};
    if (rc == 0) {
        int last = 0;
        for (int s = 0; s < n; s++) {
            last += counts[s] - 1;
            const float *row = llama_get_logits_ith(g_judge_ctx, last);
            last += 1;
            pyes[s] = ojudge_pyes_from_row(row);
            if (pyes[s] < 0.0) {
                rc = -1;
                break;
            }
        }
    }
    pthread_mutex_unlock(&g_judge_lock);
    llama_batch_free(batch);
    free(tokens);
    if (rc != 0) return false;
    for (int s = 0; s < n; s++) pyes_out[s] = pyes[s];
    if (ms_out) *ms_out = now_ms() - t0;
    return true;
}

bool ojudge_score(const char *prompt, size_t prompt_len, double *p_yes_out,
                  double *ms_out) {
    double p = 0.0;
    if (!ojudge_score_multi(&prompt, &prompt_len, 1, &p, ms_out)) return false;
    if (p_yes_out) *p_yes_out = p;
    return true;
}

void ojudge_shutdown(void) {
    bool had_model = g_judge_model != NULL;
    if (g_judge_ctx) llama_free(g_judge_ctx);
    g_judge_ctx = NULL;
    if (g_judge_model) llama_model_free(g_judge_model);
    g_judge_model = NULL;
    g_judge_vocab = NULL;
    g_judge_model_name[0] = 0;
    g_judge_ngl = 0;
    g_judge_yes_n = 0;
    g_judge_no_n = 0;
    if (had_model) llama_runtime_release();
}

#else

unsigned long long ollm_cpu_offloaded_bytes(const char *model_path, int ctx_len, int contexts) {
    (void)model_path; (void)ctx_len; (void)contexts;
    return 0;
}

void ollm_set_load_overrides(const char *tensor_override, const char *moe_cpu_experts,
                             const char *spec_mtp_gguf) {
    (void)tensor_override; (void)moe_cpu_experts; (void)spec_mtp_gguf;
}

bool ollm_init(const char *model_path, int n_gpu_layers, int ctx_len, int parallel_contexts) {
    (void)model_path; (void)n_gpu_layers; (void)ctx_len; (void)parallel_contexts;
    return false;
}
bool ollm_ready(void) { return false; }
void ollm_placement_snapshot(ollm_placement *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    snprintf(out->device, sizeof out->device, "none");
    snprintf(out->kv_type, sizeof out->kv_type, "none");
    snprintf(out->tune_class, sizeof out->tune_class, "none");
    snprintf(out->spec_source, sizeof out->spec_source, "off");
}
int ollm_suggested_contexts(void) { return 1; }
int ollm_engine_live(void) { return 0; }
const char *ollm_model_name(void) { return "none"; }
bool ollm_chat(const ochat_req *req, otoken_cb on_token, void *user, ochat_result *out) {
    (void)req; (void)on_token; (void)user; (void)out;
    return false;
}
void ollm_result_free(ochat_result *r) { (void)r; }
void ollm_shutdown(void) {}
bool oembed_init(const char *model_path, int n_gpu_layers, int ctx_len, int threads) {
    (void)model_path; (void)n_gpu_layers; (void)ctx_len; (void)threads;
    return false;
}
bool oembed_ready(void) { return false; }
const char *oembed_model_name(void) { return "none"; }
bool oembed_text(const char *text, size_t text_len, int max_dimensions,
                 oembed_result *out) {
    (void)text; (void)text_len; (void)max_dimensions; (void)out;
    return false;
}
void oembed_result_free(oembed_result *r) { (void)r; }
void oembed_shutdown(void) {}
bool ojudge_init(const char *model_path, int ctx_len, int threads) {
    (void)model_path; (void)ctx_len; (void)threads;
    return false;
}
bool ojudge_ready(void) { return false; }
const char *ojudge_model_name(void) { return "none"; }
int ojudge_ngl(void) { return 0; }
bool ojudge_score(const char *prompt, size_t prompt_len, double *p_yes_out,
                  double *ms_out) {
    (void)prompt; (void)prompt_len; (void)p_yes_out; (void)ms_out;
    return false;
}
bool ojudge_score_multi(const char **prompts, const size_t *lens, int n,
                        double *pyes_out, double *ms_out) {
    (void)prompts; (void)lens; (void)n; (void)pyes_out; (void)ms_out;
    return false;
}
unsigned long long ojudge_vram_reserve_bytes(void) { return 0; }
void ojudge_shutdown(void) {}

#endif

#ifndef USE_SD
bool osd_init(const char *model_path) { (void)model_path; return false; }
bool osd_ready(void) { return false; }
bool osd_prepare_image(oimg_req *req) { return !req->image_base64; }
bool osd_reference_edit_ready(void) { return false; }
const char *osd_model_name(void) { return "none"; }
bool osd_generate(const oimg_req *req, oimg_result *out) { (void)req; (void)out; return false; }
bool osd_try_cached_result(const oimg_req *req, oimg_result *out) { (void)req; (void)out; return false; }
void osd_result_free(oimg_result *r) { (void)r; }
bool osd_resident(void) { return false; }
int osd_set_resident(bool resident, bool wait) { (void)resident; (void)wait; return 0; }
bool osd_residency_target(void) { return false; }
size_t osd_residency_json(char *out, size_t cap) { if (cap) out[0] = 0; return 0; }
void osd_residency_settled(void) {}
#endif
