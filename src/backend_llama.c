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

enum { OLLM_VERIFY_BATCH_CAP = 17 };
enum { OLLM_MTP_DRAFT_CAP = 2 };
typedef struct {
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

static const char *spec_source_name(void) {
    if (spec_mtp_active()) return "mtp";
    if (g_spec_draft_max > 0) return "prompt-lookup";
    return "off";
}

static int spec_effective_max(void) {
    if (spec_mtp_active()) return g_spec_mtp_cfg.draft_max;
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
typedef struct {
    float probability;
} probability_capture;

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
    for (int i = 0; i < g_slot_count; i++) {
        g_slots[i].ctx = llama_init_from_model(g_model, cp);
        if (!g_slots[i].ctx) {
            for (int j = 0; j < i; j++) llama_free(g_slots[j].ctx);
            free(g_slots);
            g_slots = NULL;
            g_slot_count = 0;
            llama_model_free(g_model);
            g_model = NULL;
            llama_runtime_release();
            return false;
        }
    }

    spec_mtp_init_locked(n_gpu_layers, &cp, kv_type);

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
}

static int common_prefix(const ollm_slot *slot, const llama_token *tokens, int count) {
    int common = slot->cached_count < count ? slot->cached_count : count;
    int i = 0;
    while (i < common && slot->cached_tokens[i] == tokens[i]) i++;
    /* Re-evaluate the final shared token so logits correspond to this prompt,
     * not to the previous request's generated suffix. */
    return i > 0 ? i - 1 : 0;
}

static ollm_slot *acquire_slot(const llama_token *tokens, int count, int *cached) {
    pthread_mutex_lock(&g_slot_lock);
    for (;;) {
        ollm_slot *best = NULL;
        int best_prefix = -1;
        for (int i = 0; i < g_slot_count; i++) {
            if (!g_slots[i].busy) {
                int prefix = common_prefix(&g_slots[i], tokens, count);
                if (prefix > best_prefix) {
                    best = &g_slots[i];
                    best_prefix = prefix;
                }
            }
        }
        if (best) {
            best->busy = true;
            *cached = best_prefix;
            pthread_mutex_unlock(&g_slot_lock);
            return best;
        }
        pthread_cond_wait(&g_slot_cond, &g_slot_lock);
    }
}

static void cache_prompt(ollm_slot *slot, const llama_token *tokens, int count) {
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
        out_tokens[0] = llama_sampler_sample(smpl, ctx, -1);
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
        out_tokens[0] = llama_sampler_sample(smpl, ctx, -1);
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
        llama_token tok = llama_sampler_sample(smpl, ctx, i);
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

    if (cached_tokens > 0) {
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
    slot->cached_count = cached_tokens;
    out->cached_prompt_tokens = cached_tokens;

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    probability_capture sampled = { .probability = 1.0f };
    if (req->min_probability > 0.0f)
        llama_sampler_chain_add(smpl, llama_sampler_init(&probability_capture_i, &sampled));
    float repetition = req->repetition_penalty > 0 ? req->repetition_penalty : 1.0f;
    bool penalized = repetition != 1.0f;
    if (penalized) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(-1, repetition, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(req->top_k > 0 ? req->top_k : 40));
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
    /* Only the penalty sampler keeps history, so replaying the whole prompt
     * through the chain is wasted work on the common unpenalized path. */
    if (penalized) {
        for (int i = 0; i < n_tokens; i++) llama_sampler_accept(smpl, tokens[i]);
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

    int decoded = cached_tokens;
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
    if (ok) cache_prompt(slot, tokens, n_tokens);
    else slot->cached_count = 0;

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

    for (int i = 0; i < max_new; i++) {
        if (!ok) break;
        if (round_pos >= round_count) {
            round_count = decode_round(slot, smpl, &sampled, previous, &n_past,
                                       context, context_len,
                                       spec_effective_max() > 0 ? &gov : NULL,
                                       round_tokens, round_probs,
                                       (int)(sizeof round_tokens / sizeof *round_tokens));
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
    spec_mtp_free_locked();
    for (int i = 0; i < g_slot_count; i++) {
        if (g_slots[i].ctx) llama_free(g_slots[i].ctx);
        free(g_slots[i].cached_tokens);
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
#endif
