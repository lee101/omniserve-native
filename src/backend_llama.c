#define _GNU_SOURCE
#include "obackend.h"
#include "otext.h"
#include "otune.h"

#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
typedef struct {
    struct llama_context *ctx;
    bool busy;
    llama_token *cached_tokens;
    int cached_count;
    int cached_cap;
} ollm_slot;
static ollm_slot *g_slots;
static int g_slot_count;
static pthread_mutex_t g_slot_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_slot_cond = PTHREAD_COND_INITIALIZER;
static char g_model_name[256];
static int g_ctx_len;
static int g_batch_size;
static int g_ubatch_size;

static struct llama_model *g_embed_model;
static const struct llama_vocab *g_embed_vocab;
static struct llama_context *g_embed_ctx;
static pthread_mutex_t g_embed_lock = PTHREAD_MUTEX_INITIALIZER;
static char g_embed_model_name[256];
static int g_embed_ctx_len;

static pthread_mutex_t g_runtime_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_runtime_refs;

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

int ollm_suggested_contexts(void) {
    otune_profile profile;
    if (!g_device_desc[0]) probe_gpu_device();
    otune_profile_for(g_gpu_device_present ? g_device_desc : "cpu", &profile);
    return profile.parallel_contexts;
}

bool ollm_init(const char *model_path, int n_gpu_layers, int ctx_len, int parallel_contexts) {
    llama_runtime_acquire();
    probe_gpu_device();
    g_gpu_requested = n_gpu_layers > 0;
    g_on_gpu = g_gpu_requested && g_gpu_device_present;
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    g_model = llama_model_load_from_file(model_path, mp);
    if (!g_model) {
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
    g_batch_size = batch_env ? atoi(batch_env) : profile.n_batch;
    g_ubatch_size = ubatch_env ? atoi(ubatch_env) : profile.n_ubatch;
    if (g_batch_size < 1) g_batch_size = 1;
    if (g_batch_size > g_ctx_len) g_batch_size = g_ctx_len;
    if (g_ubatch_size < 1) g_ubatch_size = 1;
    if (g_ubatch_size > g_batch_size) g_ubatch_size = g_batch_size;
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

    for (int i = first_message; i < message_count; i++) {
        const char *source_role = messages[i].role;
        if (strcmp(source_role, "tool") == 0) continue;
        const char *role = "user";
        if (strcmp(source_role, "assistant") == 0) role = "model";
        else if (strcmp(source_role, "system") == 0 ||
                 strcmp(source_role, "developer") == 0) role = "system";
        APPEND_LITERAL("<|turn>");
        APPEND_LITERAL(role);
        APPEND_LITERAL("\n");
        if (!prompt_append_trimmed(&prompt, &length, &capacity,
                                   messages[i].content)) goto fail;
        APPEND_LITERAL("<turn|>\n");
    }
    APPEND_LITERAL("<|turn>model\n");
    if (length > INT32_MAX) goto fail;
    *formatted_len = (int32_t)length;
#undef APPEND_LITERAL
    return prompt;

fail:
#undef APPEND_LITERAL
    free(prompt);
    return NULL;
}

bool ollm_chat(const ochat_req *req, otoken_cb on_token, void *user, ochat_result *out) {
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
    slot->cached_count = cached_tokens;
    out->cached_prompt_tokens = cached_tokens;

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler *smpl = llama_sampler_chain_init(sparams);
    probability_capture sampled = { .probability = 1.0f };
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
        decoded += chunk;
    }
    if (ok) cache_prompt(slot, tokens, n_tokens);
    else slot->cached_count = 0;

    llama_token previous = LLAMA_TOKEN_NULL;
    float cumulative_probability = 1.0f;
    for (int i = 0; i < max_new; i++) {
        if (!ok) break;
        if (previous != LLAMA_TOKEN_NULL) {
            struct llama_batch token_batch = llama_batch_get_one(&previous, 1);
            if (llama_decode(ctx, token_batch) != 0) {
                ok = i > 0;
                out->finish_reason = "error";
                break;
            }
        }
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(g_vocab, tok)) break;
        cumulative_probability *= sampled.probability;
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
                size_t held = should_stop ? 0 : possible_stop_prefix(req, text, text_len);
                size_t safe_len = text_len - held;
                if (safe_len > streamed_len &&
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
        previous = tok;
    }

    if (on_token && !cancelled && text_len > streamed_len &&
        !on_token(text + streamed_len, text_len - streamed_len, user)) {
        out->finish_reason = "cancelled";
    }

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

void ollm_shutdown(void) {
    bool had_model = g_model != NULL;
    for (int i = 0; i < g_slot_count; i++) {
        if (g_slots[i].ctx) llama_free(g_slots[i].ctx);
        free(g_slots[i].cached_tokens);
    }
    free(g_slots);
    g_slots = NULL;
    g_slot_count = 0;
    if (g_model) llama_model_free(g_model);
    g_model = NULL;
    if (had_model) llama_runtime_release();
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

#else

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

#endif

#ifndef USE_SD
bool osd_init(const char *model_path) { (void)model_path; return false; }
bool osd_ready(void) { return false; }
const char *osd_model_name(void) { return "none"; }
bool osd_generate(const oimg_req *req, oimg_result *out) { (void)req; (void)out; return false; }
void osd_result_free(oimg_result *r) { (void)r; }
#endif
