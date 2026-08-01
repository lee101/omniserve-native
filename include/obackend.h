#ifndef OBACKEND_H
#define OBACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *role;
    const char *content;
} ochat_msg;

typedef struct {
    const ochat_msg *messages;
    int message_count;
    /* When set, bypass the model chat template and continue these bytes
     * directly. Legacy autocomplete and OpenAI text completions require this
     * boundary-preserving behavior. */
    const char *raw_prompt;
    /* Original completion prefix used only for boundary repair. Unlike chat
     * messages, this text is concatenated with the generated response. */
    const char *completion_prefix;
    int max_tokens;
    float temperature;
    float top_p;
    float min_p;
    float min_probability;
    int top_k;
    float repetition_penalty;
    int64_t seed;
    bool enable_thinking;
    const char *const *stop_sequences;
    int stop_count;
    int max_sentences;
} ochat_req;

typedef bool (*otoken_cb)(const char *piece, size_t len, void *user);

typedef struct {
    char *text;
    int prompt_tokens;
    int cached_prompt_tokens;
    int completion_tokens;
    /* Speculation, per request: how many tokens were guessed and how many the
     * sampler independently agreed with. Zero for both means speculation never
     * ran, which is a different situation from guessing and missing. */
    int drafted_tokens;
    int accepted_drafts;
    double elapsed_ms;
    const char *finish_reason;
} ochat_result;

/* Where the embedded weights actually ended up. A CUDA init failure makes
 * llama.cpp fall back to CPU silently, which serves requests at a small
 * fraction of GPU throughput while still reporting ready — so placement is
 * reported separately from readiness. */
typedef struct {
    bool gpu_device_present; /* a GPU compute backend is registered */
    bool gpu_requested;      /* the operator asked for offloaded layers */
    bool on_gpu;             /* offload actually happened */
    char device[96];         /* backend device description */
    char kv_type[16];        /* KV cache element type in use */
    char tune_class[16];     /* device class the batch geometry came from */
    int n_batch;
    int n_ubatch;
    bool flash_attn;
    /* Speculation, since the run: a draft length of 0 means it is off, which
     * looks the same from latency as running and never landing. */
    int spec_draft_max;
    unsigned long long spec_rounds;
    unsigned long long spec_drafted;
    unsigned long long spec_accepted;
    unsigned long long spec_saved_calls;
} ollm_placement;

/* Contexts this device class can keep resident beside the weights. Advisory:
 * the caller still clamps to configured slots. */
int ollm_suggested_contexts(void);

bool ollm_init(const char *model_path, int n_gpu_layers, int ctx_len, int parallel_contexts);
bool ollm_ready(void);
void ollm_placement_snapshot(ollm_placement *out);
const char *ollm_model_name(void);
bool ollm_chat(const ochat_req *req, otoken_cb on_token, void *user, ochat_result *out);
void ollm_result_free(ochat_result *r);
void ollm_shutdown(void);

typedef struct {
    float *values;
    int dimensions;
    int prompt_tokens;
    double elapsed_ms;
} oembed_result;

/* Encoder-only GGUF embeddings. The implementation uses mean pooling to
 * preserve text-generator.io's ModernBERT feature-extraction contract. */
bool oembed_init(const char *model_path, int n_gpu_layers, int ctx_len, int threads);
bool oembed_ready(void);
const char *oembed_model_name(void);
bool oembed_text(const char *text, size_t text_len, int max_dimensions,
                 oembed_result *out);
void oembed_result_free(oembed_result *r);
void oembed_shutdown(void);

typedef struct {
    const char *prompt;
    const char *negative_prompt;
    int width;
    int height;
    int steps;
    long seed;
} oimg_req;

typedef struct {
    unsigned char *png;
    size_t png_len;
    double elapsed_ms;
} oimg_result;

bool osd_init(const char *model_path);
bool osd_ready(void);
const char *osd_model_name(void);
bool osd_generate(const oimg_req *req, oimg_result *out);
void osd_result_free(oimg_result *r);

#ifdef __cplusplus
}
#endif

#endif
