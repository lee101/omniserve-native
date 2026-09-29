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
    /* Draft source: "mtp" when an MTP head is loaded, "prompt-lookup" when
     * drafting from the context, "off" otherwise. */
    char spec_source[16];
    unsigned long long spec_rounds;
    unsigned long long spec_drafted;
    unsigned long long spec_accepted;
    unsigned long long spec_saved_calls;
    int gpu_expert_layers;
    int cpu_expert_layers;
    char tensor_override[512];
    unsigned long long est_gpu_bytes;
} ollm_placement;

/* Contexts this device class can keep resident beside the weights. Advisory:
 * the caller still clamps to configured slots. */
int ollm_suggested_contexts(void);

typedef struct {
    char pattern[256];
    bool cpu;
} ollm_tensor_override;

typedef enum {
    OLLM_MOE_OFF = 0,
    OLLM_MOE_N,
    OLLM_MOE_ALL,
    OLLM_MOE_AUTO
} ollm_moe_mode;

int ollm_parse_tensor_overrides(const char *spec, ollm_tensor_override *out, int cap);
int ollm_parse_moe_cpu_experts(const char *spec, ollm_moe_mode *mode_out, int *n_out);

enum { OLLM_SPEC_MTP_DRAFT_MAX = 2 };

typedef struct {
    int draft_max; /* tokens per round, clamped to [0, 2] */
    float p_min;   /* minimum draft-token probability, clamped to [0, 1] */
} ollm_spec_mtp_config;

void ollm_spec_mtp_default(ollm_spec_mtp_config *cfg);
int ollm_spec_mtp_parse(ollm_spec_mtp_config *cfg, const char *draft_env,
                        const char *pmin_env);
bool ollm_tensor_is_expert(const char *name);
int ollm_tensor_block_index(const char *name);
int ollm_gpu_expert_layers(const unsigned long long *expert_bytes, int n_layers,
                           unsigned long long nonexp_bytes, unsigned long long kv_bytes,
                           unsigned long long compute_reserve, unsigned long long budget);
double ollm_kv_type_size(const char *kv_type);
void ollm_strip_channels(const char *src, char *dst, size_t cap);
void ollm_strip_channels_final(char *text, size_t cap);
bool ollm_thinking_default(void);
bool ollm_regex_balanced(const char *pattern);
int ollm_moe_cpu_pattern(int first_layer, int n_layers, char *out, size_t cap);
unsigned long long ollm_cpu_offloaded_bytes(const char *model_path, int ctx_len, int contexts);
bool ollm_moe_last_active(void);
unsigned long long ollm_moe_last_free_bytes(void);
unsigned long long ollm_moe_last_est_bytes(void);

void ollm_set_load_overrides(const char *tensor_override, const char *moe_cpu_experts,
                             const char *spec_mtp_gguf);
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

/* Safety-judge scoring (ShieldGemma-style Yes/No classifier). One prefill,
 * verdict read as P(Yes) over the first decoded token's logits. Own
 * model/context (4 parallel sequences for batched verdicts); never touches
 * the LLM pool or the MTP head. GPU placement follows
 * OMNISERVE_NATIVE_GUARD_JUDGE_NGL (default auto). */
bool ojudge_init(const char *model_path, int ctx_len, int threads);
bool ojudge_ready(void);
const char *ojudge_model_name(void);
int ojudge_ngl(void);
bool ojudge_score(const char *prompt, size_t prompt_len, double *p_yes_out,
                  double *ms_out);
/* Score n prompts (n<=4) in ONE forward pass as parallel sequences.
 * pyes_out must hold n doubles. Bit-identical verdicts to n serial calls. */
bool ojudge_score_multi(const char **prompts, const size_t *lens, int n,
                        double *pyes_out, double *ms_out);
/* VRAM bytes the judge will claim when GPU-placed. The MoE auto-placement
 * estimator subtracts this from its budget before choosing gpu layers. */
unsigned long long ojudge_vram_reserve_bytes(void);
void ojudge_shutdown(void);

typedef struct {
    const char *path;
    float scale;
} oimg_lora;

typedef struct {
    const char *prompt;
    const char *negative_prompt;
    int width;
    int height;
    int steps;
    int batch_count;
    float guidance_scale;
    int64_t seed;
    bool teleport;
    int teleport_start_step;
    const oimg_lora *loras;
    size_t lora_count;
    /* Owned by oimage_request; decoded RGB is prepared before GPU admission. */
    char *image_base64;
    unsigned char *image_pixels;
    int image_width;
    int image_height;
    float strength;
    bool cache;
} oimg_req;

typedef struct {
    unsigned char *png;
    size_t png_len;
    unsigned char **images;
    size_t *image_lens;
    size_t image_count;
    const char *format;
    bool images_malloc_owned;
    double elapsed_ms;
    bool teleport_requested;
    bool cache_requested;
    bool cache_hit;
    float denoiser_cache_threshold; /* zero disables approximate EasyCache */
    const char *denoiser_cache_mode; /* static string: easycache|taylorseer|spectrum|cache-dit|dbcache|ucache */
    bool teleport_used;
    bool teleport_cache_hit;
    bool teleport_result_cache_hit;
    int teleport_capture_step;
    int teleport_resume_step;
} oimg_result;

bool osd_init(const char *model_path);
bool osd_ready(void);
const char *osd_model_name(void);
bool osd_generate(const oimg_req *req, oimg_result *out);
bool osd_try_cached_result(const oimg_req *req, oimg_result *out);
bool osd_prepare_image(oimg_req *req);
bool osd_reference_edit_ready(void);
void osd_result_free(oimg_result *r);

#ifdef __cplusplus
}
#endif

#endif
