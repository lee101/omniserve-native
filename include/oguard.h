#ifndef OGUARD_H
#define OGUARD_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    OGUARD_NONE = 0,
    OGUARD_MINOR_SEXUAL,
    OGUARD_NONCON_SEXUAL,
    OGUARD_REALPERSON_SEXUAL,
} oguard_category;

typedef struct {
    oguard_category category;
    double score;
    char matched_rule[64];
} oguard_verdict;

void oguard_init(void);
bool oguard_enabled(void);
int oguard_names_loaded(void);
const char *oguard_category_name(oguard_category category);

oguard_verdict oguard_classify(const char *system_text, const char *user_text,
                               const char *recent_context);

typedef bool (*oguard_embed_ready_fn)(void);
typedef bool (*oguard_embed_fn)(const char *text, size_t len, float **values_out, int *dim_out);
typedef void (*oguard_embed_free_fn)(float *values);
void oguard_set_embed_backend(oguard_embed_ready_fn ready, oguard_embed_fn embed,
                              oguard_embed_free_fn efree);
void oguard_embed_warmup(void);
bool oguard_embed_ready(void);

typedef bool (*oguard_judge_ready_fn)(void);
typedef bool (*oguard_judge_fn)(const char *prompt, size_t len, double *p_yes_out,
                                double *ms_out);
typedef bool (*oguard_judge_multi_fn)(const char **prompts, const size_t *lens,
                                      int n, double *pyes_out, double *ms_out);
void oguard_set_judge_backend(oguard_judge_ready_fn ready, oguard_judge_fn score);
void oguard_set_judge_multi_backend(oguard_judge_multi_fn multi);
void oguard_judge_warmup(void);
bool oguard_judge_ready(void);

typedef struct {
    oguard_verdict lexical;
    oguard_verdict embed;
    oguard_verdict final;
    double embed_ms;
    double emb_prohib;
    double emb_benign;
    oguard_verdict judge;
    double judge_pyes;
    double judge_single[3];
    double judge_ms;
    bool judge_invoked;
    bool judge_cached;
    bool judge_ok;
    bool judge_veto;
    double emb_cat[3];
    bool sexual_ctx;
    double gate_pyes;
    double gate_ms;
    bool gate_open;
} oguard_full;

oguard_full oguard_classify_full(const char *system_text, const char *user_text,
                                 const char *recent_context);
oguard_full oguard_classify_diagnose(const char *system_text, const char *user_text,
                                     const char *recent_context);
const char *oguard_stage_name(const oguard_full *full);

typedef struct {
    bool ready;
    unsigned long long calls;
    unsigned long long cache_hits;
    unsigned long long gated_out;
    double total_ms;
    double last_ms;
} oguard_judge_stats;

void oguard_judge_stats_snapshot(oguard_judge_stats *out);
void oguard_classify_note(double ms);

oguard_verdict oguard_embed_classify(const char *system_text, const char *user_text,
                                     const char *recent_context, double *prohib_out,
                                     double *benign_out, double *ms_out);

void oguard_record_block(oguard_category category);
unsigned long long oguard_blocks(oguard_category category);
void oguard_status_json(char *buf, size_t cap);
void oguard_metrics_append(char *buf, size_t cap, size_t *len);
void oguard_text_hash(const char *system_text, const char *user_text,
                      const char *recent_context, char out_hex[65]);

#endif
