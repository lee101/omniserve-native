/* cbed - ultra-fast static int8 embeddings + kNN search in C.
 *
 * A C port of the gobed/zbed/pybed family: a token-embedding lookup table
 * (static-retrieval-mrl-en-v1 int8-quantized, 512-d) with WordPiece
 * tokenization. "Inference" is tokens -> gather rows -> mean-pool -> vector,
 * no transformer forward. Search runs integer dot products over int8 vectors.
 *
 * Intended use here: LoRA kNN search (embed LoRA descriptions once, embed a
 * prompt, return the nearest adapters) as a semantic alternative/complement to
 * BM25.
 */
#ifndef CBED_H
#define CBED_H

#include <stddef.h>
#include <stdint.h>

#define CBED_MAX_TOKENS 256

typedef struct cbed_model cbed_model;

/* Load model dir containing modelint8_512dim.safetensors + tokenizer.json.
 * Returns NULL on failure. */
cbed_model *cbed_load(const char *model_dir);
void cbed_free(cbed_model *m);

int cbed_dim(const cbed_model *m);       /* embedding dimension (512) */
int cbed_vocab_size(const cbed_model *m);

/* Embed text -> float32 vector of length cbed_dim(). out must have room. */
void cbed_embed(const cbed_model *m, const char *text, float *out);

/* Embed text -> int8 vector (len cbed_dim()) + its L2 norm, for integer search.
 * scale_out may be NULL. */
void cbed_embed_quantized(const cbed_model *m, const char *text,
                          int8_t *out, float *norm_out, float *scale_out);

/* An index of int8 embeddings for kNN. Owns its vectors. */
typedef struct cbed_index cbed_index;

cbed_index *cbed_index_new(int dim, size_t capacity_hint);
void cbed_index_free(cbed_index *ix);

/* Add one item by embedding `text` with `m`. `id` is copied. Returns item
 * count, or -1 on error. */
int cbed_index_add(cbed_index *ix, const cbed_model *m, const char *id,
                   const char *text);

typedef struct {
    const char *id;   /* borrowed from the index */
    float score;      /* cosine similarity in [-1, 1] */
} cbed_hit;

/* kNN over the index for `query`. Writes up to `k` hits (sorted desc) into
 * `hits`. Returns the number written. */
int cbed_search(const cbed_index *ix, const cbed_model *m, const char *query,
                int k, cbed_hit *hits);

/* Lower-level: search with a pre-quantized query vector. */
int cbed_search_vec(const cbed_index *ix, const int8_t *q, float q_norm,
                    int k, cbed_hit *hits);

/* int8 dot product (auto-vectorizes under -O3 -march=native). */
int32_t cbed_dot_i8(const int8_t *a, const int8_t *b, int n);

#endif /* CBED_H */
