#ifndef OIMAGE_H
#define OIMAGE_H

#include <stdbool.h>
#include <stddef.h>

#include "obackend.h"

typedef struct {
    oimg_req generation;
    char *prompt;
    char *negative_prompt;
    oimg_lora *loras;
    bool direct_lora_paths;
    int count;
} oimage_request;

void oimage_request_init(oimage_request *request);
void oimage_request_free(oimage_request *request);
int oimage_gpu_headroom_mb(void);
bool oimage_gpu_headroom_ok(double free_gib, char *error, size_t error_cap);
bool oimage_request_parse(const char *json, size_t json_len, oimage_request *request,
                          char *error, size_t error_cap);
bool oimage_openai_response(const oimg_result *result, const char *model, long long seed,
                            char **json_out, size_t *json_len_out);
/* Converts an OpenAI multipart /v1/images/edits body into the JSON reference
 * edit contract ({"prompt":..,"image_base64":..}). Masks are dropped: the
 * reference editor is global. Copies a "model" field into `model`. */
char *oimage_edit_multipart_to_json(const char *content_type, size_t content_type_len,
                                    const char *body, size_t body_len, size_t *json_len,
                                    char *model, size_t model_cap,
                                    char *error, size_t error_cap);

#endif
