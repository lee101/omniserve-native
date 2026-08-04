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
bool oimage_request_parse(const char *json, size_t json_len, oimage_request *request,
                          char *error, size_t error_cap);
bool oimage_openai_response(const oimg_result *result, const char *model, long long seed,
                            char **json_out, size_t *json_len_out);

#endif
