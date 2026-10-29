#ifndef OLORA_ROUTE_H
#define OLORA_ROUTE_H

#include <stdbool.h>
#include <stddef.h>

#define OLORA_ROUTE_MAX_LORAS 4

typedef struct {
    char name[64];
    char prefix[128];
    char suffix[256];   /* appended to the prompt (e.g. anatomy hints) */
    char negative[512]; /* used only when the request has none; ignored at CFG 1 */
    size_t lora_count;
    char ids[OLORA_ROUTE_MAX_LORAS][257]; /* cache id, or an absolute path from operator config */
    float scales[OLORA_ROUTE_MAX_LORAS];
} olora_route_pick;

bool olora_route_enabled(void);
bool olora_route_select_json(const char *json, size_t json_len, const char *prompt,
                             olora_route_pick *pick);
bool olora_route_select(const char *prompt, olora_route_pick *pick);

#endif
