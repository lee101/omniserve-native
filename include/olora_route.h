#ifndef OLORA_ROUTE_H
#define OLORA_ROUTE_H

#include <stdbool.h>
#include <stddef.h>

#define OLORA_ROUTE_MAX_LORAS 4

typedef struct {
    char name[64];
    char prefix[128];
    size_t lora_count;
    char ids[OLORA_ROUTE_MAX_LORAS][129];
    float scales[OLORA_ROUTE_MAX_LORAS];
} olora_route_pick;

bool olora_route_enabled(void);
bool olora_route_select_json(const char *json, size_t json_len, const char *prompt,
                             olora_route_pick *pick);
bool olora_route_select(const char *prompt, olora_route_pick *pick);

#endif
