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
    char prompt[256];   /* edit tasks: replaces the request prompt (the LoRA is trained on one caption) */
    int steps;          /* edit tasks: pinned recipe, 0 = leave */
    float guidance;     /* edit tasks: pinned CFG, 0 = leave */
    size_t lora_count;
    char ids[OLORA_ROUTE_MAX_LORAS][257]; /* cache id, or an absolute path from operator config */
    float scales[OLORA_ROUTE_MAX_LORAS];
} olora_route_pick;

bool olora_route_enabled(void);
bool olora_route_select_json(const char *json, size_t json_len, const char *prompt,
                             olora_route_pick *pick);
bool olora_route_select(const char *prompt, olora_route_pick *pick);
/* Reference-image edit tasks ("edit_routes": task name -> LoRA + pinned prompt/steps/CFG). */
bool olora_route_select_task_json(const char *json, size_t json_len, const char *task, olora_route_pick *pick);
bool olora_route_select_task(const char *task, olora_route_pick *pick);

#endif
