#include "admission.h"

#include <stdio.h>
#include <string.h>

Admission admit_model(const GpuSnapshot *gpu, const Model *model,
                      unsigned reserve_mib) {
    Admission result = {0};
    result.required_mib = model->weights_mib + model->runtime_mib;
    if (!gpu->available) {
        snprintf(result.reason, sizeof(result.reason),
                 "GPU telemetry is unavailable; model admission is disabled");
        return result;
    }
    if (gpu->total_mib < gpu->used_mib || gpu->total_mib - gpu->used_mib < reserve_mib) {
        snprintf(result.reason, sizeof(result.reason),
                 "GPU reserve of %u MiB is not available", reserve_mib);
        return result;
    }
    result.available_mib = gpu->total_mib - gpu->used_mib - reserve_mib;
    if (result.required_mib > result.available_mib) {
        snprintf(result.reason, sizeof(result.reason),
                 "%.80s needs %u MiB; only %u MiB is available after reserve",
                 model->label, result.required_mib, result.available_mib);
        return result;
    }
    result.allowed = true;
    snprintf(result.reason, sizeof(result.reason), "admitted with %u MiB headroom",
             result.available_mib - result.required_mib);
    return result;
}

int read_gpu_snapshot(GpuSnapshot *result) {
    char line[128] = {0};
    unsigned total = 0, used = 0;
    /* Constant command: no request or environment text reaches the shell. */
    FILE *pipe = popen("nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null", "r"); /* NOLINT(cert-env33-c) */
    if (pipe == NULL) return -1;
    int scanned = fgets(line, sizeof(line), pipe) != NULL &&
                  sscanf(line, " %u , %u", &total, &used) == 2;
    int status = pclose(pipe);
    if (!scanned || status != 0 || total < used) return -1;
    result->available = true;
    result->total_mib = total;
    result->used_mib = used;
    return 0;
}

int load_models(const char *path, Model *models, size_t capacity,
                size_t *model_count) {
    FILE *file = fopen(path, "r");
    char line[320];
    size_t count = 0;
    if (file == NULL) return -1;
    while (fgets(line, sizeof(line), file) != NULL) {
        Model model = {0};
        if (line[0] == '#' || line[0] == '\n') continue;
        if (count == capacity || sscanf(line, "%63[^,],%95[^,],%u,%u",
              model.id, model.label, &model.weights_mib, &model.runtime_mib) != 4) {
            fclose(file);
            return -1;
        }
        models[count++] = model;
    }
    fclose(file);
    *model_count = count;
    return count == 0 ? -1 : 0;
}

const Model *find_model(const Model *models, size_t count, const char *id) {
    for (size_t index = 0; index < count; ++index)
        if (strcmp(models[index].id, id) == 0) return &models[index];
    return NULL;
}
