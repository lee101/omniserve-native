#ifndef OMNISERVE_ADMISSION_H
#define OMNISERVE_ADMISSION_H

#include <stdbool.h>
#include <stddef.h>

enum { OMNISERVE_DEFAULT_RESERVE_MIB = 2048 };

typedef struct {
    bool available;
    unsigned total_mib;
    unsigned used_mib;
} GpuSnapshot;

typedef struct {
    char id[64];
    char label[96];
    unsigned weights_mib;
    unsigned runtime_mib;
} Model;

typedef struct {
    bool allowed;
    unsigned available_mib;
    unsigned required_mib;
    char reason[160];
} Admission;

Admission admit_model(const GpuSnapshot *gpu, const Model *model,
                      unsigned reserve_mib);
int read_gpu_snapshot(GpuSnapshot *result);
int load_models(const char *path, Model *models, size_t capacity,
                size_t *model_count);
const Model *find_model(const Model *models, size_t count, const char *id);

#endif
