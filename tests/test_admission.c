#include "admission.h"

#include <assert.h>
#include <string.h>

int main(void) {
    Model model = { "test", "Test model", 10240, 2048 };
    GpuSnapshot gpu = { true, 16384, 1024 };
    Admission result = admit_model(&gpu, &model, 2048);
    assert(result.allowed);
    assert(result.available_mib == 13312);
    assert(result.required_mib == 12288);
    gpu.used_mib = 3072;
    result = admit_model(&gpu, &model, 2048);
    assert(!result.allowed);
    gpu.available = false;
    result = admit_model(&gpu, &model, 2048);
    assert(!result.allowed);
    assert(strstr(result.reason, "unavailable") != NULL);

    Model models[16] = {0};
    size_t count = 0;
    assert(load_models("models/models.csv", models, 16, &count) == 0);
    const Model *anima_offload = find_model(models, count, "anima-2.9b-offload");
    const Model *anima_cuda = find_model(models, count, "anima-2.9b-bf16");
    assert(anima_offload != NULL);
    assert(anima_cuda != NULL);
    assert(anima_cuda->weights_mib + anima_cuda->runtime_mib == 24576);
    gpu = (GpuSnapshot){true, 24564, 19768};
    result = admit_model(&gpu, anima_offload, OMNISERVE_DEFAULT_RESERVE_MIB);
    assert(!result.allowed);
    assert(strstr(result.reason, "only") != NULL);
    const Model *matting = find_model(models, count, "video-matting-rvm");
    assert(matting != NULL);
    assert(matting->weights_mib + matting->runtime_mib == 1800);
    return 0;
}
