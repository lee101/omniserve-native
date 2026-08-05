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
    return 0;
}
