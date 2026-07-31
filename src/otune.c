#define _GNU_SOURCE
#include "otune.h"

#include <string.h>
#include <strings.h>

otune_device_class otune_classify(const char *device_description) {
    if (!device_description || !device_description[0]) return OTUNE_DEVICE_UNKNOWN;
    const char *d = device_description;
    if (strcasestr(d, "5090") || strcasestr(d, "5080") || strcasestr(d, "B200") ||
        strcasestr(d, "GB200") || strcasestr(d, "Blackwell")) {
        return OTUNE_DEVICE_BLACKWELL;
    }
    if (strcasestr(d, "H100") || strcasestr(d, "H200") || strcasestr(d, "Hopper")) {
        return OTUNE_DEVICE_HOPPER;
    }
    if (strcasestr(d, "4090") || strcasestr(d, "4080") || strcasestr(d, "L40") ||
        strcasestr(d, "L4") || strcasestr(d, "Ada")) {
        return OTUNE_DEVICE_ADA;
    }
    if (strcasestr(d, "3090") || strcasestr(d, "A100") || strcasestr(d, "A40") ||
        strcasestr(d, "A10") || strcasestr(d, "A6000") || strcasestr(d, "Ampere")) {
        return OTUNE_DEVICE_AMPERE;
    }
    if (strcasestr(d, "T4") || strcasestr(d, "2080") || strcasestr(d, "Turing")) {
        return OTUNE_DEVICE_TURING;
    }
    if (strcasecmp(d, "cpu") == 0 || strcasestr(d, "CPU")) return OTUNE_DEVICE_CPU;
    return OTUNE_DEVICE_UNKNOWN;
}

const char *otune_class_name(otune_device_class device_class) {
    switch (device_class) {
    case OTUNE_DEVICE_CPU: return "cpu";
    case OTUNE_DEVICE_BLACKWELL: return "blackwell";
    case OTUNE_DEVICE_ADA: return "ada";
    case OTUNE_DEVICE_AMPERE: return "ampere";
    case OTUNE_DEVICE_HOPPER: return "hopper";
    case OTUNE_DEVICE_TURING: return "turing";
    default: return "unknown";
    }
}

void otune_profile_for(const char *device_description, otune_profile *out) {
    if (!out) return;
    otune_device_class device_class = otune_classify(device_description);
    out->device_class = device_class;
    out->class_name = otune_class_name(device_class);
    switch (device_class) {
    case OTUNE_DEVICE_BLACKWELL:
        /* 32 GB and fast quantized attention: prefill wide, keep KV small so the
         * freed memory buys parallel contexts instead of idle cache. */
        out->n_batch = 4096;
        out->n_ubatch = 1024;
        out->kv_type = "q8_0";
        out->flash_attn = true;
        out->parallel_contexts = 4;
        break;
    case OTUNE_DEVICE_HOPPER:
        out->n_batch = 4096;
        out->n_ubatch = 1024;
        out->kv_type = "q8_0";
        out->flash_attn = true;
        out->parallel_contexts = 6;
        break;
    case OTUNE_DEVICE_ADA:
    case OTUNE_DEVICE_AMPERE:
        /* One profile because the two classes agree on everything this
         * function keys off. 24 GB is the binding constraint on both, not
         * compute, and quantized KV is what makes more than two contexts fit
         * beside the weights. llama.cpp compiles the q8_0/q8_0
         * flash-attention case for every architecture that has flash
         * attention at all, so Ampere has no kernel reason to carry a cache
         * twice the size — and a 3090 is bandwidth-bound at decode, so
         * halving KV traffic is a throughput win on top of the context it
         * buys. They stay separate classes because /status.tune has to name
         * the card truthfully, and because the next thing worth tuning may
         * well differ between them. */
        out->n_batch = 2048;
        out->n_ubatch = 512;
        out->kv_type = "q8_0";
        out->flash_attn = true;
        out->parallel_contexts = 3;
        break;
    case OTUNE_DEVICE_TURING:
        /* Small, bandwidth-bound, and without the newer attention kernels:
         * narrow micro-batches avoid stalling on a single long prefill. */
        out->n_batch = 1024;
        out->n_ubatch = 256;
        out->kv_type = "f16";
        out->flash_attn = false;
        out->parallel_contexts = 1;
        break;
    case OTUNE_DEVICE_CPU:
        out->n_batch = 512;
        out->n_ubatch = 128;
        out->kv_type = "f16";
        out->flash_attn = false;
        out->parallel_contexts = 1;
        break;
    default:
        out->n_batch = 2048;
        out->n_ubatch = 512;
        out->kv_type = "f16";
        out->flash_attn = true;
        out->parallel_contexts = 2;
        break;
    }
}
