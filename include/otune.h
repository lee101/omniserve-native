#ifndef OTUNE_H
#define OTUNE_H

#include <stdbool.h>

/*
 * Per-device inference defaults.
 *
 * A single tuned setting is wrong across a fleet: batch sizes that keep a 5090
 * busy stall a T4, and a KV type that is free on Blackwell costs accuracy
 * headroom on cards without fast quantized-attention kernels. These defaults
 * are keyed off the backend's device description and are only applied where the
 * operator has not set the corresponding environment variable, so explicit
 * configuration always wins.
 */

typedef enum {
    OTUNE_DEVICE_UNKNOWN = 0,
    OTUNE_DEVICE_CPU,
    OTUNE_DEVICE_BLACKWELL,  /* RTX 5090, sm_120 */
    OTUNE_DEVICE_ADA,        /* RTX 4090, L40S, sm_89 */
    OTUNE_DEVICE_AMPERE,     /* RTX 3090, A40, A100, sm_80/86 */
    OTUNE_DEVICE_HOPPER,     /* H100, H200, sm_90 */
    OTUNE_DEVICE_TURING,     /* T4, sm_75 */
} otune_device_class;

typedef struct {
    otune_device_class device_class;
    const char *class_name;
    int n_batch;
    int n_ubatch;
    const char *kv_type;
    bool flash_attn;
    int parallel_contexts; /* suggested contexts per device, capped by slots */
} otune_profile;

otune_device_class otune_classify(const char *device_description);
const char *otune_class_name(otune_device_class device_class);

/* Fills the tuned profile for a device description. Always succeeds; unknown
 * devices get the conservative Ampere-class profile. */
void otune_profile_for(const char *device_description, otune_profile *out);

#endif
