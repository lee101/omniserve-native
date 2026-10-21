#ifndef ORESIDENCY_H
#define ORESIDENCY_H

#include <stdbool.h>
#include <stddef.h>

/*
 * Demand-driven weight residency for image models that share one GPU.
 *
 * Each image gateway (embedded Z-Image on :8791, Qwen ra2 on :8792) can run its
 * model streamed (weights in RAM, uploaded over PCIe every step) or resident
 * (weights on the GPU). Resident is several times faster but the box cannot
 * hold every model resident beside the LLM, TTS and matting tenants. The
 * broker decides who is resident from the recent request mix:
 *
 *   benefit = demand (EWMA request count, half-life half_life_s) * saved_ms
 *
 * Claimants are ranked by benefit; the incumbent's benefit is multiplied by
 * switch_ratio (and it is unrevocable for min_hold_s) so the slot does not
 * flap; a revoked claimant waits min_hold_s, doubled for each consecutive grant that
 * lasted under 4 * min_hold_s (max 32x), so a mis-sized claim cannot ping-pong. Greedy admission in rank order against
 *
 *   capacity = device_free + sum(resident_mb of claimants resident now)
 *   need     = sum(resident_mb admitted) + keep_free + max(stream_mb not admitted)
 *
 * so the streamed models keep enough headroom for one render (only claimants with a
 * request inside reserve_idle_s reserve stream headroom). A claimant with
 * no request for idle_release_s is never admitted (its VRAM goes back to the
 * pool). Device free is noisy (co-tenant renders swing it by GBs), so a newcomer is
 * admitted against the minimum free of the last free_window_s plus hysteresis_mb,
 * while a holder is kept as long as it fits the current figure. A winner is only told to promote once the memory is actually free on
 * the device, i.e. after a revoked holder has demoted: two models are never
 * told to be resident on the same bytes.
 */

typedef struct ores ores;

typedef struct {
    int keep_free_mb;
    double half_life_s;
    double min_hold_s;
    double idle_release_s;
    double switch_ratio;
    double claim_ttl_s;
    double reserve_idle_s; /* a streamed claimant reserves stream_mb only this long after its last request */
    int hysteresis_mb;    /* a newcomer must fit with this much to spare; a holder only has to fit */
    double free_window_s; /* a newcomer is judged on the minimum device free seen in this window */
} ores_config;

typedef struct {
    const char *owner;
    int pid;
    int resident_mb;  /* device MB held while resident, including the render workspace */
    int stream_mb;    /* device MB one streamed render needs */
    double saved_ms;  /* per-request latency saved by residency */
    int requests;     /* requests since this claimant's previous report */
    bool resident;    /* state the claimant is in now */
} ores_report_in;

typedef struct {
    bool resident;      /* desired state */
    bool admitted;      /* won the slot (may still be pending free memory) */
    double demand;
    double benefit;
    char reason[48];
} ores_decision;

void ores_config_defaults(ores_config *cfg);
ores *ores_create(const ores_config *cfg);
void ores_destroy(ores *r);

/* Record the claimant's report and decide its desired state. device_free_mb < 0
 * (driver unreachable) always answers streamed. */
ores_decision ores_report_at(ores *r, const ores_report_in *in, double now_s, int device_free_mb);
ores_decision ores_report(ores *r, const ores_report_in *in, int device_free_mb);

size_t ores_status_json(ores *r, double now_s, char *out, size_t cap);

#endif
