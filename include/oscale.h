#ifndef OSCALE_H
#define OSCALE_H

#include <stdbool.h>
#include <stddef.h>

#include "osched.h"

/*
 * Cost- and priority-guided remote capacity control.
 *
 * The local GPU is already paid for, so it is always the cheapest lane and is
 * always tried first. Renting a remote cog is only worth doing when paid
 * traffic is queueing behind a saturated local device *and* the overflow it
 * would absorb is worth more than the instance costs. Free, subscription, and
 * background traffic are best-effort by policy: they never rent hardware, so a
 * traffic spike on the free tier can never generate a bill.
 *
 * Every lane defaults to zero instances and returns to zero when idle. The
 * decision function is pure — it takes an observation and a clock and returns
 * an action — so the expensive-if-wrong logic is testable without a GPU, a
 * network, or a provider account.
 */

#define OSCALE_MAX_LANES 8
#define OSCALE_MAX_INSTANCES 8

/* Bit per otier, so a policy can admit several tiers. */
#define OSCALE_TIER_BIT(tier) (1u << (unsigned)(tier))
#define OSCALE_TIERS_PAID_ONLY OSCALE_TIER_BIT(TIER_PAID)

typedef enum {
    OSCALE_HOLD = 0,
    OSCALE_UP,
    OSCALE_DOWN,
} oscale_action;

typedef enum {
    OSCALE_REASON_NONE = 0,
    OSCALE_REASON_DISABLED,
    OSCALE_REASON_TIER_NOT_ELIGIBLE,
    OSCALE_REASON_LOCAL_HAS_ROOM,
    OSCALE_REASON_PRESSURE_TOO_LOW,
    OSCALE_REASON_NOT_WORTH_IT,
    OSCALE_REASON_INSTANCE_CAP,
    OSCALE_REASON_SPEND_CAP,
    OSCALE_REASON_COOLDOWN,
    OSCALE_REASON_PRESSURE_SUSTAINED,
    OSCALE_REASON_IDLE,
    OSCALE_REASON_TTL_EXPIRED,
    OSCALE_REASON_NO_INSTANCES,
} oscale_reason;

const char *oscale_reason_name(oscale_reason reason);
const char *oscale_action_name(oscale_action action);

typedef struct {
    /* Identity of the rentable unit. */
    char name[24];          /* lane key, e.g. "tts" */
    char cog_template[64];  /* app.nz cog template or model id */
    char hardware[24];      /* gpu-rtx4090, gpu-rtx5090, ... */

    /* Economics. */
    double price_usd_hr;        /* what the instance costs us per hour */
    double revenue_usd_per_req; /* what one served overflow request is worth */
    double seconds_per_req;     /* how long one request occupies an instance */
    double margin;              /* required value/cost ratio before renting */

    /* Eligibility and pressure. */
    unsigned tier_mask;         /* tiers whose demand may rent hardware */
    int scale_up_queue_depth;   /* eligible waiters needed to consider renting */
    double scale_up_queue_ms;   /* worst eligible wait needed to consider it */

    /* Caps and lifecycle. */
    int max_instances;
    double max_usd_hr;          /* lane spend ceiling */
    double cooldown_s;          /* minimum gap between scale actions */
    double idle_scale_down_s;   /* idle time before releasing an instance */
    double max_instance_ttl_s;  /* absolute lifetime, billing backstop */
    bool enabled;
} oscale_policy;

typedef struct {
    int queue_depth[4];    /* waiters per tier, indexed by otier */
    double queue_ms_max[4];/* worst observed wait per tier */
    int local_permits_free;/* 0 means the local device is saturated */
    double backlog_reqs;   /* eligible requests expected over the next hour */
} oscale_observation;

typedef struct {
    bool active;
    double started_s;
    double last_used_s;
    char endpoint[256];   /* control-plane handle, empty while provisioning */
    bool ready;
} oscale_instance;

typedef struct {
    oscale_policy policy;
    oscale_instance instances[OSCALE_MAX_INSTANCES];
    int instance_count;
    double last_action_s;
    /* Accounting, surfaced on /status so a bill can be traced to a decision. */
    unsigned long long scale_ups;
    unsigned long long scale_downs;
    unsigned long long ttl_kills;
    unsigned long long refused_not_worth_it;
    unsigned long long refused_tier;
    double instance_seconds;
    double spend_usd;
    oscale_reason last_reason;
} oscale_lane;

typedef struct oscale oscale;

oscale *oscale_create(void);
void oscale_destroy(oscale *s);

/* Returns the lane index, or -1 when the table is full or the name is taken. */
int oscale_add_lane(oscale *s, const oscale_policy *policy);
int oscale_lane_count(const oscale *s);
oscale_lane *oscale_lane_at(oscale *s, int index);
oscale_lane *oscale_lane_by_name(oscale *s, const char *name);

/* Fills policy with the conservative defaults: paid-only, scale-to-zero,
 * one instance, a 1.5x margin, and a one-hour hard TTL. */
void oscale_policy_defaults(oscale_policy *policy, const char *name);

/* Pure: no allocation, no clock of its own, no I/O. */
oscale_action oscale_decide(const oscale_lane *lane, const oscale_observation *obs,
                           double now_s, oscale_reason *reason_out);

/* How much of the observed demand a lane may act on. Demand on a tier outside
 * the policy mask is invisible here by design. */
int oscale_eligible_depth(const oscale_policy *policy, const oscale_observation *obs);
double oscale_eligible_wait_ms(const oscale_policy *policy, const oscale_observation *obs);

/* Expected value of renting one more instance for an hour, against its price.
 * Returns true when the rent is justified. */
bool oscale_rent_is_justified(const oscale_policy *policy, const oscale_observation *obs,
                              double *value_usd_out, double *cost_usd_out);

/* State transitions. The caller performs the provisioning I/O and reports back,
 * so the engine stays pure and the control plane stays swappable. */
int oscale_begin_instance(oscale_lane *lane, double now_s);
void oscale_instance_ready(oscale_lane *lane, int index, const char *endpoint, double now_s);
void oscale_instance_touch(oscale_lane *lane, int index, double now_s);
void oscale_release_instance(oscale_lane *lane, int index, double now_s, oscale_reason why);
int oscale_ready_instances(const oscale_lane *lane);

/* Index of the instance that should be released next, or -1. Considers the
 * hard TTL first so a stuck instance is always reclaimed. */
int oscale_release_candidate(const oscale_lane *lane, double now_s, oscale_reason *why);

double oscale_lane_spend_rate_usd_hr(const oscale_lane *lane);

#endif
