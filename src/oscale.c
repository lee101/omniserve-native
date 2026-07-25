#define _GNU_SOURCE
#include "oscale.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct oscale {
    pthread_mutex_t lock;
    oscale_lane lanes[OSCALE_MAX_LANES];
    int lane_count;
};

const char *oscale_reason_name(oscale_reason reason) {
    switch (reason) {
    case OSCALE_REASON_DISABLED: return "disabled";
    case OSCALE_REASON_TIER_NOT_ELIGIBLE: return "tier-not-eligible";
    case OSCALE_REASON_LOCAL_HAS_ROOM: return "local-has-room";
    case OSCALE_REASON_PRESSURE_TOO_LOW: return "pressure-too-low";
    case OSCALE_REASON_NOT_WORTH_IT: return "not-worth-it";
    case OSCALE_REASON_INSTANCE_CAP: return "instance-cap";
    case OSCALE_REASON_SPEND_CAP: return "spend-cap";
    case OSCALE_REASON_COOLDOWN: return "cooldown";
    case OSCALE_REASON_PRESSURE_SUSTAINED: return "pressure-sustained";
    case OSCALE_REASON_IDLE: return "idle";
    case OSCALE_REASON_TTL_EXPIRED: return "ttl-expired";
    case OSCALE_REASON_NO_INSTANCES: return "no-instances";
    default: return "none";
    }
}

const char *oscale_action_name(oscale_action action) {
    switch (action) {
    case OSCALE_UP: return "up";
    case OSCALE_DOWN: return "down";
    default: return "hold";
    }
}

oscale *oscale_create(void) {
    oscale *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    pthread_mutex_init(&s->lock, NULL);
    return s;
}

void oscale_destroy(oscale *s) {
    if (!s) return;
    pthread_mutex_destroy(&s->lock);
    free(s);
}

void oscale_policy_defaults(oscale_policy *policy, const char *name) {
    if (!policy) return;
    memset(policy, 0, sizeof *policy);
    snprintf(policy->name, sizeof policy->name, "%s", name ? name : "");
    /* Paid-only is the whole safety story: best-effort tiers cannot rent. */
    policy->tier_mask = OSCALE_TIERS_PAID_ONLY;
    policy->margin = 1.5;
    policy->seconds_per_req = 5.0;
    policy->scale_up_queue_depth = 2;
    policy->scale_up_queue_ms = 2000.0;
    policy->max_instances = 1;
    policy->cooldown_s = 120.0;
    policy->idle_scale_down_s = 180.0;
    policy->max_instance_ttl_s = 3600.0;
    policy->enabled = false;
}

int oscale_add_lane(oscale *s, const oscale_policy *policy) {
    if (!s || !policy || !policy->name[0]) return -1;
    pthread_mutex_lock(&s->lock);
    if (s->lane_count >= OSCALE_MAX_LANES) {
        pthread_mutex_unlock(&s->lock);
        return -1;
    }
    for (int i = 0; i < s->lane_count; i++) {
        if (strcmp(s->lanes[i].policy.name, policy->name) == 0) {
            pthread_mutex_unlock(&s->lock);
            return -1;
        }
    }
    int index = s->lane_count++;
    memset(&s->lanes[index], 0, sizeof s->lanes[index]);
    s->lanes[index].policy = *policy;
    if (s->lanes[index].policy.max_instances > OSCALE_MAX_INSTANCES) {
        s->lanes[index].policy.max_instances = OSCALE_MAX_INSTANCES;
    }
    /* Background work is best-effort by definition — it runs on whatever is
     * already idle and is never worth renting hardware for. Enforced here so a
     * misconfigured tier list cannot turn a batch backlog into a bill. */
    s->lanes[index].policy.tier_mask &= ~OSCALE_TIER_BIT(TIER_BACKGROUND);
    if (!s->lanes[index].policy.tier_mask) {
        s->lanes[index].policy.tier_mask = OSCALE_TIERS_PAID_ONLY;
    }
    pthread_mutex_unlock(&s->lock);
    return index;
}

int oscale_lane_count(const oscale *s) { return s ? s->lane_count : 0; }

oscale_lane *oscale_lane_at(oscale *s, int index) {
    if (!s || index < 0 || index >= s->lane_count) return NULL;
    return &s->lanes[index];
}

oscale_lane *oscale_lane_by_name(oscale *s, const char *name) {
    if (!s || !name) return NULL;
    for (int i = 0; i < s->lane_count; i++) {
        if (strcmp(s->lanes[i].policy.name, name) == 0) return &s->lanes[i];
    }
    return NULL;
}

int oscale_eligible_depth(const oscale_policy *policy, const oscale_observation *obs) {
    if (!policy || !obs) return 0;
    int depth = 0;
    for (int tier = TIER_PAID; tier <= TIER_BACKGROUND; tier++) {
        if (policy->tier_mask & OSCALE_TIER_BIT(tier)) depth += obs->queue_depth[tier];
    }
    return depth;
}

double oscale_eligible_wait_ms(const oscale_policy *policy, const oscale_observation *obs) {
    if (!policy || !obs) return 0.0;
    double worst = 0.0;
    for (int tier = TIER_PAID; tier <= TIER_BACKGROUND; tier++) {
        if ((policy->tier_mask & OSCALE_TIER_BIT(tier)) && obs->queue_ms_max[tier] > worst) {
            worst = obs->queue_ms_max[tier];
        }
    }
    return worst;
}

bool oscale_rent_is_justified(const oscale_policy *policy, const oscale_observation *obs,
                              double *value_usd_out, double *cost_usd_out) {
    if (value_usd_out) *value_usd_out = 0.0;
    if (cost_usd_out) *cost_usd_out = 0.0;
    if (!policy || !obs) return false;
    double seconds_per_req = policy->seconds_per_req > 0 ? policy->seconds_per_req : 1.0;
    /* One instance rented for one hour can serve this many requests. */
    double capacity_reqs = 3600.0 / seconds_per_req;
    /* Only demand we actually expect counts; renting capacity nobody uses is
     * how a scale-up loses money. */
    double servable = obs->backlog_reqs < capacity_reqs ? obs->backlog_reqs : capacity_reqs;
    if (servable < 0.0) servable = 0.0;
    double value = servable * policy->revenue_usd_per_req;
    double margin = policy->margin > 0 ? policy->margin : 1.0;
    double cost = policy->price_usd_hr * margin;
    if (value_usd_out) *value_usd_out = value;
    if (cost_usd_out) *cost_usd_out = cost;
    if (policy->price_usd_hr <= 0.0) return false;
    return value >= cost;
}

double oscale_lane_spend_rate_usd_hr(const oscale_lane *lane) {
    if (!lane) return 0.0;
    int active = 0;
    for (int i = 0; i < lane->instance_count; i++) {
        if (lane->instances[i].active) active++;
    }
    return (double)active * lane->policy.price_usd_hr;
}

int oscale_ready_instances(const oscale_lane *lane) {
    if (!lane) return 0;
    int ready = 0;
    for (int i = 0; i < lane->instance_count; i++) {
        if (lane->instances[i].active && lane->instances[i].ready) ready++;
    }
    return ready;
}

static int active_instances(const oscale_lane *lane) {
    int active = 0;
    for (int i = 0; i < lane->instance_count; i++) {
        if (lane->instances[i].active) active++;
    }
    return active;
}

int oscale_release_candidate(const oscale_lane *lane, double now_s, oscale_reason *why) {
    if (why) *why = OSCALE_REASON_NONE;
    if (!lane) return -1;
    /* The hard TTL wins over everything, including active use: an instance that
     * outlives its lifetime is the failure mode that bills forever. */
    if (lane->policy.max_instance_ttl_s > 0) {
        for (int i = 0; i < lane->instance_count; i++) {
            const oscale_instance *inst = &lane->instances[i];
            if (inst->active && now_s - inst->started_s >= lane->policy.max_instance_ttl_s) {
                if (why) *why = OSCALE_REASON_TTL_EXPIRED;
                return i;
            }
        }
    }
    if (lane->policy.idle_scale_down_s > 0) {
        int oldest = -1;
        double oldest_idle = 0.0;
        for (int i = 0; i < lane->instance_count; i++) {
            const oscale_instance *inst = &lane->instances[i];
            if (!inst->active) continue;
            double reference = inst->last_used_s > 0 ? inst->last_used_s : inst->started_s;
            double idle = now_s - reference;
            if (idle >= lane->policy.idle_scale_down_s && idle > oldest_idle) {
                oldest = i;
                oldest_idle = idle;
            }
        }
        if (oldest >= 0) {
            if (why) *why = OSCALE_REASON_IDLE;
            return oldest;
        }
    }
    return -1;
}

oscale_action oscale_decide(const oscale_lane *lane, const oscale_observation *obs,
                           double now_s, oscale_reason *reason_out) {
    oscale_reason reason = OSCALE_REASON_NONE;
    oscale_action action = OSCALE_HOLD;
    if (!lane || !obs) {
        if (reason_out) *reason_out = reason;
        return action;
    }

    /* Reclaim before anything else, and without regard to the cooldown or the
     * enabled flag: a disabled lane must still shed what it already rented. */
    int release = oscale_release_candidate(lane, now_s, &reason);
    if (release >= 0) {
        if (reason_out) *reason_out = reason;
        return OSCALE_DOWN;
    }

    if (!lane->policy.enabled) {
        if (reason_out) *reason_out = OSCALE_REASON_DISABLED;
        return OSCALE_HOLD;
    }

    int depth = oscale_eligible_depth(&lane->policy, obs);
    if (depth <= 0) {
        if (reason_out) {
            /* Distinguish "nothing is waiting" from "only ineligible tiers are
             * waiting", because the second is a policy refusal worth counting. */
            int total = 0;
            for (int tier = TIER_PAID; tier <= TIER_BACKGROUND; tier++) {
                total += obs->queue_depth[tier];
            }
            *reason_out = total > 0 ? OSCALE_REASON_TIER_NOT_ELIGIBLE
                                    : OSCALE_REASON_PRESSURE_TOO_LOW;
        }
        return OSCALE_HOLD;
    }

    /* The local device is already paid for; never rent while it has room. */
    if (obs->local_permits_free > 0) {
        if (reason_out) *reason_out = OSCALE_REASON_LOCAL_HAS_ROOM;
        return OSCALE_HOLD;
    }

    if (depth < lane->policy.scale_up_queue_depth ||
        oscale_eligible_wait_ms(&lane->policy, obs) < lane->policy.scale_up_queue_ms) {
        if (reason_out) *reason_out = OSCALE_REASON_PRESSURE_TOO_LOW;
        return OSCALE_HOLD;
    }

    if (!oscale_rent_is_justified(&lane->policy, obs, NULL, NULL)) {
        if (reason_out) *reason_out = OSCALE_REASON_NOT_WORTH_IT;
        return OSCALE_HOLD;
    }

    int active = active_instances(lane);
    if (active >= lane->policy.max_instances || active >= OSCALE_MAX_INSTANCES) {
        if (reason_out) *reason_out = OSCALE_REASON_INSTANCE_CAP;
        return OSCALE_HOLD;
    }
    if (lane->policy.max_usd_hr > 0 &&
        (double)(active + 1) * lane->policy.price_usd_hr > lane->policy.max_usd_hr) {
        if (reason_out) *reason_out = OSCALE_REASON_SPEND_CAP;
        return OSCALE_HOLD;
    }
    if (lane->policy.cooldown_s > 0 && lane->last_action_s > 0 &&
        now_s - lane->last_action_s < lane->policy.cooldown_s) {
        if (reason_out) *reason_out = OSCALE_REASON_COOLDOWN;
        return OSCALE_HOLD;
    }

    if (reason_out) *reason_out = OSCALE_REASON_PRESSURE_SUSTAINED;
    return OSCALE_UP;
}

int oscale_begin_instance(oscale_lane *lane, double now_s) {
    if (!lane) return -1;
    int slot = -1;
    for (int i = 0; i < lane->instance_count; i++) {
        if (!lane->instances[i].active) { slot = i; break; }
    }
    if (slot < 0) {
        if (lane->instance_count >= OSCALE_MAX_INSTANCES) return -1;
        slot = lane->instance_count++;
    }
    memset(&lane->instances[slot], 0, sizeof lane->instances[slot]);
    lane->instances[slot].active = true;
    lane->instances[slot].started_s = now_s;
    lane->instances[slot].last_used_s = now_s;
    lane->last_action_s = now_s;
    lane->scale_ups++;
    lane->last_reason = OSCALE_REASON_PRESSURE_SUSTAINED;
    return slot;
}

void oscale_instance_ready(oscale_lane *lane, int index, const char *endpoint, double now_s) {
    if (!lane || index < 0 || index >= lane->instance_count) return;
    oscale_instance *inst = &lane->instances[index];
    if (!inst->active) return;
    inst->ready = true;
    inst->last_used_s = now_s;
    snprintf(inst->endpoint, sizeof inst->endpoint, "%s", endpoint ? endpoint : "");
}

void oscale_instance_touch(oscale_lane *lane, int index, double now_s) {
    if (!lane || index < 0 || index >= lane->instance_count) return;
    if (lane->instances[index].active) lane->instances[index].last_used_s = now_s;
}

void oscale_release_instance(oscale_lane *lane, int index, double now_s, oscale_reason why) {
    if (!lane || index < 0 || index >= lane->instance_count) return;
    oscale_instance *inst = &lane->instances[index];
    if (!inst->active) return;
    double lifetime = now_s - inst->started_s;
    if (lifetime < 0) lifetime = 0;
    lane->instance_seconds += lifetime;
    lane->spend_usd += lane->policy.price_usd_hr * lifetime / 3600.0;
    memset(inst, 0, sizeof *inst);
    lane->last_action_s = now_s;
    lane->scale_downs++;
    if (why == OSCALE_REASON_TTL_EXPIRED) lane->ttl_kills++;
    lane->last_reason = why;
}
