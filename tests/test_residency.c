#undef NDEBUG
#include "oresidency.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static ores_report_in zi(int requests, bool resident) {
    return (ores_report_in){"zimage", 1, 9700, 3000, 14000, requests, resident};
}

static ores_report_in qw(int requests, bool resident) {
    return (ores_report_in){"qwen", 2, 12000, 6000, 5000, requests, resident};
}

static void test_single_promote_hold_idle(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.hysteresis_mb = 0;
    cfg.free_window_s = 0;
    cfg.claim_ttl_s = 1e6;
    ores *r = ores_create(&cfg);
    ores_report_in z = zi(0, false);
    ores_decision d = ores_report_at(r, &z, 100, 20000);
    assert(!d.resident && strcmp(d.reason, "idle") == 0);
    z = zi(3, false);
    d = ores_report_at(r, &z, 101, 20000);
    assert(d.resident && strcmp(d.reason, "promote") == 0);
    z = zi(1, true);
    d = ores_report_at(r, &z, 110, 20000 - 9700);
    assert(d.resident && strcmp(d.reason, "holding") == 0);
    z = zi(0, true);
    d = ores_report_at(r, &z, 110 + cfg.idle_release_s + 1, 20000 - 9700);
    assert(!d.resident && strcmp(d.reason, "idle") == 0);
    ores_destroy(r);
}

static void test_capacity_keeps_stream_headroom(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.hysteresis_mb = 0;
    cfg.free_window_s = 0;
    ores *r = ores_create(&cfg);
    ores_report_in q = {"qwen", 2, 20000, 6000, 5000, 1, false};
    ores_report_at(r, &q, 100, 16000);
    ores_report_in z = zi(5, false);
    /* qwen is active: 9700 + 1024 keep + 6000 qwen stream = 16724 > 16000 */
    ores_decision d = ores_report_at(r, &z, 101, 16000);
    assert(!d.resident && !d.admitted);
    d = ores_report_at(r, &z, 102, 17000);
    assert(d.resident && d.admitted);
    ores_destroy(r);
}

static void test_switch_needs_revoke_before_promote(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.hysteresis_mb = 0;
    cfg.free_window_s = 0;
    cfg.min_hold_s = 30;
    cfg.claim_ttl_s = 1e6;
    ores *r = ores_create(&cfg);
    int total_free = 20000;
    ores_report_in z = zi(10, false), q = qw(0, false);
    ores_report_at(r, &q, 100, total_free);
    ores_decision d = ores_report_at(r, &z, 100, total_free);
    assert(d.resident);
    int free_z = total_free - 9700;
    /* both resident would need 22724 MB; Qwen burst while Z-Image goes quiet */
    q = qw(60, false);
    d = ores_report_at(r, &q, 110, free_z);
    assert(!d.resident && !d.admitted);
    q = qw(0, false);
    d = ores_report_at(r, &q, 400, free_z);
    assert(!d.resident && d.admitted && strcmp(d.reason, "pending_free") == 0);
    z = zi(0, true);
    d = ores_report_at(r, &z, 401, free_z);
    assert(!d.resident && strcmp(d.reason, "cooldown") == 0);
    d = ores_report_at(r, &q, 402, total_free);
    assert(d.resident && strcmp(d.reason, "promote") == 0);
    ores_destroy(r);
}

static void test_hysteresis(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.hysteresis_mb = 0;
    cfg.free_window_s = 0;
    cfg.min_hold_s = 0;
    ores *r = ores_create(&cfg);
    int total_free = 20000;
    ores_report_in z = zi(10, false), q = qw(0, false);
    ores_report_at(r, &q, 100, total_free);
    assert(ores_report_at(r, &z, 100, total_free).resident);
    z = zi(0, true);
    ores_report_at(r, &z, 101, total_free - 9700);
    /* z benefit 10*14000=140k * 1.5 = 210k; q 30*5000=150k: not enough to switch */
    q = qw(30, false);
    ores_decision d = ores_report_at(r, &q, 101, total_free - 9700);
    assert(!d.admitted);
    q = qw(20, false);
    d = ores_report_at(r, &q, 101, total_free - 9700);
    assert(d.admitted);
    ores_destroy(r);
}

static void test_pressure_and_unknown_device(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.hysteresis_mb = 0;
    cfg.free_window_s = 0;
    ores *r = ores_create(&cfg);
    ores_report_in z = zi(5, false);
    assert(!ores_report_at(r, &z, 10, -1).resident);
    assert(ores_report_at(r, &z, 11, 20000).resident);
    z = zi(1, true);
    assert(ores_report_at(r, &z, 12, 500).resident == false);
    char buf[2048];
    size_t n = ores_status_json(r, 12, buf, sizeof buf);
    assert(n > 0 && strstr(buf, "\"owner\":\"zimage\"") && strstr(buf, "\"revocations\":1"));
    ores_destroy(r);
}

static void test_claim_expiry(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.hysteresis_mb = 0;
    cfg.free_window_s = 0;
    ores *r = ores_create(&cfg);
    ores_report_in q = {"qwen", 2, 20000, 6000, 5000, 1, false}, z = zi(5, false);
    ores_report_at(r, &q, 100, 16000);
    assert(!ores_report_at(r, &z, 101, 16000).resident);
    assert(ores_report_at(r, &z, 100 + cfg.claim_ttl_s + 5, 16000).resident);
    ores_destroy(r);
}

static void test_quiet_claimant_reserves_nothing(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.hysteresis_mb = 0;
    cfg.free_window_s = 0;
    cfg.claim_ttl_s = 1e6;
    ores *r = ores_create(&cfg);
    /* a qwen lane too large to ever be resident here, so only its stream reserve matters */
    ores_report_in q = {"qwen", 2, 20000, 6000, 5000, 1, false}, z = zi(5, false);
    ores_report_at(r, &q, 100, 16000);
    assert(!ores_report_at(r, &z, 101, 16000).resident);
    q.requests = 0;
    ores_report_at(r, &q, 100 + cfg.reserve_idle_s + 10, 16000);
    z = zi(1, false);
    assert(ores_report_at(r, &z, 100 + cfg.reserve_idle_s + 11, 16000).resident);
    q.requests = 1;
    ores_report_at(r, &q, 100 + cfg.reserve_idle_s + 12, 16000 - 9700);
    z = zi(0, true);
    ores_decision d = ores_report_at(r, &z, 100 + cfg.reserve_idle_s + 13 + cfg.min_hold_s, 16000 - 9700);
    assert(!d.resident);
    ores_destroy(r);
}

static void test_noisy_free_does_not_flap(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.claim_ttl_s = 1e6;
    ores *r = ores_create(&cfg);
    ores_report_in z = zi(5, false);
    /* 9700 + 1024 keep = 10724; a newcomer needs that + 1024 against the window minimum */
    assert(!ores_report_at(r, &z, 100, 11000).resident);
    assert(!ores_report_at(r, &z, 101, 14000).resident);
    assert(ores_report_at(r, &z, 162, 12000).resident);
    z = zi(1, true);
    /* holder: co-tenant render drops free to 900 (+9700 own = 10600 < 10724) -> revoked; 1100 keeps it */
    assert(ores_report_at(r, &z, 163, 1100).resident);
    assert(!ores_report_at(r, &z, 164, 900).resident);
    ores_destroy(r);
}

static void test_flapping_backs_off(void) {
    ores_config cfg;
    ores_config_defaults(&cfg);
    cfg.hysteresis_mb = 0;
    cfg.free_window_s = 0;
    cfg.claim_ttl_s = 1e6;
    cfg.min_hold_s = 10;
    ores *r = ores_create(&cfg);
    ores_report_in z = zi(5, false);
    double t = 100;
    assert(ores_report_at(r, &z, t, 20000).resident);
    z = zi(1, true);
    assert(!ores_report_at(r, &z, t += 11, 500).resident); /* revoked: cooldown 10 */
    z = zi(1, false);
    assert(!ores_report_at(r, &z, t += 5, 20000).resident);
    assert(ores_report_at(r, &z, t += 6, 20000).resident);
    z = zi(1, true);
    assert(!ores_report_at(r, &z, t += 11, 500).resident); /* second short grant: cooldown 20 */
    z = zi(1, false);
    assert(!ores_report_at(r, &z, t += 15, 20000).resident);
    assert(ores_report_at(r, &z, t += 6, 20000).resident);
    ores_destroy(r);
}

int main(void) {
    test_flapping_backs_off();
    test_noisy_free_does_not_flap();
    test_quiet_claimant_reserves_nothing();
    test_single_promote_hold_idle();
    test_capacity_keeps_stream_headroom();
    test_switch_needs_revoke_before_promote();
    test_hysteresis();
    test_pressure_and_unknown_device();
    test_claim_expiry();
    puts("residency ok");
    return 0;
}
