#undef NDEBUG
#define _GNU_SOURCE
#include "ofrontier.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *POLICY =
    "{\"version\":1,\"workloads\":{\"ra2\":{\"candidates\":[{\"id\":\"local\",\"p50_ms\":12000}],"
    "\"gateway\":{\"local_p50_ms\":12000,\"remote_p50_ms\":34000,"
    "\"tiers\":{\"paid\":{\"policy\":\"cheapest_within_deadline\",\"deadline_ms\":45000},"
    "\"sub\":{\"policy\":\"fastest\"},\"free\":{\"policy\":\"local_only\"}}}}}}";

static void test_parse(void) {
    ofr_table t;
    assert(ofrontier_parse(POLICY, strlen(POLICY), "ra2", &t));
    assert(t.loaded && t.local_p50_ms == 12000 && t.remote_p50_ms == 34000);
    assert(t.policy[TIER_PAID] == OFR_POLICY_DEADLINE && t.deadline_ms[TIER_PAID] == 45000);
    assert(t.policy[TIER_SUB] == OFR_POLICY_FASTEST);
    assert(t.policy[TIER_FREE] == OFR_POLICY_LOCAL);
    assert(t.policy[TIER_BACKGROUND] == OFR_POLICY_OVERFLOW);
    assert(!ofrontier_parse(POLICY, strlen(POLICY), "yue", &t) && !t.loaded);
    assert(!ofrontier_parse("{bad", 4, "ra2", &t));
    const char *missing = "{\"workloads\":{\"ra2\":{\"gateway\":{\"local_p50_ms\":0,\"remote_p50_ms\":5}}}}";
    assert(!ofrontier_parse(missing, strlen(missing), "ra2", &t));
}

static void test_decide(void) {
    ofr_table t;
    ofrontier_parse(POLICY, strlen(POLICY), "ra2", &t);
    assert(ofrontier_decide(&t, TIER_PAID, 12000, 0) == OFR_LOCAL);
    assert(ofrontier_decide(&t, TIER_PAID, 36000, 0) == OFR_REMOTE);
    assert(ofrontier_decide(&t, TIER_PAID, 12000, 20000) == OFR_LOCAL);
    assert(ofrontier_decide(&t, TIER_PAID, 30000, 40000) == OFR_REMOTE);
    assert(ofrontier_decide(&t, TIER_PAID, 60000, 10000) == OFR_REMOTE);
    assert(ofrontier_decide(&t, TIER_SUB, 12000, 0) == OFR_LOCAL);
    assert(ofrontier_decide(&t, TIER_SUB, 24000, 0) == OFR_REMOTE);
    assert(ofrontier_decide(&t, TIER_FREE, 1e9, 0) == OFR_LOCAL);
    assert(ofrontier_decide(&t, TIER_BACKGROUND, 0, 0) == OFR_REMOTE);
    ofr_table empty;
    ofrontier_table_default(&empty);
    assert(ofrontier_decide(&empty, TIER_FREE, 0, 0) == OFR_REMOTE);
}

static void test_wait(void) {
    osched_stats st = {0};
    st.slots = 1;
    st.used_slots = 1;
    assert(ofrontier_local_wait_ms(&st, TIER_PAID, 1, 10000) == 10000);
    st.waiting[TIER_FREE] = 3;
    assert(ofrontier_local_wait_ms(&st, TIER_PAID, 1, 10000) == 10000);
    assert(ofrontier_local_wait_ms(&st, TIER_FREE, 1, 10000) == 40000);
    st.slots = 2;
    assert(ofrontier_local_wait_ms(&st, TIER_FREE, 1, 10000) == 20000);
}

static void test_reload(void) {
    char path[] = "/tmp/ofrontier-test-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    char log_path[64];
    snprintf(log_path, sizeof log_path, "%s.log", path);
    ofrontier *f = ofrontier_open(path, "ra2", log_path, 8792);
    ofr_table t;
    assert(!ofrontier_snapshot(f, &t));
    FILE *fp = fopen(path, "w");
    fputs(POLICY, fp);
    fclose(fp);
    sleep(1);
    assert(ofrontier_snapshot(f, &t) && t.policy[TIER_FREE] == OFR_POLICY_LOCAL);
    ofrontier_log(f, TIER_PAID, "local", "admitted", 1.5, 12000, 0, 200);
    ofrontier_close(f);
    fp = fopen(log_path, "r");
    char line[512] = {0};
    assert(fgets(line, sizeof line, fp));
    fclose(fp);
    assert(strstr(line, "\"workload\":\"ra2\"") && strstr(line, "\"tier\":\"paid\"") &&
           strstr(line, "\"backend\":\"local\"") && strstr(line, "\"port\":8792"));
    unlink(path);
    unlink(log_path);
}

int main(void) {
    test_parse();
    test_decide();
    test_wait();
    test_reload();
    puts("frontier ok");
    return 0;
}
