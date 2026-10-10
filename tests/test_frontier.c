#undef NDEBUG
#define _GNU_SOURCE
#include "ofrontier.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

static const char *POLICY =
    "{\"version\":1,\"workloads\":{\"ra2\":{\"candidates\":[{\"id\":\"local\",\"p50_ms\":12000}],"
    "\"gateway\":{\"local_p50_ms\":12000,\"remote_p50_ms\":34000,"
    "\"tiers\":{\"paid\":{\"policy\":\"cheapest_within_deadline\",\"deadline_ms\":45000},"
    "\"sub\":{\"policy\":\"fastest\"},\"free\":{\"policy\":\"local_only\"},"
    "\"background\":{\"policy\":\"background\"}}}}}}";

static void test_parse(void) {
    ofr_table t;
    assert(ofrontier_parse(POLICY, strlen(POLICY), "ra2", &t));
    assert(t.loaded && t.local_p50_ms == 12000 && t.remote_p50_ms == 34000);
    assert(t.policy[TIER_PAID] == OFR_POLICY_DEADLINE && t.deadline_ms[TIER_PAID] == 45000);
    assert(t.policy[TIER_SUB] == OFR_POLICY_FASTEST);
    assert(t.policy[TIER_FREE] == OFR_POLICY_LOCAL);
    assert(t.policy[TIER_BACKGROUND] == OFR_POLICY_BACKGROUND && !t.allow_overflow[TIER_BACKGROUND]);
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
    assert(ofrontier_decide(&t, TIER_BACKGROUND, 1e9, 0) == OFR_LOCAL);
    assert(ofrontier_decide(&t, TIER_BACKGROUND, 12000, 20000) == OFR_LOCAL);
    assert(ofrontier_decide(&t, TIER_BACKGROUND, 60000, 50000) == OFR_REMOTE);
    t.allow_overflow[TIER_BACKGROUND] = true;
    assert(ofrontier_decide(&t, TIER_BACKGROUND, 60000, 0) == OFR_REMOTE);
    assert(ofrontier_decide(&t, TIER_BACKGROUND, 12000, 0) == OFR_LOCAL);
    t.policy[TIER_BACKGROUND] = OFR_POLICY_OVERFLOW;
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

typedef struct {
    osched *s;
    otier tier;
    char tag;
    char *order;
    int *pos;
    pthread_mutex_t *lock;
    bool ok;
} job;

static void *run_job(void *arg) {
    job *j = arg;
    j->ok = osched_acquire_n(j->s, j->tier, 1);
    if (j->ok) {
        pthread_mutex_lock(j->lock);
        j->order[(*j->pos)++] = j->tag;
        pthread_mutex_unlock(j->lock);
        usleep(20000);
        osched_release_n(j->s, j->tier, 1);
    }
    return NULL;
}

static void wait_waiting(osched *s, int n) {
    for (int i = 0; i < 400; i++) {
        osched_stats st;
        osched_snapshot(s, &st);
        if (st.waiting[0] + st.waiting[1] + st.waiting[2] + st.waiting[3] >= n) return;
        usleep(5000);
    }
    assert(!"waiters never queued");
}

static void burst(double age_s, const otier *tiers, const char *tags, int n, int sleep_after_first_ms,
                  const char *want) {
    osched *s = osched_create(1, 5);
    osched_set_background(s, 10, age_s);
    assert(osched_acquire_n(s, TIER_PAID, 1));
    char order[16] = {0};
    int pos = 0;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    job jobs[8];
    pthread_t th[8];
    for (int i = 0; i < n; i++) {
        jobs[i] = (job){s, tiers[i], tags[i], order, &pos, &lock, false};
        pthread_create(&th[i], NULL, run_job, &jobs[i]);
        wait_waiting(s, i + 1);
        if (i == 0 && sleep_after_first_ms) usleep(sleep_after_first_ms * 1000);
    }
    osched_release_n(s, TIER_PAID, 1);
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    for (int i = 0; i < n; i++) assert(jobs[i].ok);
    if (strcmp(order, want) != 0) {
        fprintf(stderr, "order %s want %s\n", order, want);
        assert(0);
    }
    osched_destroy(s);
}

static void test_background_priority(void) {
    otier mix[4] = {TIER_BACKGROUND, TIER_FREE, TIER_PAID, TIER_BACKGROUND};
    burst(0, mix, "bfpc", 4, 0, "pfbc");
    otier aged[2] = {TIER_BACKGROUND, TIER_FREE};
    burst(0.2, aged, "bf", 2, 250, "bf");
    burst(0, aged, "bf", 2, 250, "fb");
    otier very[2] = {TIER_BACKGROUND, TIER_PAID};
    burst(0.1, very, "bp", 2, 250, "bp");
    burst(0.2, very, "bp", 2, 250, "pb");
}

static void test_background_timeout(void) {
    osched *s = osched_create(1, 0.1);
    osched_set_background(s, 0.5, 0);
    assert(osched_acquire_n(s, TIER_PAID, 1));
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    assert(!osched_acquire_n(s, TIER_BACKGROUND, 1));
    clock_gettime(CLOCK_MONOTONIC, &b);
    double waited = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) / 1e9;
    assert(waited > 0.4 && waited < 2.0);
    clock_gettime(CLOCK_MONOTONIC, &a);
    assert(!osched_acquire_n(s, TIER_FREE, 1));
    clock_gettime(CLOCK_MONOTONIC, &b);
    waited = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) / 1e9;
    assert(waited < 0.4);
    osched_release_n(s, TIER_PAID, 1);
    osched_destroy(s);
}

int main(void) {
    test_parse();
    test_decide();
    test_wait();
    test_reload();
    test_background_priority();
    test_background_timeout();
    puts("frontier ok");
    return 0;
}
