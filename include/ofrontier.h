#ifndef OFRONTIER_H
#define OFRONTIER_H

#include <stdbool.h>
#include <stddef.h>

#include "osched.h"

typedef enum {
    OFR_POLICY_OVERFLOW = 0,
    OFR_POLICY_LOCAL = 1,
    OFR_POLICY_FASTEST = 2,
    OFR_POLICY_DEADLINE = 3
} ofr_policy;

typedef enum { OFR_LOCAL = 0, OFR_REMOTE = 1 } ofr_choice;

typedef struct {
    bool loaded;
    double local_p50_ms;
    double remote_p50_ms;
    ofr_policy policy[4];
    double deadline_ms[4];
} ofr_table;

typedef struct ofrontier ofrontier;

void ofrontier_table_default(ofr_table *t);
bool ofrontier_parse(const char *json, size_t len, const char *workload, ofr_table *out);
double ofrontier_local_wait_ms(const osched_stats *st, otier tier, int permits, double local_p50_ms);
ofr_choice ofrontier_decide(const ofr_table *t, otier tier, double local_wait_ms, double deadline_ms);
const char *ofrontier_policy_name(ofr_policy p);

ofrontier *ofrontier_open(const char *policy_path, const char *workload, const char *log_path, int port);
void ofrontier_close(ofrontier *f);
bool ofrontier_snapshot(ofrontier *f, ofr_table *out);
const char *ofrontier_workload(const ofrontier *f);
void ofrontier_log(ofrontier *f, otier tier, const char *backend, const char *reason,
                   double queue_ms, double exec_ms, double local_wait_ms, int status);

#endif
