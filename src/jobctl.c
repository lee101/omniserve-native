#include "jobs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void json_string(const char *text) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
        if (*p == '"' || *p == '\\') printf("\\%c", *p);
        else if (*p == '\n') printf("\\n");
        else if (*p == '\r') printf("\\r");
        else if (*p == '\t') printf("\\t");
        else if (*p < 32) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}

static void print_job(const OmniJob *job, const char *disposition) {
    printf("{\"disposition\":"); json_string(disposition);
    printf(",\"key\":"); json_string(job->key);
    printf(",\"kind\":"); json_string(job->kind);
    printf(",\"payload\":"); json_string(job->payload);
    printf(",\"required_mib\":%u,\"priority\":%d,\"state\":", job->required_mib, job->priority);
    json_string(job->state); printf(",\"worker\":"); json_string(job->worker);
    printf(",\"gpu\":"); json_string(job->gpu);
    printf(",\"attempts\":%u,\"lease_expires\":%lld,\"result\":", job->attempts, job->lease_expires);
    json_string(job->result); printf(",\"error\":"); json_string(job->error); puts("}");
}

static unsigned parse_unsigned(const char *value, int *ok) {
    char *tail = NULL;
    unsigned long parsed = strtoul(value, &tail, 10);
    *ok = tail != value && *tail == '\0' && parsed <= 0xffffffffUL;
    return (unsigned)parsed;
}

static int usage(const char *name) {
    fprintf(stderr,
        "usage:\n"
        "  %s init DB\n"
        "  %s submit DB KEY KIND PAYLOAD REQUIRED_MIB [PRIORITY]\n"
        "  %s claim DB WORKER GPU AVAILABLE_MIB LEASE_SECONDS [KINDS_CSV]\n"
        "  %s heartbeat DB KEY WORKER LEASE_SECONDS\n"
        "  %s finish DB KEY WORKER RESULT_URI\n"
        "  %s fail DB KEY WORKER ERROR [retry]\n"
        "  %s retry DB KEY\n"
        "  %s status DB KEY\n", name, name, name, name, name, name, name, name);
    return 2;
}

int main(int argc, char **argv) {
    OmniQueue queue = {0}; OmniJob job = {0}; int result = -1, ok = 0;
    if (argc < 3) return usage(argv[0]);
    if (omni_queue_open(&queue, argv[2]) != 0) { fprintf(stderr, "cannot open queue: %s\n", argv[2]); return 1; }
    if (strcmp(argv[1], "init") == 0 && argc == 3) { puts("{\"status\":\"ready\"}"); result = 0; }
    else if (strcmp(argv[1], "submit") == 0 && (argc == 7 || argc == 8)) {
        unsigned required = parse_unsigned(argv[6], &ok); int priority = argc == 8 ? atoi(argv[7]) : 0;
        if (ok) { result = omni_job_submit(&queue, argv[3], argv[4], argv[5], required, priority, &job); if (result >= 0) { print_job(&job, result ? "created" : job.state); result = 0; } }
    } else if (strcmp(argv[1], "claim") == 0 && (argc == 7 || argc == 8)) {
        unsigned available = parse_unsigned(argv[5], &ok), lease = 0;
        if (ok) lease = parse_unsigned(argv[6], &ok);
        if (ok) { result = omni_job_claim(&queue, argv[3], argv[4], available, lease, argc == 8 ? argv[7] : "", &job); if (result >= 0) { if (result) print_job(&job, "claimed"); else puts("{\"disposition\":\"empty\"}"); result = 0; } }
    } else if (strcmp(argv[1], "heartbeat") == 0 && argc == 6) {
        unsigned lease = parse_unsigned(argv[5], &ok); if (ok) { result = omni_job_heartbeat(&queue, argv[3], argv[4], lease); if (result >= 0) { printf("{\"updated\":%s}\n", result ? "true" : "false"); result = result ? 0 : 3; } }
    } else if (strcmp(argv[1], "finish") == 0 && argc == 6) {
        result = omni_job_finish(&queue, argv[3], argv[4], argv[5]);
        if (result >= 0) { printf("{\"updated\":%s}\n", result ? "true" : "false"); result = result ? 0 : 3; }
    } else if (strcmp(argv[1], "fail") == 0 && (argc == 6 || argc == 7)) {
        result = omni_job_fail(&queue, argv[3], argv[4], argv[5], argc == 7 && strcmp(argv[6], "retry") == 0);
        if (result >= 0) { printf("{\"updated\":%s}\n", result ? "true" : "false"); result = result ? 0 : 3; }
    } else if (strcmp(argv[1], "retry") == 0 && argc == 4) {
        result = omni_job_retry(&queue, argv[3]);
        if (result >= 0) { printf("{\"updated\":%s}\n", result ? "true" : "false"); result = result ? 0 : 3; }
    } else if (strcmp(argv[1], "status") == 0 && argc == 4) {
        result = omni_job_get(&queue, argv[3], &job); if (result >= 0) { if (result) print_job(&job, "status"); else puts("{\"disposition\":\"missing\"}"); result = result ? 0 : 4; }
    } else result = usage(argv[0]);
    if (result < 0) { fprintf(stderr, "queue operation failed: %s\n", sqlite3_errmsg(queue.db)); result = 1; }
    omni_queue_close(&queue);
    return result;
}
