#ifndef OMNISERVE_JOBS_H
#define OMNISERVE_JOBS_H

#include <sqlite3.h>
#include <stddef.h>

enum {
    OMNI_JOB_KEY_SIZE = 65,
    OMNI_JOB_KIND_SIZE = 64,
    OMNI_JOB_WORKER_SIZE = 128,
    OMNI_JOB_GPU_SIZE = 64,
    OMNI_JOB_PAYLOAD_SIZE = 1024,
    OMNI_JOB_RESULT_SIZE = 1024,
    OMNI_JOB_ERROR_SIZE = 512
};

typedef struct {
    sqlite3 *db;
} OmniQueue;

typedef struct {
    char key[OMNI_JOB_KEY_SIZE];
    char kind[OMNI_JOB_KIND_SIZE];
    char payload[OMNI_JOB_PAYLOAD_SIZE];
    char state[16];
    char worker[OMNI_JOB_WORKER_SIZE];
    char gpu[OMNI_JOB_GPU_SIZE];
    char result[OMNI_JOB_RESULT_SIZE];
    char error[OMNI_JOB_ERROR_SIZE];
    unsigned required_mib;
    int priority;
    unsigned attempts;
    long long lease_expires;
} OmniJob;

/* All operations are safe across processes sharing the same SQLite database. */
int omni_queue_open(OmniQueue *queue, const char *path);
void omni_queue_close(OmniQueue *queue);

/* Returns 1 when created, 0 for an existing content key, and -1 on error. */
int omni_job_submit(OmniQueue *queue, const char *key, const char *kind,
                    const char *payload, unsigned required_mib, int priority,
                    OmniJob *job);

/*
 * Claims the highest-priority fitting job supported by kinds_csv. An empty
 * kinds list accepts all kinds. A GPU can have only one unexpired lease.
 * Returns 1 when claimed, 0 when no job fits, and -1 on error.
 */
int omni_job_claim(OmniQueue *queue, const char *worker, const char *gpu,
                   unsigned available_mib, unsigned lease_seconds,
                   const char *kinds_csv, OmniJob *job);

int omni_job_heartbeat(OmniQueue *queue, const char *key, const char *worker,
                       unsigned lease_seconds);
int omni_job_finish(OmniQueue *queue, const char *key, const char *worker,
                    const char *result_uri);
int omni_job_fail(OmniQueue *queue, const char *key, const char *worker,
                  const char *error, int retry);
int omni_job_retry(OmniQueue *queue, const char *key);
int omni_job_get(OmniQueue *queue, const char *key, OmniJob *job);

#endif
