#include "jobs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "check failed at line %d\n", __LINE__); return 1; \
} } while (0)

int main(void) {
    char path[] = "/tmp/omniserve-jobs-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0); close(fd); unlink(path);
    OmniQueue queue = {0}; OmniJob job = {0};
    const char *a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const char *b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    CHECK(omni_queue_open(&queue, path) == 0);
    CHECK(omni_job_submit(&queue, a, "video-matting", "a.json", 1500, 10, &job) == 1);
    CHECK(omni_job_submit(&queue, a, "video-matting", "different.json", 1500, 10, &job) == 0);
    CHECK(strcmp(job.payload, "a.json") == 0);
    CHECK(omni_job_submit(&queue, b, "image", "b.json", 500, 20, &job) == 1);
    CHECK(omni_job_claim(&queue, "worker-1", "node:0", 2000, 60, "video-matting", &job) == 1);
    CHECK(strcmp(job.key, a) == 0 && job.attempts == 1);
    CHECK(omni_job_claim(&queue, "worker-2", "node:0", 2000, 60, "", &job) == 0);
    CHECK(omni_job_claim(&queue, "worker-2", "node:1", 400, 60, "", &job) == 0);
    CHECK(omni_job_claim(&queue, "worker-2", "node:1", 2000, 60, "image", &job) == 1);
    CHECK(strcmp(job.key, b) == 0);
    CHECK(omni_job_finish(&queue, b, "wrong-worker", "x") == 0);
    CHECK(omni_job_finish(&queue, b, "worker-2", "result-b") == 1);
    CHECK(omni_job_heartbeat(&queue, a, "worker-1", 120) == 1);
    CHECK(omni_job_finish(&queue, a, "worker-1", "result-a") == 1);
    CHECK(omni_job_get(&queue, a, &job) == 1);
    CHECK(strcmp(job.state, "succeeded") == 0 && strcmp(job.result, "result-a") == 0);
    CHECK(omni_job_submit(&queue, a, "video-matting", "third.json", 1500, 10, &job) == 0);
    CHECK(strcmp(job.state, "succeeded") == 0);
    CHECK(omni_job_retry(&queue, a) == 0);
    CHECK(omni_job_submit(&queue, "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                          "video-matting", "c.json", 500, 1, &job) == 1);
    CHECK(omni_job_claim(&queue, "worker-3", "node:2", 1000, 60, "video-matting", &job) == 1);
    CHECK(omni_job_fail(&queue, job.key, "worker-3", "transient", 0) == 1);
    CHECK(omni_job_retry(&queue, job.key) == 1);
    CHECK(omni_job_get(&queue, job.key, &job) == 1 && strcmp(job.state, "queued") == 0);
    omni_queue_close(&queue);
    unlink(path);
    puts("job queue tests passed");
    return 0;
}
