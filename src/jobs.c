#include "jobs.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static const char *SCHEMA =
    "PRAGMA journal_mode=WAL;"
    "PRAGMA synchronous=NORMAL;"
    "PRAGMA busy_timeout=5000;"
    "CREATE TABLE IF NOT EXISTS jobs("
    " job_key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,"
    " required_mib INTEGER NOT NULL CHECK(required_mib>=0),"
    " priority INTEGER NOT NULL DEFAULT 0,"
    " state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed')),"
    " worker TEXT NOT NULL DEFAULT '', gpu TEXT NOT NULL DEFAULT '',"
    " lease_expires INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,"
    " result_uri TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',"
    " created_unix INTEGER NOT NULL, updated_unix INTEGER NOT NULL"
    ");"
    "CREATE INDEX IF NOT EXISTS jobs_ready ON jobs(state,priority DESC,created_unix);"
    "CREATE INDEX IF NOT EXISTS jobs_gpu ON jobs(state,gpu,lease_expires);";

static long long now_unix(void) { return (long long)time(NULL); }

static int valid_key(const char *key) {
    if (key == NULL || strlen(key) != 64) return 0;
    for (size_t i = 0; i < 64; ++i)
        if (!isxdigit((unsigned char)key[i])) return 0;
    return 1;
}

static void copy_text(char *dst, size_t size, const unsigned char *src) {
    snprintf(dst, size, "%s", src == NULL ? "" : (const char *)src);
}

static void read_job(sqlite3_stmt *statement, OmniJob *job) {
    memset(job, 0, sizeof(*job));
    copy_text(job->key, sizeof(job->key), sqlite3_column_text(statement, 0));
    copy_text(job->kind, sizeof(job->kind), sqlite3_column_text(statement, 1));
    copy_text(job->payload, sizeof(job->payload), sqlite3_column_text(statement, 2));
    job->required_mib = (unsigned)sqlite3_column_int64(statement, 3);
    job->priority = sqlite3_column_int(statement, 4);
    copy_text(job->state, sizeof(job->state), sqlite3_column_text(statement, 5));
    copy_text(job->worker, sizeof(job->worker), sqlite3_column_text(statement, 6));
    copy_text(job->gpu, sizeof(job->gpu), sqlite3_column_text(statement, 7));
    job->lease_expires = sqlite3_column_int64(statement, 8);
    job->attempts = (unsigned)sqlite3_column_int(statement, 9);
    copy_text(job->result, sizeof(job->result), sqlite3_column_text(statement, 10));
    copy_text(job->error, sizeof(job->error), sqlite3_column_text(statement, 11));
}

static const char *SELECT_COLUMNS =
    "job_key,kind,payload,required_mib,priority,state,worker,gpu,"
    "lease_expires,attempts,result_uri,error";

int omni_queue_open(OmniQueue *queue, const char *path) {
    if (queue == NULL || path == NULL) return -1;
    memset(queue, 0, sizeof(*queue));
    if (sqlite3_open_v2(path, &queue->db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                        SQLITE_OPEN_FULLMUTEX, NULL) != SQLITE_OK) {
        omni_queue_close(queue);
        return -1;
    }
    if (sqlite3_exec(queue->db, SCHEMA, NULL, NULL, NULL) != SQLITE_OK) {
        omni_queue_close(queue);
        return -1;
    }
    return 0;
}

void omni_queue_close(OmniQueue *queue) {
    if (queue != NULL && queue->db != NULL) sqlite3_close(queue->db);
    if (queue != NULL) queue->db = NULL;
}

int omni_job_get(OmniQueue *queue, const char *key, OmniJob *job) {
    char sql[384];
    sqlite3_stmt *statement = NULL;
    if (queue == NULL || queue->db == NULL || !valid_key(key) || job == NULL) return -1;
    snprintf(sql, sizeof(sql), "SELECT %s FROM jobs WHERE job_key=?", SELECT_COLUMNS);
    if (sqlite3_prepare_v2(queue->db, sql, -1, &statement, NULL) != SQLITE_OK) return -1;
    sqlite3_bind_text(statement, 1, key, -1, SQLITE_TRANSIENT);
    int step = sqlite3_step(statement);
    if (step == SQLITE_ROW) read_job(statement, job);
    sqlite3_finalize(statement);
    return step == SQLITE_ROW ? 1 : step == SQLITE_DONE ? 0 : -1;
}

int omni_job_submit(OmniQueue *queue, const char *key, const char *kind,
                    const char *payload, unsigned required_mib, int priority,
                    OmniJob *job) {
    static const char *sql =
        "INSERT INTO jobs(job_key,kind,payload,required_mib,priority,state,created_unix,updated_unix)"
        " VALUES(?,?,?,?,?,'queued',?,?) ON CONFLICT(job_key) DO NOTHING";
    sqlite3_stmt *statement = NULL;
    long long now = now_unix();
    if (queue == NULL || queue->db == NULL || !valid_key(key) || kind == NULL ||
        kind[0] == '\0' || payload == NULL) return -1;
    if (sqlite3_prepare_v2(queue->db, sql, -1, &statement, NULL) != SQLITE_OK) return -1;
    sqlite3_bind_text(statement, 1, key, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement, 2, kind, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement, 3, payload, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement, 4, required_mib);
    sqlite3_bind_int(statement, 5, priority);
    sqlite3_bind_int64(statement, 6, now);
    sqlite3_bind_int64(statement, 7, now);
    int step = sqlite3_step(statement);
    sqlite3_finalize(statement);
    if (step != SQLITE_DONE) return -1;
    int created = sqlite3_changes(queue->db) == 1;
    if (job != NULL && omni_job_get(queue, key, job) != 1) return -1;
    return created;
}

static int kind_allowed(const char *kind, const char *csv) {
    if (csv == NULL || csv[0] == '\0') return 1;
    const char *start = csv;
    size_t wanted = strlen(kind);
    while (*start != '\0') {
        while (*start == ',' || isspace((unsigned char)*start)) ++start;
        const char *end = start;
        while (*end != '\0' && *end != ',') ++end;
        const char *trim = end;
        while (trim > start && isspace((unsigned char)trim[-1])) --trim;
        if ((size_t)(trim - start) == wanted && strncmp(start, kind, wanted) == 0) return 1;
        start = end;
    }
    return 0;
}

static int rollback(OmniQueue *queue) {
    (void)sqlite3_exec(queue->db, "ROLLBACK", NULL, NULL, NULL);
    return -1;
}

int omni_job_claim(OmniQueue *queue, const char *worker, const char *gpu,
                   unsigned available_mib, unsigned lease_seconds,
                   const char *kinds_csv, OmniJob *job) {
    char sql[384];
    sqlite3_stmt *statement = NULL;
    char selected[OMNI_JOB_KEY_SIZE] = {0};
    long long now = now_unix();
    if (queue == NULL || queue->db == NULL || worker == NULL || worker[0] == '\0' ||
        gpu == NULL || gpu[0] == '\0' || lease_seconds == 0 || job == NULL) return -1;
    if (sqlite3_exec(queue->db, "BEGIN IMMEDIATE", NULL, NULL, NULL) != SQLITE_OK) return -1;
    if (sqlite3_prepare_v2(queue->db,
        "UPDATE jobs SET state='queued',worker='',gpu='',lease_expires=0,updated_unix=? "
        "WHERE state='running' AND lease_expires<=?", -1, &statement, NULL) != SQLITE_OK)
        return rollback(queue);
    sqlite3_bind_int64(statement, 1, now);
    sqlite3_bind_int64(statement, 2, now);
    if (sqlite3_step(statement) != SQLITE_DONE) { sqlite3_finalize(statement); return rollback(queue); }
    sqlite3_finalize(statement);

    if (sqlite3_prepare_v2(queue->db,
        "SELECT 1 FROM jobs WHERE state='running' AND gpu=? AND lease_expires>? LIMIT 1",
        -1, &statement, NULL) != SQLITE_OK) return rollback(queue);
    sqlite3_bind_text(statement, 1, gpu, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement, 2, now);
    int busy = sqlite3_step(statement) == SQLITE_ROW;
    sqlite3_finalize(statement);
    if (busy) {
        if (sqlite3_exec(queue->db, "COMMIT", NULL, NULL, NULL) != SQLITE_OK) return -1;
        return 0;
    }

    snprintf(sql, sizeof(sql),
             "SELECT %s FROM jobs WHERE state='queued' AND required_mib<=? "
             "ORDER BY priority DESC,created_unix,job_key", SELECT_COLUMNS);
    if (sqlite3_prepare_v2(queue->db, sql, -1, &statement, NULL) != SQLITE_OK)
        return rollback(queue);
    sqlite3_bind_int64(statement, 1, available_mib);
    while (sqlite3_step(statement) == SQLITE_ROW) {
        const char *kind = (const char *)sqlite3_column_text(statement, 1);
        if (kind_allowed(kind == NULL ? "" : kind, kinds_csv)) {
            copy_text(selected, sizeof(selected), sqlite3_column_text(statement, 0));
            break;
        }
    }
    sqlite3_finalize(statement);
    if (selected[0] == '\0') {
        if (sqlite3_exec(queue->db, "COMMIT", NULL, NULL, NULL) != SQLITE_OK) return -1;
        return 0;
    }

    if (sqlite3_prepare_v2(queue->db,
        "UPDATE jobs SET state='running',worker=?,gpu=?,lease_expires=?,attempts=attempts+1,"
        "updated_unix=?,error='' WHERE job_key=? AND state='queued'", -1,
        &statement, NULL) != SQLITE_OK) return rollback(queue);
    sqlite3_bind_text(statement, 1, worker, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement, 2, gpu, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement, 3, now + lease_seconds);
    sqlite3_bind_int64(statement, 4, now);
    sqlite3_bind_text(statement, 5, selected, -1, SQLITE_TRANSIENT);
    int step = sqlite3_step(statement);
    sqlite3_finalize(statement);
    if (step != SQLITE_DONE || sqlite3_changes(queue->db) != 1) return rollback(queue);
    if (sqlite3_exec(queue->db, "COMMIT", NULL, NULL, NULL) != SQLITE_OK) return -1;
    return omni_job_get(queue, selected, job);
}

int omni_job_heartbeat(OmniQueue *queue, const char *key, const char *worker,
                       unsigned lease_seconds) {
    sqlite3_stmt *statement = NULL;
    long long now = now_unix();
    if (queue == NULL || queue->db == NULL || !valid_key(key) || worker == NULL || lease_seconds == 0)
        return -1;
    if (sqlite3_prepare_v2(queue->db,
        "UPDATE jobs SET lease_expires=?,updated_unix=? WHERE job_key=? AND state='running' AND worker=?",
        -1, &statement, NULL) != SQLITE_OK) return -1;
    sqlite3_bind_int64(statement, 1, now + lease_seconds);
    sqlite3_bind_int64(statement, 2, now);
    sqlite3_bind_text(statement, 3, key, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement, 4, worker, -1, SQLITE_TRANSIENT);
    int step = sqlite3_step(statement);
    sqlite3_finalize(statement);
    return step == SQLITE_DONE && sqlite3_changes(queue->db) == 1 ? 1 : step == SQLITE_DONE ? 0 : -1;
}

static int settle(OmniQueue *queue, const char *key, const char *worker,
                  const char *state, const char *result, const char *error) {
    sqlite3_stmt *statement = NULL;
    if (queue == NULL || queue->db == NULL || !valid_key(key) || worker == NULL) return -1;
    if (sqlite3_prepare_v2(queue->db,
        "UPDATE jobs SET state=?,result_uri=?,error=?,worker='',gpu='',lease_expires=0,updated_unix=? "
        "WHERE job_key=? AND state='running' AND worker=?", -1, &statement, NULL) != SQLITE_OK) return -1;
    sqlite3_bind_text(statement, 1, state, -1, SQLITE_STATIC);
    sqlite3_bind_text(statement, 2, result == NULL ? "" : result, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement, 3, error == NULL ? "" : error, -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement, 4, now_unix());
    sqlite3_bind_text(statement, 5, key, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(statement, 6, worker, -1, SQLITE_TRANSIENT);
    int step = sqlite3_step(statement);
    sqlite3_finalize(statement);
    return step == SQLITE_DONE && sqlite3_changes(queue->db) == 1 ? 1 : step == SQLITE_DONE ? 0 : -1;
}

int omni_job_finish(OmniQueue *queue, const char *key, const char *worker,
                    const char *result_uri) {
    return settle(queue, key, worker, "succeeded", result_uri, "");
}

int omni_job_fail(OmniQueue *queue, const char *key, const char *worker,
                  const char *error, int retry) {
    return settle(queue, key, worker, retry ? "queued" : "failed", "", error);
}

int omni_job_retry(OmniQueue *queue, const char *key) {
    sqlite3_stmt *statement = NULL;
    if (queue == NULL || queue->db == NULL || !valid_key(key)) return -1;
    if (sqlite3_prepare_v2(queue->db,
        "UPDATE jobs SET state='queued',error='',worker='',gpu='',lease_expires=0,updated_unix=? "
        "WHERE job_key=? AND state='failed'", -1, &statement, NULL) != SQLITE_OK) return -1;
    sqlite3_bind_int64(statement, 1, now_unix());
    sqlite3_bind_text(statement, 2, key, -1, SQLITE_TRANSIENT);
    int step = sqlite3_step(statement);
    sqlite3_finalize(statement);
    return step == SQLITE_DONE && sqlite3_changes(queue->db) == 1 ? 1 : step == SQLITE_DONE ? 0 : -1;
}
