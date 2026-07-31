#define _GNU_SOURCE
#include "ohost.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Small enough that the floor is rechecked often during a multi-gigabyte file,
 * large enough that the syscall overhead stays irrelevant next to the I/O. */
#define OHOST_CHUNK_BYTES (64LL * 1024 * 1024)

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static ohost_stats g_stats;

bool ohost_read_meminfo(ohost_meminfo *out) {
    if (!out) return false;
    memset(out, 0, sizeof *out);
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return false;
    char line[256];
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        long value = 0;
        if (sscanf(line, "MemTotal: %ld kB", &value) == 1) { out->mem_total_kb = value; found++; }
        else if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) { out->mem_available_kb = value; found++; }
        else if (sscanf(line, "SwapTotal: %ld kB", &value) == 1) { out->swap_total_kb = value; }
        else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) { out->swap_free_kb = value; }
    }
    fclose(f);
    return found >= 2;
}

long long ohost_headroom_bytes(const ohost_meminfo *mi, int keep_free_pct) {
    if (!mi || mi->mem_total_kb <= 0) return 0;
    if (keep_free_pct < 0) keep_free_pct = 0;
    if (keep_free_pct > 90) keep_free_pct = 90;
    long long floor_kb = (long long)mi->mem_total_kb * keep_free_pct / 100;
    long long spare_kb = (long long)mi->mem_available_kb - floor_kb;
    return spare_kb > 0 ? spare_kb * 1024 : 0;
}

long long ohost_plan_bytes(long long file_size, long long headroom_bytes) {
    if (file_size <= 0 || headroom_bytes <= 0) return 0;
    return file_size < headroom_bytes ? file_size : headroom_bytes;
}

static void warm_one(const char *path, int keep_free_pct) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        pthread_mutex_lock(&g_lock);
        g_stats.files_skipped++;
        pthread_mutex_unlock(&g_lock);
        return;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        pthread_mutex_lock(&g_lock);
        g_stats.files_skipped++;
        pthread_mutex_unlock(&g_lock);
        return;
    }

    ohost_meminfo mi;
    long long budget = ohost_read_meminfo(&mi)
                           ? ohost_plan_bytes(st.st_size, ohost_headroom_bytes(&mi, keep_free_pct))
                           : 0;
    if (budget <= 0) {
        close(fd);
        pthread_mutex_lock(&g_lock);
        g_stats.files_skipped++;
        pthread_mutex_unlock(&g_lock);
        return;
    }

    long long done = 0;
    bool cut_short = false;
    while (done < budget) {
        long long chunk = budget - done;
        if (chunk > OHOST_CHUNK_BYTES) chunk = OHOST_CHUNK_BYTES;
        if (posix_fadvise(fd, (off_t)done, (off_t)chunk, POSIX_FADV_WILLNEED) != 0) break;
        done += chunk;

        /* Recheck rather than trusting the budget computed before any of this
         * landed: another tenant may have taken the headroom in the meantime,
         * and continuing would be this module causing the pressure it exists
         * to avoid. */
        if (done < budget && ohost_read_meminfo(&mi)
            && ohost_headroom_bytes(&mi, keep_free_pct) <= 0) {
            cut_short = true;
            break;
        }
    }
    close(fd);

    pthread_mutex_lock(&g_lock);
    g_stats.bytes_requested += done;
    g_stats.files_done++;
    if (cut_short) g_stats.stops_for_pressure++;
    pthread_mutex_unlock(&g_lock);
}

typedef struct {
    char *paths;
    int keep_free_pct;
} warm_job;

static void *warm_thread(void *user) {
    warm_job *job = user;
    for (char *save = NULL, *tok = strtok_r(job->paths, ",", &save); tok;
         tok = strtok_r(NULL, ",", &save)) {
        while (*tok == ' ') tok++;
        if (*tok) warm_one(tok, job->keep_free_pct);
    }
    ohost_meminfo mi;
    if (ohost_read_meminfo(&mi)) {
        pthread_mutex_lock(&g_lock);
        g_stats.mem_available_kb = mi.mem_available_kb;
        g_stats.mem_total_kb = mi.mem_total_kb;
        pthread_mutex_unlock(&g_lock);
    }
    free(job->paths);
    free(job);
    return NULL;
}

bool ohost_prefetch_start(const char *paths, int keep_free_pct) {
    if (!paths || !paths[0]) return false;
    warm_job *job = calloc(1, sizeof *job);
    if (!job) return false;
    job->paths = strdup(paths);
    job->keep_free_pct = keep_free_pct;
    if (!job->paths) { free(job); return false; }

    pthread_mutex_lock(&g_lock);
    g_stats.enabled = true;
    g_stats.keep_free_pct = keep_free_pct;
    pthread_mutex_unlock(&g_lock);

    pthread_t tid;
    if (pthread_create(&tid, NULL, warm_thread, job) != 0) {
        free(job->paths);
        free(job);
        return false;
    }
    pthread_detach(tid);
    return true;
}

void ohost_snapshot(ohost_stats *out) {
    if (!out) return;
    pthread_mutex_lock(&g_lock);
    *out = g_stats;
    pthread_mutex_unlock(&g_lock);
    ohost_meminfo mi;
    if (ohost_read_meminfo(&mi)) {
        out->mem_available_kb = mi.mem_available_kb;
        out->mem_total_kb = mi.mem_total_kb;
    }
}

size_t ohost_status_json(char *out, size_t cap) {
    if (!out || cap == 0) return 0;
    ohost_stats s;
    ohost_snapshot(&s);
    int written = snprintf(out, cap,
        "{\"enabled\":%s,\"keep_free_pct\":%d,\"bytes_requested\":%lld,"
        "\"files_done\":%d,\"files_skipped\":%d,\"stops_for_pressure\":%d,"
        "\"mem_total_mb\":%ld,\"mem_available_mb\":%ld}",
        s.enabled ? "true" : "false", s.keep_free_pct, s.bytes_requested,
        s.files_done, s.files_skipped, s.stops_for_pressure,
        s.mem_total_kb / 1024, s.mem_available_kb / 1024);
    if (written < 0) return 0;
    return (size_t)written < cap ? (size_t)written : cap - 1;
}
