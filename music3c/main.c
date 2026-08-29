#include "music3.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

enum {
    MUSIC3_INLINE_LIMIT = 8 << 20,
    MUSIC3_RESULT_SIZE = 16 << 20,
    MUSIC3_MAX_SERVE_ARGS = 64,
    MUSIC3_MAX_PREFETCH_FILES = 512,
    MUSIC3_PREFETCH_CHUNK = 8 << 20
};

static pid_t g_server_pid = -1;
static double g_server_started_at;
static const char *g_model_id;
static const char *g_model_dir;
static const char *g_port;
static pthread_mutex_t g_server_mutex = PTHREAD_MUTEX_INITIALIZER;
static char g_gpu_name[128];
static double g_prefetch_seconds;
static double g_prefetch_gib;
static int g_server_ready_before_job;

static const char *env_str(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return value && value[0] ? value : fallback;
}

static int env_int(const char *name, int fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') return fallback;
    return atoi(value);
}

static double monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static double wall_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static int read_file(const char *path, unsigned char **data, size_t *length) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return -1;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return -1; }
    long size = ftell(file);
    if (size < 0) { fclose(file); return -1; }
    rewind(file);
    unsigned char *buffer = malloc((size_t)size + 1);
    if (buffer == NULL) { fclose(file); return -1; }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) { free(buffer); fclose(file); return -1; }
    fclose(file);
    buffer[size] = '\0';
    *data = buffer;
    *length = (size_t)size;
    return 0;
}

static int write_file(const char *path, const void *data, size_t length) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) return -1;
    size_t written = fwrite(data, 1, length, file);
    fclose(file);
    return written == length ? 0 : -1;
}

static int run_logged(char **argv, const char *log_path) {
    pid_t child = fork();
    if (child == 0) {
        int fd = log_path ? open(log_path, O_CREAT | O_WRONLY | O_APPEND, 0644) : -1;
        if (fd >= 0) { dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); close(fd); }
        execvp(argv[0], argv);
        _exit(127);
    }
    if (child < 0) return -1;
    int status = 0;
    if (waitpid(child, &status, 0) < 0) return -1;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1;
}

static int curl_to_file(char **args) {
    pid_t child = fork();
    if (child == 0) { execvp("curl", args); _exit(127); }
    if (child < 0) return -1;
    int status = 0;
    if (waitpid(child, &status, 0) < 0) return -1;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : -1;
}

static int health(void) {
    char url[128];
    snprintf(url, sizeof(url), "http://127.0.0.1:%s/health", g_port);
    char *args[] = {"curl", "--silent", "--fail", "--max-time", "2", url, "-o", "/dev/null", NULL};
    return curl_to_file(args) == 0;
}

static int file_size_equals(const char *path, off_t expected) {
    struct stat info;
    return stat(path, &info) == 0 && info.st_size == expected;
}

static int ensure_model(double *download_seconds) {
    *download_seconds = 0;
    char marker[1024], flowmatching[1024];
    snprintf(marker, sizeof(marker), "%s/.omniserve-ready-v3", g_model_dir);
    snprintf(flowmatching, sizeof(flowmatching), "%s/flowmatching_vae.pth", g_model_dir);
    if (access(marker, F_OK) == 0) return 0;
    double started = monotonic_seconds();
    mkdir(g_model_dir, 0755);
    const off_t expected = 9828468476LL;
    const char *want_sha = "941f3ed9591684679e733d184308be89949abeb1b069a6e17e69a013ecec08fe";
    if (file_size_equals(flowmatching, expected)) {
        char digest[MUSIC3_SHA_SIZE];
        if (music3_sha256_file(flowmatching, digest) == 0 && strcmp(digest, want_sha) == 0) {
            /* intact checkpoint; still need the rest of the snapshot if marker is missing */
        } else if (env_int("MUSIC3_FORCE_REDOWNLOAD", 0) != 1) {
            fprintf(stderr, "Music3 checkpoint hash mismatch; set MUSIC3_FORCE_REDOWNLOAD=1 to replace the 9.8GiB file\n");
            return -1;
        } else unlink(flowmatching);
    }
    if (!file_size_equals(flowmatching, expected)) {
        char *download[] = {
            "huggingface-cli", "download", (char *)g_model_id, "flowmatching_vae.pth",
            "--local-dir", (char *)g_model_dir, NULL
        };
        if (run_logged(download, env_str("MUSIC3_SERVER_LOG", "/runpod-volume/omniserve/music3/server.log")) != 0)
            return -1;
        char digest[MUSIC3_SHA_SIZE];
        if (!file_size_equals(flowmatching, expected) || music3_sha256_file(flowmatching, digest) != 0 ||
            strcmp(digest, want_sha) != 0) {
            fprintf(stderr, "Music3 flow-matching checkpoint failed integrity verification\n");
            return -1;
        }
    }
    /* sgl-omni serves from qwen_7B/, flowmatching_vae.pth and dav.pth; the
     * diffusers-layout copies of the same weights are another 26 GiB that the
     * runtime never opens, so the slim snapshot halves the cold download. */
    char *snapshot[] = {
        "huggingface-cli", "download", (char *)g_model_id, "--local-dir", (char *)g_model_dir,
        "--max-workers", (char *)env_str("MUSIC3_DOWNLOAD_WORKERS", "16"),
        "--include", "qwen_7B/*", "dav.pth", "*.json", "*.txt", NULL, NULL
    };
    if (env_int("MUSIC3_FULL_SNAPSHOT", 0) == 1) snapshot[7] = NULL;
    if (run_logged(snapshot, env_str("MUSIC3_SERVER_LOG", "/runpod-volume/omniserve/music3/server.log")) != 0)
        return -1;
    char payload[256];
    snprintf(payload, sizeof(payload), "{\"model\":\"%s\",\"flowmatching_sha256\":\"%s\"}\n", g_model_id, want_sha);
    if (write_file(marker, payload, strlen(payload)) != 0) return -1;
    *download_seconds = monotonic_seconds() - started;
    return 0;
}

/* Network-volume reads are latency bound, so a single reader leaves most of the
 * link idle. Warming the page cache from several threads makes the serial torch
 * load that follows hit memory instead of the volume. */
typedef struct {
    char paths[MUSIC3_MAX_PREFETCH_FILES][1024];
    off_t sizes[MUSIC3_MAX_PREFETCH_FILES];
    int count;
    int next;
    off_t budget;
    off_t consumed;
    pthread_mutex_t mutex;
} Music3Prefetch;

static void prefetch_collect(Music3Prefetch *plan, const char *directory, int depth) {
    if (depth > 3 || plan->count >= MUSIC3_MAX_PREFETCH_FILES) return;
    DIR *handle = opendir(directory);
    if (handle == NULL) return;
    const char *include = env_str("MUSIC3_PREFETCH_INCLUDE", "qwen_7B,flowmatching_vae.pth,dav.pth");
    struct dirent *entry;
    while ((entry = readdir(handle)) != NULL && plan->count < MUSIC3_MAX_PREFETCH_FILES) {
        if (entry->d_name[0] == '.') continue;
        /* Only the weights the runtime actually opens are worth page cache. */
        if (depth == 0 && !music3_name_included(entry->d_name, include)) continue;
        char path[1024];
        if (snprintf(path, sizeof(path), "%s/%s", directory, entry->d_name) >= (int)sizeof(path)) continue;
        struct stat info;
        if (stat(path, &info) != 0) continue;
        if (S_ISDIR(info.st_mode)) { prefetch_collect(plan, path, depth + 1); continue; }
        if (!S_ISREG(info.st_mode) || info.st_size < (64 << 20)) continue;
        snprintf(plan->paths[plan->count], sizeof(plan->paths[0]), "%s", path);
        plan->sizes[plan->count] = info.st_size;
        plan->count++;
    }
    closedir(handle);
}

static void *prefetch_worker(void *argument) {
    Music3Prefetch *plan = argument;
    unsigned char *buffer = malloc(MUSIC3_PREFETCH_CHUNK);
    if (buffer == NULL) return NULL;
    for (;;) {
        pthread_mutex_lock(&plan->mutex);
        int index = plan->next;
        if (index >= plan->count || plan->consumed >= plan->budget) {
            pthread_mutex_unlock(&plan->mutex);
            break;
        }
        plan->next++;
        plan->consumed += plan->sizes[index];
        pthread_mutex_unlock(&plan->mutex);
        int fd = open(plan->paths[index], O_RDONLY);
        if (fd < 0) continue;
        posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
        while (read(fd, buffer, MUSIC3_PREFETCH_CHUNK) > 0) continue;
        close(fd);
    }
    free(buffer);
    return NULL;
}

static off_t prefetch_budget_bytes(void) {
    off_t configured = (off_t)env_int("MUSIC3_PREFETCH_MAX_GIB", 0) << 30;
    off_t available = 0;
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (meminfo != NULL) {
        char line[256];
        while (fgets(line, sizeof(line), meminfo) != NULL) {
            long kilobytes = 0;
            if (sscanf(line, "MemAvailable: %ld kB", &kilobytes) == 1) {
                available = (off_t)kilobytes * 1024;
                break;
            }
        }
        fclose(meminfo);
    }
    /* Leave headroom: the loader itself needs host memory for the weights. */
    off_t safe = available > 0 ? available / 2 : (off_t)32 << 30;
    if (configured > 0 && configured < safe) return configured;
    return safe;
}

static void prefetch_model(void) {
    int threads = env_int("MUSIC3_PREFETCH_THREADS", 8);
    if (threads < 1) return;
    if (threads > 32) threads = 32;
    Music3Prefetch *plan = calloc(1, sizeof(*plan));
    if (plan == NULL) return;
    pthread_mutex_init(&plan->mutex, NULL);
    plan->budget = prefetch_budget_bytes();
    prefetch_collect(plan, g_model_dir, 0);
    if (plan->count == 0 || plan->budget <= 0) { free(plan); return; }
    double started = monotonic_seconds();
    pthread_t workers[32];
    int spawned = 0;
    for (int i = 0; i < threads && i < plan->count; ++i)
        if (pthread_create(&workers[spawned], NULL, prefetch_worker, plan) == 0) spawned++;
    for (int i = 0; i < spawned; ++i) pthread_join(workers[i], NULL);
    g_prefetch_seconds = monotonic_seconds() - started;
    g_prefetch_gib = (double)plan->consumed / (double)(1 << 30);
    fprintf(stderr, "Music3 prefetched %.2f GiB in %.1fs with %d threads\n",
            g_prefetch_gib, g_prefetch_seconds, spawned);
    pthread_mutex_destroy(&plan->mutex);
    free(plan);
}

/* Space-separated extra `sgl-omni serve` flags, so acoustic dtype, attention
 * backend and solver steps are tunable per endpoint without a rebuild. */
static int append_extra_serve_args(char **args, int used, char *scratch, size_t scratch_size) {
    const char *extra = getenv("MUSIC3_SERVE_EXTRA_ARGS");
    if (extra == NULL || extra[0] == '\0') return used;
    snprintf(scratch, scratch_size, "%s", extra);
    char *save = NULL;
    for (char *token = strtok_r(scratch, " \t", &save);
         token != NULL && used < MUSIC3_MAX_SERVE_ARGS - 1;
         token = strtok_r(NULL, " \t", &save))
        args[used++] = token;
    return used;
}

/* Cost per track depends on which card the worker landed on, so the result
 * carries the GPU the platform actually gave us. */
static void detect_gpu(void) {
    if (g_gpu_name[0] != '\0') return;
    FILE *pipe = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
    if (pipe == NULL) return;
    if (fgets(g_gpu_name, sizeof(g_gpu_name), pipe) != NULL) {
        size_t length = strlen(g_gpu_name);
        while (length > 0 && (g_gpu_name[length - 1] == '\n' || g_gpu_name[length - 1] == '\r'))
            g_gpu_name[--length] = '\0';
    }
    pclose(pipe);
}

static int start_server_locked(double *download_seconds, double *start_seconds) {
    *start_seconds = 0;
    if (g_server_pid > 0 && kill(g_server_pid, 0) == 0 && health()) {
        *download_seconds = 0;
        return 0;
    }
    if (ensure_model(download_seconds) != 0) return -1;
    detect_gpu();
    mkdir(env_str("TORCHINDUCTOR_CACHE_DIR", "/tmp/music3-torchinductor"), 0755);
    prefetch_model();
    double started = monotonic_seconds();
    pid_t child = fork();
    if (child == 0) {
        /* The image WORKDIR ships its own sglang_omni checkout; CWD would
         * shadow the pinned runtime on the volume via sys.path[0]. */
        if (chdir("/") != 0) _exit(126);
        /* Checkpoint is fully local at this point; skip hub probes on boot
         * unless the operator explicitly overrides. */
        setenv("HF_HUB_OFFLINE", "1", 0);
        setenv("TRANSFORMERS_OFFLINE", "1", 0);
        const char *log_path = env_str("MUSIC3_SERVER_LOG", "/runpod-volume/omniserve/music3/server.log");
        int fd = open(log_path, O_CREAT | O_WRONLY | O_APPEND, 0644);
        if (fd >= 0) { dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); close(fd); }
        char scratch[1024];
        char *args[MUSIC3_MAX_SERVE_ARGS] = {0};
        int used = 0;
        /* Running the module directly lets a worker serve straight from the
         * runtime on the volume, with no per-boot editable install. */
        if (env_int("MUSIC3_SERVE_PYTHON_MODULE", 0) == 1) {
            args[used++] = "python3";
            args[used++] = "-m";
            args[used++] = "sglang_omni.cli";
        } else args[used++] = "sgl-omni";
        args[used++] = "serve";
        args[used++] = "--model-path";
        args[used++] = (char *)g_model_dir;
        args[used++] = "--host";
        args[used++] = "127.0.0.1";
        args[used++] = "--port";
        args[used++] = (char *)g_port;
        args[used++] = "--max-running-requests";
        args[used++] = (char *)env_str("MUSIC3_MAX_RUNNING_REQUESTS", "1");
        args[used++] = "--stages.dit_dav.factory-args.dtype";
        args[used++] = (char *)env_str("MUSIC3_ACOUSTIC_DTYPE", "bfloat16");
        used = append_extra_serve_args(args, used, scratch, sizeof(scratch));
        args[used] = NULL;
        if (getenv("MUSIC3_SERVE_DIAG")) {
            FILE *dbg = fopen("/runpod-volume/omniserve/music3/worker-diag.txt", "a");
            if (dbg) {
                fprintf(dbg, "--- serve boot ---\n");
                int saved_out = dup(STDOUT_FILENO);
                int saved_err = dup(STDERR_FILENO);
                dup2(fileno(dbg), STDOUT_FILENO);
                dup2(fileno(dbg), STDERR_FILENO);
                (void)!system("hostname; pwd; env | sort | grep -E 'PYTHONPATH|HOSTNAME'; ls /runpod-volume/ 2>&1 | head -5; git -C /runpod-volume/omniserve/music3/sglang-omni-e0c98529 rev-parse HEAD 2>&1; python3 -c 'import sglang_omni; print(sglang_omni.__file__)' 2>&1");
                for (int i = 0; args[i]; ++i) fprintf(dbg, "ARG[%d]=%s\n", i, args[i]);
                fflush(dbg);
                dup2(saved_out, STDOUT_FILENO);
                dup2(saved_err, STDERR_FILENO);
                fclose(dbg);
            }
        }
        execvp(args[0], args);
        _exit(127);
    }
    if (child < 0) return -1;
    g_server_pid = child;
    int timeout = env_int("MUSIC3_STARTUP_TIMEOUT_SECONDS", 1800);
    while (monotonic_seconds() - started < timeout) {
        int status = 0;
        pid_t ended = waitpid(child, &status, WNOHANG);
        if (ended == child) { g_server_pid = -1; return -1; }
        if (health()) {
            g_server_started_at = wall_seconds();
            *start_seconds = monotonic_seconds() - started;
            return 0;
        }
        sleep(2);
    }
    kill(child, SIGTERM);
    waitpid(child, NULL, 0);
    g_server_pid = -1;
    return -1;
}

static int start_server(double *download_seconds, double *start_seconds) {
    pthread_mutex_lock(&g_server_mutex);
    int status = start_server_locked(download_seconds, start_seconds);
    pthread_mutex_unlock(&g_server_mutex);
    return status;
}

/* Load the checkpoint while the worker is still idle rather than inside the
 * first request, so a warm-started worker answers at generation speed. */
static void *warm_start_worker(void *unused) {
    (void)unused;
    double download = 0, start = 0;
    if (start_server(&download, &start) == 0)
        fprintf(stderr, "Music3 warm start ready in %.1fs (download %.1fs)\n", start, download);
    else
        fprintf(stderr, "Music3 warm start failed; the first request will retry\n");
    return NULL;
}

static void start_warm_thread(void) {
    if (env_int("MUSIC3_WARM_START", 1) != 1) return;
    pthread_t thread;
    if (pthread_create(&thread, NULL, warm_start_worker, NULL) == 0) pthread_detach(thread);
}

static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static char *base64_encode(const unsigned char *data, size_t length) {
    size_t out_len = 4 * ((length + 2) / 3);
    char *out = malloc(out_len + 1);
    if (out == NULL) return NULL;
    size_t i = 0, j = 0;
    while (i < length) {
        unsigned octet_a = i < length ? data[i++] : 0;
        unsigned octet_b = i < length ? data[i++] : 0;
        unsigned octet_c = i < length ? data[i++] : 0;
        unsigned triple = (octet_a << 16) + (octet_b << 8) + octet_c;
        out[j++] = b64[(triple >> 18) & 63];
        out[j++] = b64[(triple >> 12) & 63];
        out[j++] = b64[(triple >> 6) & 63];
        out[j++] = b64[triple & 63];
    }
    int pad = (int)((3 - (length % 3)) % 3);
    for (int p = 0; p < pad; ++p) out[out_len - 1 - p] = '=';
    out[out_len] = '\0';
    return out;
}

static int json_escape_file(FILE *file, const char *text) {
    fputc('"', file);
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') { fputc('\\', file); fputc(*p, file); }
        else if (*p == '\n') fputs("\\n", file);
        else if (*p == '\r') fputs("\\r", file);
        else if (*p < 32) fprintf(file, "\\u%04x", *p);
        else fputc(*p, file);
    }
    fputc('"', file);
    return 0;
}

static int generate_audio(const Music3Request *request, unsigned char **audio, size_t *length, double *seconds) {
    char payload_path[] = "/tmp/music3-payload-XXXXXX";
    char audio_path[] = "/tmp/music3-audio-XXXXXX";
    int payload_fd = mkstemp(payload_path), audio_fd = mkstemp(audio_path);
    if (payload_fd < 0 || audio_fd < 0) return -1;
    close(audio_fd);
    FILE *payload = fdopen(payload_fd, "w");
    if (payload == NULL) return -1;
    fprintf(payload, "{\"model\":"); json_escape_file(payload, g_model_id);
    fprintf(payload, ",\"input\":"); json_escape_file(payload, request->lyrics);
    fprintf(payload, ",\"instructions\":"); json_escape_file(payload, request->instructions);
    fprintf(payload, ",\"response_format\":\"wav\",\"seed\":%lld,\"max_new_tokens\":%d,\"stream\":false}",
            request->seed, request->max_new_tokens);
    fclose(payload);
    char url[128], timeout[16];
    snprintf(url, sizeof(url), "http://127.0.0.1:%s/v1/audio/speech", g_port);
    snprintf(timeout, sizeof(timeout), "%d", env_int("MUSIC3_REQUEST_TIMEOUT_SECONDS", 1800));
    double started = monotonic_seconds();
    char *args[] = {
        "curl", "--silent", "--show-error", "--fail", "--max-time", timeout, "--request", "POST",
        "--header", "Content-Type: application/json", "--data-binary", payload_path, "-o", audio_path, url, NULL
    };
    /* curl --data-binary @file */
    char at_path[64];
    snprintf(at_path, sizeof(at_path), "@%s", payload_path);
    args[11] = at_path;
    int ok = curl_to_file(args);
    unlink(payload_path);
    *seconds = monotonic_seconds() - started;
    if (ok != 0) { unlink(audio_path); return -1; }
    int loaded = read_file(audio_path, audio, length);
    unlink(audio_path);
    return loaded;
}

static int upload_audio(const char *url, const unsigned char *audio, size_t length, double *seconds) {
    char path[] = "/tmp/music3-upload-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    if ((size_t)write(fd, audio, length) != length) { close(fd); unlink(path); return -1; }
    close(fd);
    char at_path[64];
    snprintf(at_path, sizeof(at_path), "@%s", path);
    double started = monotonic_seconds();
    char *args[] = {
        "curl", "--silent", "--show-error", "--fail", "--max-time", "300", "--request", "PUT",
        "--header", "Content-Type: audio/wav", "--data-binary", at_path, (char *)url, "-o", "/dev/null", NULL
    };
    int ok = curl_to_file(args);
    unlink(path);
    *seconds = monotonic_seconds() - started;
    return ok;
}

static int handle_job_json(const char *json, char **result_json) {
    Music3Request request = {0};
    if (music3_normalize_request(json, &request) != 0) {
        char *error = malloc(MUSIC3_ERROR_SIZE + 32);
        if (error == NULL) return -1;
        snprintf(error, MUSIC3_ERROR_SIZE + 32, "{\"error\":\"%s\"}", request.error);
        *result_json = error;
        return 400;
    }
    double total_started = monotonic_seconds(), download = 0, start = 0, generation = 0, upload = 0;
    long long original_seed = request.seed;
    int quality_attempts = 1;
    g_server_ready_before_job = g_server_pid > 0 && kill(g_server_pid, 0) == 0 && health();
    if (start_server(&download, &start) != 0) {
        *result_json = strdup("{\"error\":\"MiniMax-Music3 server failed to start\"}");
        return 503;
    }
    unsigned char *audio = NULL;
    size_t audio_length = 0;
    if (generate_audio(&request, &audio, &audio_length, &generation) != 0) {
        *result_json = strdup("{\"error\":\"MiniMax-Music3 returned empty audio\"}");
        return 502;
    }
    Music3WavStats stats = {0};
    if (music3_wav_statistics(audio, audio_length, &stats) != 0) {
        free(audio);
        *result_json = strdup("{\"error\":\"generated WAV is invalid\"}");
        return 502;
    }
    int retries = env_int("MUSIC3_QUALITY_RETRIES", 1);
    double threshold = atof(env_str("MUSIC3_CONTINUITY_THRESHOLD", "75"));
    for (int retry = 0; retry < retries && stats.continuity_score < threshold; ++retry) {
        Music3Request candidate_request = request;
        candidate_request.seed = original_seed <= LLONG_MAX - retry - 1 ? original_seed + retry + 1 : retry;
        unsigned char *candidate_audio = NULL;
        size_t candidate_length = 0;
        double candidate_seconds = 0;
        if (generate_audio(&candidate_request, &candidate_audio, &candidate_length, &candidate_seconds) != 0)
            break;
        generation += candidate_seconds;
        quality_attempts++;
        Music3WavStats candidate_stats = {0};
        if (music3_wav_statistics(candidate_audio, candidate_length, &candidate_stats) == 0 &&
            candidate_stats.continuity_score > stats.continuity_score) {
            free(audio);
            audio = candidate_audio;
            audio_length = candidate_length;
            stats = candidate_stats;
            request = candidate_request;
        } else free(candidate_audio);
    }
    char *inline_b64 = NULL;
    const char *audio_url = NULL;
    if (request.output_upload_url[0]) {
        if (upload_audio(request.output_upload_url, audio, audio_length, &upload) != 0) {
            free(audio);
            *result_json = strdup("{\"error\":\"audio upload failed\"}");
            return 502;
        }
        audio_url = request.output_public_url;
    } else {
        if (audio_length > (size_t)env_int("MUSIC3_MAX_INLINE_BYTES", MUSIC3_INLINE_LIMIT)) {
            free(audio);
            *result_json = strdup("{\"error\":\"output_upload_url is required for audio larger than inline limit\"}");
            return 400;
        }
        inline_b64 = base64_encode(audio, audio_length);
        if (inline_b64 == NULL) { free(audio); return -1; }
    }
    char *out = malloc(MUSIC3_RESULT_SIZE);
    if (out == NULL) { free(audio); free(inline_b64); return -1; }
    Music3Timings timings = {
        .model_download_seconds = download, .server_start_seconds = start,
        .generation_seconds = generation, .upload_seconds = upload,
        .total_seconds = monotonic_seconds() - total_started, .server_started_at = g_server_started_at,
        .prefetch_seconds = g_prefetch_seconds, .prefetch_gib = g_prefetch_gib,
        .server_ready_before_job = g_server_ready_before_job, .gpu_name = g_gpu_name,
        .quality_attempts = quality_attempts, .original_seed = original_seed,
    };
    if (music3_write_result_json(out, MUSIC3_RESULT_SIZE, &request, &stats, audio_url, inline_b64,
                                 &timings) != 0) {
        free(out); free(audio); free(inline_b64);
        *result_json = strdup("{\"error\":\"result encoding failed\"}");
        return 500;
    }
    free(audio);
    free(inline_b64);
    *result_json = out;
    return 200;
}

static int replace_id(const char *template, const char *id, char *out, size_t size) {
    const char *found = strstr(template, "$ID");
    if (found == NULL) return snprintf(out, size, "%s", template) > 0 && (size_t)snprintf(out, size, "%s", template) < size ? 0 : -1;
    if ((size_t)(found - template) + strlen(id) + strlen(found + 3) + 1 > size) return -1;
    memcpy(out, template, (size_t)(found - template));
    out[found - template] = '\0';
    strcat(out, id);
    strcat(out, found + 3);
    return 0;
}

/* RunPod's worker queue rejects an unauthenticated job-take with a 401 body
 * that otherwise looks like a job, so every control-plane call carries the
 * worker key the platform injects into the container. */
static const char *worker_auth_header(char *buffer, size_t size) {
    const char *key = getenv("RUNPOD_AI_API_KEY");
    if (key == NULL || key[0] == '\0') key = getenv("RUNPOD_API_KEY");
    if (key == NULL || key[0] == '\0') return NULL;
    snprintf(buffer, size, "Authorization: %s", key);
    return buffer;
}

static char *curl_get(const char *url) {
    char path[] = "/tmp/music3-http-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return NULL;
    close(fd);
    char auth[1024];
    const char *auth_header = worker_auth_header(auth, sizeof(auth));
    char *args[] = {
        "curl", "--silent", "--show-error", "--max-time", "30", (char *)url, "-o", path,
        NULL, NULL, NULL
    };
    if (auth_header != NULL) { args[8] = "--header"; args[9] = (char *)auth_header; }
    if (curl_to_file(args) != 0) { unlink(path); return NULL; }
    unsigned char *data = NULL;
    size_t length = 0;
    if (read_file(path, &data, &length) != 0) { unlink(path); return NULL; }
    unlink(path);
    return (char *)data;
}

static int curl_post_json(const char *url, const char *json) {
    char path[] = "/tmp/music3-post-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    if ((size_t)write(fd, json, strlen(json)) != strlen(json)) { close(fd); unlink(path); return -1; }
    close(fd);
    char at_path[64];
    snprintf(at_path, sizeof(at_path), "@%s", path);
    char auth[1024];
    const char *auth_header = worker_auth_header(auth, sizeof(auth));
    char *args[] = {
        "curl", "--silent", "--show-error", "--fail", "--max-time", "30", "--request", "POST",
        "--header", "Content-Type: application/json", "--data-binary", at_path, (char *)url, "-o", "/dev/null",
        NULL, NULL, NULL
    };
    if (auth_header != NULL) { args[15] = "--header"; args[16] = (char *)auth_header; }
    int ok = curl_to_file(args);
    unlink(path);
    return ok;
}

static char *extract_job_id(const char *json) {
    const char *key = strstr(json, "\"id\"");
    if (key == NULL) return NULL;
    const char *colon = strchr(key, ':');
    if (colon == NULL) return NULL;
    while (*colon && *colon != '"') ++colon;
    if (*colon != '"') return NULL;
    const char *start = colon + 1;
    const char *end = strchr(start, '"');
    if (end == NULL) return NULL;
    size_t length = (size_t)(end - start);
    char *id = malloc(length + 1);
    if (id == NULL) return NULL;
    memcpy(id, start, length);
    id[length] = '\0';
    return id;
}

static void append_query(char *out, size_t size, const char *base, const char *query) {
    const char *sep = strchr(base, '?') ? "&" : "?";
    snprintf(out, size, "%s%s%s", base, sep, query);
}

/* RunPod expires a job whose worker stops naming it in the ping, and a music
 * render runs for minutes, so the heartbeat has to carry the id of the job
 * currently being generated rather than an empty one. */
static pthread_mutex_t g_active_job_mutex = PTHREAD_MUTEX_INITIALIZER;
static char g_active_job_id[256];

static void set_active_job(const char *id) {
    pthread_mutex_lock(&g_active_job_mutex);
    if (id == NULL) g_active_job_id[0] = '\0';
    else snprintf(g_active_job_id, sizeof(g_active_job_id), "%s", id);
    pthread_mutex_unlock(&g_active_job_mutex);
}

static void *heartbeat_worker(void *argument) {
    const char *ping = argument;
    /* RunPod publishes this interval in milliseconds (10000 by default); read
     * as seconds it becomes a three-hour sleep and the platform reclaims the
     * job mid-render as a dead worker. */
    int interval = env_int("RUNPOD_PING_INTERVAL", 10000) / 1000;
    if (interval < 1) interval = 1;
    if (interval > 30) interval = 30;
    for (;;) {
        char job_id[256], query[320], url[2400];
        pthread_mutex_lock(&g_active_job_mutex);
        snprintf(job_id, sizeof(job_id), "%s", g_active_job_id);
        pthread_mutex_unlock(&g_active_job_mutex);
        snprintf(query, sizeof(query), "job_id=%s&retry_ping=0", job_id);
        append_query(url, sizeof(url), ping, query);
        char *ignored = curl_get(url);
        free(ignored);
        sleep((unsigned)interval);
    }
    return NULL;
}

static void start_heartbeat(void) {
    const char *ping = getenv("RUNPOD_WEBHOOK_PING");
    if (ping == NULL || ping[0] == '\0') return;
    pthread_t thread;
    if (pthread_create(&thread, NULL, heartbeat_worker, (void *)ping) == 0) pthread_detach(thread);
}

static int runpod_loop(void) {
    const char *get_url = getenv("RUNPOD_WEBHOOK_GET_JOB");
    const char *post_url = getenv("RUNPOD_WEBHOOK_POST_OUTPUT");
    if (get_url == NULL || get_url[0] == '\0' || post_url == NULL || post_url[0] == '\0') return 2;
    start_heartbeat();
    start_warm_thread();
    fprintf(stderr, "music3c polling for jobs\n");
    fflush(stderr);
    long empty_polls = 0;
    for (;;) {
        char take[2048];
        append_query(take, sizeof(take), get_url, "job_in_progress=0");
        char *job = curl_get(take);
        if (job == NULL || job[0] == '\0' || strcmp(job, "[]") == 0 || strcmp(job, "null") == 0) {
            /* A silent worker with a queued job is otherwise impossible to
             * tell apart from a crashed one, so idle polling stays visible. */
            if (++empty_polls % 60 == 0) {
                fprintf(stderr, "music3c idle: %ld empty polls, last fetch %s\n",
                        empty_polls, job == NULL ? "failed" : "empty");
                fflush(stderr);
            }
            free(job);
            sleep(1);
            continue;
        }
        char *id = extract_job_id(job);
        if (id == NULL) {
            /* Error envelopes (auth, throttling) also come back 200; treat
             * anything without a job id as idle instead of spinning on it. */
            if (++empty_polls % 60 == 0) {
                fprintf(stderr, "music3c cannot read a job id from: %.400s\n", job);
                fflush(stderr);
            }
            free(job);
            sleep(1);
            continue;
        }
        empty_polls = 0;
        fprintf(stderr, "music3c took job %s (%zu bytes)\n", id, strlen(job));
        fflush(stderr);
        set_active_job(id);
        char *result = NULL;
        int status = handle_job_json(job, &result);
        free(job);
        set_active_job(NULL);
        if (id != NULL && result != NULL) {
            char url[4096];
            /* A result envelope carries inline base64 audio, so it is far past
             * what the thread stack can hold and has to live on the heap. */
            size_t envelope_size = MUSIC3_RESULT_SIZE + 64;
            char *envelope = malloc(envelope_size);
            replace_id(post_url, id, url, sizeof(url));
            if (envelope != NULL) {
                if (status >= 400) snprintf(envelope, envelope_size, "%s", result);
                else snprintf(envelope, envelope_size, "{\"output\":%s}", result);
                (void)curl_post_json(url, envelope);
                free(envelope);
            }
        }
        free(id);
        free(result);
    }
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    g_model_id = env_str("MUSIC3_MODEL_ID", "MiniMaxAI/MiniMax-Music3");
    g_model_dir = env_str("MUSIC3_MODEL_DIR", "/runpod-volume/models/minimax-music3");
    g_port = env_str("MUSIC3_PORT", "8000");
    if (argc == 3 && strcmp(argv[1], "--job-file") == 0) {
        unsigned char *json = NULL;
        size_t length = 0;
        if (read_file(argv[2], &json, &length) != 0) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        char *result = NULL;
        int status = handle_job_json((char *)json, &result);
        free(json);
        if (result) { fputs(result, stdout); fputc('\n', stdout); free(result); }
        return status >= 400 ? 1 : 0;
    }
    /* Boot checks the binary loads in this image before handing it the job
     * loop, which the plain no-argument form would enter and never leave. */
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return 0;
    if (argc == 1 && getenv("RUNPOD_WEBHOOK_GET_JOB") != NULL) return runpod_loop();
    fprintf(stderr, "usage: music3c --job-file PATH\n");
    return 2;
}
