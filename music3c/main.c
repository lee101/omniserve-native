#include "music3.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

enum { MUSIC3_INLINE_LIMIT = 8 << 20, MUSIC3_RESULT_SIZE = 16 << 20 };

static pid_t g_server_pid = -1;
static double g_server_started_at;
static const char *g_model_id;
static const char *g_model_dir;
static const char *g_port;

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
    char *snapshot[] = {
        "huggingface-cli", "download", (char *)g_model_id, "--local-dir", (char *)g_model_dir, NULL
    };
    if (run_logged(snapshot, env_str("MUSIC3_SERVER_LOG", "/runpod-volume/omniserve/music3/server.log")) != 0)
        return -1;
    char payload[256];
    snprintf(payload, sizeof(payload), "{\"model\":\"%s\",\"flowmatching_sha256\":\"%s\"}\n", g_model_id, want_sha);
    if (write_file(marker, payload, strlen(payload)) != 0) return -1;
    *download_seconds = monotonic_seconds() - started;
    return 0;
}

static int start_server(double *download_seconds, double *start_seconds) {
    *start_seconds = 0;
    if (g_server_pid > 0 && kill(g_server_pid, 0) == 0 && health()) {
        *download_seconds = 0;
        return 0;
    }
    if (ensure_model(download_seconds) != 0) return -1;
    mkdir(env_str("TORCHINDUCTOR_CACHE_DIR", "/tmp/music3-torchinductor"), 0755);
    double started = monotonic_seconds();
    pid_t child = fork();
    if (child == 0) {
        const char *log_path = env_str("MUSIC3_SERVER_LOG", "/runpod-volume/omniserve/music3/server.log");
        int fd = open(log_path, O_CREAT | O_WRONLY | O_APPEND, 0644);
        if (fd >= 0) { dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); close(fd); }
        char *args[] = {
            "sgl-omni", "serve", "--model-path", (char *)g_model_dir, "--host", "127.0.0.1",
            "--port", (char *)g_port, "--max-running-requests", (char *)env_str("MUSIC3_MAX_RUNNING_REQUESTS", "1"),
            "--stages.dit_dav.factory-args.dtype", (char *)env_str("MUSIC3_ACOUSTIC_DTYPE", "bfloat16"),
            NULL
        };
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
    if (music3_write_result_json(out, MUSIC3_RESULT_SIZE, &request, &stats, audio_url, inline_b64,
                                 download, start, generation, upload, monotonic_seconds() - total_started,
                                 g_server_started_at) != 0) {
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

static char *curl_get(const char *url) {
    char path[] = "/tmp/music3-http-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return NULL;
    close(fd);
    char *args[] = {"curl", "--silent", "--show-error", "--max-time", "30", (char *)url, "-o", path, NULL};
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
    char *args[] = {
        "curl", "--silent", "--show-error", "--fail", "--max-time", "30", "--request", "POST",
        "--header", "Content-Type: application/json", "--data-binary", at_path, (char *)url, "-o", "/dev/null", NULL
    };
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

static void start_heartbeat(void) {
    const char *ping = getenv("RUNPOD_WEBHOOK_PING");
    if (ping == NULL || ping[0] == '\0') return;
    int interval = env_int("RUNPOD_PING_INTERVAL", 10);
    if (interval < 1) interval = 10;
    pid_t child = fork();
    if (child != 0) return;
    for (;;) {
        char url[2048];
        append_query(url, sizeof(url), ping, "job_id=&retry_ping=0");
        char *ignored = curl_get(url);
        free(ignored);
        sleep((unsigned)interval);
    }
}

static int runpod_loop(void) {
    const char *get_url = getenv("RUNPOD_WEBHOOK_GET_JOB");
    const char *post_url = getenv("RUNPOD_WEBHOOK_POST_OUTPUT");
    if (get_url == NULL || get_url[0] == '\0' || post_url == NULL || post_url[0] == '\0') return 2;
    start_heartbeat();
    for (;;) {
        char take[2048];
        append_query(take, sizeof(take), get_url, "job_in_progress=0");
        char *job = curl_get(take);
        if (job == NULL || job[0] == '\0' || strcmp(job, "[]") == 0 || strcmp(job, "null") == 0) {
            free(job);
            sleep(1);
            continue;
        }
        char *id = extract_job_id(job);
        char *result = NULL;
        int status = handle_job_json(job, &result);
        free(job);
        if (id != NULL && result != NULL) {
            char url[4096], envelope[MUSIC3_RESULT_SIZE + 64];
            replace_id(post_url, id, url, sizeof(url));
            if (status >= 400) snprintf(envelope, sizeof(envelope), "%s", result);
            else snprintf(envelope, sizeof(envelope), "{\"output\":%s}", result);
            (void)curl_post_json(url, envelope);
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
    if (argc == 1 && getenv("RUNPOD_WEBHOOK_GET_JOB") != NULL) return runpod_loop();
    fprintf(stderr, "usage: music3c --job-file PATH\n");
    return 2;
}
