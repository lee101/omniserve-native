#include "admission.h"
#include "breaker.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

enum { MAX_MODELS = 64, BUFFER_SIZE = 65536, URL_SIZE = 512, TOKEN_SIZE = 256 };

typedef struct {
    char local_url[URL_SIZE];
    char remote_url[URL_SIZE];
    char remote_bearer[URL_SIZE];
    char frontdoor_token[TOKEN_SIZE];
    CircuitBreaker local_breaker;
    CircuitBreaker remote_breaker;
} Settings;

static const char *PAGE =
"<!doctype html><html><head><meta charset=\"utf-8\"><title>OmniServe Native</title>"
"<style>body{font-family:system-ui;margin:3rem;max-width:720px;color:#182026}h1{margin-bottom:.2rem}section{border-top:1px solid #ccd6dd;padding:1.25rem 0}select,button{font:inherit;padding:.5rem}button{margin-left:.5rem}#status{white-space:pre-wrap}</style>"
"</head><body><h1>OmniServe Native</h1><p>GPU-safe local model front door</p><section><strong id=status>Loading...</strong></section><section><label for=model>Model</label> <select id=model></select><button id=load>Admit model</button></section>"
"<script>const status=document.querySelector('#status'),model=document.querySelector('#model'),load=document.querySelector('#load');async function refresh(){let r=await fetch('/v1/status'),s=await r.json();status.textContent=s.message+'\\nVRAM: '+(s.gpu.available?s.gpu.used_mib+'/'+s.gpu.total_mib+' MiB used; '+s.gpu.available_mib+' MiB admissible':'unavailable')+'\\nLocal circuit: '+s.circuits.local+'; remote circuit: '+s.circuits.remote;model.innerHTML=s.models.map(m=>`<option value=\"${m.id}\">${m.label} (${m.required_mib} MiB)</option>`).join('')}load.onclick=async()=>{let r=await fetch('/v1/models/select',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'model='+encodeURIComponent(model.value)}),x=await r.json();status.textContent=x.message;refresh()};refresh();</script></body></html>";

static void copy_env(char *out, size_t size, const char *name) {
    const char *value = getenv(name);
    if (value != NULL) snprintf(out, size, "%s", value);
}

static unsigned env_unsigned(const char *name, unsigned fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long parsed = value == NULL ? 0 : strtoul(value, &end, 10);
    return value == NULL || *value == '\0' || end == NULL || *end != '\0' || parsed > 3600 ? fallback : (unsigned)parsed;
}

static int write_all(int fd, const char *data, size_t length) {
    while (length != 0) {
        ssize_t written = write(fd, data, length);
        if (written < 0) { if (errno == EINTR) continue; return -1; }
        data += written;
        length -= (size_t)written;
    }
    return 0;
}

static const char *status_text(int status) {
    switch (status) { case 200: return "OK"; case 400: return "Bad Request"; case 401: return "Unauthorized"; case 404: return "Not Found"; case 413: return "Payload Too Large"; case 502: return "Bad Gateway"; default: return "Service Unavailable"; }
}

static void send_response(int fd, int status, const char *type, const char *body) {
    char header[256];
    int length = snprintf(header, sizeof(header), "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", status, status_text(status), type, strlen(body));
    if (length > 0 && (size_t)length < sizeof(header) && write_all(fd, header, (size_t)length) == 0) (void)write_all(fd, body, strlen(body));
}

static void write_status_json(char *out, size_t size, const Model *models, size_t count, const char *active, const Settings *settings) {
    GpuSnapshot gpu = {0};
    (void)read_gpu_snapshot(&gpu);
    unsigned admissible = gpu.available && gpu.total_mib >= gpu.used_mib + OMNISERVE_DEFAULT_RESERVE_MIB ? gpu.total_mib - gpu.used_mib - OMNISERVE_DEFAULT_RESERVE_MIB : 0;
    time_t now = time(NULL);
    int written = snprintf(out, size, "{\"message\":\"%s\",\"active_model\":\"%s\",\"gpu\":{\"available\":%s,\"total_mib\":%u,\"used_mib\":%u,\"available_mib\":%u},\"circuits\":{\"local\":\"%s\",\"remote\":\"%s\"},\"models\":[", gpu.available ? "GPU telemetry available" : "GPU telemetry unavailable; local admission disabled", active, gpu.available ? "true" : "false", gpu.total_mib, gpu.used_mib, admissible, breaker_state(&settings->local_breaker, now), breaker_state(&settings->remote_breaker, now));
    for (size_t i = 0; i < count && written > 0 && (size_t)written < size; ++i) written += snprintf(out + written, size - (size_t)written, "%s{\"id\":\"%s\",\"label\":\"%s\",\"required_mib\":%u}", i ? "," : "", models[i].id, models[i].label, models[i].weights_mib + models[i].runtime_mib);
    if (written > 0 && (size_t)written < size) (void)snprintf(out + written, size - (size_t)written, "]}");
}

static int proxy_chat(const char *url, const char *bearer, const char *request_body, char *response, size_t response_size, int *http_status) {
    int input[2], output[2];
    if (pipe(input) != 0 || pipe(output) != 0) return -1;
    pid_t child = fork();
    if (child == 0) {
        char auth[URL_SIZE + 32] = {0};
        char *args[] = { "curl", "--silent", "--show-error", "--max-time", "60", "--request", "POST", "--header", "Content-Type: application/json", "--data-binary", "@-", "--write-out", "\n__OMNISERVE_STATUS__%{http_code}", (char *)url, NULL, NULL, NULL };
        if (bearer[0] != '\0') { snprintf(auth, sizeof(auth), "Authorization: Bearer %s", bearer); args[13] = "--header"; args[14] = auth; args[15] = (char *)url; args[16] = NULL; }
        (void)dup2(input[0], STDIN_FILENO); (void)dup2(output[1], STDOUT_FILENO);
        close(input[0]); close(input[1]); close(output[0]); close(output[1]);
        execlp("curl", args[0], args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], (char *)NULL);
        _exit(127);
    }
    close(input[0]); close(output[1]);
    if (child < 0 || write_all(input[1], request_body, strlen(request_body)) != 0) { close(input[1]); close(output[0]); return -1; }
    close(input[1]);
    size_t used = 0;
    while (used + 1 < response_size) { ssize_t count = read(output[0], response + used, response_size - used - 1); if (count <= 0) break; used += (size_t)count; }
    close(output[0]); response[used] = '\0';
    int wait_status = 0; if (waitpid(child, &wait_status, 0) < 0 || !WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) return -1;
    char *marker = strrchr(response, '\n');
    if (marker == NULL || strncmp(marker, "\n__OMNISERVE_STATUS__", 20) != 0) return -1;
    *http_status = atoi(marker + 20); *marker = '\0';
    return *http_status >= 200 && *http_status < 500 ? 0 : -1;
}

static bool authorised(const char *request, const Settings *settings) {
    if (settings->frontdoor_token[0] == '\0') return false;
    char expected[TOKEN_SIZE + 32];
    snprintf(expected, sizeof(expected), "Authorization: Bearer %s", settings->frontdoor_token);
    return strstr(request, expected) != NULL;
}

/* Returns 0 for a complete request, -1 for malformed input, -2 when too large. */
static int read_http_request(int fd, char *request, size_t size) {
    size_t used = 0, required = 0;
    for (;;) {
        if (used + 1 == size) return -2;
        ssize_t received = read(fd, request + used, size - used - 1);
        if (received <= 0) return -1;
        used += (size_t)received;
        request[used] = '\0';
        char *body = strstr(request, "\r\n\r\n");
        if (body == NULL) continue;
        if (required == 0) {
            char *length = strstr(request, "\r\nContent-Length:");
            unsigned long content_length = 0;
            if (length != NULL) {
                char *end = NULL;
                content_length = strtoul(length + 17, &end, 10);
                if (end == length + 17 || content_length > size - (size_t)(body + 4 - request) - 1) return -2;
            }
            required = (size_t)(body + 4 - request) + (size_t)content_length;
        }
        if (used >= required) { request[required] = '\0'; return 0; }
    }
}

static bool local_ready(const Model *models, size_t count, const char *active) {
    const Model *model = find_model(models, count, active); GpuSnapshot gpu = {0};
    if (model == NULL || read_gpu_snapshot(&gpu) != 0) return false;
    return admit_model(&gpu, model, OMNISERVE_DEFAULT_RESERVE_MIB).allowed;
}

static void handle_chat(int fd, const char *request, const Model *models, size_t count, const char *active, Settings *settings) {
    char *body = strstr(request, "\r\n\r\n"); char response[BUFFER_SIZE] = {0}; int upstream_status = 0; time_t now = time(NULL);
    if (!authorised(request, settings)) { send_response(fd, 401, "application/json", "{\"error\":\"front door token required\"}"); return; }
    if (body == NULL) { send_response(fd, 400, "application/json", "{\"error\":\"missing request body\"}"); return; }
    body += 4;
    if (settings->local_url[0] != '\0' && breaker_allows(&settings->local_breaker, now) && local_ready(models, count, active)) {
        if (proxy_chat(settings->local_url, "", body, response, sizeof(response), &upstream_status) == 0) { breaker_record_success(&settings->local_breaker); send_response(fd, upstream_status, "application/json", response); return; }
        breaker_record_failure(&settings->local_breaker, now);
    }
    if (settings->remote_url[0] != '\0' && breaker_allows(&settings->remote_breaker, now)) {
        if (proxy_chat(settings->remote_url, settings->remote_bearer, body, response, sizeof(response), &upstream_status) == 0) { breaker_record_success(&settings->remote_breaker); send_response(fd, upstream_status, "application/json", response); return; }
        breaker_record_failure(&settings->remote_breaker, now);
    }
    send_response(fd, 503, "application/json", "{\"error\":\"no healthy AI upstream is available\"}");
}

static void handle_client(int fd, const Model *models, size_t count, char *active, Settings *settings) {
    char request[BUFFER_SIZE] = {0}, body[1024] = {0}; int request_status = read_http_request(fd, request, sizeof(request));
    if (request_status == -2) { send_response(fd, 413, "application/json", "{\"error\":\"request exceeds 64 KiB limit\"}"); return; }
    if (request_status != 0) { send_response(fd, 400, "application/json", "{\"error\":\"malformed request\"}"); return; }
    if (strncmp(request, "GET / ", 7) == 0) { send_response(fd, 200, "text/html; charset=utf-8", PAGE); return; }
    if (strncmp(request, "GET /healthz ", 13) == 0) { send_response(fd, 200, "application/json", "{\"status\":\"ok\"}"); return; }
    if (strncmp(request, "GET /v1/status ", 15) == 0) { write_status_json(body, sizeof(body), models, count, active, settings); send_response(fd, 200, "application/json", body); return; }
    if (strncmp(request, "POST /v1/chat/completions ", 26) == 0) { handle_chat(fd, request, models, count, active, settings); return; }
    if (strncmp(request, "POST /v1/models/select ", 23) == 0) {
        char *model_id = strstr(request, "\r\n\r\nmodel="); const Model *model = model_id ? find_model(models, count, model_id + 10) : NULL; GpuSnapshot gpu = {0}; Admission decision;
        (void)read_gpu_snapshot(&gpu);
        if (model == NULL) { send_response(fd, 400, "application/json", "{\"message\":\"unknown model\"}"); return; }
        decision = admit_model(&gpu, model, OMNISERVE_DEFAULT_RESERVE_MIB);
        if (!decision.allowed) { snprintf(body, sizeof(body), "{\"message\":\"%s\"}", decision.reason); send_response(fd, 503, "application/json", body); return; }
        snprintf(active, 64, "%s", model->id); snprintf(body, sizeof(body), "{\"message\":\"%s selected: %s\"}", model->label, decision.reason); send_response(fd, 200, "application/json", body); return;
    }
    send_response(fd, 404, "application/json", "{\"error\":\"not found\"}");
}

int main(int argc, char **argv) {
    const char *models_path = "models/models.csv", *listen_address = "127.0.0.1";
    unsigned port = 8080;
    Model models[MAX_MODELS]; size_t count = 0; char active[64] = ""; Settings settings = {0}; int listener, option = 1;
    struct sockaddr_in address = { .sin_family = AF_INET };
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) { fprintf(stderr, "usage: omniserve [--models PATH] [--listen ADDRESS] [--port 1024-65535]\n"); return 2; }
        if (strcmp(argv[index], "--models") == 0) models_path = argv[index + 1];
        else if (strcmp(argv[index], "--listen") == 0) listen_address = argv[index + 1];
        else if (strcmp(argv[index], "--port") == 0) {
            char tail = 0;
            if (sscanf(argv[index + 1], "%u%c", &port, &tail) != 1 || port < 1024 || port > 65535) {
                fprintf(stderr, "invalid port: %s\n", argv[index + 1]); return 2;
            }
        } else { fprintf(stderr, "unknown option: %s\n", argv[index]); return 2; }
    }
    address.sin_port = htons((uint16_t)port);
    copy_env(settings.local_url, sizeof(settings.local_url), "OMNISERVE_LOCAL_UPSTREAM"); copy_env(settings.remote_url, sizeof(settings.remote_url), "OMNISERVE_REMOTE_UPSTREAM"); copy_env(settings.remote_bearer, sizeof(settings.remote_bearer), "OMNISERVE_REMOTE_BEARER_TOKEN"); copy_env(settings.frontdoor_token, sizeof(settings.frontdoor_token), "OMNISERVE_FRONTDOOR_TOKEN");
    breaker_init(&settings.local_breaker, env_unsigned("OMNISERVE_CIRCUIT_FAILURES", 3), env_unsigned("OMNISERVE_CIRCUIT_COOLDOWN_SECONDS", 30)); breaker_init(&settings.remote_breaker, env_unsigned("OMNISERVE_CIRCUIT_FAILURES", 3), env_unsigned("OMNISERVE_CIRCUIT_COOLDOWN_SECONDS", 30));
    if (load_models(models_path, models, MAX_MODELS, &count) != 0) { fprintf(stderr, "cannot load model catalog: %s\n", models_path); return 1; }
    if (inet_pton(AF_INET, listen_address, &address.sin_addr) != 1) { fprintf(stderr, "--listen must be an IPv4 address\n"); return 2; }
    signal(SIGPIPE, SIG_IGN); listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0 || setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option)) != 0 || bind(listener, (struct sockaddr *)&address, sizeof(address)) != 0 || listen(listener, 16) != 0) { perror("server setup"); return 1; }
    printf("OmniServe listening on http://%s:%u\n", listen_address, port);
    for (;;) { int client = accept(listener, NULL, NULL); if (client >= 0) { handle_client(client, models, count, active, &settings); close(client); } else if (errno != EINTR) perror("accept"); }
}
