#include "admission.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

enum { MAX_MODELS = 64, BUFFER_SIZE = 16384 };

static const char *PAGE =
"<!doctype html><html><head><meta charset=\"utf-8\"><title>OmniServe Native</title>"
"<style>body{font-family:system-ui;margin:3rem;max-width:720px;color:#182026}"
"h1{margin-bottom:.2rem}section{border-top:1px solid #ccd6dd;padding:1.25rem 0}"
"select,button{font:inherit;padding:.5rem}button{margin-left:.5rem}#status{white-space:pre-wrap}</style>"
"</head><body><h1>OmniServe Native</h1><p>GPU-safe local model control plane</p>"
"<section><strong id=status>Loading GPU telemetry...</strong></section><section>"
"<label for=model>Model</label> <select id=model></select><button id=load>Admit model</button>"
"</section><script>async function refresh(){let r=await fetch('/v1/status');let s=await r.json();"
"status.textContent=s.message+'\\nVRAM: '+(s.gpu.available?s.gpu.used_mib+'/'+s.gpu.total_mib+' MiB used; '+s.gpu.available_mib+' MiB admissible':'unavailable');"
"model.innerHTML=s.models.map(m=>`<option value=\"${m.id}\">${m.label} (${m.required_mib} MiB)</option>`).join('')}"
"load.onclick=async()=>{let r=await fetch('/v1/models/select',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'model='+encodeURIComponent(model.value)});let x=await r.json();status.textContent=x.message;refresh()};refresh();</script></body></html>";

static int write_all(int fd, const char *data, size_t length) {
    while (length != 0) {
        ssize_t written = write(fd, data, length);
        if (written < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        data += written;
        length -= (size_t)written;
    }
    return 0;
}

static void send_response(int fd, int status, const char *type, const char *body) {
    char header[256];
    int length = snprintf(header, sizeof(header), "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", status, status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found" : "Service Unavailable", type, strlen(body));
    if (write_all(fd, header, (size_t)length) != 0) return;
    (void)write_all(fd, body, strlen(body));
}

static void write_status_json(char *out, size_t size, const Model *models, size_t count, const char *active) {
    GpuSnapshot gpu = {0};
    (void)read_gpu_snapshot(&gpu);
    unsigned admissible = gpu.available && gpu.total_mib >= gpu.used_mib + OMNISERVE_DEFAULT_RESERVE_MIB ? gpu.total_mib - gpu.used_mib - OMNISERVE_DEFAULT_RESERVE_MIB : 0;
    int written = snprintf(out, size, "{\"message\":\"%s\",\"active_model\":\"%s\",\"gpu\":{\"available\":%s,\"total_mib\":%u,\"used_mib\":%u,\"available_mib\":%u},\"models\":[", gpu.available ? "GPU telemetry available" : "GPU telemetry unavailable; admission disabled", active, gpu.available ? "true" : "false", gpu.total_mib, gpu.used_mib, admissible);
    for (size_t i = 0; i < count && written > 0 && (size_t)written < size; ++i)
        written += snprintf(out + written, size - (size_t)written, "%s{\"id\":\"%s\",\"label\":\"%s\",\"required_mib\":%u}", i ? "," : "", models[i].id, models[i].label, models[i].weights_mib + models[i].runtime_mib);
    if (written > 0 && (size_t)written < size) (void)snprintf(out + written, size - (size_t)written, "]}");
}

static void handle_client(int fd, const Model *models, size_t count, char *active) {
    char request[BUFFER_SIZE] = {0}, body[BUFFER_SIZE] = {0};
    ssize_t received = read(fd, request, sizeof(request) - 1);
    if (received <= 0) return;
    if (strncmp(request, "GET / ", 7) == 0) { send_response(fd, 200, "text/html; charset=utf-8", PAGE); return; }
    if (strncmp(request, "GET /v1/status ", 15) == 0) { write_status_json(body, sizeof(body), models, count, active); send_response(fd, 200, "application/json", body); return; }
    if (strncmp(request, "POST /v1/models/select ", 23) == 0) {
        char *model_id = strstr(request, "\r\n\r\nmodel=");
        const Model *model = model_id ? find_model(models, count, model_id + 10) : NULL;
        GpuSnapshot gpu = {0}; Admission decision;
        (void)read_gpu_snapshot(&gpu);
        if (model == NULL) { send_response(fd, 400, "application/json", "{\"message\":\"unknown model\"}"); return; }
        decision = admit_model(&gpu, model, OMNISERVE_DEFAULT_RESERVE_MIB);
        if (!decision.allowed) { snprintf(body, sizeof(body), "{\"message\":\"%s\"}", decision.reason); send_response(fd, 503, "application/json", body); return; }
        snprintf(active, 64, "%s", model->id);
        snprintf(body, sizeof(body), "{\"message\":\"%s selected: %s\"}", model->label, decision.reason);
        send_response(fd, 200, "application/json", body); return;
    }
    send_response(fd, 404, "text/plain", "not found\n");
}

int main(int argc, char **argv) {
    const char *models_path = "models/models.csv";
    Model models[MAX_MODELS]; size_t count = 0; char active[64] = "";
    int listener, option = 1; struct sockaddr_in address = { .sin_family = AF_INET, .sin_port = htons(8080), .sin_addr.s_addr = htonl(INADDR_LOOPBACK) };
    if (argc == 3 && strcmp(argv[1], "--models") == 0) models_path = argv[2];
    if (load_models(models_path, models, MAX_MODELS, &count) != 0) { fprintf(stderr, "cannot load model catalog: %s\n", models_path); return 1; }
    signal(SIGPIPE, SIG_IGN);
    listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0 || setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option)) != 0 || bind(listener, (struct sockaddr *)&address, sizeof(address)) != 0 || listen(listener, 16) != 0) { perror("server setup"); return 1; }
    printf("OmniServe listening on http://127.0.0.1:8080\n");
    for (;;) { int client = accept(listener, NULL, NULL); if (client >= 0) { handle_client(client, models, count, active); close(client); } else if (errno != EINTR) perror("accept"); }
}
