#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define RESPONSE_HEADER_MAX (64u << 10)

typedef struct {
    uint16_t port;
    const char *request;
    size_t request_len;
    uint64_t *latencies_ns;
    size_t offset;
    size_t count;
    pthread_barrier_t *ready;
    atomic_size_t *errors;
} bench_worker;

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int connect_local(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&addr, sizeof addr) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static bool write_all(int fd, const char *data, size_t len) {
    while (len) {
        ssize_t n = send(fd, data, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        data += n;
        len -= (size_t)n;
    }
    return true;
}

static const char *find_header(const char *headers, const char *name) {
    size_t name_len = strlen(name);
    const char *line = strstr(headers, "\r\n");
    if (!line) return NULL;
    line += 2;
    while (*line && line[0] != '\r') {
        const char *end = strstr(line, "\r\n");
        if (!end) return NULL;
        if ((size_t)(end - line) > name_len &&
            line[name_len] == ':' && strncasecmp(line, name, name_len) == 0) {
            const char *value = line + name_len + 1;
            while (value < end && (*value == ' ' || *value == '\t')) value++;
            return value;
        }
        line = end + 2;
    }
    return NULL;
}

static bool read_response(int fd, bool *must_close) {
    char *buf = malloc(RESPONSE_HEADER_MAX + 1);
    if (!buf) return false;
    size_t len = 0;
    char *header_end = NULL;
    while (!header_end && len < RESPONSE_HEADER_MAX) {
        ssize_t n = recv(fd, buf + len, RESPONSE_HEADER_MAX - len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            free(buf);
            return false;
        }
        if (n == 0) {
            free(buf);
            return false;
        }
        len += (size_t)n;
        buf[len] = 0;
        header_end = memmem(buf, len, "\r\n\r\n", 4);
    }
    if (!header_end) {
        free(buf);
        return false;
    }

    size_t header_len = (size_t)(header_end - buf) + 4;
    const char *cl = find_header(buf, "Content-Length");
    const char *connection = find_header(buf, "Connection");
    *must_close = connection && strncasecmp(connection, "close", 5) == 0;
    if (!cl) {
        free(buf);
        return false;
    }
    errno = 0;
    char *number_end = NULL;
    unsigned long long body_len = strtoull(cl, &number_end, 10);
    if (errno || number_end == cl || body_len > SIZE_MAX) {
        free(buf);
        return false;
    }
    size_t have = len - header_len;
    while (have < (size_t)body_len) {
        char scratch[16384];
        size_t wanted = (size_t)body_len - have;
        if (wanted > sizeof scratch) wanted = sizeof scratch;
        ssize_t n = recv(fd, scratch, wanted, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            free(buf);
            return false;
        }
        if (n == 0) {
            free(buf);
            return false;
        }
        have += (size_t)n;
    }
    free(buf);
    return true;
}

static void *worker_main(void *arg) {
    bench_worker *worker = arg;
    int fd = connect_local(worker->port);
    pthread_barrier_wait(worker->ready);
    for (size_t i = 0; i < worker->count; i++) {
        if (fd < 0) fd = connect_local(worker->port);
        uint64_t start = monotonic_ns();
        bool close_after = false;
        bool ok = fd >= 0 && write_all(fd, worker->request, worker->request_len) &&
                  read_response(fd, &close_after);
        uint64_t end = monotonic_ns();
        worker->latencies_ns[worker->offset + i] = end - start;
        if (!ok) atomic_fetch_add_explicit(worker->errors, 1, memory_order_relaxed);
        if (!ok || close_after) {
            if (fd >= 0) close(fd);
            fd = -1;
        }
    }
    if (fd >= 0) close(fd);
    return NULL;
}

static int compare_u64(const void *a, const void *b) {
    uint64_t lhs = *(const uint64_t *)a;
    uint64_t rhs = *(const uint64_t *)b;
    return (lhs > rhs) - (lhs < rhs);
}

static double percentile_us(const uint64_t *values, size_t count, double p) {
    if (!count) return 0;
    size_t index = (size_t)(p * (double)(count - 1));
    return (double)values[index] / 1000.0;
}

static long parse_long(const char *text, long fallback) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    return errno || !end || *end || value <= 0 ? fallback : value;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s PORT REQUESTS CONCURRENCY [PATH [BODY]]\n", argv[0]);
        return 2;
    }
    long port_value = parse_long(argv[1], 8791);
    long request_value = parse_long(argv[2], 100000);
    long concurrency_value = parse_long(argv[3], 32);
    if (port_value > 65535 || concurrency_value > request_value || concurrency_value > 4096) {
        fprintf(stderr, "invalid benchmark arguments\n");
        return 2;
    }
    size_t requests = (size_t)request_value;
    size_t concurrency = (size_t)concurrency_value;
    const char *path = argc >= 5 ? argv[4] : "/health";
    const char *body = argc >= 6 ? argv[5] : NULL;

    size_t request_cap = strlen(path) + (body ? strlen(body) : 0) + 512;
    char *request = malloc(request_cap);
    if (!request) return 1;
    int request_len = body
        ? snprintf(request, request_cap,
                   "POST %s HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                   "Content-Length: %zu\r\nConnection: keep-alive\r\n\r\n%s",
                   path, strlen(body), body)
        : snprintf(request, request_cap,
                   "GET %s HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n",
                   path);
    if (request_len < 0 || (size_t)request_len >= request_cap) {
        free(request);
        return 1;
    }

    uint64_t *latencies = calloc(requests, sizeof *latencies);
    pthread_t *threads = calloc(concurrency, sizeof *threads);
    bench_worker *workers = calloc(concurrency, sizeof *workers);
    if (!latencies || !threads || !workers) {
        free(request);
        free(latencies);
        free(threads);
        free(workers);
        return 1;
    }

    pthread_barrier_t ready;
    pthread_barrier_init(&ready, NULL, (unsigned)concurrency + 1);
    atomic_size_t errors = 0;
    size_t offset = 0;
    for (size_t i = 0; i < concurrency; i++) {
        size_t count = requests / concurrency + (i < requests % concurrency);
        workers[i] = (bench_worker){
            .port = (uint16_t)port_value,
            .request = request,
            .request_len = (size_t)request_len,
            .latencies_ns = latencies,
            .offset = offset,
            .count = count,
            .ready = &ready,
            .errors = &errors,
        };
        offset += count;
        if (pthread_create(&threads[i], NULL, worker_main, &workers[i]) != 0) {
            fprintf(stderr, "could not start benchmark thread\n");
            return 1;
        }
    }

    pthread_barrier_wait(&ready);
    uint64_t started = monotonic_ns();
    for (size_t i = 0; i < concurrency; i++) pthread_join(threads[i], NULL);
    uint64_t elapsed = monotonic_ns() - started;
    qsort(latencies, requests, sizeof *latencies, compare_u64);
    long double sum = 0;
    for (size_t i = 0; i < requests; i++) sum += latencies[i];
    size_t error_count = atomic_load_explicit(&errors, memory_order_relaxed);
    printf("requests=%zu concurrency=%zu errors=%zu elapsed=%.3fs rate=%.0f req/s "
           "mean=%.1fus p50=%.1fus p95=%.1fus p99=%.1fus max=%.1fus\n",
           requests, concurrency, error_count, (double)elapsed / 1e9,
           (double)requests * 1e9 / (double)elapsed,
           (double)(sum / (long double)requests) / 1000.0,
           percentile_us(latencies, requests, 0.50),
           percentile_us(latencies, requests, 0.95),
           percentile_us(latencies, requests, 0.99),
           percentile_us(latencies, requests, 1.0));

    pthread_barrier_destroy(&ready);
    free(workers);
    free(threads);
    free(latencies);
    free(request);
    return error_count ? 1 : 0;
}
