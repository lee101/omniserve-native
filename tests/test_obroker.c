/* obroker_lease: a grant is only a grant when it carries a usable lease id. */
#include "obroker.h"

#include <arpa/inet.h>
#include <assert.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

const char *otier_name(otier t) { (void)t; return "test"; } /* avoid linking osched */

static int g_listen;
static const char *g_lease_body;
static int g_releases;
static char g_release_body[256];
static int g_conns;

static void *serve(void *arg) {
    (void)arg;
    for (int i = 0; i < g_conns; i++) {
        int c = accept(g_listen, NULL, NULL);
        if (c < 0) break;
        char req[2048] = {0};
        ssize_t n = recv(c, req, sizeof req - 1, 0);
        (void)n;
        const char *body = "{}";
        if (strstr(req, "POST /v1/gpu/release")) {
            g_releases++;
            const char *b = strstr(req, "\r\n\r\n");
            snprintf(g_release_body, sizeof g_release_body, "%s", b ? b + 4 : "");
            body = "{\"released\":true}";
        } else {
            body = g_lease_body;
        }
        char out[1024];
        int len = snprintf(out, sizeof out, "HTTP/1.0 200 OK\r\nContent-Length: %zu\r\n\r\n%s",
                           strlen(body), body);
        send(c, out, (size_t)len, 0);
        close(c);
    }
    return NULL;
}

/* Runs one lease call against a canned broker reply; returns the result. */
static int run(const char *lease_body, int expect_conns, char *id, size_t cap) {
    g_listen = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(bind(g_listen, (struct sockaddr *)&a, sizeof a) == 0);
    assert(listen(g_listen, 4) == 0);
    socklen_t al = sizeof a;
    getsockname(g_listen, (struct sockaddr *)&a, &al);
    char url[64];
    snprintf(url, sizeof url, "http://127.0.0.1:%d", ntohs(a.sin_port));
    g_lease_body = lease_body;
    g_releases = 0;
    g_release_body[0] = 0;
    g_conns = expect_conns;
    pthread_t t;
    pthread_create(&t, NULL, serve, NULL);
    int waited = 0;
    int r = obroker_lease(url, "test", 1, 1000, 500, (otier)0, 30.0, 0, false, id, cap, &waited);
    pthread_join(t, NULL);
    close(g_listen);
    return r;
}

int main(void) {
    char id[64];

    assert(run("{\"granted\":true,\"mb\":1000,\"lease_id\":\"abc123\"}", 1, id, sizeof id) == 1000);
    assert(strcmp(id, "abc123") == 0 && g_releases == 0);

    /* granted but no lease_id at all */
    assert(run("{\"granted\":true,\"mb\":1000}", 1, id, sizeof id) == 0);
    assert(id[0] == 0 && g_releases == 0);

    /* empty lease id */
    assert(run("{\"granted\":true,\"mb\":1000,\"lease_id\":\"\"}", 1, id, sizeof id) == 0);
    assert(g_releases == 0);

    /* truncated: no closing quote -> not granted, nothing safe to release */
    assert(run("{\"granted\":true,\"mb\":1000,\"lease_id\":\"abc", 1, id, sizeof id) == 0);
    assert(id[0] == 0 && g_releases == 0);

    /* complete id that does not fit the caller's buffer: not granted, released whole */
    char tiny[4];
    assert(run("{\"granted\":true,\"mb\":1000,\"lease_id\":\"abcdefgh\"}", 2, tiny, sizeof tiny) == 0);
    assert(tiny[0] == 0 && g_releases == 1);
    assert(strstr(g_release_body, "abcdefgh") != NULL);

    /* not granted stays 0 */
    assert(run("{\"granted\":false}", 1, id, sizeof id) == 0);

    puts("obroker tests passed");
    return 0;
}
