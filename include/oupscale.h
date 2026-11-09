#ifndef OUPSCALE_H
#define OUPSCALE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct sockaddr;

typedef enum { OUP_GENERAL = 0, OUP_ANIME = 1 } oup_model;
typedef enum { OUP_PNG = 0, OUP_WEBP = 1, OUP_JPEG = 2 } oup_format;

#define OUP_MAX_IN_PX_X2 4194304LL
#define OUP_MAX_IN_PX_X4 2100000LL
#define OUP_MAX_OUT_SIDE 8192
#define OUP_MAX_FETCH_BYTES (48u << 20)

typedef struct {
    char *image_url;
    int scale;
    oup_model model;
    oup_format format;
} oup_req;

bool oup_parse(const char *json, size_t len, oup_req *req, char *err, size_t err_cap);
void oup_req_free(oup_req *req);
const char *oup_model_name(oup_model m);
const char *oup_format_mime(oup_format f);

int oup_check_limits(int w, int h, int scale, char *err, size_t err_cap);

bool oup_ip_public(const struct sockaddr *sa);
bool oup_url_https(const char *url);

bool oup_probe_dims(const unsigned char *b, size_t n, int *w, int *h);
bool oup_data_url_decode(const char *url, unsigned char **out, size_t *out_len);
int oup_fetch(const char *url, unsigned char **out, size_t *out_len, char *err, size_t err_cap);

void oup_box_down2(const unsigned char *src, int w, int h, int ch, unsigned char *dst);
void oup_resize_plane_bilinear(const unsigned char *src, int sw, int sh, int sstride,
                               unsigned char *dst, int dw, int dh, int dstride);

typedef struct {
    unsigned char *data;
    size_t len;
    int width, height, scale;
    const char *model;
    double ms;
} oup_result;

typedef enum {
    OUP_OK = 0,
    OUP_ERR_INPUT,
    OUP_ERR_TOO_LARGE,
    OUP_ERR_UNAVAILABLE,
    OUP_ERR_INTERNAL
} oup_status;

bool oup_backend_available(oup_model m, int scale);
int oup_backend_lease_mb(int w, int h, oup_model m, int scale);
void oup_backend_preload(void);
oup_status oup_backend_run(const unsigned char *bytes, size_t len, const oup_req *req,
                           oup_result *out, char *err, size_t err_cap);
void oup_result_free(oup_result *res);
size_t oup_backend_status_json(char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif
