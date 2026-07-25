/* CLI around omatte_estimate_fb so the pymatting eval harness can drive the C
 * (and CUDA) implementation directly:
 *
 *   omatte_cli --image img.npy --alpha alpha.npy --out-fg fg.npy [--out-bg bg.npy]
 *              [--order sequential|redblack] [--threads N] [--cuda]
 *
 * Only float32 C-order .npy files are supported, which is what the fixtures use.
 */
#include "omatte.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    float *data;
    int ndim;
    int shape[3];
    size_t count;
} npy_array;

static void npy_free(npy_array *array) {
    if (array) {
        free(array->data);
        array->data = NULL;
    }
}

static bool npy_load(const char *path, npy_array *out) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }

    unsigned char magic[10];
    if (fread(magic, 1, 10, file) != 10 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s is not a .npy file\n", path);
        fclose(file);
        return false;
    }

    size_t header_len = 0;
    if (magic[6] == 1) {
        header_len = (size_t)magic[8] | ((size_t)magic[9] << 8);
    } else {
        unsigned char extra[2];
        if (fread(extra, 1, 2, file) != 2) {
            fclose(file);
            return false;
        }
        header_len = (size_t)magic[8] | ((size_t)magic[9] << 8) | ((size_t)extra[0] << 16) |
                     ((size_t)extra[1] << 24);
    }

    char *header = calloc(header_len + 1, 1);
    if (!header || fread(header, 1, header_len, file) != header_len) {
        free(header);
        fclose(file);
        return false;
    }

    if (!strstr(header, "'<f4'") && !strstr(header, "\"<f4\"")) {
        fprintf(stderr, "%s must be float32 (found header: %s)\n", path, header);
        free(header);
        fclose(file);
        return false;
    }
    if (strstr(header, "'fortran_order': True")) {
        fprintf(stderr, "%s must be C-order\n", path);
        free(header);
        fclose(file);
        return false;
    }

    const char *shape = strstr(header, "'shape':");
    if (!shape || !(shape = strchr(shape, '('))) {
        free(header);
        fclose(file);
        return false;
    }

    out->ndim = 0;
    out->count = 1;
    const char *cursor = shape + 1;
    while (*cursor && *cursor != ')' && out->ndim < 3) {
        while (*cursor == ' ' || *cursor == ',') cursor++;
        if (*cursor == ')' || !*cursor) break;
        long value = strtol(cursor, (char **)&cursor, 10);
        out->shape[out->ndim++] = (int)value;
        out->count *= (size_t)value;
    }
    free(header);

    out->data = malloc(out->count * sizeof(float));
    if (!out->data || fread(out->data, sizeof(float), out->count, file) != out->count) {
        fprintf(stderr, "%s: truncated data\n", path);
        npy_free(out);
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

static bool npy_save(const char *path, const float *data, int ndim, const int *shape) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "cannot write %s\n", path);
        return false;
    }

    char dict[256];
    int len = 0;
    len += snprintf(dict + len, sizeof dict - len, "{'descr': '<f4', 'fortran_order': False, 'shape': (");
    size_t count = 1;
    for (int i = 0; i < ndim; i++) {
        len += snprintf(dict + len, sizeof dict - len, "%d,", shape[i]);
        count *= (size_t)shape[i];
    }
    len += snprintf(dict + len, sizeof dict - len, "), }");

    /* Header (10 bytes) + dict + padding must be a multiple of 64 bytes. */
    int total = 10 + len + 1;
    int padding = (64 - (total % 64)) % 64;

    unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
    const int header_len = len + padding + 1;
    magic[8] = (unsigned char)(header_len & 0xff);
    magic[9] = (unsigned char)((header_len >> 8) & 0xff);
    fwrite(magic, 1, 10, file);
    fwrite(dict, 1, (size_t)len, file);
    for (int i = 0; i < padding; i++) fputc(' ', file);
    fputc('\n', file);
    fwrite(data, sizeof(float), count, file);
    fclose(file);
    return true;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *image_path = NULL;
    const char *alpha_path = NULL;
    const char *out_fg = NULL;
    const char *out_bg = NULL;
    bool use_cuda = false;
    int repeat = 1;
    omatte_params params = omatte_default_params();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) image_path = argv[++i];
        else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) alpha_path = argv[++i];
        else if (strcmp(argv[i], "--out-fg") == 0 && i + 1 < argc) out_fg = argv[++i];
        else if (strcmp(argv[i], "--out-bg") == 0 && i + 1 < argc) out_bg = argv[++i];
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) params.threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (strcmp(argv[i], "--regularization") == 0 && i + 1 < argc) params.regularization = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--gradient-weight") == 0 && i + 1 < argc) params.gradient_weight = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--small-iterations") == 0 && i + 1 < argc) params.n_small_iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--big-iterations") == 0 && i + 1 < argc) params.n_big_iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "--small-size") == 0 && i + 1 < argc) params.small_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--order") == 0 && i + 1 < argc) {
            const char *order = argv[++i];
            if (strcmp(order, "redblack") == 0) params.order = OMATTE_ORDER_RED_BLACK;
            else if (strcmp(order, "sequential") == 0) params.order = OMATTE_ORDER_SEQUENTIAL;
            else { fprintf(stderr, "unknown order: %s\n", order); return 2; }
        } else if (strcmp(argv[i], "--cuda") == 0) {
            use_cuda = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            puts("usage: omatte_cli --image IMG.npy --alpha A.npy --out-fg FG.npy [--out-bg BG.npy]\n"
                 "                  [--order sequential|redblack] [--threads N] [--cuda]\n"
                 "                  [--regularization E] [--gradient-weight W] [--repeat N]\n"
                 "                  [--small-iterations N] [--big-iterations N] [--small-size N]");
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    if (!image_path || !alpha_path || !out_fg) {
        fprintf(stderr, "--image, --alpha and --out-fg are required\n");
        return 2;
    }

    npy_array image = {0};
    npy_array alpha = {0};
    if (!npy_load(image_path, &image) || !npy_load(alpha_path, &alpha)) {
        npy_free(&image);
        npy_free(&alpha);
        return 1;
    }
    if (image.ndim != 3 || alpha.ndim != 2 || image.shape[0] != alpha.shape[0] ||
        image.shape[1] != alpha.shape[1]) {
        fprintf(stderr, "shape mismatch: image %dD, alpha %dD\n", image.ndim, alpha.ndim);
        npy_free(&image);
        npy_free(&alpha);
        return 1;
    }

    const int h = image.shape[0];
    const int w = image.shape[1];
    const int depth = image.shape[2];

    float *fg = malloc((size_t)h * w * depth * sizeof(float));
    float *bg = malloc((size_t)h * w * depth * sizeof(float));
    if (!fg || !bg) {
        fprintf(stderr, "allocation failed\n");
        free(fg);
        free(bg);
        npy_free(&image);
        npy_free(&alpha);
        return 1;
    }

    /* --repeat reports the best steady-state run, which is what a server sees:
     * the first call also pays CUDA context creation. */
    int rc = 0;
    double elapsed = 0.0;
    if (repeat < 1) repeat = 1;
    for (int run = 0; run < repeat && rc == 0; run++) {
        const double started = now_ms();
        rc = use_cuda ? omatte_estimate_fb_cuda(image.data, alpha.data, h, w, depth, &params, fg, bg)
                      : omatte_estimate_fb(image.data, alpha.data, h, w, depth, &params, fg, bg);
        const double took = now_ms() - started;
        if (run == 0 || took < elapsed) elapsed = took;
    }

    if (rc != 0) {
        fprintf(stderr, "estimation failed (rc=%d%s)\n", rc,
                rc == -3 ? ", built without CUDA" : "");
    } else {
        const int shape[3] = {h, w, depth};
        if (!npy_save(out_fg, fg, 3, shape)) rc = 1;
        if (rc == 0 && out_bg && !npy_save(out_bg, bg, 3, shape)) rc = 1;
        /* The CUDA backend always uses the red-black order. */
        fprintf(stderr, "%dx%dx%d %s in %.1f ms\n", h, w, depth,
                use_cuda ? "red-black (cuda)"
                         : (params.order == OMATTE_ORDER_RED_BLACK ? "red-black" : "sequential"),
                elapsed);
    }

    free(fg);
    free(bg);
    npy_free(&image);
    npy_free(&alpha);
    return rc == 0 ? 0 : 1;
}
