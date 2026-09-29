#include "omatte.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OMATTE_MAX_DEPTH 4
#define OMATTE_MAX_THREADS 64

omatte_params omatte_default_params(void) {
    omatte_params params = {
        .regularization = 1e-5f,
        .n_small_iterations = 10,
        .n_big_iterations = 2,
        .small_size = 32,
        .gradient_weight = 1.0f,
        .threads = 0,
        .order = OMATTE_ORDER_SEQUENTIAL,
    };
    return params;
}

/* Python's round() (and therefore the reference implementation's level sizes)
 * rounds halves to even. C's roundf() rounds halves away from zero, which would
 * silently produce a different pyramid for sizes like 2.5. */
static int round_half_even(double value) {
    double floor_value = floor(value);
    double fraction = value - floor_value;
    long long result = (long long)floor_value;
    if (fraction > 0.5) {
        result += 1;
    } else if (fraction == 0.5) {
        if (result % 2 != 0) result += 1;
    }
    return (int)result;
}

/* Nearest-neighbour resize with the reference's integer index mapping. */
static void resize_nearest(float *dst, int h_dst, int w_dst, const float *src, int h_src, int w_src,
                           int depth) {
    for (int y_dst = 0; y_dst < h_dst; y_dst++) {
        int y_src = (int)(((long long)y_dst * h_src) / h_dst);
        if (y_src < 0) y_src = 0;
        if (y_src > h_src - 1) y_src = h_src - 1;
        for (int x_dst = 0; x_dst < w_dst; x_dst++) {
            int x_src = (int)(((long long)x_dst * w_src) / w_dst);
            if (x_src < 0) x_src = 0;
            if (x_src > w_src - 1) x_src = w_src - 1;
            const float *in = src + ((size_t)y_src * w_src + x_src) * depth;
            float *out = dst + ((size_t)y_dst * w_dst + x_dst) * depth;
            for (int c = 0; c < depth; c++) out[c] = in[c];
        }
    }
}

/* Solves the 2x2 system for one pixel. F and B are read in place, so the
 * caller controls the sweep order (that is what makes this Gauss-Seidel). */
static inline void solve_pixel(int y, int x, int h, int w, int depth, const float *image,
                               const float *alpha, float *F, float *B, float regularization,
                               float gradient_weight) {
    static const int dx[4] = {-1, 1, 0, 0};
    static const int dy[4] = {0, 0, -1, 1};

    const size_t index = (size_t)y * w + x;
    const float a0 = alpha[index];
    const float a1 = 1.0f - a0;

    float a00 = a0 * a0;
    const float a01 = a0 * a1;
    float a11 = a1 * a1;

    float b0[OMATTE_MAX_DEPTH];
    float b1[OMATTE_MAX_DEPTH];
    for (int c = 0; c < depth; c++) {
        const float pixel = image[index * depth + c];
        b0[c] = a0 * pixel;
        b1[c] = a1 * pixel;
    }

    for (int d = 0; d < 4; d++) {
        int x2 = x + dx[d];
        int y2 = y + dy[d];
        if (x2 < 0) x2 = 0;
        if (x2 > w - 1) x2 = w - 1;
        if (y2 < 0) y2 = 0;
        if (y2 > h - 1) y2 = h - 1;

        const size_t neighbour = (size_t)y2 * w + x2;
        const float gradient = fabsf(a0 - alpha[neighbour]);
        const float da = regularization + gradient_weight * gradient;

        a00 += da;
        a11 += da;

        for (int c = 0; c < depth; c++) {
            b0[c] += da * F[neighbour * depth + c];
            b1[c] += da * B[neighbour * depth + c];
        }
    }

    const float determinant = a00 * a11 - a01 * a01;
    const float inv_det = 1.0f / determinant;
    const float b00 = inv_det * a11;
    const float b01 = inv_det * -a01;
    const float b11 = inv_det * a00;

    for (int c = 0; c < depth; c++) {
        float f = b00 * b0[c] + b01 * b1[c];
        float b = b01 * b0[c] + b11 * b1[c];
        if (f < 0.0f) f = 0.0f;
        if (f > 1.0f) f = 1.0f;
        if (b < 0.0f) b = 0.0f;
        if (b > 1.0f) b = 1.0f;
        F[index * depth + c] = f;
        B[index * depth + c] = b;
    }
}

/* Pointers first, then the 4-byte scalars: this order leaves no padding, so the
 * job fits in one cache line and every sweep thread pulls its whole descriptor
 * in a single fetch. */
typedef struct {
    const float *image;
    const float *alpha;
    float *F;
    float *B;
    int h;
    int w;
    int depth;
    float regularization;
    float gradient_weight;
    int parity;
    int thread_index;
    int thread_count;
} sweep_job;

/* One colour class of the checkerboard. Every pixel written here has only
 * neighbours of the opposite parity, so threads never read a value another
 * thread is writing in the same pass - no locks, no races, and the result does
 * not depend on how the rows were split. */
static void *sweep_parity(void *arg) {
    sweep_job *job = (sweep_job *)arg;
    for (int y = job->thread_index; y < job->h; y += job->thread_count) {
        const int start = ((y + job->parity) & 1) ? 1 : 0;
        for (int x = start; x < job->w; x += 2) {
            solve_pixel(y, x, job->h, job->w, job->depth, job->image, job->alpha, job->F, job->B,
                        job->regularization, job->gradient_weight);
        }
    }
    return NULL;
}

static int default_thread_count(void) {
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online < 1) return 1;
    if (online > OMATTE_MAX_THREADS) return OMATTE_MAX_THREADS;
    return (int)online;
}

/* Spawning a thread per sweep only pays off once a level is big enough; the
 * pyramid starts at 1x1, where pthread_create would dominate the solve. */
#define OMATTE_MIN_PIXELS_PER_THREAD 8192

static void run_sweep(int h, int w, int depth, const float *image, const float *alpha, float *F,
                      float *B, const omatte_params *params, int threads) {
    if (params->order == OMATTE_ORDER_SEQUENTIAL) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                solve_pixel(y, x, h, w, depth, image, alpha, F, B, params->regularization,
                            params->gradient_weight);
            }
        }
        return;
    }

    const long pixels = (long)h * w;
    int useful_threads = (int)(pixels / OMATTE_MIN_PIXELS_PER_THREAD);
    if (useful_threads < 1) useful_threads = 1;
    if (useful_threads < threads) threads = useful_threads;

    for (int parity = 0; parity < 2; parity++) {
        if (threads <= 1) {
            sweep_job job = {
                .image = image, .alpha = alpha, .F = F, .B = B,
                .h = h, .w = w, .depth = depth,
                .regularization = params->regularization,
                .gradient_weight = params->gradient_weight,
                .parity = parity, .thread_index = 0, .thread_count = 1,
            };
            sweep_parity(&job);
            continue;
        }

        pthread_t workers[OMATTE_MAX_THREADS];
        sweep_job jobs[OMATTE_MAX_THREADS];
        int started = 0;
        for (int i = 0; i < threads; i++) {
            jobs[i] = (sweep_job){
                .image = image, .alpha = alpha, .F = F, .B = B,
                .h = h, .w = w, .depth = depth,
                .regularization = params->regularization,
                .gradient_weight = params->gradient_weight,
                .parity = parity, .thread_index = i, .thread_count = threads,
            };
            if (pthread_create(&workers[i], NULL, sweep_parity, &jobs[i]) != 0) break;
            started++;
        }
        /* Any threads that failed to spawn are executed inline so the sweep is
         * still complete (and still deterministic). */
        for (int i = started; i < threads; i++) sweep_parity(&jobs[i]);
        for (int i = 0; i < started; i++) pthread_join(workers[i], NULL);
    }
}

int omatte_estimate_fb(const float *image, const float *alpha, int h, int w, int depth,
                       const omatte_params *params, float *out_f, float *out_b) {
    if (!image || !alpha || !out_f || h < 1 || w < 1 || depth < 1 || depth > OMATTE_MAX_DEPTH) {
        return -1;
    }

    omatte_params defaults = omatte_default_params();
    if (!params) params = &defaults;
    int threads = params->threads > 0 ? params->threads : default_thread_count();
    if (threads > OMATTE_MAX_THREADS) threads = OMATTE_MAX_THREADS;

    /* Average colour of the confident foreground / background pixels seeds the
     * 1x1 top of the pyramid. */
    float f_mean[OMATTE_MAX_DEPTH] = {0};
    float b_mean[OMATTE_MAX_DEPTH] = {0};
    long f_count = 0;
    long b_count = 0;
    for (size_t i = 0, n = (size_t)h * w; i < n; i++) {
        const float a = alpha[i];
        if (a > 0.9f) {
            for (int c = 0; c < depth; c++) f_mean[c] += image[i * depth + c];
            f_count++;
        }
        if (a < 0.1f) {
            for (int c = 0; c < depth; c++) b_mean[c] += image[i * depth + c];
            b_count++;
        }
    }
    for (int c = 0; c < depth; c++) {
        f_mean[c] = (float)(f_mean[c] / ((double)f_count + 1e-5));
        b_mean[c] = (float)(b_mean[c] / ((double)b_count + 1e-5));
    }

    const size_t max_pixels = (size_t)h * w;
    /* Guaranteed non-zero by the h/w/depth checks above; restated here so the
     * pyramid buffers below can never be zero-sized allocations, including if a
     * future caller reaches this without going through that guard. */
    if (max_pixels == 0 || depth < 1) return -1;
    float *level_image = malloc(max_pixels * depth * sizeof(float));
    float *level_alpha = malloc(max_pixels * sizeof(float));
    float *F = malloc(max_pixels * depth * sizeof(float));
    float *B = malloc(max_pixels * depth * sizeof(float));
    float *F_prev = malloc(max_pixels * depth * sizeof(float));
    float *B_prev = malloc(max_pixels * depth * sizeof(float));
    if (!level_image || !level_alpha || !F || !B || !F_prev || !B_prev) {
        free(level_image);
        free(level_alpha);
        free(F);
        free(B);
        free(F_prev);
        free(B_prev);
        return -2;
    }

    int h_prev = 1;
    int w_prev = 1;
    for (int c = 0; c < depth; c++) {
        F_prev[c] = f_mean[c];
        B_prev[c] = b_mean[c];
    }

    const int max_dimension = h > w ? h : w;
    const int n_levels = (int)ceil(log2((double)max_dimension));

    for (int i_level = 0; i_level <= n_levels; i_level++) {
        int level_w;
        int level_h;
        if (n_levels == 0) {
            level_w = w;
            level_h = h;
        } else {
            const double t = (double)i_level / (double)n_levels;
            level_w = round_half_even(pow((double)w, t));
            level_h = round_half_even(pow((double)h, t));
        }
        if (level_w < 1) level_w = 1;
        if (level_h < 1) level_h = 1;

        resize_nearest(level_image, level_h, level_w, image, h, w, depth);
        resize_nearest(level_alpha, level_h, level_w, alpha, h, w, 1);
        resize_nearest(F, level_h, level_w, F_prev, h_prev, w_prev, depth);
        resize_nearest(B, level_h, level_w, B_prev, h_prev, w_prev, depth);

        const int n_iter = (level_w <= params->small_size && level_h <= params->small_size)
                               ? params->n_small_iterations
                               : params->n_big_iterations;

        for (int iter = 0; iter < n_iter; iter++) {
            run_sweep(level_h, level_w, depth, level_image, level_alpha, F, B, params, threads);
        }

        memcpy(F_prev, F, (size_t)level_h * level_w * depth * sizeof(float));
        memcpy(B_prev, B, (size_t)level_h * level_w * depth * sizeof(float));
        h_prev = level_h;
        w_prev = level_w;
    }

    memcpy(out_f, F, max_pixels * depth * sizeof(float));
    if (out_b) memcpy(out_b, B, max_pixels * depth * sizeof(float));

    free(level_image);
    free(level_alpha);
    free(F);
    free(B);
    free(F_prev);
    free(B_prev);
    return 0;
}

void omatte_composite(const float *fg, const float *alpha, const float *background_rgb, int h,
                      int w, int depth, float *out) {
    omatte_composite_image(fg, alpha, NULL, background_rgb, h, w, depth, out);
}

void omatte_composite_image(const float *fg, const float *alpha, const float *background,
                            const float *background_rgb, int h, int w, int depth, float *out) {
    if (!fg || !alpha || !out) return;
    for (size_t i = 0, n = (size_t)h * w; i < n; i++) {
        const float a = alpha[i];
        for (int c = 0; c < depth; c++) {
            const float back = background ? background[i * depth + c]
                                          : (background_rgb ? background_rgb[c] : 0.0f);
            out[i * depth + c] = a * fg[i * depth + c] + (1.0f - a) * back;
        }
    }
}

#ifndef OMATTE_WITH_CUDA
bool omatte_cuda_available(void) { return false; }

int omatte_estimate_fb_cuda(const float *image, const float *alpha, int h, int w, int depth,
                            const omatte_params *params, float *out_f, float *out_b) {
    (void)image;
    (void)alpha;
    (void)h;
    (void)w;
    (void)depth;
    (void)params;
    (void)out_f;
    (void)out_b;
    return -3;
}

int omatte_estimate_fb_cuda_device(const float *d_image, const float *d_alpha, int h, int w,
                                   int depth, const omatte_params *params, float *d_out_f,
                                   float *d_out_b, void *stream) {
    (void)d_image;
    (void)d_alpha;
    (void)h;
    (void)w;
    (void)depth;
    (void)params;
    (void)d_out_f;
    (void)d_out_b;
    (void)stream;
    return -3;
}

int omatte_composite_cuda_device(const float *d_fg, const float *d_alpha, const float *d_background,
                                 const float *background_rgb, int h, int w, int depth,
                                 float *d_out, void *stream) {
    (void)d_fg;
    (void)d_alpha;
    (void)d_background;
    (void)background_rgb;
    (void)h;
    (void)w;
    (void)depth;
    (void)d_out;
    (void)stream;
    return -3;
}

void omatte_cuda_release_workspace(void) {}

size_t omatte_cuda_workspace_bytes(void) { return 0; }
#endif
