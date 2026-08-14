// GPU foreground/background estimation - the same multilevel solver as
// src/omatte.c, with the whole pyramid resident on the device so a BiRefNet
// matte can be decontaminated without a round trip per level.
//
// The sweep uses the red-black (checkerboard) order: within one launch every
// written pixel only reads pixels of the opposite colour, so there is no
// read/write race between threads and the result is identical for any block
// shape or device. A naive "one thread per pixel, update in place" port of the
// sequential reference would race on exactly those neighbour reads.
//
// Two things here exist because this runs on a shared, saturated GPU rather than
// on an idle one:
//
//   * The device buffers are allocated once and reused. cudaMalloc is not free
//     and cudaFree is worse - it synchronises the whole context, so a cutout
//     freeing its pyramid would block until the segmentation or diffusion model
//     sharing that context had drained. Eight allocations and eight frees per
//     request was most of the wall clock.
//   * Nothing calls cudaDeviceSynchronize. The pass runs on a private
//     non-blocking stream (or the caller's), and only that stream is waited on.
//     A device-wide sync would make every cutout wait on work it has no
//     relationship to.

#include "omatte.h"

#include <cuda_runtime.h>
#include <pthread.h>

#define OMATTE_CUDA_MAX_DEPTH 4
// Fixed so the seed reduction is deterministic: the same input always sums in
// the same order regardless of image size or occupancy.
#define OMATTE_SEED_BLOCKS 256
#define OMATTE_SEED_THREADS 256
// One CUDA block is 1024 threads, so a level of at most 32x32 is exactly one
// thread per pixel. Levels above that keep the launch-per-sweep path.
#define OMATTE_FUSED_MAX_SIDE 32
#define OMATTE_MAX_SMALL_LEVELS 32

namespace {

// One destination pixel of the nearest-neighbour resize. Floor-aligned integer
// mapping, matching the reference exactly; a centre-aligned convention would
// shift every pyramid level by up to half a pixel.
__device__ __forceinline__ void resize_nearest_px(float *dst, int x, int y, int h_dst, int w_dst,
                                                  const float *src, int h_src, int w_src,
                                                  int depth) {
    int x_src = (int)(((long long)x * w_src) / w_dst);
    int y_src = (int)(((long long)y * h_src) / h_dst);
    x_src = min(max(x_src, 0), w_src - 1);
    y_src = min(max(y_src, 0), h_src - 1);

    const size_t in = ((size_t)y_src * w_src + x_src) * depth;
    const size_t out = ((size_t)y * w_dst + x) * depth;
    for (int c = 0; c < depth; c++) dst[out + c] = src[in + c];
}

// One pixel of one colour class. Writes F and B in place; the caller guarantees
// that no thread running concurrently reads what this one writes, which for the
// checkerboard means every concurrent thread has the opposite parity.
__device__ __forceinline__ void sweep_px(const float *image, const float *alpha, float *F, float *B,
                                         int x, int y, int h, int w, int depth,
                                         float regularization, float gradient_weight) {
    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};

    const size_t index = (size_t)y * w + x;
    const float a0 = alpha[index];
    const float a1 = 1.0f - a0;

    float a00 = a0 * a0;
    const float a01 = a0 * a1;
    float a11 = a1 * a1;

    float b0[OMATTE_CUDA_MAX_DEPTH];
    float b1[OMATTE_CUDA_MAX_DEPTH];
    for (int c = 0; c < depth; c++) {
        const float pixel = image[index * depth + c];
        b0[c] = a0 * pixel;
        b1[c] = a1 * pixel;
    }

    for (int d = 0; d < 4; d++) {
        const int x2 = min(max(x + dx[d], 0), w - 1);
        const int y2 = min(max(y + dy[d], 0), h - 1);
        const size_t neighbour = (size_t)y2 * w + x2;

        const float da = regularization + gradient_weight * fabsf(a0 - alpha[neighbour]);
        a00 += da;
        a11 += da;
        for (int c = 0; c < depth; c++) {
            b0[c] += da * F[neighbour * depth + c];
            b1[c] += da * B[neighbour * depth + c];
        }
    }

    const float inv_det = 1.0f / (a00 * a11 - a01 * a01);
    const float b00 = inv_det * a11;
    const float b01 = inv_det * -a01;
    const float b11 = inv_det * a00;

    for (int c = 0; c < depth; c++) {
        float f = b00 * b0[c] + b01 * b1[c];
        float b = b01 * b0[c] + b11 * b1[c];
        f = fminf(1.0f, fmaxf(0.0f, f));
        b = fminf(1.0f, fmaxf(0.0f, b));
        F[index * depth + c] = f;
        B[index * depth + c] = b;
    }
}

__global__ void resize_nearest_kernel(float *dst, int h_dst, int w_dst, const float *src,
                                      int h_src, int w_src, int depth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w_dst || y >= h_dst) return;
    resize_nearest_px(dst, x, y, h_dst, w_dst, src, h_src, w_src, depth);
}

__global__ void sweep_kernel(const float *image, const float *alpha, float *F, float *B, int h,
                             int w, int depth, float regularization, float gradient_weight,
                             int parity) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    if (((x + y) & 1) != parity) return;
    sweep_px(image, alpha, F, B, x, y, h, w, depth, regularization, gradient_weight);
}

// The top of the pyramid, fused.
//
// Levels whose dimensions are both <= small_size get n_small_iterations sweeps,
// which for the defaults is 6 levels x (4 resizes + 20 sweeps) = 144 launches
// that between them touch about 1400 pixels. On an idle device that is free; on
// a device already saturated by a segmentation or diffusion model every one of
// those launches queues behind someone else's work, and the fixed cost was
// measured at ~2.8 ms - more than the entire rest of the solve.
//
// A level of at most small_size x small_size fits in one block (32x32 = 1024
// threads), and within one block __syncthreads() is a full barrier over every
// thread that participates. So the whole small pyramid becomes one launch with
// barriers where the launch boundaries used to be. The loads, the stores and
// their order are unchanged, so this is the same arithmetic in the same
// sequence, not an approximation of it.
struct SmallLevels {
    int count;
    int h[OMATTE_MAX_SMALL_LEVELS];
    int w[OMATTE_MAX_SMALL_LEVELS];
};

__global__ void small_levels_kernel(const float *image, const float *alpha, int h_full, int w_full,
                                    int depth, float *level_image, float *level_alpha, float *F,
                                    float *B, float *F_prev, float *B_prev, SmallLevels levels,
                                    float regularization, float gradient_weight, int n_iter) {
    const int x = threadIdx.x;
    const int y = threadIdx.y;

    int h_prev = 1;
    int w_prev = 1;
    for (int level = 0; level < levels.count; level++) {
        const int lh = levels.h[level];
        const int lw = levels.w[level];
        const bool mine = x < lw && y < lh;

        if (mine) {
            resize_nearest_px(level_image, x, y, lh, lw, image, h_full, w_full, depth);
            resize_nearest_px(level_alpha, x, y, lh, lw, alpha, h_full, w_full, 1);
            resize_nearest_px(F, x, y, lh, lw, F_prev, h_prev, w_prev, depth);
            resize_nearest_px(B, x, y, lh, lw, B_prev, h_prev, w_prev, depth);
        }
        __syncthreads();

        for (int iter = 0; iter < n_iter; iter++) {
            for (int parity = 0; parity < 2; parity++) {
                if (mine && ((x + y) & 1) == parity) {
                    sweep_px(level_image, level_alpha, F, B, x, y, lh, lw, depth, regularization,
                             gradient_weight);
                }
                __syncthreads();
            }
        }

        // Carry this level's answer into the next level's upsample source.
        if (mine) {
            const size_t offset = ((size_t)y * lw + x) * depth;
            for (int c = 0; c < depth; c++) {
                F_prev[offset + c] = F[offset + c];
                B_prev[offset + c] = B[offset + c];
            }
        }
        __syncthreads();
        h_prev = lh;
        w_prev = lw;
    }
}

// Partial sums of the confident-foreground and confident-background colours.
// Grid is fixed at OMATTE_SEED_BLOCKS so the reduction tree - and therefore the
// float rounding - does not move with the image size.
__global__ void seed_partial_kernel(const float *__restrict__ image,
                                    const float *__restrict__ alpha, size_t pixels, int depth,
                                    double *partial_f, double *partial_b, double *partial_count) {
    __shared__ double shared_f[OMATTE_CUDA_MAX_DEPTH][OMATTE_SEED_THREADS];
    __shared__ double shared_b[OMATTE_CUDA_MAX_DEPTH][OMATTE_SEED_THREADS];
    __shared__ double shared_nf[OMATTE_SEED_THREADS];
    __shared__ double shared_nb[OMATTE_SEED_THREADS];

    const int lane = threadIdx.x;
    double local_f[OMATTE_CUDA_MAX_DEPTH] = {0, 0, 0, 0};
    double local_b[OMATTE_CUDA_MAX_DEPTH] = {0, 0, 0, 0};
    double local_nf = 0.0;
    double local_nb = 0.0;

    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + lane; i < pixels; i += stride) {
        const float a = __ldg(&alpha[i]);
        const bool is_f = a > 0.9f;
        const bool is_b = a < 0.1f;
        if (!is_f && !is_b) continue;
        for (int c = 0; c < depth; c++) {
            const double pixel = (double)__ldg(&image[i * depth + c]);
            if (is_f) local_f[c] += pixel;
            if (is_b) local_b[c] += pixel;
        }
        local_nf += is_f ? 1.0 : 0.0;
        local_nb += is_b ? 1.0 : 0.0;
    }

    for (int c = 0; c < depth; c++) {
        shared_f[c][lane] = local_f[c];
        shared_b[c][lane] = local_b[c];
    }
    shared_nf[lane] = local_nf;
    shared_nb[lane] = local_nb;
    __syncthreads();

    for (int half = OMATTE_SEED_THREADS / 2; half > 0; half >>= 1) {
        if (lane < half) {
            for (int c = 0; c < depth; c++) {
                shared_f[c][lane] += shared_f[c][lane + half];
                shared_b[c][lane] += shared_b[c][lane + half];
            }
            shared_nf[lane] += shared_nf[lane + half];
            shared_nb[lane] += shared_nb[lane + half];
        }
        __syncthreads();
    }

    if (lane == 0) {
        for (int c = 0; c < depth; c++) {
            partial_f[(size_t)blockIdx.x * OMATTE_CUDA_MAX_DEPTH + c] = shared_f[c][0];
            partial_b[(size_t)blockIdx.x * OMATTE_CUDA_MAX_DEPTH + c] = shared_b[c][0];
        }
        partial_count[blockIdx.x * 2 + 0] = shared_nf[0];
        partial_count[blockIdx.x * 2 + 1] = shared_nb[0];
    }
}

// Folds the per-block partials into the 1x1 top of the pyramid. One block, so
// the final sum order is fixed too.
__global__ void seed_finish_kernel(const double *partial_f, const double *partial_b,
                                   const double *partial_count, int depth, float *F_top,
                                   float *B_top) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double f_sum[OMATTE_CUDA_MAX_DEPTH] = {0, 0, 0, 0};
    double b_sum[OMATTE_CUDA_MAX_DEPTH] = {0, 0, 0, 0};
    double f_count = 0.0;
    double b_count = 0.0;
    for (int block = 0; block < OMATTE_SEED_BLOCKS; block++) {
        for (int c = 0; c < depth; c++) {
            f_sum[c] += partial_f[(size_t)block * OMATTE_CUDA_MAX_DEPTH + c];
            b_sum[c] += partial_b[(size_t)block * OMATTE_CUDA_MAX_DEPTH + c];
        }
        f_count += partial_count[block * 2 + 0];
        b_count += partial_count[block * 2 + 1];
    }
    for (int c = 0; c < depth; c++) {
        F_top[c] = (float)(f_sum[c] / (f_count + 1e-5));
        B_top[c] = (float)(b_sum[c] / (b_count + 1e-5));
    }
}

__global__ void composite_kernel(const float *__restrict__ fg, const float *__restrict__ alpha,
                                 const float *__restrict__ background, float b0, float b1,
                                 float b2, float b3, size_t pixels, int depth, float *out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels) return;
    const float solid[4] = {b0, b1, b2, b3};
    const float a = __ldg(&alpha[i]);
    for (int c = 0; c < depth; c++) {
        const float back = background ? __ldg(&background[i * depth + c]) : solid[c];
        out[i * depth + c] = a * fg[i * depth + c] + (1.0f - a) * back;
    }
}

// Matches Python's round-half-to-even so the GPU pyramid has exactly the same
// level sizes as the CPU and reference implementations.
int round_half_even(double value) {
    const double floor_value = floor(value);
    const double fraction = value - floor_value;
    long long result = (long long)floor_value;
    if (fraction > 0.5) result += 1;
    else if (fraction == 0.5 && (result % 2 != 0)) result += 1;
    return (int)result;
}

// Level sizes are geometric interpolations of each dimension independently, not
// successive halvings, so w and h follow separate curves and an intermediate
// level of a non-square image is not the input's aspect ratio.
void level_size(int i_level, int n_levels, int h, int w, int *out_h, int *out_w) {
    if (n_levels == 0) {
        *out_h = h;
        *out_w = w;
        return;
    }
    const double t = (double)i_level / (double)n_levels;
    int lw = round_half_even(pow((double)w, t));
    int lh = round_half_even(pow((double)h, t));
    *out_w = lw < 1 ? 1 : lw;
    *out_h = lh < 1 ? 1 : lh;
}

// Buffers whose lifetime is the process, not the request. Sized to the largest
// image seen so far and reused; the estimator's working set is bounded by the
// full-resolution image, so this settles after the first few cutouts instead of
// growing.
struct Workspace {
    cudaStream_t stream = nullptr;
    float *image = nullptr;       // full-resolution source, device copy (host API only)
    float *alpha = nullptr;
    float *level_image = nullptr;
    float *level_alpha = nullptr;
    float *F = nullptr;
    float *B = nullptr;
    float *F_prev = nullptr;
    float *B_prev = nullptr;
    double *partial_f = nullptr;
    double *partial_b = nullptr;
    double *partial_count = nullptr;
    size_t pixels = 0;            // capacity, in pixels
    size_t depth = 0;             // capacity, in channels
    bool staged = false;          // image/alpha staging buffers allocated
};

pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
Workspace g_ws;

void workspace_free_locked() {
    cudaFree(g_ws.image);
    cudaFree(g_ws.alpha);
    cudaFree(g_ws.level_image);
    cudaFree(g_ws.level_alpha);
    cudaFree(g_ws.F);
    cudaFree(g_ws.B);
    cudaFree(g_ws.F_prev);
    cudaFree(g_ws.B_prev);
    cudaFree(g_ws.partial_f);
    cudaFree(g_ws.partial_b);
    cudaFree(g_ws.partial_count);
    if (g_ws.stream) cudaStreamDestroy(g_ws.stream);
    g_ws = Workspace();
}

// Ensures capacity for pixels*depth. `need_staging` is false on the device path,
// where the caller already owns the full-resolution image and alpha.
bool ensure_stream_locked() {
    if (g_ws.stream) return true;
    // Non-blocking so this never serialises against the default stream, which
    // is where a co-resident torch model does its work.
    return cudaStreamCreateWithFlags(&g_ws.stream, cudaStreamNonBlocking) == cudaSuccess;
}

bool workspace_reserve_locked(size_t pixels, int depth, bool need_staging) {
    if (!ensure_stream_locked()) return false;
    if (pixels == 0 || depth < 1) return false;
    if (!g_ws.partial_f) {
        const size_t colour = OMATTE_SEED_BLOCKS * OMATTE_CUDA_MAX_DEPTH * sizeof(double);
        if (cudaMalloc(&g_ws.partial_f, colour) != cudaSuccess ||
            cudaMalloc(&g_ws.partial_b, colour) != cudaSuccess ||
            cudaMalloc(&g_ws.partial_count, OMATTE_SEED_BLOCKS * 2 * sizeof(double)) !=
                cudaSuccess) {
            return false;
        }
    }

    const bool grow = pixels > g_ws.pixels || (size_t)depth > g_ws.depth;
    const bool need_stage_alloc = need_staging && !g_ws.staged;
    if (!grow && !need_stage_alloc) return true;

    const size_t want_pixels = pixels > g_ws.pixels ? pixels : g_ws.pixels;
    const size_t want_depth = (size_t)depth > g_ws.depth ? (size_t)depth : g_ws.depth;
    const size_t colour_bytes = want_pixels * want_depth * sizeof(float);
    const size_t alpha_bytes = want_pixels * sizeof(float);

    float *level_image = nullptr, *level_alpha = nullptr, *F = nullptr, *B = nullptr;
    float *F_prev = nullptr, *B_prev = nullptr, *image = nullptr, *alpha = nullptr;
    const bool want_staged = need_staging || g_ws.staged;
    bool ok = true;
    if (grow) {
        ok = cudaMalloc(&level_image, colour_bytes) == cudaSuccess &&
             cudaMalloc(&level_alpha, alpha_bytes) == cudaSuccess &&
             cudaMalloc(&F, colour_bytes) == cudaSuccess &&
             cudaMalloc(&B, colour_bytes) == cudaSuccess &&
             cudaMalloc(&F_prev, colour_bytes) == cudaSuccess &&
             cudaMalloc(&B_prev, colour_bytes) == cudaSuccess;
    }
    if (ok && want_staged && (grow || need_stage_alloc)) {
        ok = cudaMalloc(&image, colour_bytes) == cudaSuccess &&
             cudaMalloc(&alpha, alpha_bytes) == cudaSuccess;
    }
    if (!ok) {
        cudaFree(level_image); cudaFree(level_alpha); cudaFree(F); cudaFree(B);
        cudaFree(F_prev); cudaFree(B_prev); cudaFree(image); cudaFree(alpha);
        return false;
    }

    // Nothing is in flight: callers hold g_lock and every launch is stream-
    // ordered behind the sync at the end of the previous request.
    if (grow) {
        cudaFree(g_ws.level_image); g_ws.level_image = level_image;
        cudaFree(g_ws.level_alpha); g_ws.level_alpha = level_alpha;
        cudaFree(g_ws.F); g_ws.F = F;
        cudaFree(g_ws.B); g_ws.B = B;
        cudaFree(g_ws.F_prev); g_ws.F_prev = F_prev;
        cudaFree(g_ws.B_prev); g_ws.B_prev = B_prev;
        g_ws.pixels = want_pixels;
        g_ws.depth = want_depth;
    }
    if (image) {
        cudaFree(g_ws.image); g_ws.image = image;
        cudaFree(g_ws.alpha); g_ws.alpha = alpha;
        g_ws.staged = true;
    }
    return true;
}

// The pyramid itself. image/alpha are full-resolution device pointers; every
// level resamples from them, never from the level above, exactly as the
// reference does.
cudaError_t run_pyramid_locked(const float *d_image, const float *d_alpha, int h, int w, int depth,
                               const omatte_params *params, cudaStream_t stream) {
    int h_prev = 1;
    int w_prev = 1;
    const int max_dimension = h > w ? h : w;
    const int n_levels = (int)ceil(log2((double)max_dimension));
    const dim3 block(32, 8);

    // Collect the leading run of levels small enough for the fused kernel. It
    // has to be a prefix: the fused kernel carries F_prev between levels
    // internally, so it cannot be resumed part-way through the pyramid.
    SmallLevels small = {0, {0}, {0}};
    int first_big = 0;
    if (params->small_size <= OMATTE_FUSED_MAX_SIDE) {
        for (int i_level = 0; i_level <= n_levels; i_level++) {
            int lw, lh;
            level_size(i_level, n_levels, h, w, &lh, &lw);
            if (lw > params->small_size || lh > params->small_size ||
                small.count >= OMATTE_MAX_SMALL_LEVELS) {
                break;
            }
            small.h[small.count] = lh;
            small.w[small.count] = lw;
            small.count++;
            first_big = i_level + 1;
        }
    }
    if (small.count > 0) {
        small_levels_kernel<<<1, dim3(OMATTE_FUSED_MAX_SIDE, OMATTE_FUSED_MAX_SIDE), 0, stream>>>(
            d_image, d_alpha, h, w, depth, g_ws.level_image, g_ws.level_alpha, g_ws.F, g_ws.B,
            g_ws.F_prev, g_ws.B_prev, small, params->regularization, params->gradient_weight,
            params->n_small_iterations);
        h_prev = small.h[small.count - 1];
        w_prev = small.w[small.count - 1];
    }

    for (int i_level = first_big; i_level <= n_levels; i_level++) {
        int level_w;
        int level_h;
        level_size(i_level, n_levels, h, w, &level_h, &level_w);

        const dim3 grid((level_w + block.x - 1) / block.x, (level_h + block.y - 1) / block.y);

        resize_nearest_kernel<<<grid, block, 0, stream>>>(g_ws.level_image, level_h, level_w,
                                                          d_image, h, w, depth);
        resize_nearest_kernel<<<grid, block, 0, stream>>>(g_ws.level_alpha, level_h, level_w,
                                                          d_alpha, h, w, 1);
        resize_nearest_kernel<<<grid, block, 0, stream>>>(g_ws.F, level_h, level_w, g_ws.F_prev,
                                                          h_prev, w_prev, depth);
        resize_nearest_kernel<<<grid, block, 0, stream>>>(g_ws.B, level_h, level_w, g_ws.B_prev,
                                                          h_prev, w_prev, depth);

        const int n_iter = (level_w <= params->small_size && level_h <= params->small_size)
                               ? params->n_small_iterations
                               : params->n_big_iterations;

        for (int iter = 0; iter < n_iter; iter++) {
            for (int parity = 0; parity < 2; parity++) {
                sweep_kernel<<<grid, block, 0, stream>>>(
                    g_ws.level_image, g_ws.level_alpha, g_ws.F, g_ws.B, level_h, level_w, depth,
                    params->regularization, params->gradient_weight, parity);
            }
        }

        const size_t level_bytes = (size_t)level_h * level_w * depth * sizeof(float);
        cudaMemcpyAsync(g_ws.F_prev, g_ws.F, level_bytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(g_ws.B_prev, g_ws.B, level_bytes, cudaMemcpyDeviceToDevice, stream);
        h_prev = level_h;
        w_prev = level_w;
    }
    return cudaGetLastError();
}

bool args_valid(const float *image, const float *alpha, int h, int w, int depth,
                const float *out_f) {
    return image && alpha && out_f && h >= 1 && w >= 1 && depth >= 1 &&
           depth <= OMATTE_CUDA_MAX_DEPTH;
}

}  // namespace

extern "C" bool omatte_cuda_available(void) {
    int devices = 0;
    return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

extern "C" void omatte_cuda_release_workspace(void) {
    pthread_mutex_lock(&g_lock);
    if (g_ws.stream) cudaStreamSynchronize(g_ws.stream);
    workspace_free_locked();
    pthread_mutex_unlock(&g_lock);
}

extern "C" size_t omatte_cuda_workspace_bytes(void) {
    pthread_mutex_lock(&g_lock);
    const size_t colour = g_ws.pixels * g_ws.depth * sizeof(float);
    const size_t alpha = g_ws.pixels * sizeof(float);
    size_t total = 0;
    if (g_ws.pixels) {
        /* level_image, F, B, F_prev, B_prev + level_alpha. */
        total += 5 * colour + alpha;
        /* image + alpha, only on the host-pointer path. */
        if (g_ws.staged) total += colour + alpha;
    }
    if (g_ws.partial_f) {
        total += OMATTE_SEED_BLOCKS * (2 * OMATTE_CUDA_MAX_DEPTH + 2) * sizeof(double);
    }
    pthread_mutex_unlock(&g_lock);
    return total;
}

extern "C" int omatte_estimate_fb_cuda(const float *image, const float *alpha, int h, int w,
                                       int depth, const omatte_params *params, float *out_f,
                                       float *out_b) {
    if (!args_valid(image, alpha, h, w, depth, out_f)) return -1;
    if (!omatte_cuda_available()) return -3;

    omatte_params defaults = omatte_default_params();
    if (!params) params = &defaults;

    // Seed colours are two reductions over the image. The host path keeps them
    // on the host so the result stays bit-identical to the CPU red-black path,
    // which matte/README.md quotes as exact; the device entry point below
    // reduces on the GPU instead, because there is no host image to read.
    float f_mean[OMATTE_CUDA_MAX_DEPTH] = {0};
    float b_mean[OMATTE_CUDA_MAX_DEPTH] = {0};
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

    const size_t pixels = (size_t)h * w;
    const size_t colour_bytes = pixels * depth * sizeof(float);
    const size_t alpha_bytes = pixels * sizeof(float);

    pthread_mutex_lock(&g_lock);
    if (!workspace_reserve_locked(pixels, depth, true)) {
        pthread_mutex_unlock(&g_lock);
        return -2;
    }
    cudaStream_t stream = g_ws.stream;

    cudaError_t status = cudaMemcpyAsync(g_ws.image, image, colour_bytes, cudaMemcpyHostToDevice,
                                         stream);
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(g_ws.alpha, alpha, alpha_bytes, cudaMemcpyHostToDevice, stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(g_ws.F_prev, f_mean, depth * sizeof(float),
                                 cudaMemcpyHostToDevice, stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(g_ws.B_prev, b_mean, depth * sizeof(float),
                                 cudaMemcpyHostToDevice, stream);
    }
    if (status == cudaSuccess) {
        status = run_pyramid_locked(g_ws.image, g_ws.alpha, h, w, depth, params, stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(out_f, g_ws.F, colour_bytes, cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess && out_b) {
        status = cudaMemcpyAsync(out_b, g_ws.B, colour_bytes, cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    pthread_mutex_unlock(&g_lock);
    return status == cudaSuccess ? 0 : -2;
}

extern "C" int omatte_estimate_fb_cuda_device(const float *d_image, const float *d_alpha, int h,
                                              int w, int depth, const omatte_params *params,
                                              float *d_out_f, float *d_out_b, void *stream_arg) {
    if (!args_valid(d_image, d_alpha, h, w, depth, d_out_f)) return -1;
    if (!omatte_cuda_available()) return -3;

    omatte_params defaults = omatte_default_params();
    if (!params) params = &defaults;

    const size_t pixels = (size_t)h * w;
    const size_t colour_bytes = pixels * depth * sizeof(float);

    pthread_mutex_lock(&g_lock);
    if (!workspace_reserve_locked(pixels, depth, false)) {
        pthread_mutex_unlock(&g_lock);
        return -2;
    }
    cudaStream_t stream = stream_arg ? (cudaStream_t)stream_arg : g_ws.stream;

    seed_partial_kernel<<<OMATTE_SEED_BLOCKS, OMATTE_SEED_THREADS, 0, stream>>>(
        d_image, d_alpha, pixels, depth, g_ws.partial_f, g_ws.partial_b, g_ws.partial_count);
    seed_finish_kernel<<<1, 1, 0, stream>>>(g_ws.partial_f, g_ws.partial_b, g_ws.partial_count,
                                            depth, g_ws.F_prev, g_ws.B_prev);

    cudaError_t status = run_pyramid_locked(d_image, d_alpha, h, w, depth, params, stream);
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(d_out_f, g_ws.F, colour_bytes, cudaMemcpyDeviceToDevice, stream);
    }
    if (status == cudaSuccess && d_out_b) {
        status = cudaMemcpyAsync(d_out_b, g_ws.B, colour_bytes, cudaMemcpyDeviceToDevice, stream);
    }
    // The workspace is shared, so the next caller must not start overwriting F
    // and B while these copies are still queued. Waiting on this one stream is
    // not a device sync: co-resident work on other streams is untouched.
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    pthread_mutex_unlock(&g_lock);
    return status == cudaSuccess ? 0 : -2;
}

extern "C" int omatte_composite_cuda_device(const float *d_fg, const float *d_alpha,
                                            const float *d_background, const float *background_rgb,
                                            int h, int w, int depth, float *d_out,
                                            void *stream_arg) {
    if (!d_fg || !d_alpha || !d_out || h < 1 || w < 1 || depth < 1 ||
        depth > OMATTE_CUDA_MAX_DEPTH) {
        return -1;
    }
    if (!omatte_cuda_available()) return -3;

    float solid[OMATTE_CUDA_MAX_DEPTH] = {0, 0, 0, 0};
    if (background_rgb) {
        for (int c = 0; c < depth; c++) solid[c] = background_rgb[c];
    }

    cudaStream_t stream = (cudaStream_t)stream_arg;
    if (!stream) {
        pthread_mutex_lock(&g_lock);
        const bool ok = ensure_stream_locked();
        stream = g_ws.stream;
        pthread_mutex_unlock(&g_lock);
        if (!ok) return -2;
    }

    const size_t pixels = (size_t)h * w;
    const int threads = 256;
    const int blocks = (int)((pixels + threads - 1) / threads);
    composite_kernel<<<blocks, threads, 0, stream>>>(d_fg, d_alpha, d_background, solid[0],
                                                     solid[1], solid[2], solid[3], pixels, depth,
                                                     d_out);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    return status == cudaSuccess ? 0 : -2;
}
