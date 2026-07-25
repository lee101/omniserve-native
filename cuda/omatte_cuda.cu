// GPU foreground/background estimation - the same multilevel solver as
// src/omatte.c, with the whole pyramid resident on the device so a BiRefNet
// matte can be decontaminated without a round trip per level.
//
// The sweep uses the red-black (checkerboard) order: within one launch every
// written pixel only reads pixels of the opposite colour, so there is no
// read/write race between threads and the result is identical for any block
// shape or device. A naive "one thread per pixel, update in place" port of the
// sequential reference would race on exactly those neighbour reads.

#include "omatte.h"

#include <cuda_runtime.h>

#define OMATTE_CUDA_MAX_DEPTH 4

namespace {

__global__ void resize_nearest_kernel(float *dst, int h_dst, int w_dst, const float *src,
                                      int h_src, int w_src, int depth) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w_dst || y >= h_dst) return;

    int x_src = (int)(((long long)x * w_src) / w_dst);
    int y_src = (int)(((long long)y * h_src) / h_dst);
    x_src = min(max(x_src, 0), w_src - 1);
    y_src = min(max(y_src, 0), h_src - 1);

    const size_t in = ((size_t)y_src * w_src + x_src) * depth;
    const size_t out = ((size_t)y * w_dst + x) * depth;
    for (int c = 0; c < depth; c++) dst[out + c] = src[in + c];
}

__global__ void sweep_kernel(const float *image, const float *alpha, float *F, float *B, int h,
                             int w, int depth, float regularization, float gradient_weight,
                             int parity) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    if (((x + y) & 1) != parity) return;

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

struct DeviceBuffers {
    float *image = nullptr;
    float *alpha = nullptr;
    float *level_image = nullptr;
    float *level_alpha = nullptr;
    float *F = nullptr;
    float *B = nullptr;
    float *F_prev = nullptr;
    float *B_prev = nullptr;

    void release() {
        cudaFree(image);
        cudaFree(alpha);
        cudaFree(level_image);
        cudaFree(level_alpha);
        cudaFree(F);
        cudaFree(B);
        cudaFree(F_prev);
        cudaFree(B_prev);
    }
};

}  // namespace

extern "C" bool omatte_cuda_available(void) {
    int devices = 0;
    return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

extern "C" int omatte_estimate_fb_cuda(const float *image, const float *alpha, int h, int w,
                                       int depth, const omatte_params *params, float *out_f,
                                       float *out_b) {
    if (!image || !alpha || !out_f || h < 1 || w < 1 || depth < 1 || depth > OMATTE_CUDA_MAX_DEPTH) {
        return -1;
    }
    if (!omatte_cuda_available()) return -3;

    omatte_params defaults = omatte_default_params();
    if (!params) params = &defaults;

    // Seed colours are two reductions over the image; doing them on the host
    // keeps the result bit-identical to the CPU path and costs one pass.
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

    DeviceBuffers dev;
    bool ok = cudaMalloc(&dev.image, colour_bytes) == cudaSuccess &&
              cudaMalloc(&dev.alpha, alpha_bytes) == cudaSuccess &&
              cudaMalloc(&dev.level_image, colour_bytes) == cudaSuccess &&
              cudaMalloc(&dev.level_alpha, alpha_bytes) == cudaSuccess &&
              cudaMalloc(&dev.F, colour_bytes) == cudaSuccess &&
              cudaMalloc(&dev.B, colour_bytes) == cudaSuccess &&
              cudaMalloc(&dev.F_prev, colour_bytes) == cudaSuccess &&
              cudaMalloc(&dev.B_prev, colour_bytes) == cudaSuccess;
    if (!ok) {
        dev.release();
        return -2;
    }

    if (cudaMemcpy(dev.image, image, colour_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev.alpha, alpha, alpha_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev.F_prev, f_mean, depth * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev.B_prev, b_mean, depth * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        dev.release();
        return -2;
    }

    int h_prev = 1;
    int w_prev = 1;
    const int max_dimension = h > w ? h : w;
    const int n_levels = (int)ceil(log2((double)max_dimension));
    const dim3 block(16, 16);

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
        level_w = level_w < 1 ? 1 : level_w;
        level_h = level_h < 1 ? 1 : level_h;

        const dim3 grid((level_w + block.x - 1) / block.x, (level_h + block.y - 1) / block.y);

        resize_nearest_kernel<<<grid, block>>>(dev.level_image, level_h, level_w, dev.image, h, w, depth);
        resize_nearest_kernel<<<grid, block>>>(dev.level_alpha, level_h, level_w, dev.alpha, h, w, 1);
        resize_nearest_kernel<<<grid, block>>>(dev.F, level_h, level_w, dev.F_prev, h_prev, w_prev, depth);
        resize_nearest_kernel<<<grid, block>>>(dev.B, level_h, level_w, dev.B_prev, h_prev, w_prev, depth);

        const int n_iter = (level_w <= params->small_size && level_h <= params->small_size)
                               ? params->n_small_iterations
                               : params->n_big_iterations;

        for (int iter = 0; iter < n_iter; iter++) {
            for (int parity = 0; parity < 2; parity++) {
                sweep_kernel<<<grid, block>>>(dev.level_image, dev.level_alpha, dev.F, dev.B,
                                              level_h, level_w, depth, params->regularization,
                                              params->gradient_weight, parity);
            }
        }

        const size_t level_bytes = (size_t)level_h * level_w * depth * sizeof(float);
        cudaMemcpy(dev.F_prev, dev.F, level_bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev.B_prev, dev.B, level_bytes, cudaMemcpyDeviceToDevice);
        h_prev = level_h;
        w_prev = level_w;
    }

    cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(out_f, dev.F, colour_bytes, cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess && out_b) {
        status = cudaMemcpy(out_b, dev.B, colour_bytes, cudaMemcpyDeviceToHost);
    }

    dev.release();
    return status == cudaSuccess ? 0 : -2;
}
