/* CUDA comparison for PCM16 -> WAV wrapping: measures the full
 * upload -> device-side wrap -> download pipeline against the CPU wrapper.
 * Build with nvcc. Usage: bench_wavwrap_cuda [mib] */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

__global__ void write_header(unsigned char *wav, unsigned riff_len,
                             unsigned rate, unsigned data_len) {
    const char tag[12] = {'R', 'I', 'F', 'F', 'W', 'A',
                          'V', 'E', 'f', 'm', 't', ' '};
    for (int i = threadIdx.x; i < 16; i += blockDim.x) {
        int dst = i < 12 ? i : 36 + (i - 12);
        wav[dst] = static_cast<unsigned char>(tag[i % 12]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto put32 = [&](int off, unsigned v) {
            wav[off] = v & 0xff;
            wav[off + 1] = (v >> 8) & 0xff;
            wav[off + 2] = (v >> 16) & 0xff;
            wav[off + 3] = (v >> 24) & 0xff;
        };
        put32(4, riff_len);
        put32(16, 16);
        wav[20] = 1; wav[21] = 0;
        wav[22] = 1; wav[23] = 0;
        put32(24, rate);
        put32(28, rate * 2);
        wav[32] = 2; wav[33] = 0;
        wav[34] = 16; wav[35] = 0;
        put32(40, data_len);
    }
}

static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err),   \
                   __FILE__, __LINE__);                                   \
            return 1;                                                     \
        }                                                                 \
    } while (0)

int main(int argc, char **argv) {
    const size_t mib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 2ull;
    const size_t pcm_len = mib << 20;
    constexpr int kRounds = 20;
    constexpr unsigned kRate = 24000;

    unsigned char *host_pcm = static_cast<unsigned char *>(malloc(pcm_len));
    unsigned char *host_wav = static_cast<unsigned char *>(malloc(pcm_len + 44));
    memset(host_pcm, 0x5a, pcm_len);

    unsigned char *dev_pcm = nullptr;
    unsigned char *dev_wav = nullptr;
    CHECK(cudaMalloc(&dev_pcm, pcm_len));
    CHECK(cudaMalloc(&dev_wav, pcm_len + 44));

    /* warmup */
    CHECK(cudaMemcpy(dev_pcm, host_pcm, pcm_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_wav + 44, dev_pcm, pcm_len, cudaMemcpyDeviceToDevice));
    write_header<<<1, 32>>>(dev_wav, 36 + pcm_len, kRate, pcm_len);
    CHECK(cudaMemcpy(host_wav, dev_wav, pcm_len + 44, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    if (memcmp(host_wav, "RIFF", 4) != 0 || memcmp(host_wav + 8, "WAVE", 4) != 0) {
        printf("GPU output invalid\n");
        return 1;
    }

    const double start = now_s();
    for (int i = 0; i < kRounds; i++) {
        CHECK(cudaMemcpy(dev_pcm, host_pcm, pcm_len, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dev_wav + 44, dev_pcm, pcm_len, cudaMemcpyDeviceToDevice));
        write_header<<<1, 32>>>(dev_wav, 36 + pcm_len, kRate, pcm_len);
        CHECK(cudaMemcpy(host_wav, dev_wav, pcm_len + 44, cudaMemcpyDeviceToHost));
    }
    CHECK(cudaDeviceSynchronize());
    const double elapsed = now_s() - start;

    printf("CUDA upload+wrap+download: %.0f MiB/s (%zu MiB payload, "
           "%d rounds, PCIe both ways included)\n",
           (static_cast<double>(mib) * kRounds) / elapsed, mib, kRounds);

    free(host_pcm);
    free(host_wav);
    cudaFree(dev_pcm);
    cudaFree(dev_wav);
    return 0;
}
