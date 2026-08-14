/* Multilevel foreground/background estimation (Germer et al. 2020).
 *
 * This is the colour-decontamination pass that runs after a segmentation model
 * (BiRefNet) produces an alpha matte: it recovers the true foreground colour
 * per pixel so green-screen / backdrop colour stops bleeding through
 * semi-transparent edges.
 *
 * The reference implementation is pymatting's `estimate_foreground_ml`. Two
 * update orders are provided:
 *
 *   OMATTE_ORDER_SEQUENTIAL - row-major in-place Gauss-Seidel, matching the
 *                             reference sweep exactly (single threaded).
 *   OMATTE_ORDER_RED_BLACK  - checkerboard Gauss-Seidel: every pixel in a
 *                             colour class reads only the other class, so the
 *                             sweep is race-free and gives identical results
 *                             for any thread count (and maps onto a GPU).
 *
 * Buffers are float32, row-major, channel-interleaved: image[(y*w + x)*depth + c].
 */
#ifndef OMATTE_H
#define OMATTE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OMATTE_ORDER_SEQUENTIAL = 0,
    OMATTE_ORDER_RED_BLACK = 1,
} omatte_order;

typedef struct {
    float regularization;    /* epsilon added to every neighbour weight */
    int n_small_iterations;  /* sweeps once both dimensions are <= small_size */
    int n_big_iterations;    /* sweeps on the larger levels */
    int small_size;          /* threshold that selects the iteration count */
    float gradient_weight;   /* how strongly alpha gradients smooth the result */
    int threads;             /* <=0 picks a default; only used by RED_BLACK */
    omatte_order order;
} omatte_params;

/* pymatting defaults: eps 1e-5, 10 small / 2 big sweeps, small_size 32, gw 1. */
omatte_params omatte_default_params(void);

/* Estimates foreground and background colours.
 *
 * image  : h*w*depth float32, values in [0,1]
 * alpha  : h*w      float32, values in [0,1]
 * out_f  : h*w*depth float32 (may not alias image)
 * out_b  : h*w*depth float32, optional (NULL to discard the background)
 *
 * Returns 0 on success, -1 on invalid arguments, -2 on allocation failure.
 */
int omatte_estimate_fb(const float *image, const float *alpha, int h, int w, int depth,
                       const omatte_params *params, float *out_f, float *out_b);

/* Composites the estimated foreground over a solid colour using alpha. */
void omatte_composite(const float *fg, const float *alpha, const float *background_rgb,
                      int h, int w, int depth, float *out);

/* Composites the estimated foreground over a full background *image* of the
 * same size. background may be NULL, in which case background_rgb (depth
 * floats) is used, and if that is NULL too the backdrop is black. */
void omatte_composite_image(const float *fg, const float *alpha, const float *background,
                            const float *background_rgb, int h, int w, int depth, float *out);

/* True when the library was built with the CUDA backend available. */
bool omatte_cuda_available(void);

/* Same contract as omatte_estimate_fb, executed on the GPU with the red-black
 * order. Returns -3 when the build has no CUDA support. */
int omatte_estimate_fb_cuda(const float *image, const float *alpha, int h, int w, int depth,
                            const omatte_params *params, float *out_f, float *out_b);

/*
 * Device-pointer entry points.
 *
 * The host-pointer call above owns four copies the caller usually does not need:
 * BiRefNet produces its alpha on the GPU and the composite is consumed there
 * too, so staging image/alpha down and F/B back up is pure round trip. These
 * take pointers that are already device memory and never touch the host.
 *
 * `stream` is a cudaStream_t, declared void * so callers (and this header) do
 * not need the CUDA headers. Pass NULL to use omatte's own private stream; pass
 * the caller's stream to keep the pass ordered against its own work without a
 * device-wide sync. Nothing here calls cudaDeviceSynchronize: with a diffusion
 * or segmentation model live in the same context that would block on work this
 * pass has no relationship to.
 *
 * The one behavioural difference from the host call is the seed colour, which
 * is reduced on the device (fixed 256-block tree, double accumulator) instead of
 * in host float32 row-major order. It is a more accurate sum, not a less
 * accurate one, and it only seeds the 1x1 top of the pyramid; measured against
 * the host path the outputs agree to ~1e-7. The host entry point still reduces
 * on the host, so the byte-identical CPU/GPU claim in matte/README.md continues
 * to hold for it.
 */
int omatte_estimate_fb_cuda_device(const float *d_image, const float *d_alpha, int h, int w,
                                   int depth, const omatte_params *params, float *d_out_f,
                                   float *d_out_b, void *stream);

/* Composites d_fg over either a full background image or a solid colour.
 * d_background is h*w*depth device floats, or NULL to use background_rgb (depth
 * floats, host memory), or both NULL for black. d_out may alias d_fg. */
int omatte_composite_cuda_device(const float *d_fg, const float *d_alpha,
                                 const float *d_background, const float *background_rgb, int h,
                                 int w, int depth, float *d_out, void *stream);

/* Releases the cached device workspace (buffers and the private stream).
 * Allocations are reused across calls precisely so a request never pays
 * cudaMalloc/cudaFree - cudaFree synchronises the whole context, which on a
 * shared GPU means waiting for someone else's model. Call this only at
 * shutdown, or when something else needs the VRAM back. */
void omatte_cuda_release_workspace(void);

/* Bytes of device memory the cached workspace currently holds. */
size_t omatte_cuda_workspace_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* OMATTE_H */
