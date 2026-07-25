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

/* True when the library was built with the CUDA backend available. */
bool omatte_cuda_available(void);

/* Same contract as omatte_estimate_fb, executed on the GPU with the red-black
 * order. Returns -3 when the build has no CUDA support. */
int omatte_estimate_fb_cuda(const float *image, const float *alpha, int h, int w, int depth,
                            const omatte_params *params, float *out_f, float *out_b);

#ifdef __cplusplus
}
#endif

#endif /* OMATTE_H */
