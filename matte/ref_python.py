"""Pure-Python/NumPy transcription of pymatting's `_estimate_fb_ml`.

This exists to *validate the written spec* (matte/README.md) against the real
pymatting implementation. It is deliberately a literal, loop-for-loop
transcription -- no vectorisation -- so it doubles as pseudo-code for the
C/CUDA port. It is slow; it is not meant to be fast.

Storage precision mirrors the numba kernel: image/alpha/F/B and the 2xD
right-hand-side scratch `b` are float32; scalar arithmetic is float64
(which is what numba's `1.0 - a0` promotion produces anyway).
"""

import numpy as np

DX = (-1, 1, 0, 0)
DY = (0, 0, -1, 1)


def resize_nearest(dst_shape, src):
    """Nearest-neighbour resize, `x_src = x_dst * w_src // w_dst` (floor), clamped.

    Works for HW and HWC. Always resamples from `src` directly.
    """
    h_dst, w_dst = dst_shape[0], dst_shape[1]
    h_src, w_src = src.shape[0], src.shape[1]

    ys = np.minimum(h_src - 1, np.maximum(0, (np.arange(h_dst) * h_src) // h_dst))
    xs = np.minimum(w_src - 1, np.maximum(0, (np.arange(w_dst) * w_src) // w_dst))

    return np.ascontiguousarray(src[np.ix_(ys, xs)] if src.ndim == 2 else src[np.ix_(ys, xs, np.arange(src.shape[2]))])


def pyramid_levels(h0, w0):
    """Return [(h, w), ...] for i_level = 0 .. n_levels inclusive."""
    n_levels = int(np.ceil(np.log2(max(w0, h0))))
    out = []
    for i_level in range(n_levels + 1):
        w = round(w0 ** (i_level / n_levels))
        h = round(h0 ** (i_level / n_levels))
        out.append((h, w))
    return out


def estimate_fb_ml_py(
    input_image,
    input_alpha,
    regularization=1e-5,
    n_small_iterations=10,
    n_big_iterations=2,
    small_size=32,
    gradient_weight=1.0,
):
    input_image = np.ascontiguousarray(input_image, dtype=np.float32)
    input_alpha = np.ascontiguousarray(input_alpha, dtype=np.float32)

    h0, w0, depth = input_image.shape

    # --- average foreground / background colour (row-major float32 accumulation)
    F_mean = np.zeros(depth, dtype=np.float32)
    B_mean = np.zeros(depth, dtype=np.float32)
    F_count = 0
    B_count = 0
    for y in range(h0):
        for x in range(w0):
            if input_alpha[y, x] > 0.9:
                for c in range(depth):
                    F_mean[c] += input_image[y, x, c]
                F_count += 1
            if input_alpha[y, x] < 0.1:
                for c in range(depth):
                    B_mean[c] += input_image[y, x, c]
                B_count += 1
    F_mean /= F_count + 1e-5
    B_mean /= B_count + 1e-5

    F_prev = np.zeros((1, 1, depth), dtype=np.float32) + F_mean
    B_prev = np.zeros((1, 1, depth), dtype=np.float32) + B_mean

    reg = np.float32(regularization)
    gw = np.float32(gradient_weight)

    for (h, w) in pyramid_levels(h0, w0):
        image = resize_nearest((h, w), input_image)
        alpha = resize_nearest((h, w), input_alpha)
        F = resize_nearest((h, w), F_prev)
        B = resize_nearest((h, w), B_prev)

        n_iter = n_small_iterations if (w <= small_size and h <= small_size) else n_big_iterations

        b = np.zeros((2, depth), dtype=np.float32)

        for _ in range(n_iter):
            # Sequential row-major sweep: Gauss-Seidel, updates are in-place.
            for y in range(h):
                for x in range(w):
                    a0 = alpha[y, x]
                    a1 = 1.0 - a0

                    a00 = np.float32(a0 * a0)  # stays float32 in numba
                    a01 = a0 * a1
                    a11 = a1 * a1

                    for c in range(depth):
                        b[0, c] = a0 * image[y, x, c]
                        b[1, c] = a1 * image[y, x, c]

                    for d in range(4):
                        x2 = max(0, min(w - 1, x + DX[d]))
                        y2 = max(0, min(h - 1, y + DY[d]))

                        gradient = abs(a0 - alpha[y2, x2])
                        da = reg + gw * gradient

                        a00 = np.float32(a00 + da)
                        a11 = a11 + da

                        for c in range(depth):
                            b[0, c] += da * F[y2, x2, c]
                            b[1, c] += da * B[y2, x2, c]

                    determinant = a00 * a11 - a01 * a01
                    inv_det = 1.0 / determinant

                    b00 = inv_det * a11
                    b01 = inv_det * -a01
                    b11 = inv_det * a00

                    for c in range(depth):
                        F_c = b00 * b[0, c] + b01 * b[1, c]
                        B_c = b01 * b[0, c] + b11 * b[1, c]

                        F[y, x, c] = max(0.0, min(1.0, F_c))
                        B[y, x, c] = max(0.0, min(1.0, B_c))

        F_prev = F
        B_prev = B

    return F_prev, B_prev
