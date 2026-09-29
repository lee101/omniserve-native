# `estimate_foreground_ml` — reference fixtures + evaluation harness

Reference data and a pass/fail harness for a C/CUDA port of pymatting's multilevel
foreground estimation (Germer et al. 2020, "Multi-Level Approach to Accurate and
Efficient Foreground Extraction").

Reference implementation: **pymatting 1.1.15**
(`pymatting/foreground/estimate_foreground_ml.py`, the numba `_estimate_fb_ml` kernel),
numpy 2.4.6, CPython 3.14.0.

Nothing here builds or contains C. This directory is inputs, golden outputs and a
scorer.

---

## 1. Layout

```
matte/
  .venv/                  pymatting numpy pillow scipy  (gitignored)
  gen_fixtures.py         regenerates fixtures/ deterministically
  ref_python.py           literal pure-Python transcription of the kernel (the spec, executable)
  example_candidate.py    worked example of the --candidate-cmd CLI contract
  ctypes_candidate.py     runs src/omatte.c as a candidate via ctypes (no C main() needed)
  eval_foreground.py      scorer: max/mean abs error, PSNR, PASS/FAIL, JSON report
  fixtures/               generated .npy + _meta.json  (716 KB, committed)
```

Setup:

```bash
uv venv matte/.venv
uv pip install --python matte/.venv/bin/python pymatting numpy pillow scipy
```

---

## 2. The algorithm (port this)

Signature and defaults:

```python
estimate_foreground_ml(image, alpha,
                       regularization=1e-5,
                       n_small_iterations=10,
                       n_big_iterations=2,
                       small_size=32,
                       return_background=False,
                       gradient_weight=1.0)
```

`image` is `(h0, w0, depth)`, `alpha` is `(h0, w0)`; both are cast to **float32** at the
entry point. Returns `F` (and `B`) with the same shape as `image`.

Solves, per pixel, for the `F`/`B` pair that best explains the observed composite
`I = a*F + (1-a)*B` while staying close to its 4 neighbours, with the neighbour coupling
weakened across alpha edges. Coarse-to-fine, so information propagates far without a
global solve.

### 2.1 Initialisation — mean colours

Before the pyramid, one pass over the full-resolution input:

```python
if input_alpha[y, x] > 0.9:   F_mean_color += input_image[y, x, :];  F_count += 1
if input_alpha[y, x] < 0.1:   B_mean_color += input_image[y, x, :];  B_count += 1
F_mean_color /= F_count + 1e-5
B_mean_color /= B_count + 1e-5
```

Strict `>` / `<`; the two tests are independent `if`s (not `elif`), though the thresholds
cannot both fire. Accumulation is float32, row-major. The `+ 1e-5` is the only guard
against an empty population — if no pixel is opaque, `F_mean_color` blows up rather than
becoming zero. Then

```python
F_prev = np.zeros((1, 1, depth)) + F_mean_color     # 1x1 images
B_prev = np.zeros((1, 1, depth)) + B_mean_color
```

### 2.2 Pyramid structure

```python
n_levels = int(np.ceil(np.log2(max(w0, h0))))

for i_level in range(n_levels + 1):     # inclusive -> n_levels+1 levels
    w = round(w0 ** (i_level / n_levels))
    h = round(h0 ** (i_level / n_levels))
```

Key points, all of which differ from a conventional halving pyramid:

- Level sizes are **geometric interpolations of each dimension independently**, not
  successive halvings. `w` and `h` follow separate curves, so non-square inputs have
  non-uniform aspect at intermediate levels (e.g. 33x17 gives level 3 = 6x4).
- Level 0 is always **1x1** (`x ** 0 == 1`). The last level, `i_level == n_levels`, is
  always exactly `w0 x h0`.
- `n_levels` is driven by `max(w0, h0)` only.
- `round()` is Python 3 semantics — **round-half-to-even**. Exact `.5` values are
  essentially unreachable for non-degenerate sizes, but a C port using
  `(int)(x + 0.5)` or `lroundf` (half-away-from-zero) is the safest place for an
  off-by-one to hide. The exact level sizes for each fixture are recorded in
  `fixtures/<case>_meta.json` under `pyramid_levels` — check against those first.
- Degenerate: `w0 == h0 == 1` gives `n_levels == 0` and a division by zero. Not
  exercised; guard it in the port.

At every level, `image` and `alpha` are resampled **from the original full-resolution
input**, never from the previous level — there is no successive-decimation error path,
and no low-pass filter of any kind:

```python
_resize_nearest_multichannel(image, input_image)    # (h, w, depth) <- (h0, w0, depth)
_resize_nearest(alpha, input_alpha)                 # (h, w)        <- (h0, w0)
```

`F`/`B` for the level are the previous level's results resampled up with the *same*
routine:

```python
_resize_nearest_multichannel(F, F_prev)             # (h, w, depth) <- (h_prev, w_prev, depth)
_resize_nearest_multichannel(B, B_prev)
```

### 2.3 The resize (both directions)

One nearest-neighbour routine serves as both downsampler and upsampler:

```python
x_src = max(0, min(w_src - 1, x_dst * w_src // w_dst))
y_src = max(0, min(h_src - 1, y_dst * h_src // h_dst))
dst[y_dst, x_dst, c] = src[y_src, x_src, c]
```

`//` is **floor** division of non-negative integers, i.e. plain integer truncation here.
This is *left/top-aligned* sampling, **not** the center-aligned
`(x_dst + 0.5) * w_src / w_dst - 0.5` convention that OpenCV/stb use. Getting this wrong
shifts every level by up to half a pixel and will not be subtle in the error metrics.
Compute in a type wide enough for `x_dst * w_src` (int32 is ample at these sizes).
The `max/min` clamp is redundant for valid sizes but harmless.

### 2.4 Iteration count per level

```python
if w <= small_size and h <= small_size:  n_iter = n_small_iterations   # 10
else:                                    n_iter = n_big_iterations     # 2
```

`and`, not `or` — **both** dimensions must be `<= 32`. In case B, level 5 is 32x26 and
still gets 10 iterations; level 6 is 64x50 and gets 2.

### 2.5 The per-pixel update

The sweep order is load-bearing:

```python
for i_iter in range(n_iter):
    for y in prange(h):          # NOTE: kernel is @njit(cache, nogil) -- parallel is NOT set,
        for x in range(w):       # so prange degrades to range: sequential row-major.
```

`_estimate_fb_ml` is decorated `@njit(..., cache=True, nogil=True)` with **no
`parallel=True`**, so `prange` is a plain `range`. Combined with `F`/`B` being updated
in place, this is **Gauss-Seidel, not Jacobi**: within one iteration the left neighbour
`(y, x-1)` and the up neighbour `(y-1, x)` read values already updated *this* iteration,
while right `(y, x+1)` and down `(y+1, x)` read values from the previous one. The shared
scratch buffer `b`, allocated once outside the loops, confirms the kernel is not
intended to run threaded.

> This is the single biggest decision for a CUDA port. A naive parallel kernel is
> Jacobi and will **not** reproduce these fixtures. Either accept a Jacobi variant and
> loosen `--atol`, or reproduce Gauss-Seidel exactly (red-black ordering is *not*
> equivalent either — it changes which neighbours are fresh).

Per pixel, with `a0 = alpha[y, x]`, `a1 = 1 - a0`:

```python
a00 = a0 * a0
a01 = a0 * a1                      # a10 == a01, matrix is symmetric
a11 = a1 * a1

for c in range(depth):
    b[0, c] = a0 * image[y, x, c]
    b[1, c] = a1 * image[y, x, c]

dx = [-1, 1, 0, 0]
dy = [ 0, 0, -1, 1]                # neighbour order: left, right, up, down

for d in range(4):
    x2 = max(0, min(w - 1, x + dx[d]))     # clamp-to-edge (replicate) borders
    y2 = max(0, min(h - 1, y + dy[d]))

    gradient = abs(a0 - alpha[y2, x2])
    da = regularization + gradient_weight * gradient

    a00 += da
    a11 += da                      # a01 is NOT touched by da

    for c in range(depth):
        b[0, c] += da * F[y2, x2, c]
        b[1, c] += da * B[y2, x2, c]

determinant = a00 * a11 - a01 * a01
inv_det = 1.0 / determinant        # no epsilon: det >= 4*reg*(...) > 0 always

b00 = inv_det *  a11
b01 = inv_det * -a01
b11 = inv_det *  a00

for c in range(depth):
    F_c = b00 * b[0, c] + b01 * b[1, c]
    B_c = b01 * b[0, c] + b11 * b[1, c]

    F[y, x, c] = max(0.0, min(1.0, F_c))     # clamp to [0,1] every update
    B[y, x, c] = max(0.0, min(1.0, B_c))     # written back in place
```

Reading that as maths — the 2x2 symmetric normal-equation system per channel is

```
[ a0^2 + S     a0*a1   ] [ F ]   [ a0*I + sum_d da_d * F(nb_d) ]
[ a0*a1     a1^2 + S   ] [ B ] = [ a1*I + sum_d da_d * B(nb_d) ]

S = sum over the 4 neighbours of  da_d = regularization + gradient_weight * |a0 - a(nb_d)|
```

solved by explicit inverse `1/det * [[a11, -a01], [-a01, a00]]`. The data term ties
`a*F + (1-a)*B` to the observed pixel; the `S` term pulls `F`/`B` toward their
neighbours, and `da` *grows* with the local alpha gradient — so coupling is
**strongest across alpha edges**, which is what smears known opaque colour into the
transparent band.

Ordering details a port must preserve to stay bit-close:

- The 4 neighbours are accumulated in the order left, right, up, down (float summation
  order).
- `a01` is computed once from the raw alphas and never receives `da`.
- Both `F` and `B` are clamped to `[0, 1]` after *every* pixel update, not once at the
  end — the clamp is inside the Gauss-Seidel feedback loop and changes the trajectory.
- `da` is the same scalar for `F` and `B` and for all channels.

### 2.6 Background

`B` is not a post-hoc `(I - a*F)/(1-a)`. It is solved **jointly** with `F` in the same
2x2 system at every pixel, every iteration, every level: same neighbour weights, same
clamp, its own pyramid carry (`B_prev`). `return_background=True` just returns the `B`
that was already being computed. There is no extra cost and no separate pass.

### 2.7 Precision

Arrays (`image`, `alpha`, `F`, `B`, the `b` scratch) are float32. In the numba kernel
some scalar temporaries promote to float64 (`1.0 - a0` and the `1.0 / determinant`
literal are float64). `ref_python.py` computes scalars in float64 with float32 array
storage and reproduces pymatting to **<= 2e-6** on all three cases — see
`pure_python_spec_check` in each meta file. So float32-throughout vs mixed precision is
worth ~1e-6, i.e. three orders of magnitude inside the default `--atol` of 2e-3.
Don't chase bit-exactness; do preserve operation order.

---

## 3. Fixtures

Deterministic, `numpy.random.default_rng(20260725)`, regenerate with
`.venv/bin/python gen_fixtures.py`. All arrays float32, C-contiguous, `[0, 1]`;
images are HWC RGB, alpha is HW.

| case | image shape | alpha shape | pyramid levels (h, w, n_iter) | content |
|---|---|---|---|---|
| A | (64, 64, 3) | (64, 64) | (1,1,10) (2,2,10) (4,4,10) (8,8,10) (16,16,10) (32,32,10) (64,64,2) | soft-edged disc alpha (r=20, 4px feather) over smooth two-colour gradients; 440 fractional-alpha px. Power-of-two, square — the easy case. |
| B | (96, 128, 3) | (96, 128) | (1,1,10) (2,2,10) (4,4,10) (7,8,10) (14,16,10) (26,32,10) (50,64,2) (96,128,2) | greenscreen: textured superellipse blob over pure green `(0,1,0)`, feathered border, **deliberate green spill** pushed into the observed image across `d in [0.70, 1.06]` — including where alpha == 1, so the estimator must actively remove green, not just composite. 1206 fractional px. Non-square, two big levels. |
| C | (17, 33, 3) | (17, 33) | (1,1,10) (2,2,10) (3,3,10) (4,6,10) (7,10,10) (11,18,10) (17,33,2) | 33x17 odd size, uniform-random image and alpha. Stresses non-power-of-two pyramids: levels are **non-square and non-monotone in aspect** (6x4, 10x7, 18x11), and `n_levels` comes from `max=33`. Row 0 cols 0..3 forced to alpha 1 and last row's last 4 to alpha 0 so the mean-colour populations are non-empty. |

Per case: `<case>_image.npy`, `<case>_alpha.npy`, `<case>_fg_ref.npy`,
`<case>_bg_ref.npy`, `<case>_meta.json` (shapes, dtype, params, md5 of every array,
level table, versions, pure-Python cross-check).

References were produced by `estimate_foreground_ml(image, alpha, return_background=True)`
with stock defaults.

Note that case B's shape is written height-first: 128x96 means w=128, h=96, so the numpy
shape is `(96, 128, 3)`.

---

## 4. Evaluation

```bash
# candidate already on disk
.venv/bin/python eval_foreground.py --case B --target fg --candidate my_fg.npy

# let the harness run your binary; it must write {out} as .npy
.venv/bin/python eval_foreground.py --case B --target fg \
    --candidate-cmd "./build/matte_ml {image} {alpha} {out} fg"

# machine-readable
.venv/bin/python eval_foreground.py --case C --target bg \
    --candidate my_bg.npy --report-json report.json
```

`--candidate-cmd` placeholders: `{out}`, `{image}`, `{alpha}`, `{case}`, `{h}`, `{w}`,
`{depth}`. Paths are shell-quoted, so the command is exec-shaped
(`prog arg arg arg`); see `example_candidate.py` for a working candidate that satisfies
the contract.

Reports max abs error, mean abs error, RMSE, PSNR (peak 1.0), count of elements over
`atol`, and the `(y, x, c)` of the worst element. **PASS requires both**
`max_abs <= --atol` (default `2e-3`) **and** `mean_abs <= --mean-atol` (default `2e-4`).
Exit 0 on PASS, 1 on FAIL, 2 on usage/IO error. Shape mismatch and non-finite values
fail hard.

Writing `.npy` from C is straightforward — 6-byte magic `\x93NUMPY`, version, an ASCII
header dict `{'descr': '<f4', 'fortran_order': False, 'shape': (96, 128, 3), }` padded
to a 64-byte boundary, then raw data. Alternatively dump raw float32 and wrap it with
`np.fromfile(...).reshape(h, w, 3)` in a two-line shim.

### Harness sanity check

Reference vs itself:

```
$ .venv/bin/python eval_foreground.py --case B --target fg --candidate fixtures/B_fg_ref.npy
case B target fg  shape (96, 128, 3)  candidate dtype float32
  max  abs error : 0.000000e+00   (atol      2.0e-03)
  mean abs error : 0.000000e+00   (mean-atol 2.0e-04)
  rmse           : 0.000000e+00
  psnr           : inf dB (exact match)
  elements > atol: 0 / 36864
  worst at (y,x,c)=(0, 0, 0)
RESULT: PASS
exit=0
```

Reference + 0.01 gaussian noise:

```
$ .venv/bin/python eval_foreground.py --case B --target fg --candidate /tmp/B_fg_perturbed.npy
case B target fg  shape (96, 128, 3)  candidate dtype float32
  max  abs error : 4.502225e-02   (atol      2.0e-03)
  mean abs error : 7.944551e-03   (mean-atol 2.0e-04)
  rmse           : 9.964109e-03
  psnr           : 40.03 dB
  elements > atol: 30938 / 36864
  worst at (y,x,c)=(86, 28, 2)
RESULT: FAIL
exit=1
```

An independent implementation (`ref_python.py`) through the `--candidate-cmd` path:

```
$ .venv/bin/python eval_foreground.py --case C --target fg \
    --candidate-cmd "./.venv/bin/python example_candidate.py {image} {alpha} {out} fg"
case C target fg  shape (17, 33, 3)  candidate dtype float32
  max  abs error : 2.980232e-07   (atol      2.0e-03)
  mean abs error : 4.670447e-08   (mean-atol 2.0e-04)
  rmse           : 6.526282e-08
  psnr           : 143.71 dB
  elements > atol: 0 / 1683
  worst at (y,x,c)=(14, 29, 2)
RESULT: PASS
exit=0
```

---

## 5. Status of the in-tree port (`src/omatte.c`)

`ctypes_candidate.py` runs the existing C implementation directly, so no C entry point is
needed to score it:

```bash
cc -O2 -shared -fPIC -Iinclude src/omatte.c -o matte/libomatte.so -lm -lpthread

cd matte && .venv/bin/python eval_foreground.py --case B --target fg \
    --candidate-cmd ".venv/bin/python ctypes_candidate.py {image} {alpha} {out} fg sequential"
```

Measured against these fixtures:

| case | target | `OMATTE_ORDER_SEQUENTIAL` max abs | result | `OMATTE_ORDER_RED_BLACK` max abs | result |
|---|---|---|---|---|---|
| A | fg | 1.907e-06 | PASS | 3.796e-03 | FAIL |
| A | bg | 1.132e-06 | PASS | | |
| B | fg | 1.550e-06 | PASS | 2.293e-02 | FAIL |
| B | bg | 1.967e-06 | PASS | | |
| C | fg | 2.980e-07 | PASS | 1.917e-01 | FAIL |
| C | bg | 4.172e-07 | PASS | | |

The sequential path reproduces pymatting to float32 rounding — identical error magnitudes
to `ref_python.py`, i.e. the residual is precision, not logic. It correctly implements
round-half-to-even level sizes, floor-aligned nearest resize, the clamp-inside-the-loop,
and the 4-neighbour stencil.

The red-black path is **thread-count deterministic** (1, 4 and 16 threads agree bit for
bit) but is a genuinely different iteration order and does **not** reproduce the
reference: it fails the default tolerance on all three cases, worst on case C where the
random alpha makes neighbour coupling maximally noisy and the 17x33 image has only two
sweeps at full resolution. This is the expected consequence of section 2.5, not a bug in
the checkerboard code. Treat red-black (and any CUDA kernel built on it) as an
*approximation* of the reference and quote its measured tolerance explicitly, rather than
relaxing `--atol` until it passes.

---

## 6. Suggested porting order

1. `resize_nearest` — floor-aligned, both directions. Test it standalone; a half-pixel
   convention error here poisons everything downstream.
2. Pyramid level sizes — assert against `pyramid_levels` in the meta files before
   writing any solver.
3. Mean colours and the 1x1 seed.
4. Sequential Gauss-Seidel update, case C first (17x33, smallest, and its odd level
   sizes catch indexing bugs that A's clean powers of two hide).
5. Case A, then case B.
6. Only then parallelise — and re-measure, because the ordering change moves the
   result away from these fixtures on its own.

If a CUDA version can't be Gauss-Seidel, record the achievable tolerance rather than
loosening the default: run the eval with an explicit `--atol`/`--mean-atol` and note the
numbers, so the gap from the CPU reference stays visible.

## 3. The C / CUDA implementation (`src/omatte.c`, `cuda/omatte_cuda.cu`)

Two sweep orders share one solver:

| order | what it is | why |
| --- | --- | --- |
| `OMATTE_ORDER_SEQUENTIAL` | row-major in-place Gauss-Seidel | matches the pymatting reference sweep exactly |
| `OMATTE_ORDER_RED_BLACK` | checkerboard Gauss-Seidel | race-free: a pixel only ever reads the opposite colour class, so any thread split (or CUDA block shape) gives the identical answer |

The CUDA backend always uses the red-black order and keeps the whole pyramid on
the device; only the seed colours are reduced on the host so they stay
bit-identical to the CPU path.

### Measured accuracy (`./matte/run_eval.sh`)

| case | sequential vs pymatting | red-black vs pymatting | CUDA vs CPU red-black |
| --- | --- | --- | --- |
| A | 1.9e-06 | 3.8e-03 | 0 (identical) |
| B | 1.6e-06 | 2.3e-02 | 0 (identical) |
| C | 3.0e-07 | 1.9e-01 | 0 (identical) |

The sequential residual is float32 rounding. The red-black gap is **not** error:
it is a different iteration order, and pymatting's defaults are not converged.
Scored against a near-converged reference (200/40 sweeps), red-black at default
iterations is as close as pymatting's own default output, and closer with more
sweeps:

| case | pymatting default | red-black default | red-black 40/10 |
| --- | --- | --- | --- |
| A | 1.6e-02 | 1.4e-02 | 4.0e-03 |
| B | 1.3e-01 | 1.2e-01 | 5.1e-02 |
| C | 3.0e-01 | 3.1e-01 | 4.5e-02 |

So: use `sequential` when you need to reproduce pymatting, `redblack`/CUDA for
throughput, and raise the iteration counts if you want a closer solve.

### Throughput (1024x1024x3, RTX 5090, best of 5)

Measured on a device already saturated by another tenant, which is the condition
this runs in.

| backend | time |
| --- | --- |
| pymatting (numba) | 886 ms |
| C sequential, 1 thread | 410 ms |
| C red-black, 16 threads | 192 ms |
| CUDA, host pointers | 9.2 ms |
| CUDA, device pointers (`omatte_estimate_fb_cuda_device`) | 3.1 ms |

The host-pointer figure was 15.7 ms before the device buffers were made
persistent and the device-wide sync was replaced with a private stream. What is
left of it is transfer and the host-side seed reduction: 2.8 ms H2D, 7.2 ms D2H
of F and B, 1.5 ms reducing the seed colours - about 11 ms of moving data around
1 ms of solving.

The device entry point removes all of it. Caller-owned device pointers, torch's
own stream, and a seed reduction on the GPU (fixed 256-block tree, double
accumulator, so it is deterministic). It agrees with the host path to 6.9e-07
max abs; the difference is the reduction order, and the device version is the
more accurate of the two. It seeds the 1x1 top of the pyramid.

Nothing here calls `cudaDeviceSynchronize`, and nothing allocates per request.
Both matter more than they look on a shared card: a device sync waits on the
co-resident segmentation or diffusion model's streams, and `cudaFree`
synchronises the whole context. Eight allocations and eight frees per request was
most of the original wall clock. The workspace is grow-only and 84 MiB for
1024x1024; `omatte_cuda_release_workspace()` returns it.

Levels small enough to fit one block (both dimensions <= 32) run in a single
fused kernel with `__syncthreads()` where the launch boundaries were, cutting 184
launches to 41. Same loads, same stores, same order - `run_eval.sh` reports
identical numbers - but it was worth measuring rather than assuming: on this
contended device one empty kernel plus a stream sync costs 2.26 ms and 184 of
them cost 2.38 ms, so launch count was not what the fixed cost was made of.
Waiting for a slot on a busy GPU was.

### Running it

```bash
cmake -S . -B build-cuda -DWITH_MATTE_CUDA=ON && cmake --build build-cuda -j
./matte/run_eval.sh                 # full accuracy matrix
ctest --test-dir build-cuda -R matte  # green-screen decontamination test
```

`workers/omatte.py` is the ctypes binding the BiRefNet worker uses; set
`BIREFNET_DECONTAMINATE=0` to turn the pass off, or pass `"decontaminate": false`
per request. `estimate_foreground` takes numpy arrays,
`estimate_foreground_torch` takes CUDA tensors and is what the worker calls;
`composite_torch` does `a*F + (1-a)*B` on the device against either a background
image or a solid colour.

### The background is an output, not a leftover

`B` is solved jointly with `F` in the same 2x2 system at every pixel, every
iteration, every level. `return_background=True` costs nothing extra, and the
result is not `(I - a*F) / (1 - a)`, which blows up as alpha approaches 1. The
solved backdrop stays in `[0, 1]` everywhere, which is what makes it usable as
input to something else - the cutout worker feeds it back to the diffusion lane
as the init image for a style-transferred replacement, so the new backdrop
inherits the original's lighting.

On a synthetic green-screen fixture the recovered backdrop came back as
`[0.027, 0.911, 0.080]` against a true `[0.02, 0.92, 0.08]`, and the subject
composited onto white had 0.048 mean error against the ideal in the fringe band
where compositing the observed pixels had 0.107.
