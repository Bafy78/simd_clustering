# SIMD and kernel mechanics

## SIMD vocabulary and data shapes

The project uses EVE vectors as the common SIMD abstraction:

* `wide_f` is a SIMD vector of `float`.
* `wide_i` is a SIMD vector of `int` with the same cardinality as `wide_f`.
* `cardinal` / `simd_cardinal()` is the number of scalar values processed by one SIMD vector.

The most important idea is that SIMD lanes almost always represent **different samples**. A single SIMD register therefore contains the same dimension for several samples at once:

```
wide_f x_d

lane:    0    1    2    3    ...
sample: s0   s1   s2   s3    ...
value: x0d  x1d  x2d  x3d    ...
```

The word **score** is used deliberately: K-Means scores are assignment distances up to an irrelevant sample-only constant, while GMM scores are weighted log probabilities.

### Static-dimensional samples

The static-dimensional path stores samples as:

```c++
eve::algo::soa_vector<kumi::tuple<float, ..., float>>
```

A SIMD load returns a Kumi tuple whose fields are `wide_f`s. For a 3-dimensional dataset, a loaded sample block looks conceptually like:

```c++
(
  [ x0 x1 x2 x3 ... ],
  [ y0 y1 y2 y3 ... ],
  [ z0 z1 z2 z3 ... ]
)
```

Dimensions are compile-time tuple fields, which allows the compiler to generate very efficient per-dimension code.

### Dynamic-dimensional samples

The dynamic-dimensional path stores samples as dimension-major columns: `samples[d][n]`, with a padded stride.

Conceptually:

```
dimension 0 -> [ d0(s0) d0(s1) d0(s2) d0(s3) ... ]
dimension 1 -> [ d1(s0) d1(s1) d1(s2) d1(s3) ... ]
dimension 2 -> [ d2(s0) d2(s1) d2(s2) d2(s3) ... ]
...
```

This keeps each dimension contiguous in memory, allowing a kernel to load `samples[d][n : n + SIMD_WIDTH]` as a single SIMD vector while streaming over dimensions.

### Centroids and means

Centroids and GMM means are stored per cluster rather than per SIMD lane.

A kernel therefore combines:

```
sample SIMD vector:
[ s0  s1  s2  s3 ... ]

cluster value:
c

broadcast(c):
[ c   c   c   c  ... ]
```

or, in the dynamic assignment backends, several cluster values packed into a layout optimized for the inner loop.

The overall pattern is therefore:

```
SIMD lanes  -> different samples
dimensions  -> streamed or tuple fields
clusters    -> scanned, tiled, or tile-packed
```

Most of the kernel design choices described later are different ways of exploiting this structure while maximizing data reuse and minimizing register pressure.

## Static-D Kumi backends

The static-D kernels all use the same basic shape:

1. load a SIMD batch of Kumi samples;
2. loop over clusters or candidates;
3. compute one SIMD score vector;
4. update SIMD accumulators, best scores, or reductions lane-wise.

### K-Means assignment

The static-D K-Means assignment backend compares every SIMD sample batch against every centroid.

For each centroid, it computes a SIMD score with one FMA chain over the Kumi fields. The current best score and best cluster index are kept as SIMD vectors. After all centroids have been scanned, the best index vector is stored to the assignment array.

This is a simple and direct shape: samples are vectorized, dimensions are unrolled by the compile-time tuple, and centroids are scanned sequentially.

### Static GMM scoring

The score is a weighted log probability instead of a K-Means assignment score.

The spherical covariance model precomputes one score constant per cluster and caches the SIMD sample norm. Scoring is then mostly a dot product with the mean plus a norm term.

The diagonal covariance model precomputes per-cluster, per-dimension quadratic and linear terms. Scoring streams over the Kumi fields and applies those terms with FMAs.

After scoring, responsibilities and sufficient statistics are accumulated as SIMD vectors, then reduced later into scalar parameters.

### Greedy K-Means++

Its distance kernel computes squared distances from a SIMD sample batch to a candidate centroid. That is used in two places:

* updating the permanent nearest-centroid distance array;
* evaluating the total cost of a trial candidate.

## Dynamic-D K-Means assignment kernels

The dynamic-D assignment kernels are more specialized because dimensions are not tuple fields. They stream dimensions from `samples[d][n]` and use an assignment-specific centroid pack.

The motivation is register pressure in the static-D path: loading one SIMD sample batch materializes one SIMD register per dimension. As `D` grows, the loaded sample tuple itself consumes more registers, before accounting for score accumulators, constants, temporaries, masks, and best-score state. Past some point, the compiler may have to spill.

The streamed dynamic-D shape avoids keeping the whole sample tuple live:

```c++
for d in D:
    x = samples[d][n : n + SIMD_WIDTH]
    update several centroid scores using x
```

At any point, the kernel mostly needs the currently loaded sample-dimension vector or vectors, plus the score accumulators it deliberately chooses to keep live. The freed registers can then be spent on keeping several centroid scores and/or several sample vectors live to improve reuse.

Both dynamic kernels precompute centroid norm constants and store centroid coefficients in a layout designed for the inner loop. The goal is to reuse each loaded sample dimension across several centroid scores before moving to the next dimension.

### Centroid-tiled kernel

The centroid-tiled kernel stores centroid coefficients dimension-major: `centroid_pack[d][k]`

For one SIMD sample block, it keeps several centroid score vectors live at the same time. For each dimension, it loads one SIMD sample vector and applies it to all currently active centroid scores.

`K_TILE` is the compile-time centroid tile width. Increasing `K_TILE` means the kernel updates more centroid scores for each loaded sample dimension, improving sample-load reuse and reducing loop overhead. The cost is that each extra centroid score is another live SIMD accumulator.

In the centroid-tiled backend, `centroid_tiles_at_once()` is effectively a multiplier on `K_TILE`:

```c++
total live centroid scores =
    K_TILE * centroid_tiles_at_once<K_TILE>()
```

So if `K_TILE = 4` and `centroid_tiles_at_once()` returns `3`, the generated grouped kernel keeps 12 centroid score vectors live.

The multiplier is chosen automatically from the available SIMD register count. The backend estimates how many centroid tile accumulators can be kept live without exhausting registers, and uses that to decide how many tiles to group together. This improves reuse by applying each loaded sample dimension to several centroid tiles before moving to the next dimension.

This backend is mainly a tiled dot-product scanner: one sample SIMD block, several centroid accumulators, dimensions streamed in the outer inner loop.

### Micro-GEMM kernel

The micro-GEMM kernel packs centroids by tile: `centroid_pack[tile][d][k_in_tile]`

It processes `N_VECTORS` SIMD sample vectors against `K_TILE` centroid columns. Conceptually, this is a small matrix multiplication between a block of samples and a block of centroids, but the score matrix is never materialized:

```text
N_VECTORS sample vectors
against
K_TILE centroid columns
```

For each dimension, the kernel loads `N_VECTORS` SIMD sample vectors and scans the centroid columns in the tile. Each loaded sample vector is reused across the centroid columns in the tile, while each centroid coefficient is reused across the `N_VECTORS` sample vectors.

Therefore, `K_TILE` increases sample-load reuse, and `N_VECTORS` increases centroid-coefficient reuse. Both also expose more independent work, but both increase register pressure because the live score count is roughly:

```c++
N_VECTORS * K_TILE
```

plus the currently loaded sample-dimension vectors. If either knob is too large, the kernel can recreate the same register-pressure problem the dynamic path was trying to avoid.

## Kernel comparison

| Kernel family | SIMD lanes mean | Cluster loop shape | Reuse strategy | Main tunables |
| ------------- | --------------- | ------------------ | -------------- | ------------- |
| [Static-D K-Means](../cpp/include/k_means/static_d/backend.hpp) | samples | scan all centroids | tuple fields are compile-time dimensions | None |
| [Static-D GMM](../cpp/include/gmm/static_d/em.hpp) | samples | scan all components | precomputed score constants / terms | covariance type |
| [Greedy K-Means++](../cpp/include/k_means/greedy_pp.hpp) | samples | scan candidate distances | reuse current min-distance array | `num_local_trials` |
| [Dynamic centroid-tiled](../cpp/include/k_means/dynamic_d/assignment_centroid_tiled.hpp) | samples | scan grouped centroid tiles | reuse one sample dimension across several centroid scores | `K_TILE` |
| [Dynamic micro-GEMM](../cpp/include/k_means/dynamic_d/assignment_micro_gemm.hpp) | samples | scan centroid tiles | reuse sample vectors across `K_TILE`; reuse coefficients across `N_VECTORS` | `N_VECTORS`, `K_TILE` |