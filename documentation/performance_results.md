# 📊 Performance results

These observations and interpretations are based on what is in `/deliverables/latest_results/`. If benchmarks are re-run and results change, this should be updated to match.

## Lloyd

### Parity

Overall parity is respected: large inertia deviations in some configurations occur when the data geometry amplifies small implementation-level floating-point and reduction-order differences, which don't make our implementation less correct.

* Huge intertia differences in very low D: The centre-box scaling, $B=10\sqrt{10/D}(K/10)^{1/D}$, reduces the tendency for randomly sampled centres to become crowded as $K$ increases. It still helps in 2-D and 3-D: the smallest centre-to-centre separation decreases more slowly than it would with a fixed sampling box, but it does not remain constant. \
The remaining effect is strongest in low dimensions. In 1-D, even with the current scaling, the expected gap between the closest pair of centres is approximately $2\sqrt{10}/K$; in 2-D and 3-D, the closest gaps also decrease with $K$, though more slowly. When these gaps become comparable to the fixed cluster standard deviation, $\sigma=1$, neighbouring Gaussian blobs overlap and their assignments become ambiguous, which can confound Lloyd's iterations. A more aggressive box scaling could maintain larger separations.

* As K increases, inertia differences increase: each centroid is based on fewer points, so a one-point numerical difference changing its assigned cluster will have a bigger impact on the updated centroid's position.

### Speedups

* Both variants slow down as $N$ increases, until they reach a plateau, whose value depends on the variant, D, and K. A representative curve will drop by 3 times between 10K and 10M $N$. \
In low K, this plateauing is more pronounced, and depends on some kind of unintentional regime change of our implementation, maybe caused by some kind of cache effect (we can see that the slow down/plateauing is not the reference by looking at the timings themselves)

* Both variants slow down as $D$ increases (excludind very low D, which are affected by parity differences anyway). Unsurprisingly, the static-D variant slows down much faster, even becoming slower than the reference. Its speedup drops by around 12 times between 5D and 100D while dynamic-D only drops by around 6 times.

* As K increases, the regime change becomes more subtle, until it completely gets smoothed out. Both versions get slower in what was previously the low-N "regime" region, and slightly advantaged in the second regime, with dynamic-D speeding back up significantly, maintaining speedup of 2 instead of 1.5.

* Overall, the dynamic-D variant is superior the static-D variant except in $D=1$ to $D=5$, and we get speedups of around 7 to 15 in the more favorable region.

### Diagnostics

#### Spill detection

#### Cachegrind

#### Other

* The executable file size of the dynamic-D variant stays approximately constant, while the static-D one increases linearly with $D$, but still with very reasonnable sizes: around 0.25 Mb in Dimension 1, compared to 1.7 Mb in Dimension 90. \
The auto variant is at first the sum of both, until it drops the inclusion of static-D when it reaches the $D=50$ hardcoded threshold.

## GMM EM

### Parity

Overall parity is respected: most configurations pass the thresolds, except Full covariance.

* For diagonal and sperical covariance types, the parity differences are explained once again by the bounding box thing and reduction differences. See [Diagnostic tool for diagonal GMM](scikit_parity_and_validation.md#-diagnostic-tool-for-diagonal-gmm).

* For full covariance, see [Full-covariance GMM and scikit-learn's centered covariance update](scikit_parity_and_validation.md#full-covariance-gmm-and-scikit-learns-centered-covariance-update)

* In the results you may, depending on the version, see parity-pressure issues at $N=15000$. This was actually a bug that was resolved later: when N was not a multiple of the SIMD width, the final masked vector iteration overwrote rather than preserved the accumulator values in masked-off lanes. On AVX-512, this effectively discarded roughly half of the accumulated lower-bound and sufficient-statistics history, distorting both the GMM lower bound and the fitted parameters. These failures should therefore not be interpreted as genuine C++/scikit-learn parity pressure.

### Speedups

* The $N$-scaling is different compared to Lloyd: instead of simply slowing down, we actually increase our speedup in medium $N$ values (around 100k), but it is attributable to scikit: our implementation takes around the same time to process one sample however many there are, with maybe a slight slowdown trend. \
For full though, we can see the same bell curve, except that our implementation is actually slowing down per sample based on $N$, so we see some kind of combinaison of both effect in the speedups, where they seem stable until high N where they drop slightly.

* In general, both static-D and dynamic-D variants are very competitive, with speedups between 20× and 60×. These gains come not only from SIMD vectorization, but also from streaming the responsibilities computed during the E-step directly into the sufficient-statistic accumulators used by the M-step, thereby avoiding the materialization and subsequent rereading of an N×K responsibility matrix as in scikit-learn. This fusion preserves batch EM semantics because the model parameters remain unchanged throughout the data pass and are updated only after all responsibilities have contributed to the accumulated statistics. It is therefore a valid specialization for our performance-oriented scope, rather than an approximation of EM, although it entails different floating-point reduction behavior and, for full covariance, requires a fallback pass when the raw-moment update is numerically unstable.

* Dynamic-D variants do better than the static-D ones, even on their preferred regions, improving speedups by 1.1 to 1.4 times. The expected behavior of reducing the slowdown as $D$ increases is also present.

* Full covariance is the biggest winner in terms of performance, with speedups reaching 120×. But it's also the most dubious in terms of parity: we're really not doing the same thing as scikit, although it is supposedly safe (and algebraically equivalent, of course).

### Diagnostics

#### Spill detection

#### Cachegrind

#### Other

* Similarly as Lloyd, the executable file size increases linearly with $D$ for the static-D variant. It is also bigger in general as it includes all covariance types, but 2.5 Mb is not anything to be concerned about anyway.


## HDBSCAN

Here $K$ is mapped to `min_cluster_size` and `min_samples` to keep the rest of the pipeline as is.

### Parity

HDBSCAN is super strong in terms of parity, with basically no friction to be noted. This can be explained in part by the fact that it (and the references) are using float64, making floating-point and reduction-order differences less impacting.

### Speedups

* Already, we notice that the overall speedups are much less impressive than the previous algorithms: between 3 and 7 in general.

* We see that overall the speedup seems stable with $N$, and even getting better. That is caused by two things: The distance stage is optimized for higher Ns with a radix selection that has high overhead. Without it it wasn't competitive at higher Ns. And at the same time the MST stage is exceptionally fast at lower Ns, counteracting the previous overhead.

* The MST stage performance is explained by how scikit-learn repeatedly allocates and filters temporary NumPy arrays at every step of Prim’s algorithm, while the C++ implementation fuses the same work into allocation-free SIMD scans over reusable buffers. This produces extreme speedups at low N, where fixed overhead dominates, but a smaller advantage as both implementations become $\mathcal{O}(N^2)$-bound at larger $N$.

* Even though the cluster selection stage is showing speedups of 20 to 50×, this doesn’t get reflected in the full pipeline because selection is a comparatively small, near-linear post-processing step; the total runtime is dominated by the quadratic distance-matrix and MST stages, especially the distance computation at high dimensionality.

### Diagnostics

#### Spill detection

#### Cachegrind



## K-Means++

### Parity

No parity check for now, so we can look at the following results, but shouldn't interpret them too much. Our implementation does match, and the algorithm is the same, so it should still be safe, but eh.

### Speedups

* The static-D variant slows down based on D and N by stages, plateauing each time, in a way that really looks like cache behavior (cf the time per sample plots), but I excluded k-means++ in the latest run so I can't confirm it. Dynamic-D variant much more smoothly with both $D$ and $N$, and its performance isn't crushed at high $D$.

* Both variants are performing very well, with speedups around maybe 15 in general, ranging from 1.5 to 40.

### Diagnostics

#### Spill detection


## Cross-compiler

These were run in earlier versions of the project, so the actual jsons aren't provided. They should be rerun.

## Cross-architecture

### ZEN 5 (AVX512) vs ZEN 3 (AVX2)

I ran the whole pipeline with g++-15 instead of 14, so the whole thing is contaminated bruh.

### ZEN 3 vs rocket lake

same issue + old