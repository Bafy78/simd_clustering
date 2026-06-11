# ✅ Scikit-learn parity and validation

This document describes what the project means by scikit-learn parity, which artifacts are compared, how C++ benchmark outputs are validated, and how to interpret parity failures.

The short version is that scikit-learn is used as the behavioral reference for the Python side of the benchmark, but parity is not treated as a proof of mathematical optimality. It is an implementation-level check that the C++ kernels and the scikit-learn calls are solving the same configured problem from the same input artifacts and producing sufficiently close metrics.

## ✔️ Validation layers

The project has two different validation layers:

1. **C++ timing-process validation**
2. **C++ vs scikit-learn parity**

These are related, but they answer different questions.

C++ timing-process validation checks whether repeated C++ benchmark processes produce the same algorithm metrics before their timing JSON files are merged. This is handled in [`python/benchmark_pipeline/metrics.py`](../python/benchmark_pipeline/metrics.py) and called from [`python/benchmark_pipeline/runner.py`](../python/benchmark_pipeline/runner.py). If the C++ timing processes disagree on the final metrics, the pipeline stops before writing the canonical metrics file.

C++ vs scikit-learn parity is computed later, during postprocessing, from the canonical C++ metrics file and the Python metrics file. This is handled in [`python/benchmark_postprocess/parity.py`](../python/benchmark_postprocess/parity.py) and inserted into the final summary by [`python/benchmark_postprocess/summary.py`](../python/benchmark_postprocess/summary.py).

The separation is intentional. Repeatability failures usually indicate nondeterminism, state leakage, or a bug in the benchmark harness. scikit-learn parity failures usually indicate either a real semantic mismatch, a numerical tolerance issue, or a known difference between the implementation under test and scikit-learn's exact behavior.

## 🧩 Phase coverage

Not every benchmark phase has a scikit-learn parity check.

| Phase           | scikit-learn reference | Metrics artifact | C++ repeat validation | C++/scikit-learn parity |
| --------------- | ---------------------- | ---------------- | --------------------- | ----------------- |
| SoA conversion  | ❌                    | ❌               | ❌                   | ❌                |
| K-Means++       | ✅                    | ❌               | ❌                   | ❌                |
| Lloyd / K-Means | ✅                    | ✅               | ✅                   | ✅                |
| GMM EM          | ✅                    | ✅               | ✅                   | ✅                |

SoA conversion is a C++ layout-preparation phase, not a scikit-learn algorithm phase.

K-Means++ has C++ and Python timing support, but it currently does not emit a metrics artifact and does not participate in parity checks. This is especially important because the measured K-Means++ phase is stochastic today: the Python benchmark calls `sklearn.cluster.kmeans_plusplus` without `random_state`, while the C++ path uses a seed from `std::random_device`.

Lloyd and GMM are the parity-bearing phases. They emit metrics files for both C++ and Python, and postprocessing compares those metrics with explicit thresholds.

## 🎯 Lloyd / K-Means parity

The Python path recomputes inertia in the benchmark script instead of relying only on `kmeans.inertia_`. Although this value is documented as being the correct final value, there can be bugs in scikit-learn that make it incorrect ([GitHub issue](https://github.com/scikit-learn/scikit-learn/issues/34074)) The C++ path also computes inertia as a scalar verification pass outside the timed algorithm.

Postprocessing currently checks Lloyd parity with the thresholds in `LLOYD_PARITY_THRESHOLDS`:

| Check                          | Threshold | Meaning                                                       |
| ------------------------------ | --------- | ------------------------------------------------------------- |
| `inertia_diff_pct`             | `1e-6`    | C++ and Python total inertia must match extremely closely, measured as a percentage of the larger magnitude. |
| `algorithm_iteration_diff_abs` | `0`       | C++ and Python must report exactly the same number of Lloyd iterations.                                      |

The parity record also carries cluster counts and per-cluster inertia for debugging, but those fields are not currently pass/fail gates in `compute_lloyd_comparison(...)`.

## 🫧 GMM EM parity

Postprocessing currently checks GMM parity with the thresholds in `GMM_PARITY_THRESHOLDS`:

| Check                          | Threshold | Meaning                                                             |
| ------------------------------ | --------- | ------------------------------------------------------------------- |
| `algorithm_iteration_diff_abs` | `0`       | C++ and Python must report exactly the same number of EM iteration. |
| `lower_bound_diff_abs`         | `1e-4`    | Final lower bounds must be close in absolute value.                 |
| `weights_max_abs_diff`         | `1e-4`    | Maximum absolute difference across mixture weights.                 |
| `means_max_abs_diff`           | `1e-3`    | Maximum absolute difference across mean coordinates.                |
| `covariances_max_rel_diff`     | `1e-2`    | Maximum relative difference across materialized covariances.        |


The covariance check uses relative error because covariance values can vary by scale. Weights and means use absolute error because their expected magnitudes are easier to interpret directly in this synthetic setup.


### Full-covariance GMM and scikit-learn's centered covariance update

For `covariance_type="full"`, scikit-learn's GaussianMixture does not estimate covariance with the raw-moment identity

```text
cov = E[x xᵀ] - μ μᵀ
```

Its current full-covariance M-step first forms component-centered residuals and then computes the weighted covariance:

```python
diff = X - means[k, :]
covariances[k] = ((resp[:, k] * diff.T) @ diff) / nk[k]
```

That difference matters for parity. The raw-moment identity is algebraically equivalent in exact arithmetic, but it can lose most of the meaningful bits when samples have a large absolute offset and the true within-component variance is small. In that case both `E[x xᵀ]` and `μ μᵀ` are large, and the covariance is recovered by subtracting two nearly equal large quantities.

This is not only a theoretical distinction. Older scikit-learn GMM code used the raw-moment formula for full covariance. It was changed in [scikit-learn PR #3945](https://github.com/scikit-learn/scikit-learn/pull/3945) after [issue #2640](https://github.com/scikit-learn/scikit-learn/issues/2640), where roundoff could produce covariance matrices that were not positive definite and caused Cholesky failures. The PR discussion explicitly states that the raw formula can lead to matrices with large negative eigenvalues, and that the centered-residual replacement may use more memory and may take longer to compute.

The cost model is different in this project. scikit-learn's EM loop already materializes the `N * K` responsibility matrix and then runs a separate M-step, so the centered full-covariance update reuses stored responsibilities. The static C++ full-covariance path is intentionally fused: responsibilities are computed while sufficient statistics are accumulated, and the implementation does not store an `N * K` responsibility matrix. An always-centered full-covariance M-step would therefore require either storing responsibilities or recomputing them in a second pass.

The fallback implementation is the compromise used here for full covariance. The first attempt uses the fused raw-moment update. Before accepting it, the implementation checks the unregularized diagonal variances for non-finite values, non-positive values, and large-offset cancellation risk. If a component is suspicious, or if its Cholesky factorization fails, that component's covariance is recomputed in a second pass using the current EM iteration's responsibilities and centered residuals around the newly estimated mean. Responsibilities are recomputed for the fallback pass rather than stored.

This means full-covariance parity should be interpreted carefully:

* if the fallback triggers, that component's covariance update is much closer to scikit-learn's centered-residual update;
* if the fallback does not trigger, the component keeps the faster raw-moment update and is not formula-level equivalent to scikit-learn;
* the fallback is intended to catch the same non-positive-definite / cancellation failure class that motivated scikit-learn's change, not to make every full-covariance M-step identical to scikit-learn's implementation.

I didn't bother including a diagnostic script here, but basically if you change the code to always run the fallback (setting `covariance_needs_stable_recompute[k] = 1` all the time), you will see that we lose about **half the performance** (each EM iteration is basically ran twice) but almost all the covariance parity pressure disappears.

## ⏱️ C++ timing-process metrics validation

C++ benchmark tasks are run once per configured timing process. Each process writes its own temporary timing JSON. For Lloyd and GMM, each process also writes its own temporary metrics file.

Before the final C++ metrics file is written, [`validate_cpp_timing_process_metrics(...)`](../python/benchmark_pipeline/metrics.py) compares the temporary metrics files.

For Lloyd, the repeated C++ metrics must agree on:

* `algorithm_iterations`
* `cluster_counts`
* `inertia`
* `cluster_inertia`
* centroid shape and centroid values

For GMM, the repeated C++ metrics must agree on:

* `phase`
* `covariance_type`
* `algorithm_iterations`
* `lower_bound`
* `lower_bounds`
* `weights`
* `means`
* `covariances`

Scalar and array comparisons use a tight internal helper, `assert_close(...)`, with default tolerances `rel_tol=1e-10` and `abs_tol=1e-8`. Some structural fields, such as cluster counts, iteration counts, phase names, and covariance type names, must match exactly.

If validation succeeds, the first timing-process metrics record becomes the canonical C++ metrics file. If validation fails, the pipeline raises an error instead of silently merging timing data whose algorithm outputs disagree.

## 📦 Postprocessing and summary output

The postprocessing entrypoint is [`python/postprocess_benchmarks.py`](../python/postprocess_benchmarks.py).

It performs four high-level steps:

1. load Lloyd and GMM metrics artifacts;
2. load raw timing records from benchmark JSON files;
3. build summary statistics, speedups, and parity records;
4. write `benchmark_summary.json`.

The postprocessor only computes parity for configurations that have the required C++ and Python metrics files. Missing metrics are treated as incomplete data, not as a successful comparison.

Parity is computed per C++ variant and per algorithm parameterization. For example, `lloyd_metrics_static_cpp_...json` and `lloyd_metrics_dynamic_cpp_...json` are both compared against the same `lloyd_metrics_reference_py_...json` scikit-learn reference.

In the final summary, each parity-bearing phase gets a `parity` block containing:

* `status`, usually `PASS` or `FAIL`;
* `failure_reasons`;
* the boolean result of each check;
* the thresholds used for that run;
* the raw compared values and derived differences.

This is deliberate. The summary should be self-describing enough that plots and notebooks do not need to reimplement the parity logic. Reporting should read the parity status and diagnostics from the summary artifact.

## 🚨 How to interpret failures

A parity failure does not automatically mean that the C++ kernel is wrong, but it does mean the corresponding benchmark result should not be treated as a clean scikit-learn-equivalent speed comparison without further investigation.

When a parity check fails, the first things to inspect are the raw metrics files, not the timing summary. The metrics files contain the actual algorithm outputs used to decide the parity status.

## 🩺 Diagnostic tool for diagonal GMM

The project includes [`python/explain_gmm_diag_scikit_parity.py`](../python/explain_gmm_diag_scikit_parity.py), a forensic script for diagonal GMM parity.

This tool is meant to explain differences between the C++ diagonal GMM path and scikit-learn's diagonal covariance behavior. It can generate comparable data and initialization artifacts, run diagnostic C++ entry points, compare traces and score calculations, and inspect fixed-responsibility M-step behavior.

After running it, it generates [a markdown file](../diagnostics_results/gmm_diag_scikit_parity_explainer/gmm_diag_scikit_parity_explanation.md) that argues if the specific tested config's difference with scikit-learn can be explained by one of its tested explanations. The main thing is that, in low-D high-N, the covariances diff with scikit-learn is generally huge, way past thresholds. But when rerunning the alg with a BLAS call in the M step, this difference disappears. And algebraically, our kernel is doing the same. So the difference in this case is very likely reduction semantics, that we won't be trying to match.

## 🚧 Non-goals and limits

The parity system has several limits:

* It does not prove that either implementation found a better clustering than the other.
* It does not validate against every scikit-learn version or every platform.
* It does not currently check K-Means++ output parity because I couldn't bother implementing it: it's not a priority.

The thresholds are also part of the experimental contract. Changing them changes the meaning of a PASS result, so threshold changes should be documented with the benchmark results that use them.