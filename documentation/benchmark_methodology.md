# 📏 Benchmark methodology

This document describes the benchmark methodology used by the pipeline: how inputs are generated, how initialization is controlled, what is included in the measured regions, and what limitations follow from those choices.

## 🌱 Dataset and initialization

### Dataset sources and materialization

The pipeline materializes each configured dataset with [`python/benchmark_pipeline/tools/dataset_gen.py`](../python/benchmark_pipeline/tools/dataset_gen.py). Synthetic benchmark cases use `source_kind="blobs"`, which generates `sklearn.datasets.make_blobs` inputs with `random_state=42`. The same pipeline can also materialize real datasets selected in the benchmark configuration through `RealDatasetConfig` / `BenchmarkCase`, currently via local or URL-provided dense arrays, OpenML, UCI ML Repository, or Hugging Face Datasets.

Synthetic blob cases remain deterministic for a fixed `(D, N, K)` configuration and unchanged generator code. Real-dataset cases are identified by an explicit dataset key and declared `(D, N, K)` shape; reproducibility depends on the selected source identifier, version/cache, and any upstream dataset changes.

For every source kind, the generator validates that the materialized input is finite numeric dense 2-D data with shape `(N, D)`, casts it to contiguous `float32`, and writes it as a flat row-major binary file. Both C++ and Python benchmarks read that same materialized binary input, so implementation comparisons are based on shared input data rather than separately loaded or regenerated datasets.

HDBSCAN uses a `float64` copy derived from the same materialized input for its benchmark path. Artifact names, summary entries, and reporting tables carry the dataset key, so results can include synthetic and real-dataset configurations side by side.

To reduce the confounding effect of synthetic data geometry, the center sampling box is scaled as a function of `D` and `K`. The scaling compensates for the fact that random center distances grow with dimension and that higher K increases center density in a fixed box. We keep cluster_std fixed at 1.0 and set $B = 10 \times \sqrt{10/D} \times (K/10)^{1/D}$, using `center_box=(-B, B)`.

### Initialization regime

The setup step also generates shared initial centers with `sklearn.cluster.kmeans_plusplus`, again using `random_state=42`.

Those seeded centers are used by the Lloyd / K-Means benchmarks. As a result, C++ and Python start from the same centroids for a given `(D, N, K)` configuration, and repeated pipeline runs regenerate the same initialization.

The GMM initialization is derived from those same K-Means++ centers only when the selected task graph contains GMM work. The generator assigns samples to their nearest initial center, derives initial mixture weights from those assignments, and estimates the requested precision parameters from the resulting groups. The GMM benchmarks therefore also start from deterministic, shared initialization artifacts.

This makes the Lloyd and GMM comparisons reproducible in the important experimental sense: the input data and initialization are fixed for each configuration.

### Consequence of K-Means++ initialization

Using K-Means++ initialization is a methodological choice, not a neutral default.

For Lloyd and GMM, the benchmark measures performance in a regime where the starting centers are usually already reasonable. This is different from measuring convergence from fully random initial centers. In practice, it can reduce the number of iterations and make the measured workload closer to a "good initialization" scenario than a worst-case or cold-start scenario.

This is an important limitation when interpreting results: the benchmark is not measuring every possible clustering regime. It measures the implementations under the configured dataset sources and K-Means++-initialized regime produced by the setup step.

### K-Means++ as a measured phase

The standalone K-Means++ benchmark is different from the seeded setup initialization.

The setup initialization is deterministic because it passes `random_state=42`. The measured K-Means++ phase currently does not use that seeded initialization path:

* the Python benchmark calls `sklearn.cluster.kmeans_plusplus` without `random_state`;
* the C++ implementation defaults to a seed from `std::random_device`.

So the K-Means++ benchmark runs on the same materialized dataset for the configuration, but the sampled centers can differ from run to run. The timing comparison therefore relies on a high number of benchmark iterations rather than identical sampled centers across repetitions.

## 📏 Measurement boundary

The benchmark distinguishes setup work from measured algorithm work.

For C++ nanobench runs, the benchmark case object is constructed before the timed `bench.run(...)` body. File reading, input loading, initial-center loading, GMM-parameter loading, and output writing happen outside the timed region. The measured region calls:

```cpp
bench_case.run_once();
bench_case.keep_alive();
```

The exact meaning of `run_once()` is defined by the C++ case adapter under [`cpp/benchmarks/cases/`](../cpp/benchmarks/cases/).

For Lloyd, `run_once()` copies the initial centroids and runs the K-Means iterations. Any work performed inside the algorithm implementation itself is included. For example, the implementations prepares centered data as part of the fit, and computes total final inertia after the fit, so these costs belongs to the measured algorithm call.

For GMM, `run_once()` runs the EM fit from the preloaded initial weights, means, and precisions. Loading those initialization arrays is not timed.

For K-Means++, `run_once()` performs the initialization procedure itself. Since this phase is stochastic today, different timing repetitions may use different sampled candidates thus taking a different amount of time.

For SoA conversion, the conversion is the thing being measured. The raw AoS data is loaded before timing, and `run_once()` copies it into the native layout. This sample-layout conversion is shared by Lloyd and GMM C++ paths, so the same SoA timing artifacts can be interpreted against either algorithm's iteration baseline.

For the isolated HDBSCAN `mreach` stage, the prepared distance matrix is treated as a reusable predecessor artifact. scikit-learn's dense mutual-reachability helper mutates its input in-place, so the Python reference copies the prepared distance matrix before calling that helper. The C++ isolated stage mirrors that staged contract by copying the prepared distance matrix into a preallocated scratch/output buffer and then running the same in-place mreach kernel used by the full C++ pipeline. The full HDBSCAN pipeline does not pay this preservation copy: after pairwise distances are computed, the distance matrix is overwritten with mutual-reachability values and consumed by the MST stage.

For Python benchmarks, each `bench_*.py` script loads the shared binary inputs before calling `pyperf.Runner.bench_func(...)`. The measured function is the wrapped scikit-learn operation. For Lloyd and GMM, estimator construction and `.fit(...)` are inside the measured function because they are part of the operation being compared. Thread-pool limiting is established outside `bench_func(...)` in the pyperf worker, so `threadpoolctl.threadpool_limits(limits=1)` context-manager setup and teardown are not part of the measured callable.

### Fixed-cost reporting

The notebook reports fixed C++ costs such as SoA conversion and standalone K-Means++ center selection against iterative algorithm baselines. For Lloyd, those costs are expressed in units of baseline Lloyd iterations. For GMM, they are expressed in units of baseline GMM EM iterations and are replicated across covariance-type parameterizations because the fixed-cost artifacts themselves are not covariance-specific.

The GMM fixed-cost plot should not be read as a full GMM-initialization benchmark. It includes K-Means++ center selection when that phase is enabled, but it does not include the extra setup work used to derive GMM weights and precision/covariance initialization artifacts.


## 🧵 Threading boundary

The current C++ SIMD implementations are single-threaded.

The benchmark is not intended to measure thread scaling, scheduling overhead, inter-thread reduction costs, or other thread-parallelism effects. In Python benchmark workers, `threadpoolctl.threadpool_limits(limits=1)` wraps the `pyperf.Runner.bench_func(...)` call, so those comparisons are made against scikit-learn with supported native thread pools limited to one thread.

Adding thread parallelism to the C++ implementations should be a separate benchmark regime rather than an implicit change to these results. The natural parallelization strategy would be similar in spirit to scikit-learn's: distribute independent sample-wise work across threads, then combine per-thread reductions for quantities such as cluster sums, responsibilities, or accumulated costs. If that is added, the benchmark should document the thread count and compare against scikit-learn under a matching thread policy.

## ⏲️ Timing and repetition

The timing policy is controlled by three fields in [`python/benchmark_pipeline/config.py`](../python/benchmark_pipeline/config.py). [`python/benchmark_pipeline/tasks.py`](../python/benchmark_pipeline/tasks.py) maps those fields to the generated Python and C++ commands.

`timing_processes` is the outer repetition level. Python receives it as pyperf `--processes`, so pyperf creates that many worker-process runs. C++ does not use pyperf to launch workers; instead, [`python/benchmark_pipeline/runner.py`](../python/benchmark_pipeline/runner.py) runs the nanobench executable that many times and merges the pyperf-shaped JSON files with [`python/benchmark_pipeline/tools/merge_pyperf_runs.py`](../python/benchmark_pipeline/tools/merge_pyperf_runs.py). The merge step stores a `timing_process_index` on each C++ run.

`timing_values` is the number of measured values requested inside each timing process. Python receives it as pyperf `--values`. C++ receives it as the nanobench epoch count.

`timing_min_time` is the target minimum duration for one measured value. Python receives it as pyperf `--min-time`; C++ receives it as nanobench `minEpochTime(...)`. This is not the duration of the whole benchmark task, and it is separate from the algorithm's own iteration count. pyperf and nanobench may internally repeat the measured function enough times to produce a value of the requested duration.

Both benchmark frontends use one warmup. Postprocessing ignores warmup/calibration data and expands only recorded JSON `values` into timing records.

## 📊 Metrics, speedups, and confidence intervals

Each language/phase/stage/variant/parameterization/configuration summary reports descriptive statistics for total time and per-algorithm-iteration time. The statistics are produced by [`python/benchmark_postprocess/stats.py`](../python/benchmark_postprocess/stats.py): count, median, mean, standard deviation, MAD, min/max, and selected percentiles. For Lloyd and GMM, per-algorithm-iteration time is total measured time divided by the iteration count stored in the metrics file. For non-iterative stages, the iteration count is treated as `1`.

Speedup is defined as `python_time / cpp_time`. Values above `1` therefore mean that the C++ implementation is faster for that phase/stage and configuration. Postprocessing reports both median-based and mean-based speedups.

Speedup intervals are computed in [`python/benchmark_postprocess/speedup.py`](../python/benchmark_postprocess/speedup.py) with a clustered bootstrap. For each bootstrap replicate, C++ and Python timing-process groups are separately sampled with replacement, and all timing values inside a selected process remain together. The requested statistic is recomputed for each sampled side, converted to a speedup ratio, and the reported interval is the percentile interval over those ratios.

The bootstrap iteration count, confidence level, and base seed are command-line options of `postprocess_benchmarks.py`. The code derives stable child seeds per configuration, phase, stage, and statistic, so unchanged inputs and postprocessing arguments produce the same intervals.

Per-algorithm-iteration speedup intervals are not bootstrapped separately. They are derived from the total-time speedup interval by scaling with the C++ and Python algorithm-iteration counts.

## ⚠️ Interpretation limits

The benchmark results should be interpreted within the limits of this methodology:

* results reflect whichever dataset sources were configured, typically controlled synthetic blob cases plus any configured real-dataset cases, and should not be read as covering all clustering workloads;
* Lloyd and GMM start from seeded K-Means++ initialization, not from arbitrary random centers;
* the standalone K-Means++ benchmark is stochastic today;
* timing results depend on the machine, compiler, build flags, and runtime environment;

The benchmark is therefore best read as a controlled comparison inside this repository's chosen regime, not as an absolute statement about all clustering workloads. \
The full limitations list/scope of this project is described in [🔭 Scope and non-goals](./limitations_and_future_work.md#-scope-and-non-goals)
