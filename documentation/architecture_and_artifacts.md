# 🧭 Architecture and artifact map

This document describes how the benchmark code is organized and which files are produced by the normal pipeline.

The main conceptual coordinates used throughout the project are `D` for dimensions, `N` for samples, and `K` for clusters or mixture components. All benchmark configurations
are identified by these three coordinates, together with the algorithm phase, implementation language, and sometimes algorithm variant. All the code also uses this naming convention.

## 🧩 Main supported algorithmic phases

The project exposes four major phases:

1. **SoA conversion**
2. **K-Means++ initialization**
3. **Lloyd / K-Means iterations**
4. **GMM EM**

Each phase has a different level of support across C++, Python, metrics, parity checks, and reporting.

| Phase           | C++ benchmark | Python reference |     Metrics | Parity / validation check |        Reporting |
| --------------- | ------------: | ---------------: | ----------: | ------------------------: | ---------------: |
| SoA conversion  |           ✅ |              ❌ |         ❌ |                       ❌ |          Limited |
| K-Means++       |           ✅ |              ✅ | Timing only |                      ❌ | Timing summaries |
| Lloyd / K-Means |           ✅ |              ✅ |         ✅ |                       ✅ |              ✅ |
| GMM EM          |           ✅ |              ✅ |         ✅ |                       ✅ |              ✅ |

The pipeline is modular: phases can be enabled or disabled depending on what is currently being studied. The project should therefore be understood as a collection of reusable benchmark stages rather than a single permanently fixed experiment script.

## 💻 C++ algorithm layer

The core implementations live under [`cpp/include`](../cpp/include)

### Benchmark entry points and case adapters

The C++ benchmark system separates generic benchmark execution from algorithm-specific setup.

The generic entry points are:
- [`cpp/benchmarks/nanobench_main.cpp`](../cpp/benchmarks/nanobench_main.cpp)
- [`cpp/benchmarks/callgrind_main.cpp`](../cpp/benchmarks/callgrind_main.cpp)
- [`cpp/benchmarks/spill_detector_main.cpp`](../cpp/benchmarks/spill_detector_main.cpp)

These files are generic dispatchers. They do not directly implement clustering algorithms. Instead, they are compiled with preprocessor definitions selecting a benchmark case header and case type, for example through:

```cpp
BENCH_CASE_HEADER
BENCH_CASE
TUPLE_SIZE
```

Algorithm-specific adapters live under [`cpp/benchmarks/cases`](../cpp/benchmarks/cases). These adapters bind command-line arguments, binary input loading, setup outside the measured region, `run_once`, benchmark metadata, keep-alive logic, and metrics writing.

### C++ case registry

The Python pipeline knows about C++ benchmark cases through a central registry in [`python/benchmark_pipeline/cpp_cases.py`](../python/benchmark_pipeline/cpp_cases.py).

The registry maps C++ benchmark case names, or `cpp_case` values, such as:
* `lloyd_static`
* `lloyd_dynamic`
* `gmm_static`
* `gmm_dynamic`
* `pp`
* `soa_static`
* `soa_dynamic`

to compile instructions, benchmark case headers, case types, and mode-specific build options.

This registry is reused by all the tools.

## 🐍 Python pipeline responsibilities

Lifecycle:

1. **Configuration:** [`python/benchmark_pipeline/config.py`](../python/benchmark_pipeline/config.py) and [`python/benchmark_pipeline/tasks.py`](../python/benchmark_pipeline/tasks.py).\
   * Define the sweep over `Dimensions`, `Samples`, `Clusters`, algorithm-specific parameters (eg. `gmm_covariance_type`), and benchmark parameters (described in more details in the [Benchmark methodology](benchmark_methodology.md#timing-and-repetition))
   * In `tasks.py`, tasks can be added, removed, or reordered in the pipeline. So you can choose not to run a specific algorithm by simply commenting out its corresponding tasks.

2. **Dataset generation:** [`python/benchmark_pipeline/tools/dataset_gen.py`](../python/benchmark_pipeline/tools/dataset_gen.py).
   * Generate input data and initialization artifacts shared by C++ and Python. It is currently using `make_blobs` and K-Means++ initial centers (converted to GMM parameters for GMM).

3. **C++ compilation**
   * Compile the required benchmark binaries for the selected cases and dimensions. As the pipeline is running the dimensions as the main loop, compilation is done once per dimension.

4. **Benchmark execution**
   * run C++ nanobench binaries;
   * run Python/scikit-learn benchmarks;
   * collect timing JSONs;

5. **Metrics collection**
   * collect algorithm outputs and raw validation metrics

6. **Postprocessing**
   * merge benchmark records;
   * compare C++ and Python metrics, including parity computation from current thresholds;
   * compute speedups;
   * build summary artifacts;

7. **Reporting**
   * load summaries;
   * transform records;
   * generate tables and plots.

The main orchestrator builds a task graph from the selected configuration. This makes it possible to focus on one or a few algorithms without changing the rest of the infrastructure.

### Artifact persistence

When running the [benchmark_pipeline](../python/benchmark_orchestrator.py), many artifacts get produced in the destination folder. Some are intermediate and are automatically cleaned up, the rest should persist until we run the [postprocessing step](../python/postprocess_benchmarks.py).

When this step is finished, it produces a summary artifact that is the canonical input for the [reporting step](../benchmark_analysis.ipynb). The postprocessing and reporting steps can both be run as many times as needed, as long as they still have access to their necessary artifacts

### Generated artifact names

Artifact names are derived from the configuration id `<D>D_<N>N_<K>K`. For example, the configuration `D=3`, `N=15000`, `K=10` uses `3D_15000N_10K`.

Persistent benchmark outputs use these patterns:

| Pattern | Meaning |
| --- | --- |
| `soa_cpp_<config>.json` | C++ SoA/native-layout timing. |
| `pp_cpp_<config>.json` | C++ K-Means++ timing. |
| `pp_py_<config>.json` | Python/scikit-learn K-Means++ timing. |
| `lloyd_cpp_<config>.json` | C++ Lloyd timing. |
| `lloyd_py_<config>.json` | Python/scikit-learn Lloyd timing. |
| `lloyd_metrics_cpp_<config>.json` | C++ Lloyd validation metrics. |
| `lloyd_metrics_py_<config>.json` | Python Lloyd validation metrics. |
| `gmm_cpp_<config>.json` | C++ GMM EM timing. |
| `gmm_py_<config>.json` | Python/scikit-learn GMM EM timing. |
| `gmm_metrics_cpp_<config>.json` | C++ GMM validation metrics. |
| `gmm_metrics_py_<config>.json` | Python GMM validation metrics. |
| `benchmark_summary.json` | Postprocessed summary consumed by reporting. |

Temporary binary inputs use these patterns:

| Pattern | Meaning |
| --- | --- |
| `data_<config>.bin` | Generated `float32` AoS dataset. |
| `init_<config>.bin` | K-Means++ initial centroids shared by Lloyd runs. |
| `gmm_weights_<config>.bin` | Initial GMM weights. |
| `gmm_means_<config>.bin` | Initial GMM means. |
| `gmm_precisions_<config>.bin` | Initial GMM precisions. |

## 🔬 Profiling and forensic tools

The project includes tools beyond normal benchmark execution.

### Callgrind profiling

The [callgrind tool](../python/callgrind_alg.py) compiles a selected C++ benchmark case in profiling mode, generates or reuses the required inputs, and runs the selected algorithm under Callgrind with Cache simulation enabled.

### Spill detection

The [spill detector tool](../python/spill_detector.py) compiles selected C++ cases to assembly and scans for stack spill/reload patterns with a regex.

What it flags should not be interpreted as confirmed harmful spills, but some kind of heuristic: these might be a spill, and if there are many of them, it might be pathological of high register pressure.

It also generates local-header-flattened version files of the code of the C++ algorithms tested for easy copy-paste into Compiler Explorer. To use that, you should probably have it locally installed as the generated assembly can be huge in high D.

### GMM diagnostics

The [`python/explain_gmm_diag_scikit_parity.py`](../python/explain_gmm_diag_scikit_parity.py) tool  investigates diagonal GMM parity behavior; see [Diagnostic tool for diagonal GMM](scikit_parity_and_validation.md#diagnostic-tool-for-diagonal-gmm).