# 🧭 Architecture and artifact map

This document describes how the benchmark code is organized and which files are produced by the normal pipeline.

The main conceptual coordinates used throughout the project are `D` for dimensions, `N` for samples, and `K` for clusters or mixture components. All benchmark configurations
are identified by these three coordinates, together with the algorithm phase, stage, implementation language, algorithm variant, and, for algorithms that need it, an algorithm parameter key. All the code also uses this naming convention.

## 🧩 Main supported algorithmic phases

The project exposes five major phases:

1. **SoA conversion**
2. **K-Means++ initialization**
3. **Lloyd / K-Means iterations**
4. **GMM EM**
5. **HDBSCAN**

Each phase has a different level of support across C++, Python, metrics, parity checks, and reporting.

| Phase           | C++ benchmark | Python reference |     Metrics | Parity / validation check |        Reporting |
| --------------- | ------------: | ---------------: | ----------: | ------------------------: | ---------------: |
| SoA conversion  |           ✅ |              ❌ |         ❌ |                       ❌ |          Limited |
| K-Means++       |           ✅ |              ✅ | Timing only |                      ❌ | Timing summaries |
| Lloyd / K-Means |           ✅ |              ✅ |         ✅ |                       ✅ |              ✅ |
| GMM EM          |           ✅ |              ✅ |         ✅ |                       ✅ |              ✅ |
| HDBSCAN         |           ✅ |              ✅ |         ✅ |                       ✅ |              ✅ |

The pipeline is modular: phases can be enabled or disabled depending on what is currently being studied. The project should therefore be understood as a collection of reusable benchmark steps rather than a single permanently fixed experiment script.

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
* ...

to compile instructions, benchmark case headers, case types, mode-specific build options, and the semantic keys needed by artifact naming:

* `phase_key`, such as `lloyd` or `gmm`;
* `stage_key`, usually `full`; HDBSCAN also uses staged keys for `distance`, `mst`, `linkage`,... ;
* `variant_key`, such as `static`, `dynamic`, or `auto`;
* `primary_input_artifact_key`, which identifies which logical setup artifact is passed to the case;
* GMM covariance support for cases that do not implement every covariance type.

The `variant_key` is deliberately shared across phases so postprocessing can pair related artifacts, for example `soa_static` with `lloyd_static` and `soa_dynamic` with `lloyd_dynamic`. Python/scikit-learn tasks use a shared `reference` variant and are compared against each C++ variant without duplicating the Python benchmark task.

Stage metadata lives in [`python/benchmark_pipeline/stages.py`](../python/benchmark_pipeline/stages.py). A stage declares the logical artifacts it consumes and which of those are predecessor/reference artifacts. [`python/benchmark_pipeline/tasks.py`](../python/benchmark_pipeline/tasks.py) turns that into `Task.input_artifacts` and `Task.output_artifacts`, so a stage can consume a dataset, generated initialization files, or a precomputed intermediate without timing the production of that input.

This registry is reused by all the tools.

## 🐍 Python pipeline responsibilities

Lifecycle:

1. **Configuration:** [`python/benchmark_pipeline/config.py`](../python/benchmark_pipeline/config.py) and [`python/benchmark_pipeline/tasks.py`](../python/benchmark_pipeline/tasks.py).\
   * Define the sweep over `Dimensions`, `Samples`, `Clusters`, algorithm-specific parameters (eg. `gmm_covariance_types`), and benchmark parameters (described in more details in the [Benchmark methodology](benchmark_methodology.md#timing-and-repetition))
   * `tasks.py` derives the task graph from the explicit config fields. To add or remove work, change the selected C++ cases and Python phase booleans in the config

2. **Dataset generation:** [`python/benchmark_pipeline/tools/dataset_gen.py`](../python/benchmark_pipeline/tools/dataset_gen.py).
   * Materialize each configured input dataset and initialization artifacts shared by C++ and Python. The dataset step supports synthetic `make_blobs` inputs plus configured real dataset sources such as local/URL dense arrays, OpenML, UCI, or Hugging Face. K-Means++ initial centers are generated after materialization. GMM initialization is generated only when at least one GMM task is enabled: weights and means are shared across covariance types, while precision files are generated only for the requested covariance types.

3. **C++ compilation**
   * Compile the required benchmark binaries for the selected cases and dimensions. As the pipeline is running the dimensions as the main loop, compilation is done once per dimension.
   * Immediately after each successful compile, record the resolved target architecture and executable size in `datasets/compile_artifacts.json`. This is done at compile time because the nanobench binary path is per C++ case and is overwritten by the next dimension.

4. **Benchmark execution**
   * run C++ nanobench binaries;
   * run Python/scikit-learn benchmarks;
   * collect timing JSONs;
   * when enabled, run one Cachegrind pass for each non-excluded C++ config/case/stage/parameterization target.

5. **Metrics collection**
   * collect algorithm outputs and raw validation metrics;
   * collect Cachegrind summary counters and annotation files when profiling is enabled.

6. **Postprocessing**
   * merge benchmark records;
   * merge Cachegrind records from `callgrind_results/`;
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

When this step is finished, it produces a summary artifact that is the canonical input for the [reporting step](../benchmark_analysis.ipynb). The postprocessing and reporting steps can both be run as many times as needed, as long as they still have access to their necessary artifacts.

Cachegrind artifacts live outside `datasets/` by default, under `callgrind_results/`, so the timing-artifact scanner does not confuse profiling JSON files with pyperf/nanobench timing files. The directory contains the raw Callgrind output, `callgrind_annotate` text output, per-run `cachegrind.*.json` summaries, and `cachegrind_manifest.json`.

## 🔬 Profiling and forensic tools

The project includes tools beyond normal benchmark execution.

### Cachegrind profiling

The [standalone callgrind tool](../python/callgrind_alg.py) compiles a selected C++ benchmark case in profiling mode, generates or reuses the required inputs, and runs the selected algorithm under Callgrind with cache simulation enabled. The pipeline and notebook report those counters as Cachegrind results.

### Spill detection

The [spill detector tool](../python/spill_detector.py) compiles selected C++ cases to assembly and scans for stack spill/reload patterns with a regex.

What it flags should not be interpreted as confirmed harmful spills, but some kind of heuristic: these might be a spill, and if there are many of them, it might be pathological of high register pressure.

It also generates local-header-flattened version files of the code of the C++ algorithms tested for easy copy-paste into Compiler Explorer. To use that, you should probably have it locally installed as the generated assembly can be huge in high D.

### GMM diagnostics

The [`python/explain_gmm_diag_scikit_parity.py`](../python/explain_gmm_diag_scikit_parity.py) tool  investigates diagonal GMM parity behavior; see [Diagnostic tool for diagonal GMM](scikit_parity_and_validation.md#-diagnostic-tool-for-diagonal-gmm).