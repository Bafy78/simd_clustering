# Run and reproduce

This document is the operational guide for reproducing the benchmark artifacts. All commands below assume they are run from the repository root.

## 1. Install system prerequisites

The C++ compile command is defined in [`python/benchmark_pipeline/cpp_cases.py`](../python/benchmark_pipeline/cpp_cases.py). By default it expects:

- a C++20 compiler named `g++-14` on `PATH`;
- an x86-64-like target where `-march=native` is appropriate;
- [EVE](https://github.com/jfalcou/eve) headers at `../eve/include`, relative to the repository root;
- [nanobench](https://github.com/martinus/nanobench) headers at `../nanobench/src/include`, relative to the repository root.

A typical workspace layout is:

```text
workspace/
├── clustering/
├── eve/
│   └── include/
└── nanobench/
    └── src/
        └── include/
```

If the compiler or dependency locations differ, update `cpp_compile_command()` in [`python/benchmark_pipeline/cpp_cases.py`](../python/benchmark_pipeline/cpp_cases.py).

Optional dependencies are only needed for the forensic scripts described in [Profiling and forensic tools](architecture_and_artifacts.md#profiling-and-forensic-tools):

- `valgrind` and `callgrind_annotate` for Callgrind profiling;
- `kcachegrind` if you want to inspect raw Callgrind files interactively;
- `rg` / ripgrep for spill detection.

## 2. Create the Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file covers the Python benchmark runners, dataset generation, postprocessing, reporting helpers, and notebook dependencies.

## 3. Review or adjust the benchmark configuration

The default sweep is controlled by `default_config()` in [`python/benchmark_pipeline/config.py`](../python/benchmark_pipeline/config.py). The task graph is built in [`python/benchmark_pipeline/tasks.py`](../python/benchmark_pipeline/tasks.py).

The runner executes the phases described in [Main supported algorithmic phases](architecture_and_artifacts.md#main-supported-algorithmic-phases). For the methodology behind the generated inputs, measured regions, repetition model, and speedup intervals, see the [Benchmark methodology](benchmark_methodology.md).

## 4. Run the benchmark sweep

```bash
python python/benchmark_orchestrator.py
```

The orchestrator prepares the datasets directory, compiles the required C++ cases for each configured dimension, then executes the configured task graph for each `(D, N, K)` combination.

Raw timing and metrics files are written under `datasets/`. Temporary binary input files are created during each configuration run and removed by the normal orchestrator. The expected artifact naming convention is documented in [Generated artifact names](architecture_and_artifacts.md#generated-artifact-names).

## 5. Postprocess benchmark outputs

The postprocessor defaults to reading `./datasets` and writing `./datasets/benchmark_summary.json`, so the normal command is:

```bash
python python/postprocess_benchmarks.py
```

Override the paths only when using non-default locations:

```bash
python python/postprocess_benchmarks.py \
  --data-dir path/to/datasets \
  --output path/to/benchmark_summary.json
```

Bootstrap and confidence-interval parameters can also be overridden when needed:

```bash
python python/postprocess_benchmarks.py \
  --bootstrap-iterations 1000 \
  --ci-level 0.95 \
  --bootstrap-seed 12345
```

The summary-generation and parity responsibilities are described in [Postprocessing and summary output](scikit_parity_and_validation.md#postprocessing-and-summary-output). The resulting `benchmark_summary.json` is the canonical input for reporting.

## 6. Open the notebook report

```bash
jupyter notebook benchmark_analysis.ipynb
```

The notebook expects to be launched from the repository root and reads `datasets/benchmark_summary.json` through the reporting helpers. You can always change its dataset folder (where the summary should be retrieved from) in the first cell.

## 7. Optional forensic tools

These tools are not part of the normal benchmark reproduction path. Their purpose is summarized in [Profiling and forensic tools](architecture_and_artifacts.md#profiling-and-forensic-tools).

### Callgrind

```bash
python python/callgrind_alg.py \
  --cpp-case lloyd_static \
  --D 3 \
  --N 15000 \
  --K 10
```

Outputs are written to `callgrind_results/`.

Cachegrind can easily mistake your cache configuration if run from a VM or WSL. So you should probably find out your real specs and pass them as arguments:

```bash
  --I1 32768,8,64 \
  --D1 32768,8,64 \
  --LL 16777216,16,64
```

### Spill detector

```bash
python python/spill_detector.py
```

Outputs are written to `spill_detector_results/`.

### Diagonal GMM parity explainer

```bash
python python/explain_gmm_diag_scikit_parity.py --force
```

Outputs are written to `diagnostics_results/`

## Troubleshooting

### `g++-14: command not found`

Install GCC 14, create a wrapper/symlink named `g++-14`, or update `cpp_compile_command()` in [`python/benchmark_pipeline/cpp_cases.py`](../python/benchmark_pipeline/cpp_cases.py).

### Missing EVE or nanobench headers

Place EVE and nanobench in the expected sibling directories or update the include paths in `cpp_compile_command()`.


### Postprocessing produces no parity records

Postprocessing computes Lloyd and GMM parity only for configurations that have the required C++ and Python metrics files. See [Validation layers](scikit_parity_and_validation.md#validation-layers) and [How to interpret failures](scikit_parity_and_validation.md#how-to-interpret-failures) before treating a missing or failed parity record as a timing result.

### Timings are unstable across runs or machines

The reproducibility limits are methodological rather than operational. See [Interpretation limits](benchmark_methodology.md#interpretation-limits) for the intended interpretation of timing differences.
