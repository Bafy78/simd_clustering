# ⚡ Clustering SIMD Benchmarks

**C++20 SIMD implementations of clustering kernels benchmarked against scikit-learn references.**

This repository compares SIMD-oriented C++ clustering implementations using EVE against Python/scikit-learn baselines, with shared datasets, reproducible initialization, parity checks, and notebook-driven reporting.

> It is a benchmark and validation codebase, not a general-purpose clustering library.

The C++ side contains optimized clustering implementations and benchmark entry points; the Python side generates shared inputs, runs reference baselines, orchestrates benchmark tasks, postprocesses timings and metrics, and drives reporting.

The project currently covers:

- (Greedy) K-Means++ (not fully)
- Lloyd / K-Means
- GMM EM

The benchmark coordinates used throughout the repository are `D` for dimensions, `N` for samples, and `K` for clusters or mixture components.

## Quick start

Create a Python environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the configured benchmark sweep:

```bash
python python/benchmark_orchestrator.py
```

Postprocess the generated artifacts:

```bash
python python/postprocess_benchmarks.py
```

Open the notebook report:

```bash
jupyter notebook benchmark_analysis.ipynb
```

The C++ benchmark build also expects a C++20 compiler plus EVE and nanobench headers. See [Run and reproduce](documentation/run_and_reproduce.md) for the expected local layout and reproduction workflow.

## Documentation index

| Document | Use it for |
| --- | --- |
| [🚀 Run and reproduce](documentation/run_and_reproduce.md) | Installing prerequisites, running the sweep, postprocessing outputs, and opening the notebook. |
| [🧭 Architecture and artifact map](documentation/architecture_and_artifacts.md) | Understanding the C++/Python pipeline, benchmark case registry, generated artifact names, and auxiliary tooling. |
| [📏 Benchmark methodology](documentation/benchmark_methodology.md) | Dataset generation, initialization policy, timing boundaries, repetitions, speedups, confidence intervals, and interpretation limits. |
| [✅ Scikit-learn parity and validation](documentation/scikit_parity_and_validation.md) | Metric files, parity checks, validation layers, failure interpretation, and diagnostic limits. |
| [⚙️ SIMD and kernel mechanics](documentation/simd_and_kernel_mechanics.md) | Static-D and dynamic-D data layouts, SIMD implementation structure, and kernel-level mechanics. |
| [📊 Performance results](documentation/performance_results.md) | General benchmark results, comparisons, and interpretations. |
| [🚧 Limitations and future work](documentation/limitations_and_future_work.md) | Project-level limitations and follow-up work that are not part of the methodology contract. |
