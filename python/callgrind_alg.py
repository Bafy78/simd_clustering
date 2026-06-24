import argparse
from pathlib import Path

from benchmark_pipeline.cachegrind import CACHEGRIND_RESULTS_DIR, require_cachegrind_tools
from benchmark_pipeline.config import PipelineOptions
from benchmark_pipeline.cpp_cases import CPP_CASES, get_cpp_case
from benchmark_pipeline.gmm_covariance import SUPPORTED_GMM_COVARIANCE_TYPES
from benchmark_pipeline.paths import DATASETS_DIR, repo_path, repo_relative_path
from benchmark_pipeline.runner import (
    compile_callgrind_binaries,
    run_cachegrind_task,
    run_command,
)
from benchmark_pipeline.tasks import (
    BenchmarkArtifacts,
    BenchmarkCase,
    build_cachegrind_task,
    build_dataset_setup_task,
)


def _single_cachegrind_options(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        timing_processes=1,
        timing_values=1,
        timing_min_time=0.0,
        gmm_covariance_types=(args.gmm_covariance_type,),
        cpp_soa_cases=(),
        cpp_pp_cases=(),
        run_python_pp=False,
        cpp_lloyd_cases=(),
        run_python_lloyd=False,
        cpp_gmm_cases=(),
        run_python_gmm=False,
        cpp_hdbscan_cases=(),
        run_python_hdbscan=False,
        hdbscan_references=("sklearn_brute",),
        hdbscan_stages=("full",),
        run_cachegrind=True,
        cachegrind_results_dir=str(args.out_dir),
        cachegrind_I1=args.I1,
        cachegrind_D1=args.D1,
        cachegrind_LL=args.LL,
        cachegrind_exclusion_rules=(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one C++ benchmark case under Cachegrind counters via Callgrind --cache-sim=yes."
    )
    parser.add_argument(
        "--cpp-case",
        choices=sorted(CPP_CASES),
        default="lloyd_static",
        help="C++ case to profile. Defaults to static Lloyd.",
    )
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument(
        "--gmm-covariance-type",
        choices=SUPPORTED_GMM_COVARIANCE_TYPES,
        default="spherical",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--datasets-dir", default=str(DATASETS_DIR))
    parser.add_argument("--out-dir", type=Path, default=repo_path(CACHEGRIND_RESULTS_DIR))

    # Optional explicit cache model. Example:
    # --I1 32768,8,64 --D1 32768,8,64 --LL 33554432,16,64
    parser.add_argument("--I1")
    parser.add_argument("--D1")
    parser.add_argument("--LL")

    args = parser.parse_args()

    require_cachegrind_tools()

    target = get_cpp_case(args.cpp_case)
    if (
        target.needs_gmm_init
        and args.gmm_covariance_type not in target.supported_gmm_covariance_types
    ):
        supported = ", ".join(target.supported_gmm_covariance_types)
        raise SystemExit(
            f"C++ case {args.cpp_case!r} does not support GMM covariance "
            f"type {args.gmm_covariance_type!r}. Supported values: {supported}"
        )

    case = BenchmarkCase(args.D, args.N, args.K)
    datasets_dir = repo_relative_path(args.datasets_dir)
    args.out_dir = repo_relative_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_compile:
        compile_callgrind_binaries(args.D, (args.cpp_case,))

    setup_gmm_covariance_types = (
        (args.gmm_covariance_type,) if target.needs_gmm_init else ()
    )

    if not args.skip_generate:
        dataset_task = build_dataset_setup_task(
            case,
            setup_gmm_covariance_types,
            datasets_dir,
        )
        run_command(dataset_task.name, dataset_task.command)

    params_key = args.gmm_covariance_type if target.needs_gmm_init else "default"
    task = build_cachegrind_task(
        case=case,
        cpp_case=args.cpp_case,
        artifacts=BenchmarkArtifacts.for_case(case, datasets_dir),
        options=_single_cachegrind_options(args),
        params_key=params_key,
    )
    run_cachegrind_task(task)

    info = task.cachegrind
    assert info is not None
    print("\nDone.")
    print(f"Cachegrind raw output:      {info.raw_output}")
    print(f"Cachegrind annotated:       {info.annotated_output}")
    print(f"Summary JSON:              {info.summary_path}")
    if info.metrics_path is not None:
        print(f"Metrics JSON:              {info.metrics_path}")
    print(f"Valgrind stdout:           {info.stdout_path}")
    print(f"Valgrind stderr:           {info.stderr_path}")
    print("")
    print("Open the raw output directly in KCachegrind, for example:")
    print(f"  kcachegrind {info.raw_output}")


if __name__ == "__main__":
    main()
