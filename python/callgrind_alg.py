import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark_pipeline.cpp_cases import (
    CPP_CASES,
    callgrind_binary_path,
    cpp_compile_command,
)
from benchmark_pipeline.gmm_covariance import SUPPORTED_GMM_COVARIANCE_TYPES
from benchmark_pipeline.paths import REPO_ROOT, repo_path
from benchmark_pipeline.runner import run_command
from benchmark_pipeline.tasks import build_pipeline, config_id, dataset_path

PROFILE_EVENTS = "Ir,Dr,D1mr,DLmr,Dw,D1mw,DLmw"


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        print(f"Required tool not found on PATH: {tool}")
        sys.exit(1)


def compile_profile_binary(D: int, cpp_case: str) -> Path:
    os.makedirs(os.path.dirname(callgrind_binary_path(cpp_case, D)), exist_ok=True)

    out = callgrind_binary_path(cpp_case, D)
    cmd = cpp_compile_command(D=D, cpp_case=cpp_case, mode="callgrind")

    print(f"Compiling Callgrind {cpp_case} binary for {D}D...")
    run_command(f"Compile Callgrind {cpp_case} binary", cmd)
    return out


def run_and_capture(
    name: str, command: list[str], stdout_path: Path, stderr_path: Path
) -> None:
    print(f"[{name}] {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    if result.returncode != 0:
        print(f"\n{name} FAILED")
        print(f"stdout: {stdout_path}")
        print(f"stderr: {stderr_path}")
        print(result.stderr)
        sys.exit(result.returncode)


def add_cache_options(command: list[str], args: argparse.Namespace) -> None:
    if args.I1:
        command.append(f"--I1={args.I1}")
    if args.D1:
        command.append(f"--D1={args.D1}")
    if args.LL:
        command.append(f"--LL={args.LL}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one C++ benchmark case under Callgrind with cache simulation enabled."
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
    parser.add_argument("--out-dir", default=repo_path("callgrind_results"))

    # Optional explicit cache model. Example:
    # --D1 32768,8,64 --LL 33554432,16,64
    parser.add_argument("--I1")
    parser.add_argument("--D1")
    parser.add_argument("--LL")

    args = parser.parse_args()

    require_tool("valgrind")
    require_tool("callgrind_annotate")

    target = CPP_CASES[args.cpp_case]
    if target.needs_gmm_init and args.gmm_covariance_type not in target.supported_gmm_covariance_types:
        supported = ", ".join(target.supported_gmm_covariance_types)
        raise SystemExit(
            f"C++ case {args.cpp_case!r} does not support GMM covariance "
            f"type {args.gmm_covariance_type!r}. Supported values: {supported}"
        )

    config_id_value = config_id(args.D, args.N, args.K)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = callgrind_binary_path(args.cpp_case, args.D)

    if not args.skip_compile:
        binary = compile_profile_binary(args.D, args.cpp_case)

    if not args.skip_generate:
        dataset_task = build_pipeline(
            args.D,
            args.N,
            args.K,
            timing_processes=1,
            timing_values=1,
            timing_min_time=0.0,
            gmm_covariance_types=(args.gmm_covariance_type,) if target.needs_gmm_init else (),
            cpp_soa_cases=(),
            run_cpp_pp=False,
            run_python_pp=False,
            cpp_lloyd_cases=(),
            run_python_lloyd=False,
            cpp_gmm_cases=(args.cpp_case,) if target.needs_gmm_init else (),
            run_python_gmm=False,
        )[0]
        run_command(dataset_task.name, dataset_task.command)

    dataset_bin = dataset_path(f"data_{config_id_value}.bin")
    init_bin = dataset_path(f"init_{config_id_value}.bin")
    gmm_weights_bin = dataset_path(f"gmm_weights_{config_id_value}.bin")
    gmm_means_bin = dataset_path(f"gmm_means_{config_id_value}.bin")
    gmm_precisions_bin = dataset_path(
        f"gmm_precisions_{args.gmm_covariance_type}_{config_id_value}.bin"
    )

    callgrind_out = out_dir / f"callgrind.{args.cpp_case}.{config_id_value}.out"
    metrics_out = out_dir / f"{args.cpp_case}_metrics_cpp.{config_id_value}.json"
    valgrind_stdout = (
        out_dir / f"valgrind.callgrind.{args.cpp_case}.{config_id_value}.stdout.txt"
    )
    valgrind_stderr = (
        out_dir / f"valgrind.callgrind.{args.cpp_case}.{config_id_value}.stderr.txt"
    )
    annotate_out = out_dir / f"callgrind.{args.cpp_case}.{config_id_value}.annotate.txt"
    annotate_err = (
        out_dir / f"callgrind.{args.cpp_case}.{config_id_value}.annotate.stderr.txt"
    )

    callgrind_cmd = [
        "valgrind",
        "--tool=callgrind",
        "--cache-sim=yes",
        "--branch-sim=no",
        "--instr-atstart=no",
        f"--callgrind-out-file={callgrind_out}",
    ]
    add_cache_options(callgrind_cmd, args)

    cpp_case_args = [
        str(binary),
        dataset_bin,
        str(args.N),
    ]

    if target.needs_clusters_arg:
        cpp_case_args.append(str(args.K))

    if target.needs_init:
        cpp_case_args.append(init_bin)

    if target.needs_gmm_init:
        cpp_case_args.extend(
            [
                gmm_weights_bin,
                gmm_means_bin,
                gmm_precisions_bin,
            ]
        )

    if target.needs_covariance_type_arg:
        cpp_case_args.append(args.gmm_covariance_type)

    if target.needs_metrics:
        cpp_case_args.append(str(metrics_out))

    callgrind_cmd.extend(cpp_case_args)

    run_and_capture(
        f"Callgrind {args.cpp_case}",
        callgrind_cmd,
        valgrind_stdout,
        valgrind_stderr,
    )

    callgrind_annotate_cmd = [
        "callgrind_annotate",
        f"--show={PROFILE_EVENTS}",
        "--sort=D1mr,D1mw,DLmr,DLmw,Ir",
        "--inclusive=yes",
        "--tree=both",
        "--threshold=0.0",
        "--context=8",
        str(callgrind_out),
    ]

    run_and_capture(
        "callgrind_annotate",
        callgrind_annotate_cmd,
        annotate_out,
        annotate_err,
    )

    print("\nDone.")
    print(f"Callgrind raw output:      {callgrind_out}")
    print(f"Callgrind annotated:       {annotate_out}")
    if target.needs_metrics:
        print(f"Metrics JSON:              {metrics_out}")
    print(f"Valgrind stdout:           {valgrind_stdout}")
    print(f"Valgrind stderr:           {valgrind_stderr}")
    print("")
    print("Open the raw output directly in KCachegrind, for example:")
    print(f"  kcachegrind {callgrind_out}")


if __name__ == "__main__":
    main()
