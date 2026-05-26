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
from benchmark_pipeline.paths import REPO_ROOT, repo_path
from benchmark_pipeline.runner import run_command
from benchmark_pipeline.tasks import build_pipeline, config_id, dataset_path

PROFILE_EVENTS = "Ir,Dr,D1mr,DLmr,Dw,D1mw,DLmw"


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        print(f"Required tool not found on PATH: {tool}")
        sys.exit(1)


def compile_profile_binary(dim: int, alg: str) -> Path:
    os.makedirs(os.path.dirname(callgrind_binary_path(alg, dim)), exist_ok=True)

    out = callgrind_binary_path(alg, dim)
    cmd = cpp_compile_command(dim=dim, alg=alg, mode="callgrind")

    print(f"Compiling Callgrind {alg} binary for {dim}D...")
    run_command(f"Compile Callgrind {alg} binary", cmd)
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
        "--alg",
        choices=sorted(CPP_CASES),
        default="lloyd",
        help="C++ case to profile. Defaults to static Lloyd for backward compatibility.",
    )
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
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

    target = CPP_CASES[args.alg]
    case = config_id(args.dim, args.n_samples, args.n_clusters)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = callgrind_binary_path(args.alg, args.dim)

    if not args.skip_compile:
        binary = compile_profile_binary(args.dim, args.alg)

    if not args.skip_generate:
        dataset_task = build_pipeline(
            args.dim,
            args.n_samples,
            args.n_clusters,
            bench_processes=1,
            bench_values=1,
            bench_min_time=0.0,
        )[0]
        run_command(dataset_task.name, dataset_task.command)

    dataset_bin = dataset_path(f"data_{case}.bin")
    init_bin = dataset_path(f"init_{case}.bin")

    callgrind_out = out_dir / f"callgrind.{args.alg}.{case}.out"
    metrics_out = out_dir / f"{args.alg}_metrics_cpp.{case}.json"
    valgrind_stdout = out_dir / f"valgrind.callgrind.{args.alg}.{case}.stdout.txt"
    valgrind_stderr = out_dir / f"valgrind.callgrind.{args.alg}.{case}.stderr.txt"
    annotate_out = out_dir / f"callgrind.{args.alg}.{case}.annotate.txt"
    annotate_err = out_dir / f"callgrind.{args.alg}.{case}.annotate.stderr.txt"

    callgrind_cmd = [
        "valgrind",
        "--tool=callgrind",
        "--cache-sim=yes",
        "--branch-sim=no",
        "--instr-atstart=no",
        f"--callgrind-out-file={callgrind_out}",
    ]
    add_cache_options(callgrind_cmd, args)

    case_args = [
        str(binary),
        dataset_bin,
        str(args.n_samples),
    ]

    if target.needs_clusters_arg:
        case_args.append(str(args.n_clusters))

    if target.needs_init:
        case_args.append(init_bin)

    if target.needs_metrics:
        case_args.append(str(metrics_out))

    callgrind_cmd.extend(case_args)

    run_and_capture(
        f"Callgrind {args.alg}",
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
