import argparse
import os
import subprocess
import sys
from pathlib import Path

from benchmark_pipeline.paths import BIN_DIR, REPO_ROOT, repo_path
from benchmark_pipeline.runner import run_command
from benchmark_pipeline.tasks import build_pipeline, config_id, dataset_path


def compile_cachegrind_binary(dim: int) -> Path:
    os.makedirs(BIN_DIR, exist_ok=True)

    src = repo_path("cpp", "benchmarks", "profile_lloyd_cachegrind.cpp")
    out = BIN_DIR / f"profile_lloyd_cachegrind_{dim}D.bin"

    cmd = [
        "g++-14",
        "-O3",
        "-g",
        "-march=native",
        "-std=c++20",
        "-I../eve/include",
        "-I../nanobench/src/include",
        f"-DTUPLE_SIZE={dim}",
        "-DKMEANS_K_TILE=5",
        "-DKMEANS_M_GROUP=2",
        src,
        "-o",
        str(out),
    ]

    print(f"Compiling Cachegrind Lloyd binary for {dim}D...")
    run_command("Compile Cachegrind Lloyd binary", cmd)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--out-dir", default=repo_path("cachegrind_results"))

    # Optional explicit cache model. Example:
    # --D1 32768,8,64 --LL 33554432,16,64
    parser.add_argument("--I1")
    parser.add_argument("--D1")
    parser.add_argument("--LL")

    args = parser.parse_args()

    case = config_id(args.dim, args.n_samples, args.n_clusters)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = BIN_DIR / f"profile_lloyd_cachegrind_{args.dim}D.bin"

    if not args.skip_compile:
        binary = compile_cachegrind_binary(args.dim)

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

    cg_out = out_dir / f"cachegrind.lloyd.{case}.out"
    metrics_out = out_dir / f"lloyd_metrics_cpp.{case}.json"
    valgrind_stdout = out_dir / f"valgrind.lloyd.{case}.stdout.txt"
    valgrind_stderr = out_dir / f"valgrind.lloyd.{case}.stderr.txt"
    annotate_out = out_dir / f"cachegrind.lloyd.{case}.annotate.txt"
    annotate_err = out_dir / f"cachegrind.lloyd.{case}.annotate.stderr.txt"

    valgrind_cmd = [
        "valgrind",
        "--tool=cachegrind",
        "--cache-sim=yes",
        "--branch-sim=no",
        "--instr-at-start=no",
        f"--cachegrind-out-file={cg_out}",
    ]

    if args.I1:
        valgrind_cmd.append(f"--I1={args.I1}")
    if args.D1:
        valgrind_cmd.append(f"--D1={args.D1}")
    if args.LL:
        valgrind_cmd.append(f"--LL={args.LL}")

    valgrind_cmd.extend(
        [
            str(binary),
            dataset_bin,
            str(args.n_samples),
            str(args.n_clusters),
            init_bin,
            str(metrics_out),
        ]
    )

    run_and_capture(
        "Cachegrind Lloyd",
        valgrind_cmd,
        valgrind_stdout,
        valgrind_stderr,
    )

    annotate_cmd = [
        "cg_annotate",
        "--show=Ir,Dr,D1mr,DLmr,Dw,D1mw,DLmw",
        "--sort=D1mr,D1mw,DLmr,DLmw,Ir",
        "--threshold=0.0",
        "--context=8",
        str(cg_out),
    ]

    run_and_capture(
        "cg_annotate",
        annotate_cmd,
        annotate_out,
        annotate_err,
    )

    print("\nDone.")
    print(f"Raw Cachegrind output: {cg_out}")
    print(f"Annotated report:      {annotate_out}")
    print(f"Metrics JSON:          {metrics_out}")
    print(f"Valgrind stderr:       {valgrind_stderr}")


if __name__ == "__main__":
    main()
