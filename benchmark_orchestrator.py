import os
import sys
import subprocess
from dataclasses import dataclass
from typing import List

@dataclass
class Task:
    name: str
    command: List[str]

def compile_cpp_binaries(dim: int):
    """Compiles the independent C++ benchmarks for a specific dimension."""
    print(f"\n{'='*50}")
    print(f"--- Compiling C++ Binaries for {dim}D ---")
    print(f"{'='*50}")
    
    os.makedirs("./bin", exist_ok=True)
    
    cpp_targets = [
        {"src": "cpp/benchmarks/bench_soa.cpp", "bin": "./bin/bench_soa.bin"},
        {"src": "cpp/benchmarks/bench_pp.cpp", "bin": "./bin/bench_pp.bin"},
        {"src": "cpp/benchmarks/bench_lloyd.cpp", "bin": "./bin/bench_lloyd.bin"}
    ]
    
    for target in cpp_targets:
        cmd = [
            "g++-14", "-O3", "-march=native", "-std=c++20",
            "-I../eve/include", "-I../nanobench/src/include", 
            f"-DTUPLE_SIZE={dim}",
            target["src"], "-o", target["bin"]
        ]
        print(f"Compiling {target['src']}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed for {target['src']}:\n{result.stderr}")
            sys.exit(1)

def build_pipeline(dim: int, n_samples: int, n_clusters: int) -> List[Task]:
    """Defines the strict 'contract' of tasks for a single configuration."""
    config_id = f"{dim}D_{n_samples}S_{n_clusters}K"
    
    # Standardized file paths
    dataset_bin = f"./datasets/data_{config_id}.bin"
    init_centroids_bin = f"./datasets/init_{config_id}.bin"
    
    tasks = []

    # Phase A: Setup (The Orchestrator generates the baseline)
    tasks.append(Task(
        name="Setup: Generate Dataset & Initial Centroids",
        command=[
            sys.executable, "python/dataset_gen.py",
            "--n-samples", str(n_samples),
            "--n-features", str(dim),
            "--n-clusters", str(n_clusters),
            "--dataset-out", dataset_bin,
            "--centroids-out", init_centroids_bin
        ]
    ))

    # Phase B: Independent Benchmarks
    tasks.append(Task(
        name="C++: AoS to SoA Tax",
        command=[
            "./bin/bench_soa.bin", 
            dataset_bin, str(n_samples), 
            f"./datasets/soa_cpp_{config_id}.json"
        ]
    ))
    
    tasks.append(Task(
        name="C++: K-Means++ Initialization",
        command=[
            "./bin/bench_pp.bin", 
            dataset_bin, str(n_samples), str(n_clusters), 
            f"./datasets/pp_cpp_{config_id}.json"
        ]
    ))

    tasks.append(Task(
        name="Python: K-Means++ Initialization",
        command=[
            sys.executable, "python/bench_pp.py",
            "--dataset-bin", dataset_bin,
            "--n-samples", str(n_samples),
            "--n-features", str(dim),
            "--n-clusters", str(n_clusters),
            "--output", f"./datasets/pp_py_{config_id}.json"
        ]
    ))

    # Phase C: Dependent Benchmarks (Lloyd Iterations)
    tasks.append(Task(
        name="C++: Lloyd Iterations",
        command=[
            "./bin/bench_lloyd.bin",
            dataset_bin, str(n_samples), str(n_clusters),
            init_centroids_bin,
            f"./datasets/results_cpp_{config_id}.txt",
            f"./datasets/lloyd_cpp_{config_id}.json"
        ]
    ))

    tasks.append(Task(
        name="Python: Lloyd Iterations",
        command=[
            sys.executable, "python/bench_lloyd.py",
            "--dataset-bin", dataset_bin,
            "--n-samples", str(n_samples),
            "--n-features", str(dim),
            "--n-clusters", str(n_clusters),
            "--init-centroids-bin", init_centroids_bin,
            "--result-file", f"./datasets/results_py_{config_id}.txt",
            "--output", f"./datasets/lloyd_py_{config_id}.json"
        ]
    ))

    return tasks

def execute_pipeline(dim: int, n_samples: int, n_clusters: int):
    """Executes the task list for a specific configuration."""
    print(f"\n--- Running Config: {dim}D | {n_samples} Samples | {n_clusters} Clusters ---")
    
    pipeline = build_pipeline(dim, n_samples, n_clusters)
    
    for task in pipeline:
        # If the command specifies an output file, remove it so pyperf doesn't complain
        if "--output" in task.command:
            output_idx = task.command.index("--output") + 1
            output_path = task.command[output_idx]
            if os.path.exists(output_path):
                os.remove(output_path)
        
        print(f"[{task.name}] Running...")
        result = subprocess.run(task.command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\nTask '{task.name}' FAILED!")
            print(f"Command: {' '.join(task.command)}")
            print(f"Return Code: {result.returncode}")
            print(f"Stdout:\n{result.stdout}")
            print(f"Stderr:\n{result.stderr}")
            sys.exit(1)

if __name__ == "__main__":
    os.makedirs("./datasets", exist_ok=True)
    
    test_dimensions = [2, 3, 5, 8, 12, 20]
    test_samples = [10000, 100000, 1000000]
    test_clusters = [3, 5, 10, 20]

    for dim in test_dimensions:
        compile_cpp_binaries(dim)
        
        for n in test_samples:
            for k in test_clusters:
                execute_pipeline(dim, n, k)
                
    print("\nAll benchmarking finished successfully!")