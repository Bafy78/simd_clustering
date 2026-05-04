import subprocess
import sys
import os
import csv
import numpy as np
from sklearn.datasets import make_blobs

def create_and_save_dataset(n_samples, n_features, n_clusters, output_filename):
    print(f"Generating {n_samples} samples in {n_features}D...")

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    X, _, *_ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters
    )
    X_float32 = X.astype(np.float32)
    X_float32.tofile(output_filename)
    print(f"Saved binary file to {output_filename}")

def benchmark_config(binary_name, dim, n_samples, n_clusters):
    config_name = f"{dim}D_{n_samples}S_{n_clusters}K"

    print(f"\n{'-'*50}")
    print(f"--- Benchmarking Config: {config_name} ---")
    print(f"{'-'*50}")
    
    data_file = f"./datasets/dataset_{config_name}.bin"
    cpp_out_file = f"./datasets/results_cpp_{config_name}.txt"
    py_out_file = f"./datasets/results_py_{config_name}.txt"
    cpp_bench_file = f"./datasets/bench_cpp_{config_name}"
    py_bench_file = f"./datasets/bench_py_{config_name}.json"

    # Clean up existing bench files to prevent appending issues in nanobench/pyperf
    for f_path in [
        cpp_bench_file + "_aos_to_soa.json", 
        cpp_bench_file + "_kmeans_fit.json", 
        py_bench_file
    ]:
        if os.path.exists(f_path):
            os.remove(f_path)
    
    create_and_save_dataset(n_samples, dim, n_clusters, data_file)
    
    # Execute the C++
    run_cmd = [binary_name, data_file, str(n_samples), str(n_clusters), cpp_out_file, cpp_bench_file]
    print("Running C++ (Nanobench)...")
    run_process = subprocess.run(run_cmd, check=True)
    if run_process.returncode != 0:
        print(f"Execution failed for config {config_name}:\n{run_process.stderr}")
        sys.exit(1)
    
    # Execute the Python
    python_script = "python/k-means_sklearn.py" 
    py_cmd = [
        sys.executable, python_script, 
        "--output", py_bench_file, 
        "--binary-file", data_file,     
        "--n-samples", str(n_samples),  
        "--n-features", str(dim),       
        "--n-clusters", str(n_clusters),
        "--result-file", py_out_file
    ]
    print("Running Python (Pyperf)...")
    py_process = subprocess.run(py_cmd, check=True)
    if py_process.returncode != 0:
        print(f"Python Execution failed for config {config_name}:\n{py_process.stderr}")
        sys.exit(1)
    
    # Return manifest dictionary (no calculations)
    return {
        "Dimensions": dim,
        "Samples": n_samples,
        "Clusters": n_clusters,
        "Data File": data_file,
        "CPP Out File": cpp_out_file,
        "Py Out File": py_out_file,
        "CPP Bench Base": cpp_bench_file,
        "Py Bench File": py_bench_file
    }

def build_and_run_cpp(dimensions, sample_sizes, cluster_counts, summary_filename="benchmark_manifest.csv"):
    cpp_source = "cpp/k-means.cpp"
    binary_name = "./kmeans_benchmark.bin"
    
    all_results = []

    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"--- Compiling for Dimension: {dim}D ---")
        print(f"{'='*60}")
        
        # Compile the C++ binary ONCE per dimension
        compile_cmd = [
            "g++-14", "-O3", "-march=native", "-std=c++20",
            "-I../eve/include", "-I../nanobench/src/include", f"-DTUPLE_SIZE={dim}",
            cpp_source, "-o", binary_name
        ]
        print(f"Compiling: {' '.join(compile_cmd)}")
        compile_process = subprocess.run(compile_cmd, check=True)
        if compile_process.returncode != 0:
            print(f"Compilation failed for dimension {dim}:\n{compile_process.stderr}")
            sys.exit(1)

        # Iterate through sample sizes and cluster configurations
        for n_samples in sample_sizes:
            for n_clusters in cluster_counts:
                config_results = benchmark_config(binary_name, dim, n_samples, n_clusters)
                all_results.append(config_results)

    # Save manifest for the Jupyter Notebook
    with open(summary_filename, 'w', newline='') as csvfile:
        fieldnames = [
            "Dimensions", "Samples", "Clusters", 
            "Data File", "CPP Out File", "Py Out File", 
            "CPP Bench Base", "Py Bench File"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    
    print(f"\nAll benchmarking finished! Run manifest saved to: {summary_filename}")

if __name__ == "__main__":
    test_dimensions = [2, 3, 10, 50]
    test_samples = [10000, 100000, 1000000]
    test_clusters = [3, 5, 10]

    build_and_run_cpp(test_dimensions, test_samples, test_clusters)