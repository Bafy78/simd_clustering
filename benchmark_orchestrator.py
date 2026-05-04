import subprocess
import sys
import os
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
        centers=n_clusters,
        random_state=42
    )
    X_float32 = X.astype(np.float32)
    X_float32.tofile(output_filename)
    print(f"Saved binary file to {output_filename}")

def build_and_run_cpp(dimensions, n_samples, n_clusters):
    cpp_source = "cpp/k-means.cpp"
    binary_name = "./kmeans_benchmark.bin"
    
    for dim in dimensions:
        print(f"\n{'='*40}")
        print(f"--- Benchmarking Dimension: {dim} ---")
        print(f"{'='*40}")
        
        # 1. Generate the dataset for this dimension
        data_file = f"./datasets/dataset_{dim}D.bin"
        create_and_save_dataset(n_samples, dim, n_clusters, data_file)
        
        # 2. Compile the C++ binary for this dimension
        compile_cmd = [
            "g++-14", "-O3", "-march=native", "-std=c++20",
            "-I../eve/include", f"-DTUPLE_SIZE={dim}",
            cpp_source, "-o", binary_name
        ]
        
        print(f"Compiling: {' '.join(compile_cmd)}")
        compile_process = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compile_process.returncode != 0:
            print(f"Compilation failed for dimension {dim}:\n{compile_process.stderr}")
            sys.exit(1)
            
        # 3. Execute the binary, passing the runtime arguments
        run_cmd = [binary_name, data_file, str(n_samples), str(n_clusters)]
        print(f"Running: {' '.join(run_cmd)}")
        
        run_process = subprocess.run(run_cmd, capture_output=True, text=True)
        if run_process.returncode != 0:
            print(f"Execution failed for dimension {dim}:\n{run_process.stderr}")
            sys.exit(1)
            
        print(run_process.stdout)

        # 4. Execute the Python (scikit-learn) benchmark
        python_script = "python/k-means_sklearn.py" 
        
        py_cmd = [
            sys.executable, 
            python_script, 
            data_file, 
            str(n_samples), 
            str(dim),
            str(n_clusters)
        ]
        
        print(f"Running Python: {' '.join(py_cmd)}")
        py_process = subprocess.run(py_cmd, capture_output=True, text=True)
        
        if py_process.returncode != 0:
            print(f"Python Execution failed for dimension {dim}:\n{py_process.stderr}")
            sys.exit(1)
            
        print("--- Python Scikit-Learn Output ---")
        print(py_process.stdout)

if __name__ == "__main__":
    test_dimensions = [2, 3, 10, 50]
    n_samples = 10000  # Start small to verify, scale up later
    n_clusters = 5
    
    build_and_run_cpp(test_dimensions, n_samples, n_clusters)