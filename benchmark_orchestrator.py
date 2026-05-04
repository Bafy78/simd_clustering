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

def compute_inertia(data_file, result_file, n_samples, n_features):
    """Reads the result file and calculates the sum of squared distances to centroids."""
    raw_data = np.fromfile(data_file, dtype=np.float32).reshape((n_samples, n_features))
    
    with open(result_file, 'r') as f:
        lines = f.read().splitlines()
        
    centroids = []
    clusters = {}
    
    mode = None
    for line in lines:
        if line == "[Centroids]":
            mode = "centroids"
            continue
        elif line == "[Clusters]":
            mode = "clusters"
            continue
            
        if mode == "centroids":
            centroids.append([float(x) for x in line.split()])
        elif mode == "clusters":
            parts = line.split(":")
            k = int(parts[0])
            indices_str = parts[1].strip()
            clusters[k] = [int(x) for x in indices_str.split()] if indices_str else []
            
    centroids = np.array(centroids, dtype=np.float32)
    inertia = 0.0
    
    for k, indices in clusters.items():
        if not indices:
            continue
        pts = raw_data[indices]
        c = centroids[k]
        # Sum of squared Euclidean distances
        inertia += np.sum((pts - c) ** 2)
        
    return inertia

def build_and_run_cpp(dimensions, n_samples, n_clusters):
    cpp_source = "cpp/k-means.cpp"
    binary_name = "./kmeans_benchmark.bin"
    
    for dim in dimensions:
        print(f"\n{'='*40}")
        print(f"--- Benchmarking Dimension: {dim} ---")
        print(f"{'='*40}")
        
        data_file = f"./datasets/dataset_{dim}D.bin"
        cpp_out_file = f"./datasets/results_cpp_{dim}D.txt"
        py_out_file = f"./datasets/results_py_{dim}D.txt"
        
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
        run_cmd = [binary_name, data_file, str(n_samples), str(n_clusters), cpp_out_file]
        print(f"Running: {' '.join(run_cmd)}")
        
        run_process = subprocess.run(run_cmd, capture_output=True, text=True)
        if run_process.returncode != 0:
            print(f"Execution failed for dimension {dim}:\n{run_process.stderr}")
            sys.exit(1)
        
        # 4. Execute the Python (scikit-learn) benchmark
        python_script = "python/k-means_sklearn.py" 
        py_cmd = [
            sys.executable, 
            python_script, 
            data_file, 
            str(n_samples), 
            str(dim),
            str(n_clusters),
            py_out_file
        ]
        
        print(f"Running Python: {' '.join(py_cmd)}")
        py_process = subprocess.run(py_cmd, capture_output=True, text=True)
        
        if py_process.returncode != 0:
            print(f"Python Execution failed for dimension {dim}:\n{py_process.stderr}")
            sys.exit(1)
            
        # 5. Evaluate and Compare
        cpp_inertia = compute_inertia(data_file, cpp_out_file, n_samples, dim)
        py_inertia = compute_inertia(data_file, py_out_file, n_samples, dim)
        
        print(f"C++ EVE Inertia:      {cpp_inertia:.2f}")
        print(f"Scikit-Learn Inertia: {py_inertia:.2f}")
        
        # Calculate percentage difference
        diff = abs(cpp_inertia - py_inertia) / max(cpp_inertia, py_inertia) * 100
        print(f"Difference:           {diff:.4f}%")

if __name__ == "__main__":
    test_dimensions = [2, 3, 10, 50]
    n_samples = 10000  # Start small to verify, scale up later
    n_clusters = 5
    
    build_and_run_cpp(test_dimensions, n_samples, n_clusters)