import re
from pathlib import Path
import json
import pandas as pd
import numpy as np

from .benchmark_constants import *

def extract_config_params(filename):
    """Extracts Dimension, Samples, and Clusters from a string using regex."""
    match = re.search(r"(\d+)D_(\d+)S_(\d+)K", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

def extract_iterations_from_file(filepath):
    """Parses the Lloyd Iteration count from a results text file."""
    if not filepath.exists():
        return 1
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Look for the tag and grab the number immediately following it
            match = re.search(r"\[Lloyd Iterations\]\s*\n\s*(\d+)", content)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse iterations from {filepath.name}: {e}")
    return 1

def load_benchmark_data(data_dir=Path("./datasets")):
    records = []
    path = Path(data_dir)
    
    for filepath in path.glob("*.json"):
        dim, samples, clusters = extract_config_params(filepath.name)
        if dim is None:
            print(f"Skipping malformed file: {filepath.name}")
            continue
            
        phase_key = filepath.name.split("_")[0]
        lang_key = filepath.name.split("_")[1]

        txt_filename = f"results_{lang_key}_{dim}D_{samples}S_{clusters}K.txt"
        txt_filepath = path / txt_filename
        
        iterations = extract_iterations_from_file(txt_filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for bench in data.get("benchmarks", []):
            for run in bench.get("runs", []):
                for val in run.get("values", []):
                    records.append({
                        COL_PHASE: PHASE_MAP.get(phase_key, phase_key),
                        COL_LANGUAGE: LANG_CPP if lang_key == "cpp" else LANG_PY,
                        COL_DIMENSIONS: dim,
                        COL_SAMPLES: samples,
                        COL_CLUSTERS: clusters,
                        COL_ITERATIONS: iterations if phase_key == "lloyd" else 1,
                        COL_TIME_S: val,
                        COL_CONFIGURATION: f"{dim}D | {samples}S | {clusters}K",
                    })
                    
    df = pd.DataFrame(records)
    df[COL_PHASE] = pd.Categorical(
        df[COL_PHASE],
        categories=list(PHASE_MAP.values()),
        ordered=True,
    )

    df[COL_LANGUAGE] = pd.Categorical(
        df[COL_LANGUAGE],
        categories=[LANG_CPP, LANG_PY],
        ordered=True,
    )
    
    return df

def read_result_iterations_and_inertia(result_file, raw_data):
    centroids = []
    inertia = 0.0
    iterations = None
    mode = None

    with open(result_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "[Lloyd Iterations]":
                mode = "iterations"
                continue
            elif line == "[Centroids]":
                mode = "centroids"
                continue
            elif line == "[Clusters]":
                mode = "clusters"
                continue

            if mode == "iterations":
                iterations = int(line)
                mode = None

            elif mode == "centroids":
                centroids.append(np.fromstring(line, sep=" "))

            elif mode == "clusters":
                left, sep, right = line.partition(":")
                if not sep:
                    continue

                indices_str = right.strip()
                if not indices_str:
                    continue

                k = int(left)
                indices = np.fromstring(indices_str, dtype=np.intp, sep=" ")

                diff = raw_data[indices] - centroids[k]

                inertia += np.einsum("ij,ij->", diff, diff)

    return iterations, float(inertia)