"""
Single-entry diagonal GMM scikit-parity explainer.

It compiles a few narrow C++ helpers, runs the minimum probes needed to explain
the scikit parity gap, and writes one Markdown/JSON report.

The report tests four diagnostic claims:
  1. A full EM run has a covariance gap vs scikit, already visible at algorithm iteration 1.
  2. Raw scoring / normalization differences appear too small to be the primary
     explanation for that gap on a sampled subset.
  3. With the exact same scikit responsibilities, a current-like streaming
     M-step reproduces most of the algorithm-iteration-1 covariance seed gap in both magnitude
     and error-vector shape.
  4. With the same responsibilities, a C++ BLAS/SGEMM M-step matches NumPy's
     matmul M-step, supporting reduction semantics rather than a covariance
     formula / precision interpretation bug as the explanation for this diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from threadpoolctl import threadpool_limits

from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _compute_precision_cholesky_from_precisions,
    _estimate_gaussian_parameters,
    _estimate_log_gaussian_prob,
)
from sklearn.utils.extmath import row_norms
from scipy.special import logsumexp as scipy_logsumexp


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], *, cwd: Path, label: str) -> None:
    log(f"[{label}] " + " ".join(shlex.quote(str(x)) for x in cmd))
    p = subprocess.run(cmd, cwd=str(cwd), text=True)
    if p.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {p.returncode}")


def choose_compiler(requested: str | None) -> str:
    if requested:
        return requested
    return "g++-14" if shutil.which("g++-14") else "g++"


def write_json(path: Path, obj: Any) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(conv(obj), f, indent=2)
        f.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def fnum(x: Any, digits: int = 6) -> str:
    if x is None:
        return "n/a"
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-4 or abs(x) >= 1e5:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def pct(x: Any, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"


def ratio(num: Any, den: Any) -> float | None:
    if num is None or den is None:
        return None
    den = float(den)
    if den == 0:
        return None
    return float(num) / den


def nested(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def arr_stats(a: np.ndarray, b: np.ndarray | None = None) -> dict[str, Any]:
    av = np.asarray(a, dtype=np.float64)
    if b is None:
        diff = av.ravel()
        ref = None
    else:
        bv = np.asarray(b, dtype=np.float64)
        diff = (av - bv).ravel()
        ref = bv.ravel()
    ad = np.abs(diff)
    out: dict[str, Any] = {
        "mean_abs": float(np.mean(ad)) if ad.size else 0.0,
        "median_abs": float(np.median(ad)) if ad.size else 0.0,
        "p90_abs": float(np.percentile(ad, 90)) if ad.size else 0.0,
        "p95_abs": float(np.percentile(ad, 95)) if ad.size else 0.0,
        "p99_abs": float(np.percentile(ad, 99)) if ad.size else 0.0,
        "max_abs": float(np.max(ad)) if ad.size else 0.0,
        "signed_mean": float(np.mean(diff)) if diff.size else 0.0,
        "a_gt_b_count": int(np.sum(diff > 0)) if b is not None else None,
        "b_gt_a_count": int(np.sum(diff < 0)) if b is not None else None,
    }
    if ref is not None:
        rel = ad / np.maximum(np.abs(ref), 1e-12)
        out.update(
            {
                "mean_rel": float(np.mean(rel)) if rel.size else 0.0,
                "median_rel": float(np.median(rel)) if rel.size else 0.0,
                "p90_rel": float(np.percentile(rel, 90)) if rel.size else 0.0,
                "p95_rel": float(np.percentile(rel, 95)) if rel.size else 0.0,
                "p99_rel": float(np.percentile(rel, 99)) if rel.size else 0.0,
                "max_rel": float(np.max(rel)) if rel.size else 0.0,
            }
        )
    return out


def error_vector_stats(observed: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    """Compare two same-shaped error vectors, not only their scalar norms."""
    obs = np.asarray(observed, dtype=np.float64).ravel()
    exp = np.asarray(expected, dtype=np.float64).ravel()
    residual = obs - exp
    obs_abs = np.abs(obs)
    exp_abs = np.abs(exp)
    residual_abs = np.abs(residual)

    obs_norm = float(np.linalg.norm(obs))
    exp_norm = float(np.linalg.norm(exp))
    denom = obs_norm * exp_norm
    cosine = float(np.dot(obs, exp) / denom) if denom else None
    if obs.size > 1 and np.std(obs) > 0.0 and np.std(exp) > 0.0:
        pearson = float(np.corrcoef(obs, exp)[0, 1])
    else:
        pearson = None

    nonzero = (obs != 0.0) | (exp != 0.0)
    if np.any(nonzero):
        sign_agreement = float(
            np.mean(np.signbit(obs[nonzero]) == np.signbit(exp[nonzero]))
        )
    else:
        sign_agreement = None

    obs_p95 = float(np.percentile(obs_abs, 95)) if obs_abs.size else 0.0
    exp_p95 = float(np.percentile(exp_abs, 95)) if exp_abs.size else 0.0
    residual_p95 = float(np.percentile(residual_abs, 95)) if residual_abs.size else 0.0
    residual_max = float(np.max(residual_abs)) if residual_abs.size else 0.0

    return {
        "size": int(obs.size),
        "observed_p95_abs": obs_p95,
        "expected_p95_abs": exp_p95,
        "expected_to_observed_p95_ratio": ratio(exp_p95, obs_p95),
        "residual_p95_abs": residual_p95,
        "residual_to_observed_p95_ratio": ratio(residual_p95, obs_p95),
        "residual_max_abs": residual_max,
        "cosine_similarity": cosine,
        "pearson_correlation": pearson,
        "sign_agreement_fraction": sign_agreement,
    }


def precision_cholesky_from_precisions(precisions: np.ndarray) -> np.ndarray:
    # scikit-learn 1.6+ accepts xp=; older versions do not. Passing xp=np fixes
    # the newer signature while the TypeError fallback keeps older installs usable.
    try:
        return _compute_precision_cholesky_from_precisions(precisions, "diag", xp=np)
    except TypeError:
        return _compute_precision_cholesky_from_precisions(precisions, "diag")


def compute_precision_cholesky(covariances: np.ndarray) -> np.ndarray:
    try:
        return _compute_precision_cholesky(covariances, "diag", xp=np)
    except TypeError:
        return _compute_precision_cholesky(covariances, "diag")


def nearest_center_labels(
    X: np.ndarray, centers: np.ndarray, max_distance_cells: int = 25_000_000
) -> np.ndarray:
    N, D = X.shape
    K = centers.shape[0]
    chunk = max(1, max_distance_cells // max(1, K * D))
    labels = np.empty(N, dtype=np.intp)
    c64 = centers.astype(np.float64)
    cn = np.einsum("kd,kd->k", c64, c64)
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        xs = X[start:stop].astype(np.float64)
        dist = -2.0 * (xs @ c64.T) + cn
        labels[start:stop] = np.argmin(dist, axis=1)
    return labels


def estimate_diag_precisions(
    X: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    reg_covar: float,
) -> np.ndarray:
    K, D = means.shape
    sq_sums = np.zeros((K, D), dtype=np.float64)
    m64 = means.astype(np.float64)
    for start in range(0, X.shape[0], 1_000_000):
        stop = min(start + 1_000_000, X.shape[0])
        lab = labels[start:stop]
        diff_sq = (X[start:stop].astype(np.float64) - m64[lab]) ** 2
        for d in range(D):
            sq_sums[:, d] += np.bincount(lab, weights=diff_sq[:, d], minlength=K)
    cov = sq_sums / counts.astype(np.float64)[:, None] + reg_covar
    return 1.0 / cov


@dataclass
class Paths:
    repo: Path
    work: Path
    config_id: str
    data: Path
    weights: Path
    means: Path
    precisions: Path
    cpp_trace_json: Path
    py_trace_json: Path
    cpp_score_json: Path
    sample_indices: Path
    resp_bin: Path
    cpp_same_json: Path
    cpp_blas_json: Path


def make_paths(repo: Path, work: Path, args: argparse.Namespace) -> Paths:
    config_id = f"{args.D}D_{args.N}N_{args.K}K_seed{args.seed}"
    return Paths(
        repo=repo,
        work=work,
        config_id=config_id,
        data=work / f"data_{config_id}.bin",
        weights=work / f"gmm_weights_{config_id}.bin",
        means=work / f"gmm_means_{config_id}.bin",
        precisions=work / f"gmm_precisions_diag_{config_id}.bin",
        cpp_trace_json=work / f"cpp_trace_diag_{config_id}.json",
        py_trace_json=work / f"py_trace_diag_{config_id}.json",
        cpp_score_json=work
        / f"cpp_score_dump_diag_{config_id}_score_N{args.score_N}.json",
        sample_indices=work
        / f"score_sample_indices_{config_id}_score_N{args.score_N}.txt",
        resp_bin=work / f"fixed_resp_diag_{config_id}_sklearn_private.bin",
        cpp_same_json=work / f"cpp_same_resp_mstep_diag_{config_id}.json",
        cpp_blas_json=work / f"cpp_blas_same_resp_mstep_diag_{config_id}.json",
    )


def generate_data_and_init(p: Paths, args: argparse.Namespace) -> None:
    if (
        p.data.exists()
        and p.weights.exists()
        and p.means.exists()
        and p.precisions.exists()
        and not args.force_data
    ):
        log("[setup] using existing data/init files")
        return
    log("[setup] generating dataset and initial diagonal GMM parameters")
    X, _ = make_blobs(
        n_samples=args.N,
        n_features=args.D,
        centers=args.K,
        random_state=args.seed,
    )
    X = np.asarray(X, dtype=np.float32)
    centers, _ = kmeans_plusplus(X, n_clusters=args.K, random_state=args.seed)
    means = np.asarray(centers, dtype=np.float32)
    labels = nearest_center_labels(X, means)
    counts = np.bincount(labels, minlength=args.K).astype(np.int64)
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        raise RuntimeError(
            f"initialization produced empty clusters: {empty[:20].tolist()}"
        )
    weights = (counts.astype(np.float64) / float(args.N)).astype(np.float32)
    precisions = estimate_diag_precisions(
        X, labels, means, counts, args.reg_covar
    ).astype(np.float32)
    p.work.mkdir(parents=True, exist_ok=True)
    X.tofile(p.data)
    weights.tofile(p.weights)
    means.tofile(p.means)
    precisions.tofile(p.precisions)


def compile_cpp(
    p: Paths,
    args: argparse.Namespace,
    source_name: str,
    output_name: str,
    *,
    blas: bool = False,
) -> Path:
    src = p.repo / "cpp" / "diagnostics" / source_name
    out = p.work / output_name
    cxx = choose_compiler(args.compiler)
    flags = args.cxxflags or ["-O3", "-march=native", "-std=c++20"]
    cmd = [cxx, *flags, f"-DTUPLE_SIZE={args.D}"]
    if source_name in {"gmm_diag_trace.cpp", "gmm_diag_score_dump.cpp"}:
        cmd += ["-I../eve/include", "-Icpp/include"]
    if blas and args.blas_include:
        cmd += ["-I", str(args.blas_include)]
    cmd += [str(src), "-o", str(out)]
    if blas:
        if args.blas_flags:
            cmd += shlex.split(args.blas_flags)
        else:
            cmd += ["-lopenblas"]
    run(cmd, cwd=p.repo, label=f"compile-{source_name}")
    return out


def run_cpp_trace(p: Paths, args: argparse.Namespace) -> dict[str, Any]:
    exe = compile_cpp(p, args, "gmm_diag_trace.cpp", f"gmm_diag_trace_{args.D}D.bin")
    run(
        [
            str(exe),
            str(p.data),
            str(args.N),
            str(args.K),
            str(p.weights),
            str(p.means),
            str(p.precisions),
            str(p.cpp_trace_json),
            str(args.max_iter),
            repr(float(args.tol)),
            repr(float(args.reg_covar)),
        ],
        cwd=p.repo,
        label="run-cpp-trace",
    )
    return load_json(p.cpp_trace_json)


def run_python_trace(p: Paths, args: argparse.Namespace) -> dict[str, Any]:
    log("[py-trace] running sklearn-style diagonal EM trace")
    X = np.memmap(p.data, dtype=np.float32, mode="r", shape=(args.N, args.D))
    weights = np.memmap(p.weights, dtype=np.float32, mode="r", shape=(args.K,)).copy()
    means = np.memmap(
        p.means, dtype=np.float32, mode="r", shape=(args.K, args.D)
    ).copy()
    precisions = np.memmap(
        p.precisions, dtype=np.float32, mode="r", shape=(args.K, args.D)
    ).copy()
    covariances = (1.0 / precisions).astype(np.float32)
    precisions_chol = precision_cholesky_from_precisions(precisions)
    algorithm_iterations: list[dict[str, Any]] = []
    lower_bounds: list[float] = []
    lower_bound = -math.inf
    converged = False

    def body() -> None:
        nonlocal weights, means, covariances, precisions_chol, lower_bound, converged
        for it in range(1, args.max_iter + 1):
            prev = lower_bound
            weighted = _estimate_log_gaussian_prob(
                X, means, precisions_chol, "diag"
            ) + np.log(weights)
            log_prob_norm = scipy_logsumexp(weighted, axis=1)
            with np.errstate(under="ignore"):
                log_resp = weighted - log_prob_norm[:, None]
                resp = np.exp(log_resp)
            lower_bound = float(np.mean(log_prob_norm))
            nk_raw = np.sum(resp, axis=0)
            eps10 = np.asarray(10 * np.finfo(resp.dtype).eps, dtype=resp.dtype)
            nk = nk_raw + eps10
            sum_x = resp.T @ X
            sum_x2 = resp.T @ (X * X)
            avg_x = sum_x / nk[:, None]
            avg_x2 = sum_x2 / nk[:, None]
            mean_sq = avg_x * avg_x
            cov_from_stats = avg_x2 - mean_sq + args.reg_covar
            weights_from_nk = nk / np.sum(nk)
            algorithm_iterations.append(
                {
                    "iter": it,
                    "lower_bound": lower_bound,
                    "weights_before_m_step": weights.copy(),
                    "means_before_m_step": means.copy(),
                    "covariances_before_m_step": covariances.copy(),
                    "precisions_before_m_step": precisions_chol**2,
                    "nk_raw": nk_raw,
                    "nk": nk,
                    "weights_from_nk": weights_from_nk,
                    "sum_x": sum_x,
                    "sum_x2": sum_x2,
                    "avg_x": avg_x,
                    "avg_x2": avg_x2,
                    "mean_sq": mean_sq,
                    "cov_from_stats": cov_from_stats,
                }
            )
            sk_nk, sk_means, sk_covs = _estimate_gaussian_parameters(
                X, resp, args.reg_covar, "diag"
            )
            weights = sk_nk / np.sum(sk_nk)
            means = sk_means
            covariances = sk_covs
            precisions_chol = compute_precision_cholesky(covariances)
            lower_bounds.append(lower_bound)
            if abs(lower_bound - prev) < args.tol:
                converged = True
                break

    if threadpool_limits is None:
        body()
    else:
        with threadpool_limits(limits=1):
            body()
    trace = {
        "schema_version": 1,
        "phase": "gmm",
        "diagnostic": "py_sklearn_diag_trace",
        "D": args.D,
        "N": args.N,
        "K": args.K,
        "algorithm_iterations": algorithm_iterations,
        "final": {
            "algorithm_iterations": len(algorithm_iterations),
            "converged": converged,
            "lower_bound": lower_bound,
            "lower_bounds": lower_bounds,
            "weights": weights,
            "means": means,
            "covariances": covariances,
            "precisions": precisions_chol**2,
        },
    }
    write_json(p.py_trace_json, trace)
    return trace


def compare_full_trace(cpp: dict[str, Any], py: dict[str, Any]) -> dict[str, Any]:
    n = min(len(cpp["algorithm_iterations"]), len(py["algorithm_iterations"]))
    per: list[dict[str, Any]] = []
    for i in range(n):
        ci, pi = cpp["algorithm_iterations"][i], py["algorithm_iterations"][i]
        row: dict[str, Any] = {
            "iter": int(ci["iter"]),
            "lower_bound_cpp_minus_py": float(ci["lower_bound"] - pi["lower_bound"]),
            "lower_bound_abs_diff": abs(float(ci["lower_bound"] - pi["lower_bound"])),
        }
        for out_key, trace_key in [
            ("covariance", "cov_from_stats"),
            ("avg_x2", "avg_x2"),
            ("mean_sq", "mean_sq"),
            ("mean", "avg_x"),
            ("nk", "nk"),
            ("sum_x", "sum_x"),
            ("sum_x2", "sum_x2"),
        ]:
            row[out_key] = arr_stats(
                np.asarray(ci[trace_key]), np.asarray(pi[trace_key])
            )
        cov_diff = np.asarray(ci["cov_from_stats"], dtype=np.float64) - np.asarray(
            pi["cov_from_stats"], dtype=np.float64
        )
        row["cpp_cov_gt_py_count"] = int(np.sum(cov_diff > 0))
        row["py_cov_gt_cpp_count"] = int(np.sum(cov_diff < 0))
        per.append(row)
    cf, pf = cpp["final"], py["final"]
    final = {
        "covariance": arr_stats(
            np.asarray(cf["covariances"]), np.asarray(pf["covariances"])
        ),
        "means": arr_stats(np.asarray(cf["means"]), np.asarray(pf["means"])),
        "weights": arr_stats(np.asarray(cf["weights"]), np.asarray(pf["weights"])),
    }
    cov_diff = np.asarray(cf["covariances"], dtype=np.float64) - np.asarray(
        pf["covariances"], dtype=np.float64
    )
    final["cpp_cov_gt_py_count"] = int(np.sum(cov_diff > 0))
    final["py_cov_gt_cpp_count"] = int(np.sum(cov_diff < 0))
    raw: dict[str, Any] = {}
    if n > 0:
        raw = {
            "cpp_iter1_cov_from_stats": np.asarray(
                cpp["algorithm_iterations"][0]["cov_from_stats"]
            ),
            "py_iter1_cov_from_stats": np.asarray(
                py["algorithm_iterations"][0]["cov_from_stats"]
            ),
        }
    return {
        "algorithm_iterations": {
            "cpp": cf["algorithm_iterations"],
            "py": pf["algorithm_iterations"],
            "compared": n,
            "cpp_converged": cf.get("converged"),
            "py_converged": pf.get("converged"),
        },
        "lower_bound": {
            "cpp": cf["lower_bound"],
            "py": pf["lower_bound"],
            "abs_diff": abs(float(cf["lower_bound"] - pf["lower_bound"])),
            "cpp_minus_py": float(cf["lower_bound"] - pf["lower_bound"]),
        },
        "per_algorithm_iteration": per,
        "final": final,
        "raw": raw,
    }


def write_sample_indices(p: Paths, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed + 12345)
    score_N = min(args.score_N, args.N)
    idx = np.sort(rng.choice(args.N, size=score_N, replace=False).astype(np.int64))
    p.sample_indices.write_text("\n".join(str(int(i)) for i in idx) + "\n")
    return idx


def run_score_sanity(
    p: Paths, args: argparse.Namespace, idx: np.ndarray
) -> dict[str, Any]:
    exe = compile_cpp(
        p, args, "gmm_diag_score_dump.cpp", f"gmm_diag_score_dump_{args.D}D.bin"
    )
    run(
        [
            str(exe),
            str(p.data),
            str(args.N),
            str(args.K),
            str(p.weights),
            str(p.means),
            str(p.precisions),
            str(p.sample_indices),
            str(p.cpp_score_json),
        ],
        cwd=p.repo,
        label="run-cpp-score-dump",
    )
    cpp = load_json(p.cpp_score_json)
    X_full = np.memmap(p.data, dtype=np.float32, mode="r", shape=(args.N, args.D))
    X = np.asarray(X_full[idx], dtype=np.float32)
    weights = np.memmap(p.weights, dtype=np.float32, mode="r", shape=(args.K,)).copy()
    means = np.memmap(
        p.means, dtype=np.float32, mode="r", shape=(args.K, args.D)
    ).copy()
    precisions = np.memmap(
        p.precisions, dtype=np.float32, mode="r", shape=(args.K, args.D)
    ).copy()
    precisions_chol = precision_cholesky_from_precisions(precisions)
    py_score = _estimate_log_gaussian_prob(X, means, precisions_chol, "diag") + np.log(
        weights
    )
    cpp_score = np.asarray(cpp["scores"]["eve_current"], dtype=np.float64)
    py_score64 = np.asarray(py_score, dtype=np.float64)

    # Normalize both score matrices exactly the same way to isolate score impact on resp.
    def softmax_from_scores(s: np.ndarray) -> np.ndarray:
        m = np.max(s, axis=1, keepdims=True)
        u = np.exp(s - m)
        return u / np.sum(u, axis=1, keepdims=True)

    cpp_resp = softmax_from_scores(cpp_score)
    py_resp = softmax_from_scores(py_score64)
    argmax_changes = int(
        np.sum(np.argmax(cpp_resp, axis=1) != np.argmax(py_resp, axis=1))
    )
    return {
        "score_N": int(idx.size),
        "eve_cardinal": cpp.get("eve_cardinal"),
        "score_cpp_current_vs_sklearn": arr_stats(cpp_score, py_score64),
        "resp_from_scores_cpp_current_vs_sklearn": arr_stats(cpp_resp, py_resp),
        "argmax_changed_count": argmax_changes,
        "argmax_changed_percent": 100.0 * argmax_changes / max(1, idx.size),
        "cpp_score_json": str(p.cpp_score_json),
    }


def generate_fixed_resp(p: Paths, args: argparse.Namespace) -> dict[str, Any]:
    if p.resp_bin.exists() and not args.force_resp:
        log("[resp] using existing fixed resp.bin")
        resp = np.memmap(
            p.resp_bin,
            dtype=np.float32,
            mode="r",
            shape=(args.N, args.K),
        )
        row_sum = np.asarray(resp[: min(args.N, 10000)].sum(axis=1))
        return {
            "row_sum_max_abs_err_sample": float(np.max(np.abs(row_sum - 1.0))),
            "path": str(p.resp_bin),
        }
    log("[resp] generating fixed sklearn responsibilities")
    X = np.memmap(p.data, dtype=np.float32, mode="r", shape=(args.N, args.D))
    weights = np.memmap(p.weights, dtype=np.float32, mode="r", shape=(args.K,)).copy()
    means = np.memmap(
        p.means, dtype=np.float32, mode="r", shape=(args.K, args.D)
    ).copy()
    precisions = np.memmap(
        p.precisions, dtype=np.float32, mode="r", shape=(args.K, args.D)
    ).copy()
    precisions_chol = precision_cholesky_from_precisions(precisions)
    resp = np.memmap(p.resp_bin, dtype=np.float32, mode="w+", shape=(args.N, args.K))
    max_err = 0.0
    mn, mx = math.inf, -math.inf
    for start in range(0, args.N, args.resp_chunk_samples):
        stop = min(start + args.resp_chunk_samples, args.N)
        scores = _estimate_log_gaussian_prob(
            X[start:stop], means, precisions_chol, "diag"
        ) + np.log(weights)
        lpn = scipy_logsumexp(scores, axis=1)
        block = np.exp(scores - lpn[:, None]).astype(np.float32)
        resp[start:stop] = block
        max_err = max(max_err, float(np.max(np.abs(block.sum(axis=1) - 1.0))))
        mn = min(mn, float(block.min()))
        mx = max(mx, float(block.max()))
        log(f"[resp] samples {stop}/{args.N}")
    resp.flush()
    return {
        "row_sum_max_abs_err": max_err,
        "min": mn,
        "max": mx,
        "path": str(p.resp_bin),
    }


def py_mstep_reference(p: Paths, args: argparse.Namespace) -> dict[str, Any]:
    log("[py-mstep] NumPy matmul reference")
    X = np.memmap(p.data, dtype=np.float32, mode="r", shape=(args.N, args.D))
    resp = np.memmap(p.resp_bin, dtype=np.float32, mode="r", shape=(args.N, args.K))
    nk_raw = np.sum(resp, axis=0)
    eps10 = np.asarray(10 * np.finfo(resp.dtype).eps, dtype=resp.dtype)
    nk = nk_raw + eps10
    sum_x = resp.T @ X
    sum_x2 = resp.T @ (X * X)
    means = sum_x / nk[:, None]
    avg_x2 = sum_x2 / nk[:, None]
    mean_sq = means * means
    cov = avg_x2 - mean_sq + args.reg_covar
    weights = nk / np.sum(nk)
    return {
        "nk_raw": np.asarray(nk_raw),
        "nk": np.asarray(nk),
        "weights": np.asarray(weights),
        "sum_x": np.asarray(sum_x),
        "sum_x2": np.asarray(sum_x2),
        "means": np.asarray(means),
        "avg_x2": np.asarray(avg_x2),
        "mean_sq": np.asarray(mean_sq),
        "covariances": np.asarray(cov),
    }


def run_cpp_same_resp(p: Paths, args: argparse.Namespace) -> dict[str, Any]:
    exe = compile_cpp(
        p,
        args,
        "gmm_diag_same_resp_mstep.cpp",
        f"gmm_diag_same_resp_mstep_{args.D}D.bin",
    )
    run(
        [
            str(exe),
            str(p.data),
            str(args.N),
            str(args.K),
            str(p.resp_bin),
            str(p.cpp_same_json),
            repr(float(args.reg_covar)),
        ],
        cwd=p.repo,
        label="run-cpp-same-resp-mstep",
    )
    return load_json(p.cpp_same_json)


def run_cpp_blas(p: Paths, args: argparse.Namespace) -> dict[str, Any]:
    exe = compile_cpp(
        p,
        args,
        "gmm_diag_blas_same_resp_mstep.cpp",
        f"gmm_diag_blas_same_resp_mstep_{args.D}D.bin",
        blas=True,
    )
    run(
        [
            str(exe),
            str(p.data),
            str(args.N),
            str(args.K),
            str(p.resp_bin),
            str(p.cpp_blas_json),
            repr(float(args.reg_covar)),
        ],
        cwd=p.repo,
        label="run-cpp-blas-mstep",
    )
    return load_json(p.cpp_blas_json)


def flat_variant(v: dict[str, Any], K: int, D: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in [
        "nk_raw",
        "nk",
        "weights",
        "sum_x",
        "sum_x2",
        "means",
        "avg_x2",
        "mean_sq",
        "covariances",
    ]:
        a = np.asarray(v[key], dtype=np.float64)
        if key not in {"nk_raw", "nk", "weights"}:
            a = a.reshape(K, D)
        out[key] = a
    return out


def compare_mstep_variants(
    cpp_json: dict[str, Any], ref: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    comp: dict[str, Any] = {}
    for name, variant in cpp_json["variants"].items():
        vv = flat_variant(variant, args.K, args.D)
        comp[name] = {field: arr_stats(vv[field], ref[field]) for field in ref}
    return comp


def compare_covariance_error_vectors(
    full: dict[str, Any],
    cpp_same: dict[str, Any],
    ref: dict[str, Any],
    current_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    iter1_cpp = np.asarray(
        full["raw"]["cpp_iter1_cov_from_stats"], dtype=np.float64
    ).reshape(args.K, args.D)
    iter1_py = np.asarray(
        full["raw"]["py_iter1_cov_from_stats"], dtype=np.float64
    ).reshape(args.K, args.D)
    same_current = flat_variant(cpp_same["variants"][current_name], args.K, args.D)[
        "covariances"
    ]
    ref_cov = np.asarray(ref["covariances"], dtype=np.float64)

    full_seed_delta = iter1_cpp - iter1_py
    same_resp_delta = same_current - ref_cov
    return error_vector_stats(full_seed_delta, same_resp_delta)


def choose_current_like(comp: dict[str, Any], eve_cardinal: int | None) -> str:
    if eve_cardinal is not None:
        lane_name = f"cpp_lane_bucket_float_fma_w{eve_cardinal}"
        if lane_name in comp:
            return lane_name
    for name in [
        "cpp_sample_major_float_fma",
        "cpp_sample_major_float_mul_add",
    ]:
        if name in comp:
            return name
    for name in comp:
        if name.startswith("cpp_"):
            return name
    return next(iter(comp))


def render_report(
    args: argparse.Namespace,
    p: Paths,
    full: dict[str, Any],
    score: dict[str, Any],
    same_comp: dict[str, Any],
    blas_comp: dict[str, Any],
    resp_info: dict[str, Any],
    current_name: str,
    cov_error_vector: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    iter1 = full["per_algorithm_iteration"][0]
    final = full["final"]
    iter1_p95 = nested(iter1, "covariance", "p95_abs")
    iter1_rel = nested(iter1, "covariance", "p95_rel")
    final_p95 = nested(final, "covariance", "p95_abs")
    final_rel = nested(final, "covariance", "p95_rel")
    lower_abs = nested(full, "lower_bound", "abs_diff")
    amplification = ratio(final_p95, iter1_p95)

    current_cov = same_comp[current_name]["covariances"]
    current_p95 = current_cov["p95_abs"]
    current_rel = current_cov.get("p95_rel")
    current_gt = current_cov.get("a_gt_b_count", 0) or 0
    current_lt = current_cov.get("b_gt_a_count", 0) or 0
    seed_ratio = ratio(current_p95, iter1_p95)
    final_ratio = ratio(current_p95, final_p95)

    blas_name = (
        "cpp_cblas_sgemm_nk_loop_float"
        if "cpp_cblas_sgemm_nk_loop_float" in blas_comp
        else min(blas_comp, key=lambda n: blas_comp[n]["covariances"]["p95_abs"])
    )
    blas_cov = blas_comp[blas_name]["covariances"]
    blas_p95 = blas_cov["p95_abs"]
    blas_seed_ratio = ratio(blas_p95, iter1_p95)
    blas_final_ratio = ratio(blas_p95, final_p95)

    score_resp = score["resp_from_scores_cpp_current_vs_sklearn"]
    score_p99 = score_resp["p99_abs"]
    score_max = score_resp["max_abs"]
    score_argmax = score["argmax_changed_count"]
    score_N = score.get("score_N")
    eve_cardinal = score.get("eve_cardinal")

    scoring_unlikely_primary = score_argmax == 0 and score_p99 < 1e-5
    mstep_magnitude_matches = seed_ratio is not None and 0.5 <= seed_ratio <= 1.5
    vector_residual_ratio = cov_error_vector.get("residual_to_observed_p95_ratio")
    vector_cosine = cov_error_vector.get("cosine_similarity")
    vector_sign_agreement = cov_error_vector.get("sign_agreement_fraction")
    vector_matches = (
        vector_residual_ratio is not None
        and vector_residual_ratio <= 0.25
        and (vector_cosine is None or vector_cosine >= 0.80)
        and (vector_sign_agreement is None or vector_sign_agreement >= 0.80)
    )
    blas_closes = blas_seed_ratio is not None and blas_seed_ratio < 0.05
    diagnostic_supported = mstep_magnitude_matches and vector_matches and blas_closes

    if diagnostic_supported:
        verdict = (
            "For this diagnostic, the scikit parity gap is best explained by "
            "M-step reduction semantics, not by an algebraic diagonal covariance "
            "formula bug."
        )
    else:
        failing = []
        if not mstep_magnitude_matches:
            failing.append(
                "same-responsibility M-step magnitude does not match the algorithm-iteration-1 seed"
            )
        if not vector_matches:
            failing.append(
                "same-responsibility M-step error vector does not match the algorithm-iteration-1 error vector"
            )
        if not blas_closes:
            failing.append("BLAS/SGEMM does not close the seed gap")
        verdict = (
            "For this diagnostic, the evidence does not support M-step "
            "reduction semantics as the primary explanation for the scikit parity gap."
        )
        if failing:
            verdict += " Failed checks: " + "; ".join(failing) + "."

    if scoring_unlikely_primary:
        verdict += (
            " Scoring/normalization differences look unlikely to be the primary cause "
            "on the selected samples."
        )
    else:
        verdict += (
            " Scoring/normalization may still contribute on the selected samples."
        )

    obj = {
        "schema_version": 3,
        "config": {
            "D": args.D,
            "N": args.N,
            "K": args.K,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "reg_covar": args.reg_covar,
        },
        "verdict": verdict,
        "diagnostic_supported": diagnostic_supported,
        "full_em": {
            "iter1_cov_p95_abs": iter1_p95,
            "iter1_cov_p95_rel": iter1_rel,
            "final_cov_p95_abs": final_p95,
            "final_cov_p95_rel": final_rel,
            "lower_bound_abs_diff": lower_abs,
            "amplification_final_over_iter1": amplification,
        },
        "scoring_sanity": {
            "score_N": score_N,
            "resp_p99_abs": score_p99,
            "resp_max_abs": score_max,
            "argmax_changed_count": score_argmax,
            "unlikely_primary_on_sample": scoring_unlikely_primary,
            "eve_cardinal": eve_cardinal,
        },
        "same_resp_streaming_mstep": {
            "variant": current_name,
            "eve_cardinal": eve_cardinal,
            "cov_p95_abs": current_p95,
            "cov_p95_rel": current_rel,
            "ratio_to_iter1_seed": seed_ratio,
            "ratio_to_final_gap": final_ratio,
            "a_gt_b_count": current_gt,
            "b_gt_a_count": current_lt,
            "magnitude_matches_iter1_seed": mstep_magnitude_matches,
            "covariance_error_vector": cov_error_vector,
            "error_vector_matches_iter1_seed": vector_matches,
        },
        "blas_oracle": {
            "variant": blas_name,
            "cov_p95_abs": blas_p95,
            "ratio_to_iter1_seed": blas_seed_ratio,
            "ratio_to_final_gap": blas_final_ratio,
            "nk_sgemv_cov_p95_abs": nested(
                blas_comp.get("cpp_cblas_sgemm_nk_sgemv", {}), "covariances", "p95_abs"
            ),
            "nk_sgemm_ones_cov_p95_abs": nested(
                blas_comp.get("cpp_cblas_sgemm_nk_sgemm_ones", {}),
                "covariances",
                "p95_abs",
            ),
        },
        "resp_info": resp_info,
        "files": {
            "cpp_trace_json": str(p.cpp_trace_json),
            "py_trace_json": str(p.py_trace_json),
            "cpp_score_json": str(p.cpp_score_json),
            "resp_bin": str(p.resp_bin),
            "cpp_same_json": str(p.cpp_same_json),
            "cpp_blas_json": str(p.cpp_blas_json),
        },
    }

    md: list[str] = []
    md.append("# Diagonal GMM scikit parity explainer")
    md.append("")
    md.append(f"Case: `{args.D}D`, `{args.N}N`, `{args.K}K`, seed `{args.seed}`.")
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(verdict)
    md.append("")
    md.append("")
    md.append("## Evidence summary")
    md.append("")
    md.append("| Check | Result | Interpretation |")
    md.append("|---|---:|---|")
    md.append(
        f"| Full EM algorithm-iteration-1 covariance p95 abs diff | `{fnum(iter1_p95)}` | Initial seed gap |"
    )
    md.append(
        f"| Full EM final covariance p95 abs diff | `{fnum(final_p95)}` | Gap after EM amplification (`{fnum(amplification)}×`) |"
    )
    md.append(
        f"| Full EM lower-bound abs diff | `{fnum(lower_abs)}` | Model likelihood remains close |"
    )
    md.append(
        f"| Score-derived responsibility p99 abs diff | `{fnum(score_p99)}` over `{score_N}` samples | {'Makes scoring unlikely as the primary cause on the selected samples' if scoring_unlikely_primary else 'Scoring may contribute on the selected samples'} |"
    )
    md.append(
        f"| Score-derived argmax changes | `{score_argmax}` | {'No assignment winner changes on selected samples' if score_argmax == 0 else 'Some winner changes'} |"
    )
    md.append(
        f"| Same-resp current-like M-step cov p95 abs diff | `{fnum(current_p95)}` | `{pct(seed_ratio)}` of algorithm-iteration-1 seed using `{current_name}` |"
    )
    md.append(
        f"| Same-resp covariance error-vector residual p95 | `{fnum(cov_error_vector.get('residual_p95_abs'))}` | `{pct(vector_residual_ratio)}` of algorithm-iteration-1 seed p95 |"
    )
    md.append(
        f"| Same-resp covariance error-vector cosine | `{fnum(vector_cosine)}` | {'Error vector is aligned with algorithm-iteration-1 seed' if vector_matches else 'Error vector alignment check failed'} |"
    )
    md.append(
        f"| BLAS oracle cov p95 abs diff | `{fnum(blas_p95)}` | `{pct(blas_seed_ratio)}` of algorithm-iteration-1 seed |"
    )
    md.append("")
    md.append(
        "## Why this supports the M-step explanation for this diagnostic"
        if diagnostic_supported
        else "## Why this does not establish the M-step explanation for this diagnostic"
    )
    md.append("")
    if diagnostic_supported:
        md.append(
            "The strongest comparison is the same-responsibility M-step test. Both implementations receive the exact same scikit-generated `resp` matrix. That bypasses scoring, exponentials, logsumexp, and cluster matching. The current-like streaming M-step reproduces the first algorithm iteration covariance difference in both p95 magnitude and covariance error-vector shape. Then the BLAS oracle, using `sgemm` for `resp.T @ X` and `resp.T @ X²` with a loop-float `nk`, nearly removes that difference."
        )
    else:
        md.append(
            "The same-responsibility M-step test is the main causal check because both implementations receive the exact same scikit-generated `resp` matrix. In this run, one or more required checks failed, so the report should not claim that M-step reduction semantics explain the scikit parity gap for this diagnostic. Inspect the magnitude, error-vector residual, and BLAS rows above to see which part failed."
        )
    md.append("")
    if diagnostic_supported:
        md.append(
            "So the remaining scikit mismatch in this diagnostic is best described as: fused streaming SIMD sufficient-stat accumulation follows a different float32 reduction order than scikit/NumPy's BLAS-backed matrix products. The covariance formula `E[x²] - E[x]² + reg_covar` then amplifies small mean/second-moment differences, and later EM algorithm iterations amplify the initial seed."
        )
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append(
        "- `nk` should not be BLAS-ified for parity here; the oracle intentionally uses SGEMM for `sum_x` / `sum_x2` and a float loop for `nk`."
    )
    md.append(
        "- This report does not recommend switching the production path to BLAS. It only uses BLAS as a parity probe for this diagnostic."
    )
    md.append(
        "- If the diagnostic checks pass, document this as a numerical reduction-semantics gap rather than a formula bug. If they fail, investigate the failed check before making that claim."
    )
    md.append("")
    return "\n".join(md) + "\n", obj


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Explain diagonal GMM scikit parity gaps in one standalone diagnostic."
    )
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument(
        "--workdir",
        type=Path,
        default=Path("diagnostics_results/gmm_diag_scikit_parity_explainer"),
    )
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--N", type=int, default=2_000_000)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--reg-covar", type=float, default=1e-6)
    ap.add_argument(
        "--score-sample-N",
        dest="score_N",
        type=int,
        default=65536,
    )
    ap.add_argument(
        "--resp-chunk-samples",
        dest="resp_chunk_samples",
        type=int,
        default=65536,
    )
    ap.add_argument("--compiler", default=None)
    ap.add_argument("--cxxflags", nargs="*", default=None)
    ap.add_argument("--blas-flags", default=None)
    ap.add_argument("--blas-include", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Remove workdir first.")
    ap.add_argument("--force-data", action="store_true")
    ap.add_argument("--force-resp", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    work = (
        args.workdir if args.workdir.is_absolute() else repo / args.workdir
    ).resolve()
    if args.force and work.exists():
        log(f"[setup] removing {work}")
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    p = make_paths(repo, work, args)

    generate_data_and_init(p, args)
    cpp_trace = run_cpp_trace(p, args)
    py_trace = run_python_trace(p, args)
    full = compare_full_trace(cpp_trace, py_trace)
    write_json(work / "full_em_trace_comparison.json", full)

    idx = write_sample_indices(p, args)
    score = run_score_sanity(p, args, idx)
    write_json(work / "score_sanity_comparison.json", score)

    resp_info = generate_fixed_resp(p, args)
    py_ref = py_mstep_reference(p, args)
    cpp_same = run_cpp_same_resp(p, args)
    same_comp = compare_mstep_variants(cpp_same, py_ref, args)
    write_json(work / "same_resp_streaming_mstep_comparison.json", same_comp)
    current_name = choose_current_like(same_comp, score.get("eve_cardinal"))
    cov_error_vector = compare_covariance_error_vectors(
        full, cpp_same, py_ref, current_name, args
    )
    write_json(
        work / "same_resp_covariance_error_vector_comparison.json", cov_error_vector
    )

    cpp_blas = run_cpp_blas(p, args)
    blas_comp = compare_mstep_variants(cpp_blas, py_ref, args)
    write_json(work / "blas_oracle_mstep_comparison.json", blas_comp)

    report_md, report_obj = render_report(
        args,
        p,
        full,
        score,
        same_comp,
        blas_comp,
        resp_info,
        current_name,
        cov_error_vector,
    )
    (work / "gmm_diag_scikit_parity_explanation.md").write_text(report_md)
    (work / "gmm_diag_scikit_parity_explanation.txt").write_text(report_md)
    write_json(work / "gmm_diag_scikit_parity_explanation.json", report_obj)
    print("\n" + report_md)
    log(f"[done] report: {work / 'gmm_diag_scikit_parity_explanation.md'}")


if __name__ == "__main__":
    main()
