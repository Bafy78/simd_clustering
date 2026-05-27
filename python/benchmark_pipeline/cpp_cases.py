from dataclasses import dataclass
from pathlib import Path

from benchmark_pipeline.paths import BIN_DIR, repo_path


@dataclass(frozen=True)
class CppCase:
    name: str
    case_struct: str
    case_header: str
    needs_init: bool
    needs_metrics: bool
    needs_clusters_arg: bool
    needs_gmm_init: bool = False
    needs_covariance_type_arg: bool = False


CPP_CASES: dict[str, CppCase] = {
    "lloyd_static": CppCase(
        name="lloyd_static",
        case_struct="static_lloyd_case",
        case_header="cases/static_lloyd_case.hpp",
        needs_init=True,
        needs_metrics=True,
        needs_clusters_arg=True,
    ),
    "lloyd_dynamic": CppCase(
        name="lloyd_dynamic",
        case_struct="dynamic_lloyd_case",
        case_header="cases/dynamic_lloyd_case.hpp",
        needs_init=True,
        needs_metrics=True,
        needs_clusters_arg=True,
    ),
    "gmm_static": CppCase(
        name="gmm_static",
        case_struct="static_gmm_case",
        case_header="cases/static_gmm_case.hpp",
        needs_init=False,
        needs_metrics=True,
        needs_clusters_arg=True,
        needs_gmm_init=True,
        needs_covariance_type_arg=True,
    ),
    "pp": CppCase(
        name="pp",
        case_struct="static_pp_case",
        case_header="cases/static_pp_case.hpp",
        needs_init=False,
        needs_metrics=False,
        needs_clusters_arg=True,
    ),
    "soa_static": CppCase(
        name="soa_static",
        case_struct="static_soa_case",
        case_header="cases/static_soa_case.hpp",
        needs_init=False,
        needs_metrics=False,
        needs_clusters_arg=False,
    ),
    "soa_dynamic": CppCase(
        name="soa_dynamic",
        case_struct="dynamic_soa_case",
        case_header="cases/dynamic_soa_case.hpp",
        needs_init=False,
        needs_metrics=False,
        needs_clusters_arg=False,
    ),
}


def get_cpp_case(alg: str) -> CppCase:
    try:
        return CPP_CASES[alg]
    except KeyError as exc:
        valid = ", ".join(sorted(CPP_CASES))
        raise ValueError(
            f"Unknown C++ benchmark case '{alg}'. Valid cases: {valid}"
        ) from exc


def nanobench_binary_path(alg: str) -> str:
    case = get_cpp_case(alg)
    return str(BIN_DIR / f"bench_{case.name}.bin")


def callgrind_binary_path(alg: str, dim: int) -> Path:
    case = get_cpp_case(alg)
    return BIN_DIR / f"profile_{case.name}_callgrind_{dim}D.bin"


def cpp_compile_command(*, dim: int, alg: str, mode: str) -> list[str]:
    if mode == "nanobench":
        src = repo_path("cpp", "benchmarks", "nanobench_main.cpp")
        out = nanobench_binary_path(alg)
    elif mode == "callgrind":
        src = repo_path("cpp", "benchmarks", "callgrind_main.cpp")
        out = str(callgrind_binary_path(alg, dim))
    else:
        raise ValueError(f"Unknown C++ benchmark mode '{mode}'")

    case = get_cpp_case(alg)

    return [
        "g++-14",
        "-O3",
        *(["-g"] if mode == "callgrind" else []),
        "-march=native",
        "-std=c++20",
        "-I../eve/include",
        "-I../nanobench/src/include",
        f"-DTUPLE_SIZE={dim}",
        "-DKMEANS_K_TILE=5",
        "-DKMEANS_M_GROUP=2",
        f'-DKMEANS_BENCH_CASE_HEADER="{case.case_header}"',
        f"-DKMEANS_BENCH_CASE={case.case_struct}",
        src,
        "-o",
        out,
    ]
