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


def spill_detector_assembly_path(
    alg: str,
    dim: int,
    out_dir: str | Path | None = None,
    gmm_covariance_type: str | None = None,
) -> Path:
    case = get_cpp_case(alg)
    root = (
        Path(out_dir)
        if out_dir is not None
        else Path(repo_path("spill_detector_results"))
    )
    covariance_suffix = (
        f".{gmm_covariance_type}" if alg == "gmm_static" and gmm_covariance_type else ""
    )
    return root / f"asm.{case.name}{covariance_suffix}.{dim}D.s"


def cpp_compile_command(
    *,
    dim: int,
    alg: str,
    mode: str,
    output: str | Path | None = None,
    extra_defines: list[str] | None = None,
) -> list[str]:
    if mode == "nanobench":
        src = repo_path("cpp", "benchmarks", "nanobench_main.cpp")
        out = str(output) if output is not None else nanobench_binary_path(alg)
    elif mode == "callgrind":
        src = repo_path("cpp", "benchmarks", "callgrind_main.cpp")
        out = (
            str(output) if output is not None else str(callgrind_binary_path(alg, dim))
        )
    elif mode == "assembly":
        src = repo_path("cpp", "benchmarks", "spill_detector_main.cpp")
        out = (
            str(output)
            if output is not None
            else str(spill_detector_assembly_path(alg, dim))
        )
    else:
        raise ValueError(f"Unknown C++ benchmark mode '{mode}'")

    case = get_cpp_case(alg)

    return [
        "g++-14",
        "-O3",
        *(["-g"] if mode == "callgrind" else []),
        *(["-S", "-masm=intel"] if mode == "assembly" else []),
        "-march=native",
        "-std=c++20",
        "-I../eve/include",
        "-I../nanobench/src/include",
        f"-DTUPLE_SIZE={dim}",
        "-DKMEANS_K_TILE=5",
        "-DKMEANS_M_GROUP=2",
        f'-DBENCH_CASE_HEADER="{case.case_header}"',
        f"-DBENCH_CASE={case.case_struct}",
        *(extra_defines or []),
        src,
        "-o",
        out,
    ]
