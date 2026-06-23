from dataclasses import dataclass
from pathlib import Path

from benchmark_metadata import FULL_STAGE_KEY, HDBSCAN_DISTANCE_STAGE_KEY
from benchmark_pipeline.gmm_covariance import SUPPORTED_GMM_COVARIANCE_TYPES
from benchmark_pipeline.paths import BIN_DIR, repo_path


@dataclass(frozen=True)
class CppCase:
    name: str
    case_struct: str
    case_header: str
    needs_init: bool
    needs_metrics: bool
    needs_clusters_arg: bool
    phase_key: str
    variant_key: str
    display_name: str
    stage_keys: tuple[str, ...] = (FULL_STAGE_KEY,)
    needs_gmm_init: bool = False
    needs_covariance_type_arg: bool = False
    supported_gmm_covariance_types: tuple[str, ...] = ()

    def supports_stage(self, stage_key: str) -> bool:
        return stage_key in self.stage_keys


CPP_CASES: dict[str, CppCase] = {
    "lloyd_static": CppCase(
        name="lloyd_static",
        case_struct="static_lloyd_case",
        case_header="cases/static_lloyd_case.hpp",
        needs_init=True,
        needs_metrics=True,
        needs_clusters_arg=True,
        phase_key="lloyd",
        variant_key="static",
        display_name="Lloyd static C++",
    ),
    "lloyd_dynamic": CppCase(
        name="lloyd_dynamic",
        case_struct="dynamic_lloyd_case",
        case_header="cases/dynamic_lloyd_case.hpp",
        needs_init=True,
        needs_metrics=True,
        needs_clusters_arg=True,
        phase_key="lloyd",
        variant_key="dynamic",
        display_name="Lloyd dynamic C++",
    ),
    "lloyd_auto": CppCase(
        name="lloyd_auto",
        case_struct="auto_lloyd_case",
        case_header="cases/auto_lloyd_case.hpp",
        needs_init=True,
        needs_metrics=True,
        needs_clusters_arg=True,
        phase_key="lloyd",
        variant_key="auto",
        display_name="Lloyd auto C++",
    ),
    "gmm_static": CppCase(
        name="gmm_static",
        case_struct="static_gmm_case",
        case_header="cases/static_gmm_case.hpp",
        needs_init=False,
        needs_metrics=True,
        needs_clusters_arg=True,
        phase_key="gmm",
        variant_key="static",
        display_name="GMM static C++",
        needs_gmm_init=True,
        needs_covariance_type_arg=True,
        supported_gmm_covariance_types=SUPPORTED_GMM_COVARIANCE_TYPES,
    ),
    "gmm_dynamic": CppCase(
        name="gmm_dynamic",
        case_struct="dynamic_gmm_case",
        case_header="cases/dynamic_gmm_case.hpp",
        needs_init=False,
        needs_metrics=True,
        needs_clusters_arg=True,
        phase_key="gmm",
        variant_key="dynamic",
        display_name="GMM dynamic C++",
        needs_gmm_init=True,
        needs_covariance_type_arg=True,
        supported_gmm_covariance_types=SUPPORTED_GMM_COVARIANCE_TYPES,
    ),
    "pp_static": CppCase(
        name="pp_static",
        case_struct="static_pp_case",
        case_header="cases/static_pp_case.hpp",
        needs_init=False,
        needs_metrics=False,
        needs_clusters_arg=True,
        phase_key="pp",
        variant_key="static",
        display_name="K-Means++ static C++",
    ),
    "pp_dynamic": CppCase(
        name="pp_dynamic",
        case_struct="dynamic_pp_case",
        case_header="cases/dynamic_pp_case.hpp",
        needs_init=False,
        needs_metrics=False,
        needs_clusters_arg=True,
        phase_key="pp",
        variant_key="dynamic",
        display_name="K-Means++ dynamic C++",
    ),
    "soa_static": CppCase(
        name="soa_static",
        case_struct="static_soa_case",
        case_header="cases/static_soa_case.hpp",
        needs_init=False,
        needs_metrics=False,
        needs_clusters_arg=False,
        phase_key="soa",
        variant_key="static",
        display_name="AoS to static SoA C++",
    ),
    "soa_dynamic": CppCase(
        name="soa_dynamic",
        case_struct="dynamic_soa_case",
        case_header="cases/dynamic_soa_case.hpp",
        needs_init=False,
        needs_metrics=False,
        needs_clusters_arg=False,
        phase_key="soa",
        variant_key="dynamic",
        display_name="AoS to dynamic SoA C++",
    ),
    "hdbscan_static": CppCase(
        name="hdbscan_static",
        case_struct="static_hdbscan_case",
        case_header="cases/static_hdbscan_case.hpp",
        needs_init=False,
        needs_metrics=True,
        needs_clusters_arg=False,
        phase_key="hdbscan",
        variant_key="static",
        display_name="HDBSCAN static C++",
        stage_keys=(HDBSCAN_DISTANCE_STAGE_KEY,),
    ),
}


def get_cpp_case(cpp_case: str) -> CppCase:
    try:
        return CPP_CASES[cpp_case]
    except KeyError as exc:
        valid = ", ".join(sorted(CPP_CASES))
        raise ValueError(
            f"Unknown C++ benchmark case '{cpp_case}'. Valid cases: {valid}"
        ) from exc


def nanobench_binary_path(cpp_case: str) -> str:
    cpp_case_def = get_cpp_case(cpp_case)
    return str(BIN_DIR / f"bench_{cpp_case_def.name}.bin")


def callgrind_binary_path(cpp_case: str, D: int) -> Path:
    cpp_case_def = get_cpp_case(cpp_case)
    return BIN_DIR / f"profile_{cpp_case_def.name}_callgrind_{D}D.bin"


def spill_detector_assembly_path(
    cpp_case: str,
    D: int,
    out_dir: str | Path | None = None,
    gmm_covariance_type: str | None = None,
) -> Path:
    cpp_case_def = get_cpp_case(cpp_case)
    root = (
        Path(out_dir)
        if out_dir is not None
        else Path(repo_path("spill_detector_results"))
    )
    covariance_suffix = (
        f".{gmm_covariance_type}"
        if cpp_case_def.needs_gmm_init and gmm_covariance_type
        else ""
    )
    return root / f"asm.{cpp_case_def.name}{covariance_suffix}.{D}D.s"


def cpp_compile_command(
    D: int,
    cpp_case: str,
    mode: str,
    output: str | Path | None = None,
    extra_defines: list[str] | None = None,
) -> list[str]:
    if mode == "nanobench":
        src = repo_path("cpp", "benchmarks", "nanobench_main.cpp")
        out = str(output) if output is not None else nanobench_binary_path(cpp_case)
    elif mode == "callgrind":
        src = repo_path("cpp", "benchmarks", "callgrind_main.cpp")
        out = (
            str(output)
            if output is not None
            else str(callgrind_binary_path(cpp_case, D))
        )
    elif mode == "assembly":
        src = repo_path("cpp", "benchmarks", "spill_detector_main.cpp")
        out = (
            str(output)
            if output is not None
            else str(spill_detector_assembly_path(cpp_case, D))
        )
    else:
        raise ValueError(f"Unknown C++ benchmark mode '{mode}'")

    cpp_case_def = get_cpp_case(cpp_case)

    return [
        "g++-14",
        "-O3",
        *(["-g"] if mode == "callgrind" else []),
        *(["-S", "-masm=intel"] if mode == "assembly" else []),
        ("-march=x86-64-v3" if mode == "callgrind" else "-march=native"),
        "-std=c++20",
        "-I../eve/include",
        "-I../nanobench/src/include",
        f"-DTUPLE_SIZE={D}",
        "-DKMEANS_N_GROUP=2",
        "-DKMEANS_K_TILE=5",
        "-DKMEANS_PP_N_VECTORS=2",
        "-DKMEANS_PP_LOCAL_TRIAL_TILE=5",
        "-DGMM_N_GROUP=2",
        "-DGMM_K_TILE=5",
        f'-DBENCH_CASE_HEADER="{cpp_case_def.case_header}"',
        f"-DBENCH_CASE={cpp_case_def.case_struct}",
        *(extra_defines or []),
        src,
        "-o",
        out,
    ]
