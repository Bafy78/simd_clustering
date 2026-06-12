import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from benchmark_pipeline.cpp_cases import get_cpp_case, nanobench_binary_path
from benchmark_pipeline.gmm_covariance import validate_gmm_covariance_types
from benchmark_pipeline.paths import DATASETS_DIR, repo_path, repo_relative_path

NO_PARAMS = "default"
REFERENCE_VARIANT = "reference"


@dataclass
class Task:
    name: str
    command: list[str]
    cpp_case: Optional[str] = None
    cpp_json_arg: Optional[int] = None
    cpp_metrics_arg: Optional[int] = None


def config_id(D: int, N: int, K: int) -> str:
    return f"{D}D_{N}N_{K}K"


def configuration_label(D: int, N: int, K: int) -> str:
    return f"{D}D | {N}N | {K}K"


def dataset_path(filename: str, datasets_dir: str | Path = DATASETS_DIR) -> str:
    return str(repo_relative_path(datasets_dir) / filename)


def gmm_precisions_filename(covariance_type: str, case_id: str) -> str:
    return f"gmm_precisions_{covariance_type}_{case_id}.bin"


def gmm_precisions_path(
    covariance_type: str,
    case_id: str,
    datasets_dir: str | Path = DATASETS_DIR,
) -> str:
    return dataset_path(gmm_precisions_filename(covariance_type, case_id), datasets_dir)


def artifact_name_parts(
    phase_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
) -> list[str]:
    parts = [phase_key, variant_key]

    if params_key != NO_PARAMS:
        parts.append(params_key)

    parts.extend([language_key, case_id])
    return parts


def timing_artifact_name(
    phase_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
) -> str:
    return "_".join(
        artifact_name_parts(
            phase_key,
            variant_key,
            language_key,
            case_id,
            params_key,
        )
    ) + ".json"


def metrics_artifact_name(
    phase_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
) -> str:
    return "_".join(
        [phase_key, "metrics"]
        + artifact_name_parts(
            "",
            variant_key,
            language_key,
            case_id,
            params_key,
        )[1:]
    ) + ".json"


def timing_artifact_path(
    phase_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
    datasets_dir: str | Path = DATASETS_DIR,
) -> str:
    return dataset_path(
        timing_artifact_name(
            phase_key,
            variant_key,
            language_key,
            case_id,
            params_key,
        ),
        datasets_dir,
    )


def metrics_artifact_path(
    phase_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
    datasets_dir: str | Path = DATASETS_DIR,
) -> str:
    return dataset_path(
        metrics_artifact_name(
            phase_key,
            variant_key,
            language_key,
            case_id,
            params_key,
        ),
        datasets_dir,
    )


def cpp_task_name(cpp_case: str, params_key: str = NO_PARAMS) -> str:
    case = get_cpp_case(cpp_case)
    if case.phase_key == "gmm" and params_key != NO_PARAMS:
        return f"C++: {case.display_name} ({params_key} covariance)"
    return f"C++: {case.display_name}"


def add_cpp_case_task(
    tasks: list[Task],
    cpp_case: str,
    command: list[str],
    cpp_json_arg: int,
    cpp_metrics_arg: int | None = None,
    params_key: str = NO_PARAMS,
) -> None:
    tasks.append(
        Task(
            name=cpp_task_name(cpp_case, params_key),
            command=command,
            cpp_case=cpp_case,
            cpp_json_arg=cpp_json_arg,
            cpp_metrics_arg=cpp_metrics_arg,
        )
    )


def build_pipeline(
    D: int,
    N: int,
    K: int,
    timing_processes: int,
    timing_values: int,
    timing_min_time: float,
    gmm_covariance_types: tuple[str, ...],
    cpp_soa_cases: tuple[str, ...],
    run_cpp_pp: bool,
    run_python_pp: bool,
    cpp_lloyd_cases: tuple[str, ...],
    run_python_lloyd: bool,
    cpp_gmm_cases: tuple[str, ...],
    run_python_gmm: bool,
    datasets_dir: str | Path = DATASETS_DIR,
) -> list[Task]:
    """Defines the strict contract of tasks for a single D/N/K configuration."""
    validate_gmm_covariance_types(gmm_covariance_types)

    case_id = config_id(D, N, K)

    dataset_bin = dataset_path(f"data_{case_id}.bin", datasets_dir)
    init_centroids_bin = dataset_path(f"init_{case_id}.bin", datasets_dir)
    gmm_weights_bin = dataset_path(f"gmm_weights_{case_id}.bin", datasets_dir)
    gmm_means_bin = dataset_path(f"gmm_means_{case_id}.bin", datasets_dir)

    for cpp_case in cpp_gmm_cases:
        case = get_cpp_case(cpp_case)
        unsupported = sorted(
            set(gmm_covariance_types) - set(case.supported_gmm_covariance_types)
        )
        if unsupported:
            supported = ", ".join(case.supported_gmm_covariance_types)
            raise ValueError(
                f"C++ case {cpp_case!r} does not support GMM covariance "
                f"type(s) {unsupported}. Supported values: {supported}"
            )

    needs_gmm_init = bool(cpp_gmm_cases) or run_python_gmm
    if needs_gmm_init and not gmm_covariance_types:
        raise ValueError(
            "At least one GMM covariance type is required when GMM tasks are enabled."
        )

    setup_command = [
        sys.executable,
        repo_path("python", "benchmark_pipeline", "tools", "dataset_gen.py"),
        "--D",
        str(D),
        "--N",
        str(N),
        "--K",
        str(K),
        "--dataset-out",
        dataset_bin,
        "--centroids-out",
        init_centroids_bin,
    ]

    if needs_gmm_init:
        setup_command.extend([
            "--gmm-weights-out",
            gmm_weights_bin,
            "--gmm-means-out",
            gmm_means_bin,
        ])

        for covariance_type in gmm_covariance_types:
            setup_command.extend(
                [
                    "--gmm-precisions-out",
                    covariance_type,
                    gmm_precisions_path(covariance_type, case_id, datasets_dir),
                ]
            )

    setup_name = "Setup: Generate Dataset & K-Means++ Init"
    if needs_gmm_init:
        setup_name += " & GMM Init"

    tasks: list[Task] = [Task(name=setup_name, command=setup_command)]

    for cpp_case in cpp_soa_cases:
        case = get_cpp_case(cpp_case)
        add_cpp_case_task(
            tasks,
            cpp_case,
            [
                nanobench_binary_path(cpp_case),
                dataset_bin,
                str(N),
                timing_artifact_path(
                    case.phase_key,
                    case.variant_key,
                    "cpp",
                    case_id,
                    datasets_dir=datasets_dir,
                ),
                str(timing_values),
                str(timing_min_time),
            ],
            cpp_json_arg=3,
        )

    if run_cpp_pp:
        cpp_case = "pp"
        case = get_cpp_case(cpp_case)
        add_cpp_case_task(
            tasks,
            cpp_case,
            [
                nanobench_binary_path(cpp_case),
                dataset_bin,
                str(N),
                str(K),
                timing_artifact_path(
                    case.phase_key,
                    case.variant_key,
                    "cpp",
                    case_id,
                    datasets_dir=datasets_dir,
                ),
                str(timing_values),
                str(timing_min_time),
            ],
            cpp_json_arg=4,
        )

    if run_python_pp:
        tasks.append(
            Task(
                name="Python: K-Means++ Initialization",
                command=[
                    sys.executable,
                    repo_path("python", "benchmark_pipeline", "benches", "bench_pp.py"),
                    "--dataset-bin",
                    dataset_bin,
                    "--D",
                    str(D),
                    "--N",
                    str(N),
                    "--K",
                    str(K),
                    "--processes",
                    str(timing_processes),
                    "--values",
                    str(timing_values),
                    "--min-time",
                    str(timing_min_time),
                    "--output",
                    timing_artifact_path(
                        "pp",
                        REFERENCE_VARIANT,
                        "py",
                        case_id,
                        datasets_dir=datasets_dir,
                    ),
                ],
            )
        )

    for cpp_case in cpp_lloyd_cases:
        case = get_cpp_case(cpp_case)
        add_cpp_case_task(
            tasks,
            cpp_case,
            [
                nanobench_binary_path(cpp_case),
                dataset_bin,
                str(N),
                str(K),
                init_centroids_bin,
                metrics_artifact_path(
                    case.phase_key,
                    case.variant_key,
                    "cpp",
                    case_id,
                    datasets_dir=datasets_dir,
                ),
                timing_artifact_path(
                    case.phase_key,
                    case.variant_key,
                    "cpp",
                    case_id,
                    datasets_dir=datasets_dir,
                ),
                str(timing_values),
                str(timing_min_time),
            ],
            cpp_metrics_arg=5,
            cpp_json_arg=6,
        )

    if run_python_lloyd:
        tasks.append(
            Task(
                name="Python: Lloyd Algorithm",
                command=[
                    sys.executable,
                    repo_path("python", "benchmark_pipeline", "benches", "bench_lloyd.py"),
                    "--dataset-bin",
                    dataset_bin,
                    "--D",
                    str(D),
                    "--N",
                    str(N),
                    "--K",
                    str(K),
                    "--init-centroids-bin",
                    init_centroids_bin,
                    "--metrics-file",
                    metrics_artifact_path(
                        "lloyd",
                        REFERENCE_VARIANT,
                        "py",
                        case_id,
                        datasets_dir=datasets_dir,
                    ),
                    "--processes",
                    str(timing_processes),
                    "--values",
                    str(timing_values),
                    "--min-time",
                    str(timing_min_time),
                    "--output",
                    timing_artifact_path(
                        "lloyd",
                        REFERENCE_VARIANT,
                        "py",
                        case_id,
                        datasets_dir=datasets_dir,
                    ),
                ],
            )
        )

    for covariance_type in gmm_covariance_types:
        gmm_precisions_bin = gmm_precisions_path(covariance_type, case_id, datasets_dir)

        for cpp_case in cpp_gmm_cases:
            case = get_cpp_case(cpp_case)
            add_cpp_case_task(
                tasks,
                cpp_case,
                [
                    nanobench_binary_path(cpp_case),
                    dataset_bin,
                    str(N),
                    str(K),
                    gmm_weights_bin,
                    gmm_means_bin,
                    gmm_precisions_bin,
                    covariance_type,
                    metrics_artifact_path(
                        case.phase_key,
                        case.variant_key,
                        "cpp",
                        case_id,
                        covariance_type,
                        datasets_dir=datasets_dir,
                    ),
                    timing_artifact_path(
                        case.phase_key,
                        case.variant_key,
                        "cpp",
                        case_id,
                        covariance_type,
                        datasets_dir=datasets_dir,
                    ),
                    str(timing_values),
                    str(timing_min_time),
                ],
                cpp_metrics_arg=8,
                cpp_json_arg=9,
                params_key=covariance_type,
            )

        if run_python_gmm:
            tasks.append(
                Task(
                    name=f"Python: GaussianMixture EM ({covariance_type} covariance)",
                    command=[
                        sys.executable,
                        repo_path("python", "benchmark_pipeline", "benches", "bench_gmm.py"),
                        "--dataset-bin",
                        dataset_bin,
                        "--D",
                        str(D),
                        "--N",
                        str(N),
                        "--K",
                        str(K),
                        "--covariance-type",
                        covariance_type,
                        "--gmm-weights-bin",
                        gmm_weights_bin,
                        "--gmm-means-bin",
                        gmm_means_bin,
                        "--gmm-precisions-bin",
                        gmm_precisions_bin,
                        "--metrics-file",
                        metrics_artifact_path(
                            "gmm",
                            REFERENCE_VARIANT,
                            "py",
                            case_id,
                            covariance_type,
                            datasets_dir=datasets_dir,
                        ),
                        "--processes",
                        str(timing_processes),
                        "--values",
                        str(timing_values),
                        "--min-time",
                        str(timing_min_time),
                        "--output",
                        timing_artifact_path(
                            "gmm",
                            REFERENCE_VARIANT,
                            "py",
                            case_id,
                            covariance_type,
                            datasets_dir=datasets_dir,
                        ),
                    ],
                )
            )

    return tasks
