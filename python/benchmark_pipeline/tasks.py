import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark_pipeline.config import PipelineOptions
from benchmark_pipeline.cpp_cases import get_cpp_case, nanobench_binary_path
from benchmark_pipeline.gmm_covariance import validate_gmm_covariance_types
from benchmark_pipeline.paths import DATASETS_DIR, repo_path, repo_relative_path

NO_PARAMS = "default"
REFERENCE_VARIANT = "reference"


@dataclass(frozen=True, order=True)
class BenchmarkCase:
    D: int
    N: int
    K: int

    @property
    def case_id(self) -> str:
        return f"{self.D}D_{self.N}N_{self.K}K"

    @property
    def label(self) -> str:
        return f"{self.D}D | {self.N}N | {self.K}K"

    def dimension_args(self) -> list[str]:
        return ["--D", str(self.D), "--N", str(self.N), "--K", str(self.K)]


def benchmark_case(D: int, N: int, K: int) -> BenchmarkCase:
    return BenchmarkCase(D=D, N=N, K=K)


@dataclass
class Task:
    name: str
    command: list[str]
    cpp_case: str | None = None
    cpp_json_arg: int | None = None
    cpp_metrics_arg: int | None = None


def config_id(D: int, N: int, K: int) -> str:
    return benchmark_case(D, N, K).case_id


def configuration_label(D: int, N: int, K: int) -> str:
    return benchmark_case(D, N, K).label


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


@dataclass(frozen=True)
class BenchmarkArtifacts:
    case_id: str
    datasets_dir: str | Path = DATASETS_DIR

    @classmethod
    def for_case(
        cls,
        case: BenchmarkCase,
        datasets_dir: str | Path = DATASETS_DIR,
    ) -> "BenchmarkArtifacts":
        return cls(case.case_id, datasets_dir)

    def dataset(self, filename: str) -> str:
        return dataset_path(filename, self.datasets_dir)

    @property
    def dataset_bin(self) -> str:
        return self.dataset(f"data_{self.case_id}.bin")

    @property
    def init_centroids_bin(self) -> str:
        return self.dataset(f"init_{self.case_id}.bin")

    @property
    def gmm_weights_bin(self) -> str:
        return self.dataset(f"gmm_weights_{self.case_id}.bin")

    @property
    def gmm_means_bin(self) -> str:
        return self.dataset(f"gmm_means_{self.case_id}.bin")

    def gmm_precisions_bin(self, covariance_type: str) -> str:
        return gmm_precisions_path(covariance_type, self.case_id, self.datasets_dir)

    def timing(
        self,
        phase_key: str,
        variant_key: str,
        language_key: str,
        params_key: str = NO_PARAMS,
    ) -> str:
        return timing_artifact_path(
            phase_key,
            variant_key,
            language_key,
            self.case_id,
            params_key,
            self.datasets_dir,
        )

    def metrics(
        self,
        phase_key: str,
        variant_key: str,
        language_key: str,
        params_key: str = NO_PARAMS,
    ) -> str:
        return metrics_artifact_path(
            phase_key,
            variant_key,
            language_key,
            self.case_id,
            params_key,
            self.datasets_dir,
        )


def cpp_task_name(cpp_case: str, params_key: str = NO_PARAMS) -> str:
    case = get_cpp_case(cpp_case)
    if case.phase_key == "gmm" and params_key != NO_PARAMS:
        return f"C++: {case.display_name} ({params_key} covariance)"
    return f"C++: {case.display_name}"


def cpp_timing_args(options: PipelineOptions) -> list[str]:
    return [str(options.timing_values), str(options.timing_min_time)]


def python_pyperf_args(options: PipelineOptions, output: str) -> list[str]:
    return [
        "--processes",
        str(options.timing_processes),
        "--values",
        str(options.timing_values),
        "--min-time",
        str(options.timing_min_time),
        "--output",
        output,
    ]


def add_cpp_case_task(
    tasks: list[Task],
    cpp_case: str,
    command_args: list[str],
    *,
    json_out: str,
    options: PipelineOptions,
    metrics_out: str | None = None,
    params_key: str = NO_PARAMS,
) -> None:
    command = [nanobench_binary_path(cpp_case), *command_args]
    cpp_metrics_arg = None

    if metrics_out is not None:
        cpp_metrics_arg = len(command)
        command.append(metrics_out)

    cpp_json_arg = len(command)
    command.append(json_out)
    command.extend(cpp_timing_args(options))

    tasks.append(
        Task(
            name=cpp_task_name(cpp_case, params_key),
            command=command,
            cpp_case=cpp_case,
            cpp_json_arg=cpp_json_arg,
            cpp_metrics_arg=cpp_metrics_arg,
        )
    )


def build_dataset_setup_task(
    case: BenchmarkCase,
    gmm_covariance_types: tuple[str, ...] = (),
    datasets_dir: str | Path = DATASETS_DIR,
) -> Task:
    validate_gmm_covariance_types(gmm_covariance_types)

    artifacts = BenchmarkArtifacts.for_case(case, datasets_dir)

    setup_command = [
        sys.executable,
        repo_path("python", "benchmark_pipeline", "tools", "dataset_gen.py"),
        *case.dimension_args(),
        "--dataset-out",
        artifacts.dataset_bin,
        "--centroids-out",
        artifacts.init_centroids_bin,
    ]

    if gmm_covariance_types:
        setup_command.extend(
            [
                "--gmm-weights-out",
                artifacts.gmm_weights_bin,
                "--gmm-means-out",
                artifacts.gmm_means_bin,
            ]
        )

        for covariance_type in gmm_covariance_types:
            setup_command.extend(
                [
                    "--gmm-precisions-out",
                    covariance_type,
                    artifacts.gmm_precisions_bin(covariance_type),
                ]
            )

    setup_name = "Setup: Generate Dataset & K-Means++ Init"
    if gmm_covariance_types:
        setup_name += " & GMM Init"

    return Task(name=setup_name, command=setup_command)


def _validate_cpp_gmm_cases(
    cpp_gmm_cases: tuple[str, ...],
    gmm_covariance_types: tuple[str, ...],
) -> None:
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


def build_pipeline(
    case: BenchmarkCase,
    options: PipelineOptions,
    datasets_dir: str | Path = DATASETS_DIR,
) -> list[Task]:
    """Defines the strict contract of tasks for a single D/N/K configuration."""
    validate_gmm_covariance_types(options.gmm_covariance_types)
    _validate_cpp_gmm_cases(options.cpp_gmm_cases, options.gmm_covariance_types)

    needs_gmm_init = bool(options.cpp_gmm_cases) or options.run_python_gmm
    if needs_gmm_init and not options.gmm_covariance_types:
        raise ValueError(
            "At least one GMM covariance type is required when GMM tasks are enabled."
        )

    artifacts = BenchmarkArtifacts.for_case(case, datasets_dir)
    setup_gmm_covariance_types = (
        options.gmm_covariance_types if needs_gmm_init else ()
    )
    tasks: list[Task] = [
        build_dataset_setup_task(
            case,
            setup_gmm_covariance_types,
            datasets_dir,
        )
    ]

    for cpp_case in options.cpp_soa_cases:
        cpp_case_info = get_cpp_case(cpp_case)
        add_cpp_case_task(
            tasks,
            cpp_case,
            [
                artifacts.dataset_bin,
                str(case.N),
            ],
            json_out=artifacts.timing(
                cpp_case_info.phase_key,
                cpp_case_info.variant_key,
                "cpp",
            ),
            options=options,
        )

    if options.run_cpp_pp:
        cpp_case = "pp"
        cpp_case_info = get_cpp_case(cpp_case)
        add_cpp_case_task(
            tasks,
            cpp_case,
            [
                artifacts.dataset_bin,
                str(case.N),
                str(case.K),
            ],
            json_out=artifacts.timing(
                cpp_case_info.phase_key,
                cpp_case_info.variant_key,
                "cpp",
            ),
            options=options,
        )

    if options.run_python_pp:
        tasks.append(
            Task(
                name="Python: K-Means++ Initialization",
                command=[
                    sys.executable,
                    repo_path("python", "benchmark_pipeline", "benches", "bench_pp.py"),
                    "--dataset-bin",
                    artifacts.dataset_bin,
                    *case.dimension_args(),
                    *python_pyperf_args(
                        options,
                        artifacts.timing("pp", REFERENCE_VARIANT, "py"),
                    ),
                ],
            )
        )

    for cpp_case in options.cpp_lloyd_cases:
        cpp_case_info = get_cpp_case(cpp_case)
        add_cpp_case_task(
            tasks,
            cpp_case,
            [
                artifacts.dataset_bin,
                str(case.N),
                str(case.K),
                artifacts.init_centroids_bin,
            ],
            metrics_out=artifacts.metrics(
                cpp_case_info.phase_key,
                cpp_case_info.variant_key,
                "cpp",
            ),
            json_out=artifacts.timing(
                cpp_case_info.phase_key,
                cpp_case_info.variant_key,
                "cpp",
            ),
            options=options,
        )

    if options.run_python_lloyd:
        tasks.append(
            Task(
                name="Python: Lloyd Algorithm",
                command=[
                    sys.executable,
                    repo_path("python", "benchmark_pipeline", "benches", "bench_lloyd.py"),
                    "--dataset-bin",
                    artifacts.dataset_bin,
                    *case.dimension_args(),
                    "--init-centroids-bin",
                    artifacts.init_centroids_bin,
                    "--metrics-file",
                    artifacts.metrics("lloyd", REFERENCE_VARIANT, "py"),
                    *python_pyperf_args(
                        options,
                        artifacts.timing("lloyd", REFERENCE_VARIANT, "py"),
                    ),
                ],
            )
        )

    for covariance_type in options.gmm_covariance_types:
        gmm_precisions_bin = artifacts.gmm_precisions_bin(covariance_type)

        for cpp_case in options.cpp_gmm_cases:
            cpp_case_info = get_cpp_case(cpp_case)
            add_cpp_case_task(
                tasks,
                cpp_case,
                [
                    artifacts.dataset_bin,
                    str(case.N),
                    str(case.K),
                    artifacts.gmm_weights_bin,
                    artifacts.gmm_means_bin,
                    gmm_precisions_bin,
                    covariance_type,
                ],
                metrics_out=artifacts.metrics(
                    cpp_case_info.phase_key,
                    cpp_case_info.variant_key,
                    "cpp",
                    covariance_type,
                ),
                json_out=artifacts.timing(
                    cpp_case_info.phase_key,
                    cpp_case_info.variant_key,
                    "cpp",
                    covariance_type,
                ),
                options=options,
                params_key=covariance_type,
            )

        if options.run_python_gmm:
            tasks.append(
                Task(
                    name=f"Python: GaussianMixture EM ({covariance_type} covariance)",
                    command=[
                        sys.executable,
                        repo_path("python", "benchmark_pipeline", "benches", "bench_gmm.py"),
                        "--dataset-bin",
                        artifacts.dataset_bin,
                        *case.dimension_args(),
                        "--covariance-type",
                        covariance_type,
                        "--gmm-weights-bin",
                        artifacts.gmm_weights_bin,
                        "--gmm-means-bin",
                        artifacts.gmm_means_bin,
                        "--gmm-precisions-bin",
                        gmm_precisions_bin,
                        "--metrics-file",
                        artifacts.metrics(
                            "gmm",
                            REFERENCE_VARIANT,
                            "py",
                            covariance_type,
                        ),
                        *python_pyperf_args(
                            options,
                            artifacts.timing(
                                "gmm",
                                REFERENCE_VARIANT,
                                "py",
                                covariance_type,
                            ),
                        ),
                    ],
                )
            )

    return tasks
