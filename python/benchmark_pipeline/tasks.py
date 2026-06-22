import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark_pipeline.config import PipelineOptions
from benchmark_pipeline.exclusions import (
    BenchmarkExclusionRule,
    excluded_phase_keys_for_case,
    is_cachegrind_excluded,
)
from benchmark_pipeline.cpp_cases import get_cpp_case, nanobench_binary_path
from benchmark_pipeline.cpp_cases import callgrind_binary_path
from benchmark_pipeline.cachegrind import (
    add_cache_options,
    cache_model_record,
    cachegrind_summary_path,
    cachegrind_file_stem,
)
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
    kind: str = "subprocess"
    cpp_case: str | None = None
    cpp_json_arg: int | None = None
    cpp_metrics_arg: int | None = None
    cachegrind: "CachegrindTaskInfo | None" = None


@dataclass(frozen=True)
class CachegrindTaskInfo:
    cpp_case: str
    D: int
    N: int
    K: int
    params_key: str
    cache_model: dict[str, str | None]
    raw_output: str
    annotated_output: str
    stdout_path: str
    stderr_path: str
    annotate_stderr_path: str
    summary_path: str
    metrics_path: str | None = None


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


def enabled_phase_keys_for_options(options: PipelineOptions) -> set[str]:
    phase_keys: set[str] = set()

    if options.cpp_soa_cases:
        phase_keys.add("soa")
    if options.cpp_pp_cases or options.run_python_pp:
        phase_keys.add("pp")
    if options.cpp_lloyd_cases or options.run_python_lloyd:
        phase_keys.add("lloyd")
    if options.cpp_gmm_cases or options.run_python_gmm:
        phase_keys.add("gmm")

    return phase_keys


@dataclass(frozen=True, order=True)
class CppTarget:
    cpp_case: str
    params_key: str = NO_PARAMS

    @property
    def phase_key(self) -> str:
        return get_cpp_case(self.cpp_case).phase_key

    @property
    def variant_key(self) -> str:
        return get_cpp_case(self.cpp_case).variant_key


def active_cpp_targets_for_case(
    case: BenchmarkCase,
    options: PipelineOptions,
    exclusion_rules: tuple[BenchmarkExclusionRule, ...] = (),
) -> list[CppTarget]:
    """Return concrete C++ targets that normal timing would run for one D/N/K."""
    enabled_phase_keys = enabled_phase_keys_for_options(options)
    excluded_phase_keys = excluded_phase_keys_for_case(
        D=case.D,
        N=case.N,
        K=case.K,
        rules=exclusion_rules,
        phase_keys=enabled_phase_keys,
    )
    active_phase_keys = enabled_phase_keys - excluded_phase_keys

    targets: list[CppTarget] = []

    if "soa" in active_phase_keys:
        targets.extend(CppTarget(cpp_case) for cpp_case in options.cpp_soa_cases)
    if "pp" in active_phase_keys:
        targets.extend(CppTarget(cpp_case) for cpp_case in options.cpp_pp_cases)
    if "lloyd" in active_phase_keys:
        targets.extend(CppTarget(cpp_case) for cpp_case in options.cpp_lloyd_cases)
    if "gmm" in active_phase_keys:
        for covariance_type in options.gmm_covariance_types:
            targets.extend(
                CppTarget(cpp_case, covariance_type)
                for cpp_case in options.cpp_gmm_cases
            )

    return sorted(targets)


def active_cachegrind_targets_for_case(
    case: BenchmarkCase,
    options: PipelineOptions,
    exclusion_rules: tuple[BenchmarkExclusionRule, ...] = (),
) -> list[CppTarget]:
    if not options.run_cachegrind:
        return []

    return [
        target
        for target in active_cpp_targets_for_case(case, options, exclusion_rules)
        if not is_cachegrind_excluded(
            D=case.D,
            N=case.N,
            K=case.K,
            phase_key=target.phase_key,
            cpp_case=target.cpp_case,
            params_key=target.params_key,
            rules=options.cachegrind_exclusion_rules,
        )
    ]


def cachegrind_model_for_options(
    options: PipelineOptions,
) -> dict[str, str | None]:
    return cache_model_record(
        I1=options.cachegrind_I1,
        D1=options.cachegrind_D1,
        LL=options.cachegrind_LL,
    )


def cpp_case_runtime_args(
    *,
    cpp_case: str,
    case: BenchmarkCase,
    artifacts: BenchmarkArtifacts,
    params_key: str = NO_PARAMS,
    metrics_out: str | None = None,
) -> list[str]:
    case_info = get_cpp_case(cpp_case)
    command_args = [
        artifacts.dataset_bin,
        str(case.N),
    ]

    if case_info.needs_clusters_arg:
        command_args.append(str(case.K))

    if case_info.needs_init:
        command_args.append(artifacts.init_centroids_bin)

    if case_info.needs_gmm_init:
        if params_key == NO_PARAMS:
            raise ValueError(
                f"C++ case {cpp_case!r} needs a GMM covariance params_key."
            )
        command_args.extend(
            [
                artifacts.gmm_weights_bin,
                artifacts.gmm_means_bin,
                artifacts.gmm_precisions_bin(params_key),
            ]
        )

    if case_info.needs_covariance_type_arg:
        command_args.append(params_key)

    if case_info.needs_metrics:
        if metrics_out is None:
            raise ValueError(f"C++ case {cpp_case!r} needs a metrics output path.")
        command_args.append(metrics_out)

    return command_args

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
            kind="cpp_timing",
            cpp_case=cpp_case,
            cpp_json_arg=cpp_json_arg,
            cpp_metrics_arg=cpp_metrics_arg,
        )
    )


def cachegrind_task_name(cpp_case: str, params_key: str = NO_PARAMS) -> str:
    case = get_cpp_case(cpp_case)
    suffix = ""
    if params_key != NO_PARAMS:
        suffix = f" ({params_key} covariance)"
    return f"Cachegrind: {case.display_name}{suffix}"


def build_cachegrind_task(
    *,
    case: BenchmarkCase,
    cpp_case: str,
    artifacts: BenchmarkArtifacts,
    options: PipelineOptions,
    params_key: str = NO_PARAMS,
) -> Task:
    case_info = get_cpp_case(cpp_case)
    out_dir = repo_relative_path(options.cachegrind_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = cachegrind_file_stem(cpp_case, params_key, case.case_id)
    raw_output = out_dir / f"{stem}.out"
    annotated_output = out_dir / f"{stem}.annotate.txt"
    stdout_path = out_dir / f"valgrind.{stem}.stdout.txt"
    stderr_path = out_dir / f"valgrind.{stem}.stderr.txt"
    annotate_stderr_path = out_dir / f"{stem}.annotate.stderr.txt"
    summary_path = cachegrind_summary_path(
        out_dir,
        cpp_case,
        params_key,
        case.case_id,
    )

    metrics_path: Path | None = None
    if case_info.needs_metrics:
        metrics_path = out_dir / f"{cpp_case}_metrics_cpp.{params_key}.{case.case_id}.json"

    command = [
        "valgrind",
        "--tool=callgrind",
        "--cache-sim=yes",
        "--branch-sim=no",
        "--instr-atstart=no",
        f"--callgrind-out-file={raw_output}",
    ]
    add_cache_options(
        command,
        I1=options.cachegrind_I1,
        D1=options.cachegrind_D1,
        LL=options.cachegrind_LL,
    )
    command.append(str(callgrind_binary_path(cpp_case, case.D)))
    command.extend(
        cpp_case_runtime_args(
            cpp_case=cpp_case,
            case=case,
            artifacts=artifacts,
            params_key=params_key,
            metrics_out=str(metrics_path) if metrics_path is not None else None,
        )
    )

    return Task(
        name=cachegrind_task_name(cpp_case, params_key),
        command=command,
        kind="cachegrind",
        cpp_case=cpp_case,
        cachegrind=CachegrindTaskInfo(
            cpp_case=cpp_case,
            D=case.D,
            N=case.N,
            K=case.K,
            params_key=params_key,
            cache_model=cachegrind_model_for_options(options),
            raw_output=str(raw_output),
            annotated_output=str(annotated_output),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            annotate_stderr_path=str(annotate_stderr_path),
            summary_path=str(summary_path),
            metrics_path=str(metrics_path) if metrics_path is not None else None,
        ),
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
    exclusion_rules: tuple[BenchmarkExclusionRule, ...] = (),
) -> list[Task]:
    """Defines the strict contract of tasks for a single D/N/K configuration."""
    validate_gmm_covariance_types(options.gmm_covariance_types)
    _validate_cpp_gmm_cases(options.cpp_gmm_cases, options.gmm_covariance_types)

    enabled_phase_keys = enabled_phase_keys_for_options(options)
    excluded_phase_keys = excluded_phase_keys_for_case(
        D=case.D,
        N=case.N,
        K=case.K,
        rules=exclusion_rules,
        phase_keys=enabled_phase_keys,
    )
    active_phase_keys = enabled_phase_keys - excluded_phase_keys

    needs_gmm_init = "gmm" in active_phase_keys
    if ("gmm" in enabled_phase_keys) and not options.gmm_covariance_types:
        raise ValueError(
            "At least one GMM covariance type is required when GMM tasks are enabled."
        )

    if not active_phase_keys:
        return []

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

    if "soa" in active_phase_keys:
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

    if "pp" in active_phase_keys:
        for cpp_case in options.cpp_pp_cases:
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

    if "lloyd" in active_phase_keys:
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

    if "gmm" in active_phase_keys:
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

    for target in active_cachegrind_targets_for_case(
        case,
        options,
        exclusion_rules,
    ):
        tasks.append(
            build_cachegrind_task(
                case=case,
                cpp_case=target.cpp_case,
                artifacts=artifacts,
                options=options,
                params_key=target.params_key,
            )
        )

    return tasks
