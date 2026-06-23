import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark_pipeline.config import PipelineOptions
from benchmark_pipeline.exclusions import (
    BenchmarkExclusionRule,
    excluded_phase_stage_keys_for_case,
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
from benchmark_metadata import (
    FULL_STAGE_KEY,
    LANGUAGE_CPP_KEY,
    LANGUAGE_PY_KEY,
    NO_PARAMS,
    REFERENCE_VARIANT,
    stage_display_name,
)
from benchmark_pipeline.stages import (
    DATASET_ARTIFACT,
    GMM_MEANS_ARTIFACT,
    GMM_PRECISIONS_ARTIFACT,
    GMM_WEIGHTS_ARTIFACT,
    INIT_CENTROIDS_ARTIFACT,
    get_stage_spec,
    stage_keys_for_phase,
)


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
    phase_key: str | None = None
    stage_key: str = FULL_STAGE_KEY
    input_artifacts: tuple[str, ...] = ()
    output_artifacts: tuple[str, ...] = ()
    cpp_json_arg: int | None = None
    cpp_metrics_arg: int | None = None
    cachegrind: "CachegrindTaskInfo | None" = None


@dataclass(frozen=True)
class CachegrindTaskInfo:
    cpp_case: str
    stage_key: str
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
    stage_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
) -> list[str]:
    parts = [phase_key, stage_key, variant_key]

    if params_key != NO_PARAMS:
        parts.append(params_key)

    parts.extend([language_key, case_id])
    return parts


def timing_artifact_name(
    phase_key: str,
    stage_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
) -> str:
    return "_".join(
        artifact_name_parts(
            phase_key,
            stage_key,
            variant_key,
            language_key,
            case_id,
            params_key,
        )
    ) + ".json"


def metrics_artifact_name(
    phase_key: str,
    stage_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
) -> str:
    return "_".join(
        [phase_key, stage_key, "metrics"]
        + artifact_name_parts(
            "",
            "",
            variant_key,
            language_key,
            case_id,
            params_key,
        )[2:]
    ) + ".json"


def timing_artifact_path(
    phase_key: str,
    stage_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
    datasets_dir: str | Path = DATASETS_DIR,
) -> str:
    return dataset_path(
        timing_artifact_name(
            phase_key,
            stage_key,
            variant_key,
            language_key,
            case_id,
            params_key,
        ),
        datasets_dir,
    )


def metrics_artifact_path(
    phase_key: str,
    stage_key: str,
    variant_key: str,
    language_key: str,
    case_id: str,
    params_key: str = NO_PARAMS,
    datasets_dir: str | Path = DATASETS_DIR,
) -> str:
    return dataset_path(
        metrics_artifact_name(
            phase_key,
            stage_key,
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

    def generic_stage_artifact(
        self,
        stage_key: str,
        artifact_key: str,
        params_key: str = NO_PARAMS,
    ) -> str:
        params_part = "" if params_key == NO_PARAMS else f"_{params_key}"
        return self.dataset(
            f"{stage_key}_{artifact_key}{params_part}_{self.case_id}.bin"
        )

    def artifact(
        self,
        artifact_key: str,
        *,
        stage_key: str = FULL_STAGE_KEY,
        params_key: str = NO_PARAMS,
    ) -> str:
        if artifact_key == DATASET_ARTIFACT:
            return self.dataset_bin
        if artifact_key == INIT_CENTROIDS_ARTIFACT:
            return self.init_centroids_bin
        if artifact_key == GMM_WEIGHTS_ARTIFACT:
            return self.gmm_weights_bin
        if artifact_key == GMM_MEANS_ARTIFACT:
            return self.gmm_means_bin
        if artifact_key == GMM_PRECISIONS_ARTIFACT:
            if params_key == NO_PARAMS:
                raise ValueError("gmm_precisions artifacts require a params_key")
            return self.gmm_precisions_bin(params_key)
        return self.generic_stage_artifact(stage_key, artifact_key, params_key)

    def timing(
        self,
        phase_key: str,
        stage_key: str,
        variant_key: str,
        language_key: str,
        params_key: str = NO_PARAMS,
    ) -> str:
        return timing_artifact_path(
            phase_key,
            stage_key,
            variant_key,
            language_key,
            self.case_id,
            params_key,
            self.datasets_dir,
        )

    def metrics(
        self,
        phase_key: str,
        stage_key: str,
        variant_key: str,
        language_key: str,
        params_key: str = NO_PARAMS,
    ) -> str:
        return metrics_artifact_path(
            phase_key,
            stage_key,
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


def enabled_stage_keys_by_phase_for_options(
    options: PipelineOptions,
) -> dict[str, tuple[str, ...]]:
    return {
        phase_key: stage_keys_for_phase(phase_key)
        for phase_key in enabled_phase_keys_for_options(options)
    }


@dataclass(frozen=True, order=True)
class CppTarget:
    cpp_case: str
    params_key: str = NO_PARAMS

    @property
    def phase_key(self) -> str:
        return get_cpp_case(self.cpp_case).phase_key

    @property
    def stage_key(self) -> str:
        return get_cpp_case(self.cpp_case).stage_key

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
    stage_keys_by_phase = enabled_stage_keys_by_phase_for_options(options)
    excluded_phase_stage_keys = excluded_phase_stage_keys_for_case(
        D=case.D,
        N=case.N,
        K=case.K,
        rules=exclusion_rules,
        phase_keys=enabled_phase_keys,
        stage_keys_by_phase=stage_keys_by_phase,
    )

    targets: list[CppTarget] = []

    if "soa" in enabled_phase_keys:
        targets.extend(CppTarget(cpp_case) for cpp_case in options.cpp_soa_cases)
    if "pp" in enabled_phase_keys:
        targets.extend(CppTarget(cpp_case) for cpp_case in options.cpp_pp_cases)
    if "lloyd" in enabled_phase_keys:
        targets.extend(CppTarget(cpp_case) for cpp_case in options.cpp_lloyd_cases)
    if "gmm" in enabled_phase_keys:
        for covariance_type in options.gmm_covariance_types:
            targets.extend(
                CppTarget(cpp_case, covariance_type)
                for cpp_case in options.cpp_gmm_cases
            )

    return sorted(
        target
        for target in targets
        if (target.phase_key, target.stage_key) not in excluded_phase_stage_keys
    )


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
            stage_key=target.stage_key,
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
        artifacts.artifact(
            case_info.primary_input_artifact_key,
            stage_key=case_info.stage_key,
            params_key=params_key,
        ),
        str(case.N),
    ]

    if case_info.needs_clusters_arg:
        command_args.append(str(case.K))

    if case_info.needs_init:
        command_args.append(
            artifacts.artifact(
                INIT_CENTROIDS_ARTIFACT,
                stage_key=case_info.stage_key,
                params_key=params_key,
            )
        )

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
    stage_suffix = ""
    if case.stage_key != FULL_STAGE_KEY:
        stage_suffix = f" [{stage_display_name(case.stage_key)}]"
    if case.phase_key == "gmm" and params_key != NO_PARAMS:
        return f"C++: {case.display_name}{stage_suffix} ({params_key} covariance)"
    return f"C++: {case.display_name}{stage_suffix}"


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


def cpp_case_timing_artifact(
    *,
    cpp_case: str,
    artifacts: BenchmarkArtifacts,
    params_key: str = NO_PARAMS,
) -> str:
    case_info = get_cpp_case(cpp_case)
    return artifacts.timing(
        case_info.phase_key,
        case_info.stage_key,
        case_info.variant_key,
        LANGUAGE_CPP_KEY,
        params_key,
    )


def cpp_case_metrics_artifact(
    *,
    cpp_case: str,
    artifacts: BenchmarkArtifacts,
    params_key: str = NO_PARAMS,
) -> str | None:
    case_info = get_cpp_case(cpp_case)
    if not case_info.needs_metrics:
        return None

    return artifacts.metrics(
        case_info.phase_key,
        case_info.stage_key,
        case_info.variant_key,
        LANGUAGE_CPP_KEY,
        params_key,
    )


def build_cpp_case_task(
    *,
    cpp_case: str,
    case: BenchmarkCase,
    artifacts: BenchmarkArtifacts,
    options: PipelineOptions,
    params_key: str = NO_PARAMS,
) -> Task:
    metrics_out = cpp_case_metrics_artifact(
        cpp_case=cpp_case,
        artifacts=artifacts,
        params_key=params_key,
    )
    runtime_args = cpp_case_runtime_args(
        cpp_case=cpp_case,
        case=case,
        artifacts=artifacts,
        params_key=params_key,
        metrics_out=metrics_out,
    )

    command = [nanobench_binary_path(cpp_case), *runtime_args]
    cpp_metrics_arg = len(command) - 1 if metrics_out is not None else None

    cpp_json_arg = len(command)
    command.append(
        cpp_case_timing_artifact(
            cpp_case=cpp_case,
            artifacts=artifacts,
            params_key=params_key,
        )
    )
    command.extend(cpp_timing_args(options))

    case_info = get_cpp_case(cpp_case)
    stage_spec = get_stage_spec(case_info.phase_key, case_info.stage_key)
    input_artifacts = tuple(
        artifacts.artifact(artifact_key, stage_key=case_info.stage_key, params_key=params_key)
        for artifact_key in stage_spec.input_artifact_keys
    )
    if case_info.needs_init:
        input_artifacts = (
            *input_artifacts,
            artifacts.artifact(
                INIT_CENTROIDS_ARTIFACT,
                stage_key=case_info.stage_key,
                params_key=params_key,
            ),
        )
    if case_info.needs_gmm_init:
        input_artifacts = (
            *input_artifacts,
            artifacts.gmm_weights_bin,
            artifacts.gmm_means_bin,
            artifacts.gmm_precisions_bin(params_key),
        )

    output_artifacts = (metrics_out,) if metrics_out is not None else ()

    return Task(
        name=cpp_task_name(cpp_case, params_key),
        command=command,
        kind="cpp_timing",
        cpp_case=cpp_case,
        phase_key=case_info.phase_key,
        stage_key=case_info.stage_key,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        cpp_json_arg=cpp_json_arg,
        cpp_metrics_arg=cpp_metrics_arg,
    )


def add_cpp_case_task(
    tasks: list[Task],
    *,
    cpp_case: str,
    case: BenchmarkCase,
    artifacts: BenchmarkArtifacts,
    options: PipelineOptions,
    params_key: str = NO_PARAMS,
) -> None:
    tasks.append(
        build_cpp_case_task(
            cpp_case=cpp_case,
            case=case,
            artifacts=artifacts,
            options=options,
            params_key=params_key,
        )
    )


def cachegrind_task_name(cpp_case: str, params_key: str = NO_PARAMS) -> str:
    case = get_cpp_case(cpp_case)
    suffix = ""
    if params_key != NO_PARAMS:
        suffix = f" ({params_key} covariance)"
    stage_suffix = ""
    if case.stage_key != FULL_STAGE_KEY:
        stage_suffix = f" [{stage_display_name(case.stage_key)}]"
    return f"Cachegrind: {case.display_name}{stage_suffix}{suffix}"


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

    stem = cachegrind_file_stem(cpp_case, case_info.stage_key, params_key, case.case_id)
    raw_output = out_dir / f"{stem}.out"
    annotated_output = out_dir / f"{stem}.annotate.txt"
    stdout_path = out_dir / f"valgrind.{stem}.stdout.txt"
    stderr_path = out_dir / f"valgrind.{stem}.stderr.txt"
    annotate_stderr_path = out_dir / f"{stem}.annotate.stderr.txt"
    summary_path = cachegrind_summary_path(
        out_dir,
        cpp_case,
        case_info.stage_key,
        params_key,
        case.case_id,
    )

    metrics_path: Path | None = None
    if case_info.needs_metrics:
        metrics_path = out_dir / (
            f"{case_info.phase_key}_{case_info.stage_key}_metrics_"
            f"{case_info.variant_key}_{params_key}_cpp_{case.case_id}.json"
        )

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
        phase_key=case_info.phase_key,
        stage_key=case_info.stage_key,
        cachegrind=CachegrindTaskInfo(
            cpp_case=cpp_case,
            stage_key=case_info.stage_key,
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

    produced = [artifacts.dataset_bin, artifacts.init_centroids_bin]
    if gmm_covariance_types:
        produced.extend([artifacts.gmm_weights_bin, artifacts.gmm_means_bin])
        produced.extend(
            artifacts.gmm_precisions_bin(covariance_type)
            for covariance_type in gmm_covariance_types
        )

    return Task(
        name=setup_name,
        command=setup_command,
        phase_key=None,
        stage_key=FULL_STAGE_KEY,
        output_artifacts=tuple(produced),
    )


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
    stage_keys_by_phase = enabled_stage_keys_by_phase_for_options(options)
    excluded_phase_keys = excluded_phase_keys_for_case(
        D=case.D,
        N=case.N,
        K=case.K,
        rules=exclusion_rules,
        phase_keys=enabled_phase_keys,
        stage_keys_by_phase=stage_keys_by_phase,
    )
    excluded_phase_stage_keys = excluded_phase_stage_keys_for_case(
        D=case.D,
        N=case.N,
        K=case.K,
        rules=exclusion_rules,
        phase_keys=enabled_phase_keys,
        stage_keys_by_phase=stage_keys_by_phase,
    )
    active_phase_keys = enabled_phase_keys - excluded_phase_keys

    def stage_is_active(phase_key: str, stage_key: str = FULL_STAGE_KEY) -> bool:
        return phase_key in active_phase_keys and (phase_key, stage_key) not in excluded_phase_stage_keys

    def cpp_case_is_active(cpp_case: str) -> bool:
        case_info = get_cpp_case(cpp_case)
        return stage_is_active(case_info.phase_key, case_info.stage_key)

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
            if not cpp_case_is_active(cpp_case):
                continue
            add_cpp_case_task(
                tasks,
                cpp_case=cpp_case,
                case=case,
                artifacts=artifacts,
                options=options,
            )

    if "pp" in active_phase_keys:
        for cpp_case in options.cpp_pp_cases:
            if not cpp_case_is_active(cpp_case):
                continue
            add_cpp_case_task(
                tasks,
                cpp_case=cpp_case,
                case=case,
                artifacts=artifacts,
                options=options,
            )

        if options.run_python_pp and stage_is_active("pp"):
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
                            artifacts.timing(
                                "pp",
                                FULL_STAGE_KEY,
                                REFERENCE_VARIANT,
                                LANGUAGE_PY_KEY,
                            ),
                        ),
                    ],
                    phase_key="pp",
                    stage_key=FULL_STAGE_KEY,
                    input_artifacts=(artifacts.dataset_bin,),
                )
            )

    if "lloyd" in active_phase_keys:
        for cpp_case in options.cpp_lloyd_cases:
            if not cpp_case_is_active(cpp_case):
                continue
            add_cpp_case_task(
                tasks,
                cpp_case=cpp_case,
                case=case,
                artifacts=artifacts,
                options=options,
            )

        if options.run_python_lloyd and stage_is_active("lloyd"):
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
                        artifacts.metrics(
                            "lloyd",
                            FULL_STAGE_KEY,
                            REFERENCE_VARIANT,
                            LANGUAGE_PY_KEY,
                        ),
                        *python_pyperf_args(
                            options,
                            artifacts.timing(
                                "lloyd",
                                FULL_STAGE_KEY,
                                REFERENCE_VARIANT,
                                LANGUAGE_PY_KEY,
                            ),
                        ),
                    ],
                    phase_key="lloyd",
                    stage_key=FULL_STAGE_KEY,
                    input_artifacts=(artifacts.dataset_bin, artifacts.init_centroids_bin),
                    output_artifacts=(
                        artifacts.metrics(
                            "lloyd",
                            FULL_STAGE_KEY,
                            REFERENCE_VARIANT,
                            LANGUAGE_PY_KEY,
                        ),
                    ),
                )
            )

    if "gmm" in active_phase_keys:
        for covariance_type in options.gmm_covariance_types:
            gmm_precisions_bin = artifacts.gmm_precisions_bin(covariance_type)

            for cpp_case in options.cpp_gmm_cases:
                if not cpp_case_is_active(cpp_case):
                    continue
                add_cpp_case_task(
                    tasks,
                    cpp_case=cpp_case,
                    case=case,
                    artifacts=artifacts,
                    options=options,
                    params_key=covariance_type,
                )

            if options.run_python_gmm and stage_is_active("gmm"):
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
                                FULL_STAGE_KEY,
                                REFERENCE_VARIANT,
                                LANGUAGE_PY_KEY,
                                covariance_type,
                            ),
                            *python_pyperf_args(
                                options,
                                artifacts.timing(
                                    "gmm",
                                    FULL_STAGE_KEY,
                                    REFERENCE_VARIANT,
                                    LANGUAGE_PY_KEY,
                                    covariance_type,
                                ),
                            ),
                        ],
                        phase_key="gmm",
                        stage_key=FULL_STAGE_KEY,
                        input_artifacts=(
                            artifacts.dataset_bin,
                            artifacts.gmm_weights_bin,
                            artifacts.gmm_means_bin,
                            gmm_precisions_bin,
                        ),
                        output_artifacts=(
                            artifacts.metrics(
                                "gmm",
                                FULL_STAGE_KEY,
                                REFERENCE_VARIANT,
                                LANGUAGE_PY_KEY,
                                covariance_type,
                            ),
                        ),
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
