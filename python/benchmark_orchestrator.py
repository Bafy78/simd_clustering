from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmark_pipeline.config import BenchmarkConfig, default_config

from benchmark_pipeline.cachegrind import (
    build_cachegrind_exclusion_record,
    build_cachegrind_manifest,
    cachegrind_manifest_path,
    prepare_cachegrind_results_dir,
    require_cachegrind_tools,
    write_json,
)
from benchmark_pipeline.exclusions import (
    EXCLUSIONS_FILENAME,
    build_exclusion_manifest,
    write_exclusion_manifest,
)
from benchmark_pipeline.runner import (
    compile_callgrind_binaries,
    compile_cpp_binaries,
    execute_pipeline,
    prepare_datasets_dir,
)
from benchmark_pipeline.tasks import (
    BenchmarkCase,
    CppTarget,
    active_cachegrind_targets_for_case,
    active_cpp_targets_for_case,
    cachegrind_model_for_options,
    enabled_phase_keys_for_options,
    enabled_stage_keys_by_phase_for_options,
)


def _iter_config_cases(config: BenchmarkConfig) -> Iterable[BenchmarkCase]:
    for D in config.test_Ds:
        for N in config.test_Ns:
            for K in config.test_Ks:
                yield BenchmarkCase(D, N, K)

    for dataset in config.real_datasets:
        yield BenchmarkCase(
            D=dataset.D,
            N=dataset.N,
            K=dataset.K,
            dataset=dataset.key,
            dataset_source_kind=dataset.source,
            dataset_source_path=dataset.path,
            dataset_source_url=dataset.url,
            dataset_source_format=dataset.format,
            downloads_dir=config.downloads_dir,
            openml_data_id=dataset.data_id,
            openml_name=dataset.name,
            openml_version=dataset.version,
            uci_dataset_id=dataset.dataset_id,
            hf_repo=dataset.repo,
            hf_config=dataset.hf_config,
            hf_split=dataset.split,
            hf_feature_column=dataset.feature_column,
            feature_columns=dataset.feature_columns,
        )


def _cases_for_dimension(
    cases: Iterable[BenchmarkCase],
    D: int,
) -> list[BenchmarkCase]:
    return [case for case in cases if case.D == D]


def _targets_for_dimension(
    config: BenchmarkConfig,
    D: int,
    cases: Iterable[BenchmarkCase] | None,
    target_builder,
) -> set[CppTarget]:
    targets: set[CppTarget] = set()
    cases = list(_iter_config_cases(config)) if cases is None else cases

    for case in _cases_for_dimension(cases, D):
        targets.update(
            target_builder(
                case,
                config.pipeline,
                config.exclusion_rules,
            )
        )

    return targets


def cpp_cases_for_dimension(
    config: BenchmarkConfig,
    D: int,
    cases: Iterable[BenchmarkCase] | None = None,
) -> set[str]:
    return {
        target.cpp_case
        for target in _targets_for_dimension(config, D, cases, active_cpp_targets_for_case)
    }


def cachegrind_cpp_cases_for_dimension(
    config: BenchmarkConfig,
    D: int,
    cases: Iterable[BenchmarkCase] | None = None,
) -> set[str]:
    return {
        target.cpp_case
        for target in _targets_for_dimension(config, D, cases, active_cachegrind_targets_for_case)
    }


def build_cachegrind_manifest_for_config(config: BenchmarkConfig) -> dict[str, Any]:
    options = config.pipeline
    planned_records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    rules = tuple(options.cachegrind_exclusion_rules)

    if options.run_cachegrind:
        for case in _iter_config_cases(config):
            for target in active_cpp_targets_for_case(
                case,
                options,
                config.exclusion_rules,
            ):
                exclusion = build_cachegrind_exclusion_record(
                    dataset=case.dataset,
                    D=case.D,
                    N=case.N,
                    K=case.K,
                    cpp_case=target.cpp_case,
                    stage_key=target.stage_key,
                    params_key=target.params_key,
                    rules=rules,
                )
                if exclusion is not None:
                    exclusions.append(exclusion)
                    continue

                planned_records.append(
                    {
                        "dataset": case.dataset,
                        "D": int(case.D),
                        "N": int(case.N),
                        "K": int(case.K),
                        "config_id": case.case_id,
                        "cpp_case": target.cpp_case,
                        "phase_key": target.phase_key,
                        "stage_key": target.stage_key,
                        "variant_key": target.variant_key,
                        "params_key": target.params_key,
                    }
                )

    return build_cachegrind_manifest(
        enabled=options.run_cachegrind,
        results_dir=options.cachegrind_results_dir,
        cache_model=cachegrind_model_for_options(options),
        planned_records=planned_records,
        exclusions=exclusions,
    )


def write_benchmark_exclusion_manifest(
    config: BenchmarkConfig,
    datasets_dir: Path,
) -> None:
    manifest = build_exclusion_manifest(
        cases=_iter_config_cases(config),
        rules=config.exclusion_rules,
        phase_keys=enabled_phase_keys_for_options(config.pipeline),
        stage_keys_by_phase=enabled_stage_keys_by_phase_for_options(config.pipeline),
    )
    path = datasets_dir / EXCLUSIONS_FILENAME
    write_exclusion_manifest(path, manifest)

    if manifest["exclusions"]:
        print(
            f"Registered {len(manifest['exclusions'])} configured benchmark "
            f"phase/stage exclusions in {path}."
        )


def prepare_cachegrind(config: BenchmarkConfig) -> None:
    if config.pipeline.run_cachegrind:
        require_cachegrind_tools()
        prepare_cachegrind_results_dir(config.pipeline.cachegrind_results_dir)

    manifest = build_cachegrind_manifest_for_config(config)
    manifest_path = cachegrind_manifest_path(config.pipeline.cachegrind_results_dir)
    write_json(manifest_path, manifest)

    if config.pipeline.run_cachegrind:
        print(
            "Registered "
            f"{manifest['planned_record_count']} Cachegrind target(s) and "
            f"{manifest['exclusion_count']} Cachegrind exclusion(s) in {manifest_path}."
        )


def run_benchmark_suite(config: BenchmarkConfig) -> None:
    prepare_cachegrind(config)
    datasets_dir = prepare_datasets_dir(config.datasets_dir)
    write_benchmark_exclusion_manifest(config, datasets_dir)

    cases = list(_iter_config_cases(config))

    for D in sorted({case.D for case in cases}):
        cases_for_D = _cases_for_dimension(cases, D)

        compile_cpp_binaries(
            D,
            cpp_cases_for_dimension(config, D, cases_for_D),
            datasets_dir=datasets_dir,
        )

        if config.pipeline.run_cachegrind:
            compile_callgrind_binaries(
                D,
                cachegrind_cpp_cases_for_dimension(config, D, cases_for_D),
            )

        for case in sorted(cases_for_D):
            execute_pipeline(
                case,
                config.pipeline,
                datasets_dir=datasets_dir,
                keep_inputs=config.keep_inputs,
                exclusion_rules=config.exclusion_rules,
            )

    print("\nAll benchmarking finished successfully!")


if __name__ == "__main__":
    run_benchmark_suite(default_config())
