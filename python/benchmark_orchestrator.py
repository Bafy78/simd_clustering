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


def _targets_for_dimension(
    config: BenchmarkConfig,
    D: int,
    target_builder,
) -> set[CppTarget]:
    targets: set[CppTarget] = set()

    for N in config.test_Ns:
        for K in config.test_Ks:
            targets.update(
                target_builder(
                    BenchmarkCase(D, N, K),
                    config.pipeline,
                    config.exclusion_rules,
                )
            )

    return targets


def cpp_cases_for_dimension(config: BenchmarkConfig, D: int) -> set[str]:
    return {
        target.cpp_case
        for target in _targets_for_dimension(config, D, active_cpp_targets_for_case)
    }


def cachegrind_cpp_cases_for_dimension(config: BenchmarkConfig, D: int) -> set[str]:
    return {
        target.cpp_case
        for target in _targets_for_dimension(config, D, active_cachegrind_targets_for_case)
    }


def _iter_config_cases(config: BenchmarkConfig) -> Iterable[BenchmarkCase]:
    for D in config.test_Ds:
        for N in config.test_Ns:
            for K in config.test_Ks:
                yield BenchmarkCase(D, N, K)


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
                    D=case.D,
                    N=case.N,
                    K=case.K,
                    cpp_case=target.cpp_case,
                    params_key=target.params_key,
                    rules=rules,
                )
                if exclusion is not None:
                    exclusions.append(exclusion)
                    continue

                planned_records.append(
                    {
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
        test_Ds=config.test_Ds,
        test_Ns=config.test_Ns,
        test_Ks=config.test_Ks,
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

    for D in config.test_Ds:
        compile_cpp_binaries(
            D,
            cpp_cases_for_dimension(config, D),
            datasets_dir=datasets_dir,
        )

        if config.pipeline.run_cachegrind:
            compile_callgrind_binaries(
                D,
                cachegrind_cpp_cases_for_dimension(config, D),
            )

        for N in config.test_Ns:
            for K in config.test_Ks:
                execute_pipeline(
                    BenchmarkCase(D, N, K),
                    config.pipeline,
                    datasets_dir=datasets_dir,
                    keep_inputs=config.keep_inputs,
                    exclusion_rules=config.exclusion_rules,
                )

    print("\nAll benchmarking finished successfully!")


if __name__ == "__main__":
    run_benchmark_suite(default_config())
