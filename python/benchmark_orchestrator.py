from benchmark_pipeline.config import BenchmarkConfig, default_config
from benchmark_pipeline.exclusions import (
    EXCLUSIONS_FILENAME,
    build_exclusion_manifest,
    is_phase_excluded,
    write_exclusion_manifest,
)
from benchmark_pipeline.runner import (
    compile_cpp_binaries,
    execute_pipeline,
    prepare_datasets_dir,
)
from benchmark_pipeline.tasks import BenchmarkCase, enabled_phase_keys_for_options


def cpp_cases_for_dimension(config: BenchmarkConfig, D: int) -> set[str]:
    options = config.pipeline
    cpp_cases: set[str] = set()

    for N in config.test_Ns:
        for K in config.test_Ks:
            active_phase_keys = {
                phase_key
                for phase_key in enabled_phase_keys_for_options(options)
                if not is_phase_excluded(
                    D=D,
                    N=N,
                    K=K,
                    phase_key=phase_key,
                    rules=config.exclusion_rules,
                )
            }

            if "soa" in active_phase_keys:
                cpp_cases.update(options.cpp_soa_cases)
            if "pp" in active_phase_keys:
                cpp_cases.update(options.cpp_pp_cases)
            if "lloyd" in active_phase_keys:
                cpp_cases.update(options.cpp_lloyd_cases)
            if "gmm" in active_phase_keys:
                cpp_cases.update(options.cpp_gmm_cases)

    return cpp_cases


def main() -> None:
    config = default_config()
    datasets_dir = prepare_datasets_dir(config.datasets_dir)

    exclusion_manifest = build_exclusion_manifest(
        test_Ds=config.test_Ds,
        test_Ns=config.test_Ns,
        test_Ks=config.test_Ks,
        rules=config.exclusion_rules,
        phase_keys=enabled_phase_keys_for_options(config.pipeline),
    )
    write_exclusion_manifest(datasets_dir / EXCLUSIONS_FILENAME, exclusion_manifest)
    if exclusion_manifest["exclusions"]:
        print(
            f"Registered {len(exclusion_manifest['exclusions'])} configured "
            f"benchmark phase exclusions in {datasets_dir / EXCLUSIONS_FILENAME}."
        )

    for D in config.test_Ds:
        compile_cpp_binaries(D, cpp_cases_for_dimension(config, D))

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
    main()
