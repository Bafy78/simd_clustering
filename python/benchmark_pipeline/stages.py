"""Stage metadata and input-artifact planning for benchmark phases.

A phase can expose one or more benchmarkable stages. The currently implemented
phases each expose only the ``full`` stage, but the task graph and artifact
identity use this module so future staged phases do not need to overload phase
names.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_metadata import (
    FULL_STAGE_KEY,
    HDBSCAN_DISTANCE_STAGE_KEY,
    HDBSCAN_LINKAGE_STAGE_KEY,
    HDBSCAN_MREACH_STAGE_KEY,
    HDBSCAN_MST_STAGE_KEY,
    HDBSCAN_SELECT_STAGE_KEY,
    PHASE_KEYS,
    phase_stage_keys,
)


DATASET_ARTIFACT = "dataset"
HDBSCAN_DATASET_ARTIFACT = "hdbscan_dataset"
INIT_CENTROIDS_ARTIFACT = "init_centroids"
GMM_WEIGHTS_ARTIFACT = "gmm_weights"
GMM_MEANS_ARTIFACT = "gmm_means"
GMM_PRECISIONS_ARTIFACT = "gmm_precisions"
HDBSCAN_DISTANCE_MATRIX_ARTIFACT = "distance_matrix"
HDBSCAN_MREACH_MATRIX_ARTIFACT = "mreach_matrix"
HDBSCAN_MST_EDGES_ARTIFACT = "mst_edges"
HDBSCAN_LINKAGE_TREE_ARTIFACT = "single_linkage_tree"
HDBSCAN_STAGE_ARTIFACT_KEYS = (
    HDBSCAN_DISTANCE_MATRIX_ARTIFACT,
    HDBSCAN_MREACH_MATRIX_ARTIFACT,
    HDBSCAN_MST_EDGES_ARTIFACT,
    HDBSCAN_LINKAGE_TREE_ARTIFACT,
)


@dataclass(frozen=True)
class StageSpec:
    phase_key: str
    stage_key: str
    input_artifact_keys: tuple[str, ...]
    reference_input_artifact_keys: tuple[str, ...] = ()

    @property
    def is_full(self) -> bool:
        return self.stage_key == FULL_STAGE_KEY

    @property
    def isolates_stage(self) -> bool:
        """True when predecessor artifacts should be precomputed outside timing."""
        return not self.is_full and bool(self.reference_input_artifact_keys)


DEFAULT_STAGE_SPECS: dict[tuple[str, str], StageSpec] = {
    (phase_key, stage_key): StageSpec(
        phase_key=phase_key,
        stage_key=stage_key,
        input_artifact_keys=(DATASET_ARTIFACT,),
    )
    for phase_key in PHASE_KEYS
    for stage_key in phase_stage_keys(phase_key)
}

DEFAULT_STAGE_SPECS.update(
    {
        ("hdbscan", FULL_STAGE_KEY): StageSpec(
            phase_key="hdbscan",
            stage_key=FULL_STAGE_KEY,
            input_artifact_keys=(HDBSCAN_DATASET_ARTIFACT,),
        ),
        ("hdbscan", HDBSCAN_DISTANCE_STAGE_KEY): StageSpec(
            phase_key="hdbscan",
            stage_key=HDBSCAN_DISTANCE_STAGE_KEY,
            input_artifact_keys=(HDBSCAN_DATASET_ARTIFACT,),
        ),
        ("hdbscan", HDBSCAN_MREACH_STAGE_KEY): StageSpec(
            phase_key="hdbscan",
            stage_key=HDBSCAN_MREACH_STAGE_KEY,
            input_artifact_keys=(HDBSCAN_DISTANCE_MATRIX_ARTIFACT,),
            reference_input_artifact_keys=(HDBSCAN_DATASET_ARTIFACT,),
        ),
        ("hdbscan", HDBSCAN_MST_STAGE_KEY): StageSpec(
            phase_key="hdbscan",
            stage_key=HDBSCAN_MST_STAGE_KEY,
            input_artifact_keys=(HDBSCAN_MREACH_MATRIX_ARTIFACT,),
            reference_input_artifact_keys=(HDBSCAN_DATASET_ARTIFACT,),
        ),
        ("hdbscan", HDBSCAN_LINKAGE_STAGE_KEY): StageSpec(
            phase_key="hdbscan",
            stage_key=HDBSCAN_LINKAGE_STAGE_KEY,
            input_artifact_keys=(HDBSCAN_MST_EDGES_ARTIFACT,),
            reference_input_artifact_keys=(HDBSCAN_DATASET_ARTIFACT,),
        ),
        ("hdbscan", HDBSCAN_SELECT_STAGE_KEY): StageSpec(
            phase_key="hdbscan",
            stage_key=HDBSCAN_SELECT_STAGE_KEY,
            input_artifact_keys=(HDBSCAN_LINKAGE_TREE_ARTIFACT,),
            reference_input_artifact_keys=(HDBSCAN_DATASET_ARTIFACT,),
        ),
    }
)


def get_stage_spec(phase_key: str, stage_key: str = FULL_STAGE_KEY) -> StageSpec:
    try:
        return DEFAULT_STAGE_SPECS[(phase_key, stage_key)]
    except KeyError as exc:
        valid = ", ".join(
            f"{phase}:{stage}"
            for phase, stage in sorted(DEFAULT_STAGE_SPECS)
        )
        raise ValueError(
            f"Unknown benchmark stage {phase_key!r}/{stage_key!r}. "
            f"Known phase/stage pairs: {valid}"
        ) from exc


def stage_keys_for_phase(phase_key: str) -> tuple[str, ...]:
    return tuple(
        stage_key
        for stage_key in phase_stage_keys(phase_key)
        if (phase_key, stage_key) in DEFAULT_STAGE_SPECS
    )
