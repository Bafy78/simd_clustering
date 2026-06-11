SUPPORTED_GMM_COVARIANCE_TYPES: tuple[str, ...] = ("full", "diag", "spherical")


def validate_gmm_covariance_type(covariance_type: str) -> None:
    if covariance_type in SUPPORTED_GMM_COVARIANCE_TYPES:
        return

    valid = ", ".join(SUPPORTED_GMM_COVARIANCE_TYPES)
    raise ValueError(
        f"Unsupported GMM covariance type {covariance_type!r}; valid values: {valid}"
    )


def validate_gmm_covariance_types(covariance_types: tuple[str, ...]) -> None:
    for covariance_type in covariance_types:
        validate_gmm_covariance_type(covariance_type)

    if len(set(covariance_types)) != len(covariance_types):
        raise ValueError(f"GMM covariance types must be unique, got {covariance_types!r}")


def spill_detector_define(covariance_type: str | None) -> list[str]:
    if covariance_type is None:
        return []

    validate_gmm_covariance_type(covariance_type)
    return [f"-DSPILL_GMM_COVARIANCE_{covariance_type.upper()}"]
