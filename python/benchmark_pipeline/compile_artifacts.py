import json
import re
import subprocess
from pathlib import Path
from typing import Any

from benchmark_pipeline.cpp_cases import get_cpp_case
from benchmark_pipeline.paths import DATASETS_DIR, REPO_ROOT, repo_relative_path

COMPILE_ARTIFACTS_FILENAME = "compile_artifacts.json"

_GCC_TARGET_LINE_RE = re.compile(
    r"^\s*-(?P<option>march|mcpu)=\s+(?P<value>\S+)\s*$"
)
_CLANG_TARGET_CPU_RE = re.compile(r'"-target-cpu"\s+"(?P<value>[^"]+)"')
_ARCHITECTURE_OPTIONS = ("-march", "-mcpu")


def compile_artifacts_path(datasets_dir: str | Path = DATASETS_DIR) -> Path:
    return repo_relative_path(datasets_dir) / COMPILE_ARTIFACTS_FILENAME


def _repo_relative_string(path: str | Path) -> str:
    resolved = Path(path).resolve()

    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _architecture_option(command: list[str]) -> tuple[str, str] | None:
    for index, argument in reversed(list(enumerate(command))):
        for option in _ARCHITECTURE_OPTIONS:
            prefix = f"{option}="
            if argument.startswith(prefix):
                return option, argument[len(prefix) :]
            if argument == option and index + 1 < len(command):
                return option, command[index + 1]

    return None


def _gcc_resolved_native_architecture(compiler: str, option: str) -> str | None:
    result = subprocess.run(
        [compiler, f"{option}=native", "-Q", "--help=target"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    if result.returncode != 0:
        return None

    target_option = option.lstrip("-")
    for line in result.stdout.splitlines():
        match = _GCC_TARGET_LINE_RE.match(line)
        if not match:
            continue
        if match.group("option") != target_option:
            continue

        value = match.group("value")
        if value and value != "native":
            return value

    return None


def _clang_resolved_native_architecture(compiler: str) -> str | None:
    result = subprocess.run(
        [compiler, "-march=native", "-###", "-E", "-x", "c++", "-"],
        input="",
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    # Clang intentionally writes the cc1 command line to stderr with -###.
    output = "\n".join([result.stdout, result.stderr])
    match = _CLANG_TARGET_CPU_RE.search(output)
    if match is None:
        return None

    value = match.group("value")
    if not value or value == "native":
        return None

    return value


def resolved_compiled_architecture(command: list[str]) -> str:
    architecture_option = _architecture_option(command)
    if architecture_option is None:
        raise RuntimeError(
            "The C++ compile command does not specify -march or -mcpu, so the "
            "compiled architecture cannot be reported."
        )

    option, value = architecture_option
    if value != "native":
        return value

    compiler = command[0]
    resolved = _gcc_resolved_native_architecture(compiler, option)
    if resolved is None and option == "-march":
        resolved = _clang_resolved_native_architecture(compiler)

    if resolved is None:
        raise RuntimeError(
            f"Could not resolve {option}=native in the C++ compile command. "
            "Use an explicit architecture flag, or use a compiler whose native "
            "target can be queried."
        )

    return resolved


def compile_artifact_record(
    *,
    D: int,
    cpp_case: str,
    command: list[str],
    binary_path: str | Path,
) -> dict[str, Any]:
    case = get_cpp_case(cpp_case)
    binary = Path(binary_path)

    if not binary.exists():
        raise FileNotFoundError(f"Compiled binary not found: {binary}")

    architecture = resolved_compiled_architecture(command)
    architecture_option = _architecture_option(command)
    if architecture_option is None:
        raise AssertionError(
            "resolved_compiled_architecture returned without an architecture option"
        )
    option, value = architecture_option

    return {
        "D": int(D),
        "cpp_case": case.name,
        "phase_key": case.phase_key,
        "stage_keys": list(case.stage_keys),
        "variant_key": case.variant_key,
        "architecture": architecture,
        "architecture_flag": f"{option}={value}",
        "executable_size_bytes": int(binary.stat().st_size),
        "binary_path": _repo_relative_string(binary),
        "compile_command": command,
    }


def _artifact_key(record: dict[str, Any]) -> tuple[int, str]:
    return int(record["D"]), str(record["cpp_case"])


def _empty_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "description": (
            "C++ nanobench compile artifacts captured immediately after each "
            "dimension-specific compilation."
        ),
        "records": [],
    }


def load_compile_artifacts(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return _empty_payload()

    with path.open("r") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected compile artifact JSON object in {path}")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported compile artifact schema in {path}")
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"Expected compile artifact records list in {path}")

    return payload


def write_compile_artifact_record(
    path: str | Path,
    record: dict[str, Any],
) -> None:
    path = Path(path)
    payload = load_compile_artifacts(path)

    records_by_key = {
        _artifact_key(existing): existing for existing in payload["records"]
    }
    records_by_key[_artifact_key(record)] = record
    payload["records"] = [records_by_key[key] for key in sorted(records_by_key)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
