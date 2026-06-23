import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark_pipeline.config import default_config
from benchmark_pipeline.gmm_covariance import (
    SUPPORTED_GMM_COVARIANCE_TYPES,
    spill_detector_define,
)
from benchmark_pipeline.cpp_cases import (
    CPP_CASES,
    cpp_compile_command,
    spill_detector_assembly_path,
)
from benchmark_pipeline.paths import REPO_ROOT, repo_path, repo_relative_path

SPILL_DETECTOR_PATTERN = (
    r"(?m)(?="
    r"(?P<pair>"
    r"^\s*v(?:mov[au]ps|movdqa|movdqu)\s+"
    r"(?:YMMWORD PTR\s+)?(?P<slot>[+-]?\d+\[(?:rbp|rsp)\])\s*,\s*ymm[0-9]+\s*$\n"
    r"(?:(?!^\s*\.L[.$A-Za-z0-9_]+:)[^\n]*\n){0,300}?"
    r"^\s*(?:"
    r"v(?:mov[au]ps|movdqa|movdqu)\s+ymm[0-9]+\s*,\s*"
    r"(?:YMMWORD PTR\s+)?(?P=slot)\s*"
    r"|"
    r"v(?:"
    r"addps|subps|mulps|divps|minps|maxps|"
    r"fmadd(?:132|213|231)ps|"
    r"fnmadd(?:132|213|231)ps"
    r")\b\s+.*(?:YMMWORD PTR\s+)?(?P=slot)\s*"
    r")$"
    r")"
    r")"
)


class SpillDetectorError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True, order=True)
class SpillScanTarget:
    cpp_case: str
    D: int
    gmm_covariance_type: str | None = None


@dataclass(frozen=True)
class SpillScanResult:
    cpp_case: str
    D: int
    gmm_covariance_type: str | None
    candidate_reload_pairs: int
    assembly_file: str
    rg_output_file: str
    rg_returncode: int


@dataclass(frozen=True)
class AsmFunction:
    symbol: str
    text: str


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        raise SpillDetectorError(f"Required tool not found on PATH: {tool}")


def case_tag(cpp_case: str, D: int, gmm_covariance_type: str | None = None) -> str:
    covariance_suffix = (
        f".{gmm_covariance_type}"
        if cpp_case in {"gmm_static", "gmm_dynamic"} and gmm_covariance_type
        else ""
    )
    return f"{cpp_case}{covariance_suffix}.{D}D"


def instrumented_source_path(
    cpp_case: str,
    D: int,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
) -> Path:
    return out_dir / f"instrumented.{case_tag(cpp_case, D, gmm_covariance_type)}.cpp"


def run_once_assembly_path(
    cpp_case: str,
    D: int,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
) -> Path:
    return out_dir / f"run_once.{case_tag(cpp_case, D, gmm_covariance_type)}.s"


def parse_compile_defines(cmd: Iterable[str]) -> dict[str, str]:
    defines: dict[str, str] = {}
    iterator = iter(cmd)

    for token in iterator:
        if token == "-D":
            define = next(iterator, "")
        elif token.startswith("-D"):
            define = token[2:]
        else:
            continue

        if not define:
            continue

        if "=" in define:
            name, value = define.split("=", 1)
        else:
            name, value = define, "1"

        if name:
            defines[name] = value

    return defines


def parse_compile_include_dirs(cmd: Iterable[str]) -> list[Path]:
    include_dirs: list[Path] = []
    iterator = iter(cmd)

    for token in iterator:
        if token == "-I":
            raw_path = next(iterator, "")
        elif token.startswith("-I"):
            raw_path = token[2:]
        else:
            continue

        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        include_dirs.append(path.resolve())

    return include_dirs


def compile_command_source(cmd: Iterable[str]) -> Path:
    sources = [
        Path(token) for token in cmd if Path(token).suffix in {".cc", ".cpp", ".cxx"}
    ]
    if len(sources) != 1:
        raise RuntimeError(
            "Expected exactly one C++ source in compile command, got: "
            + ", ".join(str(source) for source in sources)
        )

    source = sources[0]
    if not source.is_absolute():
        source = REPO_ROOT / source
    return source.resolve()


def replace_compile_source(cmd: list[str], replacement_source: Path) -> list[str]:
    replaced: list[str] = []
    replacement_done = False

    for token in cmd:
        if Path(token).suffix in {".cc", ".cpp", ".cxx"}:
            replaced.append(str(replacement_source))
            replacement_done = True
        else:
            replaced.append(token)

    if not replacement_done:
        raise RuntimeError("Could not find a C++ source path in compile command")

    return replaced


def define_lines(defines: dict[str, str]) -> list[str]:
    return [f"#define {name} {value}" for name, value in sorted(defines.items())]


def strip_include_delimiters(include_target: str) -> str | None:
    if (
        len(include_target) >= 2
        and include_target[0] == '"'
        and include_target[-1] == '"'
    ):
        return include_target[1:-1]
    return None


def resolve_macro_include(include_target: str, defines: dict[str, str]) -> str:
    if include_target in defines:
        return defines[include_target]
    return include_target


def is_repo_local_cpp_file(path: Path) -> bool:
    try:
        path.resolve().relative_to((REPO_ROOT / "cpp").resolve())
        return True
    except ValueError:
        return False


def resolve_local_include(
    *,
    include_target: str,
    current_file: Path,
    include_dirs: list[Path],
    defines: dict[str, str],
) -> Path | None:
    include_target = resolve_macro_include(include_target, defines)
    relative_include = strip_include_delimiters(include_target)
    if relative_include is None:
        return None

    search_dirs = [
        current_file.parent,
        REPO_ROOT / "cpp" / "benchmarks",
        REPO_ROOT / "cpp" / "include",
        REPO_ROOT / "cpp",
        *include_dirs,
    ]

    for directory in search_dirs:
        candidate = (directory / relative_include).resolve()
        if (
            candidate.exists()
            and candidate.is_file()
            and is_repo_local_cpp_file(candidate)
        ):
            return candidate

    return None


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


INCLUDE_RE = re.compile(
    r"^(?P<indent>\s*)#\s*include\s+(?P<target><[^>]+>|\"[^\"]+\"|[A-Za-z_]\w*)\s*(?P<comment>//.*)?$"
)


def flatten_local_includes(
    *,
    source: Path,
    include_dirs: list[Path],
    defines: dict[str, str],
    already_included: set[Path] | None = None,
) -> list[str]:
    if already_included is None:
        already_included = set()

    source = source.resolve()
    if source in already_included:
        return [f"// ===== skipped duplicate {repo_relative(source)} ====="]

    already_included.add(source)
    lines: list[str] = []

    for line in source.read_text().splitlines():
        if line.strip() == "#pragma once":
            continue

        match = INCLUDE_RE.match(line)
        if not match:
            lines.append(line)
            continue

        include_target = match.group("target")
        local_include = resolve_local_include(
            include_target=include_target,
            current_file=source,
            include_dirs=include_dirs,
            defines=defines,
        )

        if local_include is None:
            lines.append(line)
            continue

        label = repo_relative(local_include)
        lines.append(f"// ===== begin {label} =====")
        lines.extend(
            flatten_local_includes(
                source=local_include,
                include_dirs=include_dirs,
                defines=defines,
                already_included=already_included,
            )
        )
        lines.append(f"// ===== end {label} =====")

    return lines


RUN_ONCE_DEFINITION_RE = re.compile(
    r"^(?P<indent>\s*)(?P<decl>(?:[A-Za-z_:<>~*&]+\s+)+run_once\s*\([^;]*\).*)$"
)


def add_noinline_to_run_once(lines: list[str]) -> list[str]:
    instrumented: list[str] = []

    for line in lines:
        match = RUN_ONCE_DEFINITION_RE.match(line)
        if match and "__attribute__((noinline))" not in line:
            line = (
                f"{match.group('indent')}__attribute__((noinline)) "
                f"{match.group('decl')}"
            )

        instrumented.append(line)

    return instrumented


def flattened_source_lines_for_compile(
    *,
    cpp_case: str,
    D: int,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
) -> list[str]:
    assembly = spill_detector_assembly_path(cpp_case, D, out_dir, gmm_covariance_type)
    cmd = cpp_compile_command(
        D=D,
        cpp_case=cpp_case,
        mode="assembly",
        output=assembly,
        extra_defines=spill_detector_define(gmm_covariance_type),
    )

    source = compile_command_source(cmd)
    defines = parse_compile_defines(cmd)
    include_dirs = parse_compile_include_dirs(cmd)

    body = add_noinline_to_run_once(
        flatten_local_includes(
            source=source,
            include_dirs=include_dirs,
            defines=defines,
        )
    )

    covariance_label = f" {gmm_covariance_type}" if gmm_covariance_type else ""
    header = [
        "// Generated by spill_detector.py.",
        f"// C++ case: {cpp_case}{covariance_label} {D}D",
        "// Local project includes are expanded; standard/library includes are left intact.",
        "// The function named run_once() is marked __attribute__((noinline)) in this generated copy only.",
        "",
        *define_lines(defines),
        "",
    ]

    return header + body


def write_instrumented_source(
    *,
    cpp_case: str,
    D: int,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
    output_path: Path | None = None,
) -> Path:
    if output_path is None:
        output_path = instrumented_source_path(
            cpp_case,
            D,
            out_dir,
            gmm_covariance_type,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            flattened_source_lines_for_compile(
                cpp_case=cpp_case,
                D=D,
                out_dir=out_dir,
                gmm_covariance_type=gmm_covariance_type,
            )
        )
        + "\n"
    )
    return output_path


def compile_assembly(
    cpp_case: str,
    D: int,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
) -> Path:
    assembly = spill_detector_assembly_path(cpp_case, D, out_dir, gmm_covariance_type)
    assembly.parent.mkdir(parents=True, exist_ok=True)

    cmd = cpp_compile_command(
        D=D,
        cpp_case=cpp_case,
        mode="assembly",
        output=assembly,
        extra_defines=spill_detector_define(gmm_covariance_type),
    )
    source = write_instrumented_source(
        cpp_case=cpp_case,
        D=D,
        out_dir=out_dir,
        gmm_covariance_type=gmm_covariance_type,
    )
    cmd = replace_compile_source(cmd, source)

    covariance_label = f" {gmm_covariance_type}" if gmm_covariance_type else ""
    print(f"Compiling assembly for {cpp_case}{covariance_label} {D}D...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)

    if result.returncode != 0:
        lines = [
            f"\nAssembly compile failed for {cpp_case}{covariance_label} {D}D",
            f"Command: {' '.join(cmd)}",
        ]
        if result.stdout:
            lines.extend(["", "stdout:", result.stdout])
        if result.stderr:
            lines.extend(["", "stderr:", result.stderr])
        raise SpillDetectorError("\n".join(lines), exit_code=result.returncode)

    return assembly


def clear_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)


ASM_TYPE_RE = re.compile(r"^\s*\.type\s+([^,\s]+),\s*@function\s*$")
ASM_SIZE_RE = re.compile(r"^\s*\.size\s+([^,\s]+),")
ASM_DIRECT_BRANCH_RE = re.compile(r"^\s*(?:call|jmp)\s+([^\s#]+)")


def normalize_asm_branch_target(raw_target: str) -> str | None:
    target = raw_target.removesuffix("@PLT")

    # Indirect calls/jumps and local labels are not function-symbol callees.
    if target.startswith("*") or target.startswith("."):
        return None

    return target


RUN_ONCE_SYMBOL_RE = re.compile(r"(?<![0-9])\d+run_once[A-Za-z0-9_]*")


def parse_asm_functions(assembly: Path) -> list[AsmFunction]:
    lines = assembly.read_text(errors="replace").splitlines(keepends=True)
    raw_functions: list[tuple[str, str]] = []
    index = 0

    while index < len(lines):
        type_match = ASM_TYPE_RE.match(lines[index])
        if not type_match:
            index += 1
            continue

        symbol = type_match.group(1)
        label_index = index + 1

        while label_index < len(lines):
            if lines[label_index].startswith(symbol + ":"):
                break
            if ASM_TYPE_RE.match(lines[label_index]):
                label_index = -1
                break
            label_index += 1

        if label_index < 0 or label_index >= len(lines):
            index += 1
            continue

        end_index = label_index + 1
        while end_index < len(lines):
            size_match = ASM_SIZE_RE.match(lines[end_index])
            if size_match and size_match.group(1) == symbol:
                break
            end_index += 1

        if end_index >= len(lines):
            index = label_index + 1
            continue

        raw_functions.append((symbol, "".join(lines[label_index : end_index + 1])))
        index = end_index + 1

    return [
        AsmFunction(
            symbol=symbol,
            text=text,
        )
        for symbol, text in raw_functions
    ]


def is_run_once_function(function: AsmFunction) -> bool:
    return RUN_ONCE_SYMBOL_RE.search(function.symbol) is not None


def local_call_targets(function: AsmFunction, known_symbols: set[str]) -> set[str]:
    targets: set[str] = set()

    for line in function.text.splitlines():
        match = ASM_DIRECT_BRANCH_RE.match(line)
        if not match:
            continue

        target = normalize_asm_branch_target(match.group(1))
        if target is not None and target in known_symbols:
            targets.add(target)

    return targets


def select_run_once_reachable_functions(
    functions: list[AsmFunction],
) -> list[AsmFunction]:
    by_symbol = {function.symbol: function for function in functions}
    known_symbols = set(by_symbol)

    roots = [
        function.symbol for function in functions if is_run_once_function(function)
    ]

    if not roots:
        raise RuntimeError(
            "No emitted assembly function matching mangled component '8run_once' was found. "
            "Check that the generated source successfully marked run_once noinline."
        )

    selected: set[str] = set()
    worklist = list(roots)

    while worklist:
        symbol = worklist.pop()

        if symbol in selected or symbol not in by_symbol:
            continue

        selected.add(symbol)

        for target in local_call_targets(by_symbol[symbol], known_symbols):
            if target not in selected:
                worklist.append(target)

    return [function for function in functions if function.symbol in selected]


def run_once_assembly_text(assembly: Path) -> str:
    functions = parse_asm_functions(assembly)
    selected = select_run_once_reachable_functions(functions)

    return "\n".join(f"# {function.symbol}\n{function.text}" for function in selected)


def write_run_once_assembly(
    *,
    cpp_case: str,
    D: int,
    assembly: Path,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
) -> Path:
    output_path = run_once_assembly_path(cpp_case, D, out_dir, gmm_covariance_type)
    output_path.write_text(run_once_assembly_text(assembly))
    return output_path


def count_candidate_reload_pairs(assembly: Path, pattern: str) -> int:
    text = assembly.read_text(errors="replace")
    regex = re.compile(pattern, re.MULTILINE)
    return sum(1 for _ in regex.finditer(text))


def run_rg_scan(
    *,
    rg: str,
    pattern: str,
    cpp_case: str,
    D: int,
    assembly: Path,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
) -> tuple[Path, int]:
    tag = case_tag(cpp_case, D, gmm_covariance_type)
    output_path = out_dir / f"rg.{tag}.txt"
    cmd = [rg, "-nUP", pattern, str(assembly)]

    covariance_label = f" {gmm_covariance_type}" if gmm_covariance_type else ""
    print(f"Scanning assembly for {cpp_case}{covariance_label} {D}D...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)

    output_path.write_text(result.stdout)

    # rg returns 1 when the pattern has no matches. That is a normal scan outcome.
    if result.returncode not in (0, 1):
        lines = [
            f"\nrg failed for {cpp_case}{covariance_label} {D}D",
            f"Command: {' '.join(cmd)}",
            f"rg output: {output_path}",
        ]
        if result.stderr:
            lines.extend(["", "stderr:", result.stderr])
        raise SpillDetectorError("\n".join(lines), exit_code=result.returncode)

    return output_path, result.returncode


def summary_payload(results: list[SpillScanResult]) -> dict[str, object]:
    total = sum(result.candidate_reload_pairs for result in results)
    return {
        "schema_version": 1,
        "description": (
            "candidate_reload_pairs counts matches of the "
            "configured YMM stack-store/reload regex in run_once-rooted reachable assembly."
        ),
        "total_candidate_reload_pairs": total,
        "results": [asdict(result) for result in results],
    }


def spill_detection_status(payload: dict[str, object]) -> str:
    total = int(payload.get("total_candidate_reload_pairs") or 0)
    return "PASS" if total == 0 else "FAIL"


def write_summary(results: list[SpillScanResult], out_dir: Path) -> tuple[Path, Path]:
    summary_json = out_dir / "summary.json"
    summary_txt = out_dir / "summary.txt"

    payload = summary_payload(results)
    total = int(payload["total_candidate_reload_pairs"])

    summary_json.write_text(json.dumps(payload, indent=2) + "\n")

    if results:
        cpp_case_width = max(
            len("cpp_case"), *(len(result.cpp_case) for result in results)
        )
        cov_width = max(
            len("gmm_covariance_type"),
            *(len(result.gmm_covariance_type or "-") for result in results),
        )
        dim_width = max(len("D"), *(len(str(result.D)) for result in results))
        count_width = max(
            len("candidate_reload_pairs"),
            *(len(str(result.candidate_reload_pairs)) for result in results),
        )
    else:
        cpp_case_width = len("cpp_case")
        cov_width = len("gmm_covariance_type")
        dim_width = len("D")
        count_width = len("candidate_reload_pairs")

    lines = [
        "candidate_reload_pairs counts matches of the configured",
        "YMM stack-store/reload regex in run_once-rooted reachable assembly.",
        "",
        f"{'cpp_case':<{cpp_case_width}}  {'gmm_covariance_type':<{cov_width}}  {'D':>{dim_width}}  "
        f"{'candidate_reload_pairs':>{count_width}}  rg_output",
        f"{'-' * cpp_case_width}  {'-' * cov_width}  {'-' * dim_width}  {'-' * count_width}  ---------",
    ]

    for result in sorted(
        results,
        key=lambda item: (item.cpp_case, item.gmm_covariance_type or "", item.D),
    ):
        lines.append(
            f"{result.cpp_case:<{cpp_case_width}}  "
            f"{(result.gmm_covariance_type or '-'):<{cov_width}}  "
            f"{result.D:>{dim_width}}  "
            f"{result.candidate_reload_pairs:>{count_width}}  "
            f"{result.rg_output_file}"
        )

    lines.extend(
        [
            "",
            f"total_candidate_reload_pairs: {total}",
        ]
    )

    summary_txt.write_text("\n".join(lines) + "\n")
    return summary_json, summary_txt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile C++ benchmark cases to Intel-syntax assembly across D values, "
            "then scan run_once-rooted reachable assembly with the YMM stack-store/reload ripgrep pattern."
        )
    )
    parser.add_argument(
        "--cpp-case",
        choices=sorted(CPP_CASES),
        nargs="+",
        default=sorted(CPP_CASES),
        help="C++ case(s) to scan. Defaults to all registered C++ cases.",
    )
    parser.add_argument(
        "--D",
        dest="D_values",
        type=int,
        nargs="+",
        default=default_config().test_Ds,
        help="D values to compile and scan. Defaults to the benchmark config D values.",
    )
    parser.add_argument(
        "--gmm-covariance-types",
        choices=SUPPORTED_GMM_COVARIANCE_TYPES,
        nargs="+",
        default=list(SUPPORTED_GMM_COVARIANCE_TYPES),
        help=(
            "Covariance type(s) to compile separately for GMM cases. Defaults to all supported"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=repo_path("spill_detector_results"),
        help="Directory for assembly, rg outputs, and summaries.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Reuse existing assembly files in --out-dir instead of compiling them.",
    )
    parser.add_argument(
        "--rg",
        default="rg",
        help="ripgrep executable to use. Defaults to rg.",
    )
    parser.add_argument(
        "--pattern",
        default=SPILL_DETECTOR_PATTERN,
        help="Override the default YMM stack-store/reload PCRE pattern.",
    )
    return parser.parse_args()


def scan_variants(cpp_case: str, gmm_covariance_types: list[str]) -> list[str | None]:
    case = CPP_CASES[cpp_case]

    if not case.needs_gmm_init:
        return [None]

    supported = set(case.supported_gmm_covariance_types)
    return [cov for cov in gmm_covariance_types if cov in supported]


def scan_targets(
    targets: Iterable[SpillScanTarget],
    *,
    out_dir: Path,
    skip_compile: bool = False,
    rg: str = "rg",
    pattern: str = SPILL_DETECTOR_PATTERN,
    clear_outputs: bool = True,
) -> list[SpillScanResult]:
    require_tool(rg)

    out_dir = repo_relative_path(out_dir)
    if skip_compile:
        out_dir.mkdir(parents=True, exist_ok=True)
    elif clear_outputs:
        clear_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: list[SpillScanResult] = []

    for target in sorted(set(targets)):
        cpp_case = target.cpp_case
        D = target.D
        gmm_covariance_type = target.gmm_covariance_type

        if skip_compile:
            assembly = spill_detector_assembly_path(
                cpp_case,
                D,
                out_dir,
                gmm_covariance_type,
            )
            if not assembly.exists():
                covariance_label = (
                    f" {gmm_covariance_type}" if gmm_covariance_type else ""
                )
                raise SpillDetectorError(
                    f"Missing assembly file for {cpp_case}{covariance_label} "
                    f"{D}D: {assembly}"
                )
        else:
            assembly = compile_assembly(cpp_case, D, out_dir, gmm_covariance_type)

        scan_assembly = write_run_once_assembly(
            cpp_case=cpp_case,
            D=D,
            assembly=assembly,
            out_dir=out_dir,
            gmm_covariance_type=gmm_covariance_type,
        )

        rg_output, rg_returncode = run_rg_scan(
            rg=rg,
            pattern=pattern,
            cpp_case=cpp_case,
            D=D,
            assembly=scan_assembly,
            out_dir=out_dir,
            gmm_covariance_type=gmm_covariance_type,
        )

        candidate_reload_pairs = count_candidate_reload_pairs(scan_assembly, pattern)

        results.append(
            SpillScanResult(
                cpp_case=cpp_case,
                D=D,
                gmm_covariance_type=gmm_covariance_type,
                candidate_reload_pairs=candidate_reload_pairs,
                assembly_file=str(assembly),
                rg_output_file=str(rg_output),
                rg_returncode=rg_returncode,
            )
        )

    return results


def targets_from_cli_args(
    *,
    cpp_cases: Iterable[str],
    D_values: Iterable[int],
    gmm_covariance_types: list[str],
) -> list[SpillScanTarget]:
    targets: list[SpillScanTarget] = []

    for cpp_case in cpp_cases:
        for gmm_covariance_type in scan_variants(cpp_case, gmm_covariance_types):
            for D in D_values:
                targets.append(
                    SpillScanTarget(
                        cpp_case=cpp_case,
                        D=int(D),
                        gmm_covariance_type=gmm_covariance_type,
                    )
                )

    return targets


def benchmark_record_scan_targets(
    records: Iterable[dict[str, object]],
) -> list[SpillScanTarget]:
    cpp_case_by_phase_stage_variant = {
        (case.phase_key, stage_key, case.variant_key): cpp_case
        for cpp_case, case in CPP_CASES.items()
        for stage_key in case.stage_keys
    }

    targets: set[SpillScanTarget] = set()

    for record in records:
        if record.get("language_key") != "cpp":
            continue

        phase_key = str(record.get("phase_key"))
        stage_key = str(record["stage_key"])
        variant_key = str(record.get("variant_key"))
        cpp_case = cpp_case_by_phase_stage_variant.get(
            (phase_key, stage_key, variant_key)
        )
        if cpp_case is None:
            continue

        case = CPP_CASES[cpp_case]
        params_key = str(record.get("params_key", "default"))
        gmm_covariance_type = params_key if case.needs_gmm_init else None

        if (
            gmm_covariance_type is not None
            and gmm_covariance_type not in case.supported_gmm_covariance_types
        ):
            continue

        targets.add(
            SpillScanTarget(
                cpp_case=cpp_case,
                D=int(record["dimensions"]),
                gmm_covariance_type=gmm_covariance_type,
            )
        )

    return sorted(targets)


def main() -> None:
    args = parse_args()
    out_dir = repo_relative_path(args.out_dir)
    targets = targets_from_cli_args(
        cpp_cases=args.cpp_case,
        D_values=args.D_values,
        gmm_covariance_types=args.gmm_covariance_types,
    )

    try:
        results = scan_targets(
            targets,
            out_dir=out_dir,
            skip_compile=args.skip_compile,
            rg=args.rg,
            pattern=args.pattern,
        )
    except SpillDetectorError as exc:
        print(exc)
        sys.exit(exc.exit_code)

    summary_json, summary_txt = write_summary(results, out_dir)

    print()
    print(summary_txt.read_text(), end="")
    print()
    print("Done.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary text: {summary_txt}")


if __name__ == "__main__":
    main()
