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
from benchmark_pipeline.cpp_cases import (
    CPP_CASES,
    cpp_compile_command,
    spill_detector_assembly_path,
)
from benchmark_pipeline.paths import REPO_ROOT, repo_path

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

GMM_COVARIANCE_TYPES = ("spherical", "diag", "full")


@dataclass(frozen=True)
class SpillScanResult:
    cpp_case: str
    D: int
    gmm_covariance_type: str | None
    candidate_reload_pairs: int
    assembly_file: str
    rg_output_file: str
    rg_returncode: int


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        print(f"Required tool not found on PATH: {tool}")
        sys.exit(1)


def case_tag(cpp_case: str, D: int, gmm_covariance_type: str | None = None) -> str:
    covariance_suffix = (
        f".{gmm_covariance_type}"
        if cpp_case in {"gmm_static", "gmm_dynamic"} and gmm_covariance_type
        else ""
    )
    return f"{cpp_case}{covariance_suffix}.{D}D"


def gmm_covariance_define(gmm_covariance_type: str | None) -> list[str]:
    if gmm_covariance_type == "spherical":
        return ["-DSPILL_GMM_COVARIANCE_SPHERICAL"]
    if gmm_covariance_type == "diag":
        return ["-DSPILL_GMM_COVARIANCE_DIAG"]
    if gmm_covariance_type == "full":
        return ["-DSPILL_GMM_COVARIANCE_FULL"]
    return []


def godbolt_source_path(
    cpp_case: str,
    D: int,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
) -> Path:
    return out_dir / f"godbolt.{case_tag(cpp_case, D, gmm_covariance_type)}.cpp"


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


def emit_godbolt_source(
    *,
    cpp_case: str,
    D: int,
    out_dir: Path,
    gmm_covariance_type: str | None = None,
) -> Path:
    output_path = godbolt_source_path(cpp_case, D, out_dir, gmm_covariance_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assembly = spill_detector_assembly_path(cpp_case, D, out_dir, gmm_covariance_type)
    cmd = cpp_compile_command(
        D=D,
        cpp_case=cpp_case,
        mode="assembly",
        output=assembly,
        extra_defines=gmm_covariance_define(gmm_covariance_type),
    )

    source = compile_command_source(cmd)
    defines = parse_compile_defines(cmd)
    include_dirs = parse_compile_include_dirs(cmd)

    body = flatten_local_includes(
        source=source,
        include_dirs=include_dirs,
        defines=defines,
    )

    covariance_label = f" {gmm_covariance_type}" if gmm_covariance_type else ""
    header = [
        "// Generated by spill_detector.py --emit-godbolt-source.",
        f"// C++ case: {cpp_case}{covariance_label} {D}D",
        "// Local project includes are expanded; standard/library includes are left intact.",
        "",
        *define_lines(defines),
        "",
    ]

    output_path.write_text("\n".join(header + body) + "\n")
    print(f"Wrote Godbolt source for {cpp_case}{covariance_label} {D}D: {output_path}")
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
        extra_defines=gmm_covariance_define(gmm_covariance_type),
    )

    covariance_label = f" {gmm_covariance_type}" if gmm_covariance_type else ""
    print(f"Compiling assembly for {cpp_case}{covariance_label} {D}D...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)

    if result.returncode != 0:
        print(f"\nAssembly compile failed for {cpp_case}{covariance_label} {D}D")
        print(f"Command: {' '.join(cmd)}")
        if result.stdout:
            print("\nstdout:")
            print(result.stdout)
        if result.stderr:
            print("\nstderr:")
            print(result.stderr)
        sys.exit(result.returncode)

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
        print(f"\nrg failed for {cpp_case}{covariance_label} {D}D")
        print(f"Command: {' '.join(cmd)}")
        print(f"rg output: {output_path}")
        if result.stderr:
            print("\nstderr:")
            print(result.stderr)
        sys.exit(result.returncode)

    return output_path, result.returncode


def write_summary(results: list[SpillScanResult], out_dir: Path) -> tuple[Path, Path]:
    summary_json = out_dir / "summary.json"
    summary_txt = out_dir / "summary.txt"

    total = sum(result.candidate_reload_pairs for result in results)
    payload = {
        "schema_version": 1,
        "description": (
            "candidate_reload_pairs counts non-overlapping matches of the "
            "configured YMM stack-store/reload regex in each generated assembly file."
        ),
        "total_candidate_reload_pairs": total,
        "results": [asdict(result) for result in results],
    }

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
        "candidate_reload_pairs counts non-overlapping matches of the configured",
        "YMM stack-store/reload regex in each generated assembly file.",
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
            "then scan each .s file with the YMM stack-store/reload ripgrep pattern."
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
        choices=GMM_COVARIANCE_TYPES,
        nargs="+",
        default=list(GMM_COVARIANCE_TYPES),
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
        "--emit-godbolt-source",
        action="store_true",
        help=(
            "Also write a local-header-flattened .cpp source for each variant. "
            "Standard and third-party library includes are left intact for Compiler Explorer."
        ),
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
    if cpp_case == "gmm_static":
        return gmm_covariance_types
    if cpp_case == "gmm_dynamic":
        return [cov for cov in gmm_covariance_types if cov in {"spherical", "diag"}]
    return [None]


def main() -> None:
    args = parse_args()

    require_tool(args.rg)

    out_dir = Path(args.out_dir)
    if args.skip_compile:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        clear_output_dir(out_dir)

    results: list[SpillScanResult] = []

    for cpp_case in args.cpp_case:
        for gmm_covariance_type in scan_variants(cpp_case, args.gmm_covariance_types):
            for D in args.D_values:
                if args.skip_compile:
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
                        print(
                            f"Missing assembly file for {cpp_case}{covariance_label} {D}D: {assembly}"
                        )
                        sys.exit(1)
                else:
                    assembly = compile_assembly(
                        cpp_case, D, out_dir, gmm_covariance_type
                    )

                if args.emit_godbolt_source:
                    emit_godbolt_source(
                        cpp_case=cpp_case,
                        D=D,
                        out_dir=out_dir,
                        gmm_covariance_type=gmm_covariance_type,
                    )

                rg_output, rg_returncode = run_rg_scan(
                    rg=args.rg,
                    pattern=args.pattern,
                    cpp_case=cpp_case,
                    D=D,
                    assembly=assembly,
                    out_dir=out_dir,
                    gmm_covariance_type=gmm_covariance_type,
                )

                candidate_reload_pairs = count_candidate_reload_pairs(
                    assembly, args.pattern
                )

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

    summary_json, summary_txt = write_summary(results, out_dir)

    print()
    print(summary_txt.read_text(), end="")
    print()
    print("Done.")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary text: {summary_txt}")


if __name__ == "__main__":
    main()
