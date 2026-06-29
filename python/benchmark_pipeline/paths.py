from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_DIR.parent
DATASETS_DIR = REPO_ROOT / "datasets"
DOWNLOADS_DIR = REPO_ROOT / "downloads"
BIN_DIR = REPO_ROOT / "bin"


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def repo_relative_path(path: str | Path) -> Path:
    """Resolve relative user-facing paths from the repository root."""
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path.resolve()
