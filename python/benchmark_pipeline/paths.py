from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_DIR.parent
DATASETS_DIR = REPO_ROOT / "datasets"
BIN_DIR = REPO_ROOT / "bin"


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))
