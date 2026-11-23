"""Repository paths shared by notebooks and scripts."""

import os
import sys
from pathlib import Path


def _is_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src/parasite_detection").is_dir()


def _find_project_root() -> Path:
    configured_root = os.environ.get("PARASITE_DETECTION_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if _is_project_root(root):
            return root
        raise RuntimeError(f"PARASITE_DETECTION_ROOT is not a project root: {root}")

    for location in (Path(__file__).resolve(), Path(sys.prefix).resolve(), Path.cwd().resolve()):
        for candidate in (location, *location.parents):
            if _is_project_root(candidate):
                return candidate

    raise RuntimeError(
        "Could not locate the project root. Set PARASITE_DETECTION_ROOT to the repository path."
    )


ROOT = _find_project_root()
RAW_DATA = ROOT / "data/raw/parasites-1s07h"
DATA = ROOT / "data/converted/parasites-1s07h"
DATASET_YAML = DATA / "dataset.yaml"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
