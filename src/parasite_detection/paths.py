"""Repository paths shared by notebooks and scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data/raw/parasites-1s07h"
DATA = ROOT / "data/converted/parasites-1s07h"
DATASET_YAML = DATA / "dataset.yaml"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
