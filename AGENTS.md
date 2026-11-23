# Parasite detection experiments

Compare Faster R-CNN v2 (ResNet-50 FPN) and YOLOv10-S on `Francesco/parasites-1s07h`.
This is an experimental university assignment. Keep changes simple and useful for the experiments.

- Set up with `bash scripts/setup_env.sh cpu` or `bash scripts/setup_env.sh cu128`.
- Activate the environment with `source .venv/bin/activate` in each new terminal, then use `python` and `jupyter` directly.
- Download with `python scripts/download_dataset.py`; prepare with `python scripts/prepare_dataset.py`.
- Keep original Hugging Face train/validation/test splits. Their pinned source is in `configs/dataset.json`.
- Data stages: `data/raw/` contains original downloads; `data/converted/` contains extracted images and COCO/YOLO annotations with unchanged image content. `data/preprocessed/` holds the team's variants; `scripts/deduplicate_train.py` writes `parasites-1s07h-dedup/` (train without near duplicates of val/test; val/test unchanged).
- COCO/torchvision use class IDs 1–8 (0 is background); YOLO uses 0–7. The generated `class_mapping.json` records both.
- Put notebooks in `notebooks/`, shared Python in `src/parasite_detection/`, and experiment settings in `configs/`.
- Store lightweight metrics and figures in `results/`, and local weights in `models/`. Keep downloads, weights, caches, and credentials out of new commits.
- Record actual experiment settings and results; distinguish runs completed from plans. Respect teammates' changes.
- The shared `semantic-commits-and-push` skill is in `.agents/skills/`; Claude's adapter is in `.claude/skills/`.
