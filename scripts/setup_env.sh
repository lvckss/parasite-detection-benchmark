#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

backend="${1:-cpu}"
if [[ "$backend" != "cpu" && "$backend" != "cu128" ]]; then
    echo "Usage: $0 [cpu|cu128]" >&2
    exit 2
fi
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
uv sync --locked --extra "$backend"
.venv/bin/python -m ipykernel install --sys-prefix --name vision-p5 --display-name "VISION P5"
echo "Environment ready ($backend). Run: .venv/bin/jupyter lab"
echo 'For Jupyter outside this environment, register the kernel with:'
echo '.venv/bin/python -m ipykernel install --user --name vision-p5 --display-name "VISION P5"'
