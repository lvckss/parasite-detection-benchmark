#!/usr/bin/env bash
# detenemos la preparación si falla un comando o falta una variable
set -euo pipefail
# trabajamos desde la raíz aunque el script se lance desde otra carpeta
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# elegimos las dependencias de cpu o de cuda según el argumento recibido
backend="${1:-cpu}"
if [[ "$backend" != "cpu" && "$backend" != "cu128" ]]; then
    echo "Usage: $0 [cpu|cu128]" >&2
    exit 2
fi
# uv instala las versiones fijadas en el archivo de dependencias
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
uv sync --locked --extra "$backend"
# activamos el entorno para registrar su intérprete como kernel
source .venv/bin/activate
# jupyter mostrará este entorno con el nombre vision p5
python -m ipykernel install --sys-prefix --name vision-p5 --display-name "VISION P5"
# mostramos los comandos necesarios para usar el entorno en otra terminal
echo "Environment ready ($backend). Activate it in your terminal:"
echo 'source .venv/bin/activate'
echo 'Then run: jupyter lab'
echo 'For Jupyter outside this environment, register the kernel with:'
# el registro de usuario también permite encontrar el kernel desde otros entornos
echo 'python -m ipykernel install --user --name vision-p5 --display-name "VISION P5"'
