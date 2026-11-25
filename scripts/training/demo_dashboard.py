"""genera un informe de prueba sin cargar datasets ni modelos"""

import argparse
from pathlib import Path

from parasite_detection.dashboard.demo import create_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="carpeta nueva para la simulación")
    parser.add_argument("--delay", type=float, default=0.04, help="pausa entre batches simulados")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    dashboard = create_demo(args.output, delay=args.delay, epochs=args.epochs)
    print(f"Demo guardada en {dashboard.report_dir / 'report.html'}")


if __name__ == "__main__":
    main()
