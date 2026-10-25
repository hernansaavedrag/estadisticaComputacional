
from pathlib import Path
import json
import subprocess
import sys

RAIZ = Path(__file__).resolve().parents[2]
CARPETA = RAIZ / "S3/resultados/rodrigo_bootstrap_lotes"
WORKER = RAIZ / "S3/src/rodrigo_bootstrap_worker.py"

def completadas():
    return sum(
        len(json.loads(p.read_text(encoding="utf-8"))["registros"])
        for p in CARPETA.glob("lote_*.json")
    )

while completadas() < 10000:
    antes = completadas()
    proceso = subprocess.run(
        [sys.executable, str(WORKER)],
        cwd=RAIZ, capture_output=True, text=True,
    )
    despues = completadas()

    if proceso.returncode != 0 or despues <= antes:
        print(proceso.stdout)
        print(proceso.stderr)
        raise SystemExit(
            f"Ejecución detenida. Código: {proceso.returncode}. "
            f"Intentos guardados: {despues}."
        )

    print(f"Guardados: {despues}/10000", flush=True)

print("Los 10.000 intentos están guardados.")
