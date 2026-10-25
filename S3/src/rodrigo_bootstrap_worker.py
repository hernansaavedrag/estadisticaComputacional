
import os
for nombre in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[nombre] = "1"
from IPython.display import display

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from threadpoolctl import threadpool_limits

limite_hilos = threadpool_limits(limits=1)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# Localizar el repositorio.
RAIZ = next(
    (
        carpeta
        for carpeta in [Path.cwd(), *Path.cwd().parents]
        if (
            carpeta / "datos" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
        ).is_file()
    ),
    None,
)

if RAIZ is None:
    raise FileNotFoundError("No se encontró la raíz del repositorio.")

DIRECTORIO_S3 = RAIZ / "S3"
RESULTADOS = DIRECTORIO_S3 / "resultados"
FIGURAS = DIRECTORIO_S3 / "figuras"

RESULTADOS.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

# Reutilizar exactamente los procedimientos de Lenin.
sys.path.insert(0, str(DIRECTORIO_S3 / "src"))

from lenin_modelado import (
    ImputadorRegresion,
    PreparadorDiseno,
    ajustar_logit,
    codificar_objetivo,
    pesos_balanceados,
    reajustar_modelo_final,
    tabla_coeficientes,
)

config_final = json.loads(
    (RESULTADOS / "lenin_configuracion_modelo_final.json")
    .read_text(encoding="utf-8")
)

SEMILLA = config_final["semilla"]
UMBRAL = config_final["umbral"]
ESPECIFICACIONES = config_final["especificaciones"]
MODELO_FINAL = config_final["modelo_seleccionado"]

df_mcar = pd.read_csv(
    RAIZ / config_final["datos_MCAR"]
).set_index("EmployeeNumber")

particion = pd.read_csv(
    RAIZ / config_final["particion"]
).set_index("EmployeeNumber")

train_mcar = df_mcar.loc[
    particion.index[particion["Conjunto"] == "Entrenamiento"]
].copy()

test_mcar = df_mcar.loc[
    particion.index[particion["Conjunto"] == "Prueba"]
].copy()

y_train = codificar_objetivo(train_mcar["Attrition"])
y_test = codificar_objetivo(test_mcar["Attrition"])

assert len(train_mcar) == 1029
assert len(test_mcar) == 441
assert set(train_mcar.index).isdisjoint(test_mcar.index)

print("Modelo final seleccionado por Lenin:", MODELO_FINAL)
print("Entrenamiento:", len(train_mcar))
print("Prueba:", len(test_mcar))

display(pd.DataFrame([
    {
        "Modelo": nombre,
        "Predictores": ", ".join(variables),
    }
    for nombre, variables in ESPECIFICACIONES.items()
]))

resultado_referencia, preparador_referencia, _, _ = (
    reajustar_modelo_final(RAIZ)
)
VARIABLES_BOOT = config_final["variables"]
TERMINOS_BOOT = list(resultado_referencia.params.index)

def ajustar_remuestra(datos):
    y = codificar_objetivo(datos['Attrition'])
    if y.nunique() != 2:
        raise ValueError('La remuestra no contiene ambas clases.')
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        imputador_b = ImputadorRegresion(config_final['imputacion']['variable'], config_final['imputacion']['predictores']).ajustar(datos)
        datos_imp = imputador_b.transformar(datos)
        preparador_b = PreparadorDiseno(VARIABLES_BOOT).ajustar(datos_imp)
        X_b = preparador_b.transformar(datos_imp)
        resultado_b = ajustar_logit(X_b, y, pesos_balanceados(y))
    mensajes = [f'{aviso.category.__name__}: {aviso.message}' for aviso in avisos]
    if not resultado_b.converged:
        raise RuntimeError('El ajuste no convergió.')
    if any(('converg' in mensaje.lower() or 'separation' in mensaje.lower() for mensaje in mensajes)):
        raise RuntimeError(' | '.join(mensajes))
    coeficientes = resultado_b.params.copy()
    intercepto = float(coeficientes['const'])
    for variable in preparador_b.numericas:
        beta = float(resultado_b.params[variable])
        media_b = preparador_b.medias_[variable]
        de_b = preparador_b.desviaciones_[variable]
        media_ref = preparador_referencia.medias_[variable]
        de_ref = preparador_referencia.desviaciones_[variable]
        coeficientes[variable] = beta * de_ref / de_b
        intercepto += beta * (media_ref - media_b) / de_b
    coeficientes['const'] = intercepto
    coeficientes = coeficientes.reindex(TERMINOS_BOOT)
    if not np.isfinite(coeficientes.to_numpy()).all():
        raise ValueError('Coeficientes no finitos.')
    return (coeficientes, mensajes)
import hashlib

B = 10_000
INTENTOS_POR_EJECUCION = 100
TAMANO_LOTE = 100

CARPETA_LOTES = RESULTADOS / "rodrigo_bootstrap_lotes"
CARPETA_LOTES.mkdir(parents=True, exist_ok=True)

# Identificar los datos, la configuración y la escala de referencia.
contenido_control = (
    train_mcar.to_csv()
    + json.dumps(config_final, sort_keys=True)
    + resultado_referencia.params.to_json()
    + preparador_referencia.parametros().to_csv(index=False)
    + "bootstrap_por_replica_v1"
)

huella = hashlib.sha256(
    contenido_control.encode("utf-8")
).hexdigest()

# Recuperar el avance de ejecuciones anteriores.
historial = []

for archivo in sorted(CARPETA_LOTES.glob("lote_*.json")):
    lote_guardado = json.loads(archivo.read_text(encoding="utf-8"))

    if lote_guardado["huella"] != huella:
        raise ValueError(
            "Los lotes guardados pertenecen a otra configuración."
        )

    historial.extend(lote_guardado["registros"])

ids_completados = sorted(r["Replica"] for r in historial)

assert ids_completados == list(
    range(1, len(ids_completados) + 1)
), "Hay réplicas duplicadas o discontinuas en los lotes."

inicio = len(historial) + 1
fin = min(B, inicio + INTENTOS_POR_EJECUCION - 1)

if inicio > B:
    print("Ya se completaron los 10.000 intentos.")

else:
    print(f"Continuando desde la réplica {inicio} hasta la {fin}.")

    for primero in range(inicio, fin + 1, TAMANO_LOTE):
        ultimo = min(primero + TAMANO_LOTE - 1, fin)
        registros_lote = []

        for replica in range(primero, ultimo + 1):
            rng_replica = np.random.default_rng(
                np.random.SeedSequence([SEMILLA, 30, replica])
            )

            indices = rng_replica.integers(
                0, len(train_mcar), size=len(train_mcar)
            )
            muestra = train_mcar.iloc[indices].reset_index(drop=True)

            try:
                coeficientes, avisos = ajustar_remuestra(muestra)

                registro = {
                    "Replica": replica,
                    "Estado": "Valida",
                    "Detalle": " | ".join(avisos),
                    "Coeficientes": coeficientes.to_dict(),
                }

            except Exception as error:
                registro = {
                    "Replica": replica,
                    "Estado": "Fallida",
                    "Detalle": (
                        f"{type(error).__name__}: {error}"
                    )[:2000],
                    "Coeficientes": None,
                }

            registros_lote.append(registro)

        destino = CARPETA_LOTES / (
            f"lote_{primero:05d}_{ultimo:05d}.json"
        )
        temporal = destino.with_suffix(".tmp")

        temporal.write_text(
            json.dumps(
                {
                    "huella": huella,
                    "registros": registros_lote,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        temporal.replace(destino)

        historial.extend(registros_lote)
        validas = sum(r["Estado"] == "Valida" for r in historial)

        print(
            f"Guardadas: {len(historial)}/{B} | "
            f"Válidas: {validas} | "
            f"Fallidas: {len(historial) - validas}"
        )

print("Los lotes completados quedan guardados en disco.")