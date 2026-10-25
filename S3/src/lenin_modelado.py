"""Funciones de modelamiento logístico para la Sumativa 3 (IBM HR Attrition).

Responsable: Lenin Morales.

El módulo reúne todo lo necesario para volver a ajustar el modelo final:

1. ``ImputadorRegresion``: imputa MonthlyIncome con una regresión lineal
   ajustada solo con los ingresos observados del conjunto de ajuste.
2. ``PreparadorDiseno``: codifica OverTime y Department, aplica las
   transformaciones declaradas en el nombre (``log_X``, ``log1p_X``,
   ``ind_X_k``) y estandariza con media y desviación del conjunto de ajuste.
3. ``pesos_balanceados``: pesos n / (2 * n_clase) calculados con el
   conjunto de ajuste.
4. ``ajustar_logit``: regresión logística sin ponderar (máxima verosimilitud)
   o ponderada (GLM binomial con pesos y covarianza sándwich HC0).
5. ``stepwise``, ``mejor_subconjunto``: selección por AIC/BIC.
6. ``preparar_pliegue`` y ``metricas_clasificacion``: validación cruzada sin
   fuga de información.
7. ``reajustar_modelo_final``: reconstruye el modelo final desde la
   configuración guardada.

Ningún paso utiliza Attrition para imputar, ni datos de validación o prueba
para aprender parámetros.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CODIFICACION_OBJETIVO = {"No": 0, "Yes": 1}
CODIFICACION_OVERTIME = {"No": 0, "Yes": 1}
DEPARTAMENTO_REFERENCIA = "Research & Development"

# Cada variable candidata es un "bloque": se agrega o elimina completa.
# Department aporta dos columnas indicadoras (referencia: R&D).
COLUMNAS_DEPARTAMENTO = ["Dept_Human Resources", "Dept_Sales"]


def codificar_objetivo(serie: pd.Series) -> pd.Series:
    """Convierte Attrition (No/Yes) en 0/1 y verifica la codificación."""
    y = serie.map(CODIFICACION_OBJETIVO)
    if y.isna().any():
        raise ValueError("Attrition contiene valores fuera de No/Yes.")
    return y.astype(int)


# ---------------------------------------------------------------------------
# 1. Imputación por regresión
# ---------------------------------------------------------------------------
class ImputadorRegresion:
    """Imputa una variable con una regresión lineal múltiple (OLS).

    ``ajustar`` utiliza solo filas con la variable observada. ``transformar``
    reemplaza únicamente los faltantes y conserva los valores observados.
    """

    def __init__(self, variable: str, predictores: list[str]):
        self.variable = variable
        self.predictores = list(predictores)
        self.coeficientes_: pd.Series | None = None

    def ajustar(self, datos: pd.DataFrame) -> "ImputadorRegresion":
        observados = datos.dropna(subset=[self.variable])
        X = sm.add_constant(
            observados[self.predictores].astype(float), has_constant="add"
        )
        modelo = sm.OLS(observados[self.variable].astype(float), X).fit()
        self.coeficientes_ = modelo.params
        self.n_ajuste_ = int(modelo.nobs)
        return self

    def transformar(self, datos: pd.DataFrame) -> pd.DataFrame:
        if self.coeficientes_ is None:
            raise RuntimeError("Primero se debe ajustar el imputador.")
        resultado = datos.copy()
        faltantes = resultado[self.variable].isna()
        if faltantes.any():
            X = sm.add_constant(
                resultado.loc[faltantes, self.predictores].astype(float),
                has_constant="add",
            )
            predicciones = X.to_numpy() @ self.coeficientes_.to_numpy()
            if not np.isfinite(predicciones).all() or (predicciones <= 0).any():
                raise ValueError("Imputación no finita o no positiva.")
            resultado.loc[faltantes, self.variable] = predicciones
        return resultado

    def parametros(self) -> dict:
        return {
            "variable": self.variable,
            "predictores": self.predictores,
            "coeficientes": {k: float(v) for k, v in self.coeficientes_.items()},
            "registros_ajuste": self.n_ajuste_,
        }


# ---------------------------------------------------------------------------
# 2. Matriz de diseño: codificación y estandarización
# ---------------------------------------------------------------------------
def valores_base(datos: pd.DataFrame, variable: str) -> pd.Series:
    """Valores de una variable antes de estandarizar.

    Admite transformaciones declaradas en el nombre:
    - ``log_X``: logaritmo natural de X (X > 0).
    - ``log1p_X``: ln(1 + X), para variables con ceros.
    - ``ind_X_k``: indicadora 1 si X == k, 0 en otro caso.
    """
    if variable.startswith("log1p_"):
        return np.log1p(datos[variable.removeprefix("log1p_")].astype(float))
    if variable.startswith("log_"):
        base = datos[variable.removeprefix("log_")].astype(float)
        if (base <= 0).any():
            raise ValueError(f"{variable}: logaritmo de valores no positivos.")
        return np.log(base)
    if variable.startswith("ind_"):
        base, nivel = variable.removeprefix("ind_").rsplit("_", 1)
        return (datos[base] == int(nivel)).astype(float)
    return datos[variable].astype(float)


class PreparadorDiseno:
    """Construye la matriz de diseño del modelo logístico.

    - OverTime: 0 = No, 1 = Sí (sin estandarizar).
    - Department: indicadoras de Human Resources y Sales (referencia R&D).
    - Indicadoras ``ind_X_k``: 0/1, sin estandarizar.
    - Resto de variables (numéricas, ordinales o transformadas con log):
      puntaje z con la media y la desviación estándar muestral (ddof = 1)
      del conjunto de ajuste.
    """

    def __init__(self, variables: list[str]):
        self.variables = list(variables)
        self.medias_: dict[str, float] = {}
        self.desviaciones_: dict[str, float] = {}

    @property
    def numericas(self) -> list[str]:
        return [
            v for v in self.variables
            if v not in ("OverTime", "Department") and not v.startswith("ind_")
        ]

    def ajustar(self, datos: pd.DataFrame) -> "PreparadorDiseno":
        for variable in self.numericas:
            columna = valores_base(datos, variable)
            self.medias_[variable] = float(columna.mean())
            self.desviaciones_[variable] = float(columna.std(ddof=1))
            if self.desviaciones_[variable] == 0:
                raise ValueError(f"{variable} es constante en el ajuste.")
        return self

    def transformar(self, datos: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=datos.index)
        for variable in self.variables:
            if variable == "OverTime":
                X["OverTime"] = datos["OverTime"].map(CODIFICACION_OVERTIME)
            elif variable == "Department":
                for columna in COLUMNAS_DEPARTAMENTO:
                    nivel = columna.removeprefix("Dept_")
                    X[columna] = (datos["Department"] == nivel).astype(float)
            elif variable.startswith("ind_"):
                X[variable] = valores_base(datos, variable)
            else:
                X[variable] = (
                    valores_base(datos, variable) - self.medias_[variable]
                ) / self.desviaciones_[variable]
        if X.isna().any().any():
            raise ValueError("La matriz de diseño contiene faltantes.")
        return sm.add_constant(X.astype(float), has_constant="add")

    def parametros(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Variable": self.numericas,
            "Media ajuste": [self.medias_[v] for v in self.numericas],
            "Desviacion ajuste": [self.desviaciones_[v] for v in self.numericas],
        })


def columnas_de(variables: list[str]) -> list[str]:
    """Columnas de diseño (sin intercepto) asociadas a una lista de variables."""
    columnas = []
    for variable in variables:
        columnas += COLUMNAS_DEPARTAMENTO if variable == "Department" else [variable]
    return columnas


# ---------------------------------------------------------------------------
# 3. Pesos y ajuste
# ---------------------------------------------------------------------------
def pesos_balanceados(y: pd.Series) -> dict[int, float]:
    """Pesos n / (2 n_clase): cada clase aporta la mitad del peso total."""
    n = len(y)
    conteos = y.value_counts()
    return {int(c): float(n / (2 * conteos[c])) for c in (0, 1)}


def ajustar_logit(X: pd.DataFrame, y: pd.Series, pesos: dict | None = None):
    """Ajusta la regresión logística.

    Sin pesos: ``sm.Logit`` por máxima verosimilitud (inferencia clásica).
    Con pesos: GLM binomial con ``var_weights`` y covarianza sándwich HC0.
    Los pesos por clase no son frecuencias reales; por eso no se usa la
    covarianza clásica para la versión ponderada.
    """
    if pesos is None:
        resultado = sm.Logit(y, X).fit(disp=False, maxiter=200)
        if not resultado.mle_retvals["converged"]:
            raise RuntimeError("El modelo logístico no convergió.")
        return resultado
    w = y.map(pesos).to_numpy(dtype=float)
    modelo = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=w)
    return modelo.fit(cov_type="HC0", maxiter=200)


def criterio_informacion(resultado, criterio: str) -> float:
    return float(resultado.aic if criterio == "aic" else resultado.bic)


# ---------------------------------------------------------------------------
# 4. Selección de variables (siempre sin ponderar)
# ---------------------------------------------------------------------------
def _ajuste_criterio(X_completo, y, variables, criterio):
    X = X_completo[["const", *columnas_de(variables)]]
    return criterio_informacion(ajustar_logit(X, y), criterio)


def stepwise(
    X_completo: pd.DataFrame,
    y: pd.Series,
    candidatas: list[str],
    criterio: str = "aic",
    inicio: list[str] | None = None,
):
    """Selección paso a paso bidireccional por AIC o BIC.

    En cada paso evalúa agregar cualquier variable fuera del modelo o quitar
    cualquiera dentro; aplica el movimiento que más reduce el criterio y se
    detiene cuando ninguno lo reduce. Department entra o sale como bloque.
    Devuelve las variables seleccionadas (en orden de las candidatas) y la
    traza de pasos.
    """
    actuales = list(inicio or [])
    valor_actual = _ajuste_criterio(X_completo, y, actuales, criterio)
    traza = [{"Paso": 0, "Movimiento": "inicio", "Variable": "-",
              "Criterio": valor_actual, "Variables": ", ".join(actuales) or "(solo intercepto)"}]

    for paso in range(1, 2 * len(candidatas) + 1):
        opciones = []
        for variable in candidatas:
            if variable in actuales:
                propuesta = [v for v in actuales if v != variable]
                movimiento = "quitar"
            else:
                propuesta = [*actuales, variable]
                movimiento = "agregar"
            opciones.append((
                _ajuste_criterio(X_completo, y, propuesta, criterio),
                movimiento, variable, propuesta,
            ))
        mejor_valor, movimiento, variable, propuesta = min(opciones, key=lambda o: o[0])
        if mejor_valor >= valor_actual - 1e-9:
            break
        actuales, valor_actual = propuesta, mejor_valor
        traza.append({"Paso": paso, "Movimiento": movimiento, "Variable": variable,
                      "Criterio": valor_actual, "Variables": ", ".join(actuales)})

    seleccion = [v for v in candidatas if v in actuales]
    return seleccion, pd.DataFrame(traza)


def mejor_subconjunto(
    X_completo: pd.DataFrame,
    y: pd.Series,
    candidatas: list[str],
    criterio: str = "bic",
):
    """Evalúa todos los subconjuntos de candidatas (2^k modelos).

    Devuelve la mejor selección y la tabla completa ordenada por el criterio.
    """
    filas = []
    for k in range(len(candidatas) + 1):
        for subconjunto in combinations(candidatas, k):
            resultado = ajustar_logit(
                X_completo[["const", *columnas_de(list(subconjunto))]], y
            )
            filas.append({
                "Variables": ", ".join(subconjunto) or "(solo intercepto)",
                "Numero variables": k,
                "Parametros": int(resultado.df_model + 1),
                "AIC": float(resultado.aic),
                "BIC": float(resultado.bic),
            })
    tabla = pd.DataFrame(filas).sort_values(criterio.upper()).reset_index(drop=True)
    mejor = tabla.loc[0, "Variables"]
    seleccion = [] if mejor == "(solo intercepto)" else mejor.split(", ")
    return [v for v in candidatas if v in seleccion], tabla


# ---------------------------------------------------------------------------
# 5. Validación cruzada sin fuga
# ---------------------------------------------------------------------------
def preparar_pliegue(
    ajuste_mcar: pd.DataFrame,
    validacion_mcar: pd.DataFrame,
    variables: list[str],
    variable_imputada: str,
    predictores_imputacion: list[str],
):
    """Imputa y estandariza un pliegue aprendiendo solo del ajuste.

    Recibe los datos MCAR originales (con faltantes). Devuelve X_ajuste,
    X_validacion, el imputador y el preparador ajustados.
    """
    imputador = ImputadorRegresion(variable_imputada, predictores_imputacion)
    imputador.ajustar(ajuste_mcar)
    ajuste = imputador.transformar(ajuste_mcar)
    validacion = imputador.transformar(validacion_mcar)

    preparador = PreparadorDiseno(variables).ajustar(ajuste)
    return (
        preparador.transformar(ajuste),
        preparador.transformar(validacion),
        imputador,
        preparador,
    )


def metricas_clasificacion(y_real, probabilidades, umbral: float = 0.5) -> dict:
    """AUC (no depende del umbral) y métricas de la clase salida = 1."""
    prediccion = (np.asarray(probabilidades) >= umbral).astype(int)
    return {
        "AUC": roc_auc_score(y_real, probabilidades),
        "Recall": recall_score(y_real, prediccion, zero_division=0),
        "Precision": precision_score(y_real, prediccion, zero_division=0),
        "F1": f1_score(y_real, prediccion, zero_division=0),
        "Exactitud": float(np.mean(prediccion == np.asarray(y_real))),
        "Predichos salida": int(prediccion.sum()),
    }


# ---------------------------------------------------------------------------
# 6. Reajuste del modelo final desde la configuración guardada
# ---------------------------------------------------------------------------
def tabla_coeficientes(resultado) -> pd.DataFrame:
    """Coeficientes, errores estándar, OR e intervalos del 95 %."""
    intervalos = np.asarray(resultado.conf_int(alpha=0.05))
    params = np.asarray(resultado.params)
    tabla = pd.DataFrame({
        "Coeficiente": params,
        "Error estandar": np.asarray(resultado.bse),
        "z": np.asarray(resultado.tvalues),
        "Valor p": np.asarray(resultado.pvalues),
        "IC95 coef inferior": intervalos[:, 0],
        "IC95 coef superior": intervalos[:, 1],
        "OR": np.exp(params),
        "IC95 OR inferior": np.exp(intervalos[:, 0]),
        "IC95 OR superior": np.exp(intervalos[:, 1]),
    }, index=pd.Index(resultado.model.exog_names, name="Termino"))
    return tabla


def reajustar_modelo_final(raiz: Path | str, ruta_configuracion: Path | str | None = None):
    """Vuelve a ajustar el modelo final con el entrenamiento completo.

    Lee ``S3/resultados/lenin_configuracion_modelo_final.json``, recupera los
    datos MCAR y la partición de Formativa 2, imputa y estandariza con el
    entrenamiento, recalcula los pesos y ajusta la especificación guardada.
    Devuelve (resultado, preparador, imputador, configuracion).
    """
    raiz = Path(raiz)
    ruta_configuracion = Path(
        ruta_configuracion
        or raiz / "S3" / "resultados" / "lenin_configuracion_modelo_final.json"
    )
    config = json.loads(ruta_configuracion.read_text(encoding="utf-8"))

    df_mcar = pd.read_csv(raiz / config["datos_MCAR"]).set_index("EmployeeNumber")
    particion = pd.read_csv(raiz / config["particion"]).set_index("EmployeeNumber")
    train = df_mcar.loc[particion.index[particion["Conjunto"] == "Entrenamiento"]]

    imputador = ImputadorRegresion(
        config["imputacion"]["variable"], config["imputacion"]["predictores"]
    ).ajustar(train)
    train_imputado = imputador.transformar(train)

    preparador = PreparadorDiseno(config["variables"]).ajustar(train_imputado)
    X = preparador.transformar(train_imputado)
    y = codificar_objetivo(train_imputado[config["objetivo"]])

    pesos = pesos_balanceados(y) if config["ponderacion"] == "balanceada" else None
    resultado = ajustar_logit(X, y, pesos)
    return resultado, preparador, imputador, config
