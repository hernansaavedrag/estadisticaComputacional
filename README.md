# Estadística Computacional — MCDI501

Proyecto de la asignatura Estadística Computacional del Magíster en Ciencia de Datos e Inteligencia Artificial de la Universidad Andrés Bello.

**Equipo:** Hernán Saavedra, Lenin Salas y Rodrigo Gómez.

## Problema de estudio

¿Qué características laborales y personales se asocian con la salida de empleados en el conjunto IBM HR Employee Attrition?

La variable de interés es `Attrition`: `Yes` indica salida y `No`, permanencia. Se estudian asociaciones e incertidumbre estadística; los resultados no establecen causalidad.

## Conjunto de datos

- **Fuente:** [IBM HR Analytics Employee Attrition & Performance — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
- **Naturaleza:** datos sintéticos; no representan una muestra probabilística de una empresa real.
- **Dimensiones:** 1.470 registros y 35 columnas.
- **Salida:** 237 registros (16,12 %); permanencia: 1.233.
- **Calidad del original:** sin faltantes ni filas duplicadas.
- **Columnas constantes:** `EmployeeCount`, `Over18` y `StandardHours`.
- **Identificador:** `EmployeeNumber`, utilizado para trazabilidad e identificación de observaciones influyentes, sin tratarlo como predictor numérico.
- **Ingreso mensual:** se expresa en unidades monetarias del dataset, sin asignarle una moneda no verificada.

El CSV original se conserva; los análisis trabajan sobre copias en memoria. La simulación de faltantes de la Formativa 1 es un ejercicio separado: S1 y S2 utilizan el dataset original completo.

## Entregas y archivos principales

| Etapa | Contenido | Ubicación |
| --- | --- | --- |
| Formativa 1 | Exploración inicial, clasificación de variables y análisis estadístico | [Formativa1](Formativa1/) |
| Sumativa 1 | Descriptiva, correlaciones, IC del 95 % para tres medias, chi-cuadrado y Welch | [Notebook de S1](Sumativa1/Sumativa1_IBM_HR.ipynb) |
| Referencia S1 | Copia de referencia para la continuidad solicitada en S2 | [S1](S1/) |
| Sumativa 2 | Notebook integrado con resultados guardados de los tres integrantes | [S2_IBM_HR_Validacion.ipynb](S2/notebooks/S2_IBM_HR_Validacion.ipynb) |
| Figuras S2 | Gráficos exportados por los procedimientos | [S2/figuras](S2/figuras/) |
| Resultados S2 | Referencias S1, réplicas, intervalos, simulaciones y sensibilidad | [S2/resultados](S2/resultados/) |
| Documentos | Informes disponibles de las entregas | [docs](docs/) |
| Entorno | Versiones de las dependencias | [requirements.txt](requirements.txt) |

`Sumativa1/` se conserva para mantener los enlaces de la entrega anterior; `S1/` contiene la copia de referencia requerida para esta fase.

## Sumativa 2: simulación y remuestreo

El notebook integrado organiza el trabajo en cuatro bloques:

1. **Bloque A — Referencias de S1 y Monte Carlo:** reproduce las estimaciones previas y simula 10.000 escenarios de salida con tamaños y proporciones de S1. Examina convergencia, error Monte Carlo y distribuciones de escenarios.
2. **Bloque B — Bootstrap:** utiliza 10.000 remuestras para las medias de ingreso, edad y antigüedad; compara IC clásicos, percentil y BCa. Evalúa cuatro correlaciones de Spearman mediante remuestreo pareado, con intervalos y gráficos de estabilidad.
3. **Bloque C — Permutación y robustez:** valida la comparación de ingresos mediante 10.000 permutaciones del estadístico de Welch, identifica observaciones influyentes mediante jackknife y examina sensibilidad a la exclusión conjunta de ingresos extremos.
4. **Bloque D — Insumos para S3:** reúne estimaciones, intervalos, resultados de validación, observaciones influyentes y recomendaciones metodológicas.

### Distribución del trabajo

| Integrante | Aporte principal |
| --- | --- |
| Hernán Saavedra | Referencias S1, Monte Carlo, convergencia e integración del proyecto. |
| Lenin Salas | Bootstrap percentil y BCa de medias, estabilidad de correlaciones y visualizaciones. |
| Rodrigo Gómez | Permutación, jackknife y análisis de sensibilidad. |
| Equipo completo | Discusión de resultados, revisión e informe técnico. |

Los notebooks individuales se conservan en [S2/notebooks](S2/notebooks/) como registro de los aportes. El archivo destinado a la entrega conjunta es **`S2_IBM_HR_Validacion.ipynb`**.

### Resultados e interpretación

- Los IC bootstrap de las tres medias concuerdan aproximadamente con los IC clásicos de S1 bajo los supuestos del análisis.
- Las tres correlaciones con experiencia total conservan un signo positivo estable; la correlación entre distancia e ingreso es compatible con una asociación monótona pequeña o nula.
- Welch y permutación coinciden en rechazar la igualdad de ingresos medios con α = 0,05. El valor p mínimo de la permutación con corrección +1 es 1/10.001; no permite resolver valores tan pequeños como el p paramétrico.
- Monte Carlo produce una media de 236,90 salidas frente a una esperanza de 237. El rango central de escenarios es aproximadamente [210; 264], condicionado a parámetros fijos; no es un intervalo de confianza de un parámetro.
- La diferencia de ingresos permanece positiva al omitir un empleado a la vez. Al excluir conjuntamente los ingresos sobre el límite IQR, disminuye un 39,14 % y sigue siendo significativa: la dirección se mantiene en los escenarios examinados, pero la magnitud requiere cautela.

### Archivos de entrada para Sumativa 3

- [S2_resultados_validados_S3.csv](S2/resultados/S2_resultados_validados_S3.csv): siete estimaciones —tres medias y cuatro correlaciones— con intervalos, valoración y recomendaciones.
- [S2_reporte_validado_S3.json](S2/resultados/S2_reporte_validado_S3.json): reporte complementario con permutación, sensibilidad, Monte Carlo y empleados influyentes.
- [rodrigo_jackknife_influencia.csv](S2/resultados/rodrigo_jackknife_influencia.csv): influencia individual sobre la media global y la diferencia entre grupos.

## Ejecución reproducible

Entorno utilizado para la integración: **Python 3.11**. Las versiones de las bibliotecas están fijadas en `requirements.txt`.

Desde la raíz del repositorio, crear un entorno virtual:

```bash
python -m venv .venv
```

En Windows, si hay varias versiones instaladas, se puede utilizar `py -3.11 -m venv .venv`.

Instalar las dependencias con el intérprete del entorno:

**Windows — Git Bash o PowerShell:**

```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Linux o macOS:**

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

En VS Code, abrir `S2/notebooks/S2_IBM_HR_Validacion.ipynb`, seleccionar el kernel `.venv` y ejecutar todas las celdas en orden desde un kernel reiniciado. Guardar al finalizar. El bootstrap de correlaciones utiliza matrices grandes y puede requerir tiempo y memoria adicional.

El notebook localiza la raíz del repositorio y exporta sus resultados a `S2/resultados/` y `S2/figuras/`. Fija la semilla base 42 y utiliza flujos aleatorios separados por procedimiento. La carpeta `.venv/` está excluida de Git mediante `.gitignore`.

## Alcance y estado de entrega

El remuestreo supone observaciones independientes; no corrige sesgos de selección ni convierte los datos sintéticos en representativos. La simulación Monte Carlo mantiene fijos los parámetros estimados y no incorpora toda su incertidumbre. Los ingresos extremos se conservan en el análisis principal; el filtro IQR es un escenario de sensibilidad, no una limpieza de errores demostrados.

**Estado:** análisis computacional integrado y archivos para S3 disponibles. El informe técnico de Sumativa 2 está pendiente de elaboración y revisión en la plantilla del profesor.

La entrega en Canvas debe incluir **PDF de máximo 7 páginas + notebook `.ipynb` + enlace al repositorio**. La disponibilidad de los archivos en GitHub no sustituye esa entrega.
