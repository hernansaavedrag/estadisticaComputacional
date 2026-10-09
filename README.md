# Estadística Computacional

Repositorio de trabajo de la asignatura **Estadística Computacional (MCDI501)** del Magíster en Ciencia de Datos e Inteligencia Artificial de la Universidad Andrés Bello.

Contiene los datos, el notebook de análisis y el informe de la **Formativa 1**, desarrollada con el conjunto de datos **IBM HR Employee Attrition**.

## Formativa 1: rotación de empleados

El trabajo explora las características de los empleados y su relación preliminar con la salida de la empresa. La pregunta que orienta el análisis es:

> ¿Qué características laborales y personales se asocian con la rotación de empleados en este conjunto de datos?

La variable de interés es `Attrition`, que indica si el empleado dejó la empresa (`Yes`) o permaneció en ella (`No`). En esta etapa se realiza un análisis exploratorio e inferencial; no se entrena un modelo predictivo.

## Archivos

| Ubicación | Contenido |
| --- | --- |
| [Formativa1/Formativa1_IBM_HR_Paso_a_Paso.ipynb](Formativa1/Formativa1_IBM_HR_Paso_a_Paso.ipynb) | Notebook con código, tablas, gráficos y resultados estadísticos. |
| [datos/WA_Fn-UseC_-HR-Employee-Attrition.csv](datos/WA_Fn-UseC_-HR-Employee-Attrition.csv) | Dataset original utilizado en el análisis. |
| [docs/Informe_Formativa1_IBM_HR.pdf](docs/Informe_Formativa1_IBM_HR.pdf) | Informe de la actividad en formato PDF. |

## Conjunto de datos

**IBM HR Employee Attrition** es un dataset sintético con **1.470 registros y 35 columnas**. Incluye variables como edad, ingreso mensual, distancia al trabajo, antigüedad, departamento, cargo, satisfacción laboral y horas extra.

- **Fuente:** [IBM HR Analytics Employee Attrition & Performance — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
- **Variable de interés:** `Attrition`.
- **Distribución:** 237 empleados con salida y 1.233 sin salida; aproximadamente un 16,12 % de salida.
- **Calidad inicial:** sin valores faltantes ni filas duplicadas.
- **Columnas constantes:** `EmployeeCount`, `Over18` y `StandardHours`.
- **Identificador:** `EmployeeNumber` se excluye del análisis estadístico.

El archivo original se conserva intacto. Las transformaciones se realizan sobre una copia en memoria.

## Análisis del notebook

1. Revisión de calidad: dimensiones, faltantes, duplicados y columnas constantes.
2. Clasificación de variables según su tipo estadístico.
3. Simulación de faltantes MCAR: se oculta aleatoriamente el 10 % de los valores de `MonthlyIncome` y `DistanceFromHome`, utilizando la semilla 42.
4. Estadística descriptiva: frecuencias, media, mediana, desviación estándar, cuartiles y asimetría; comparación según `Attrition`.
5. Visualización de distribuciones y porcentajes de salida por grupo.
6. Estimación puntual e intervalos de confianza del 95 %: Wilson para la proporción de salida y t de Student para el ingreso mensual medio observado.
7. Prueba chi-cuadrado de Pearson para la asociación entre `OverTime` y `Attrition`, con nivel de significancia de 0,05 y V de Cramér como medida de asociación.

Los cálculos de ingreso y distancia utilizan los valores observados, sin imputación. Los valores ocultados se reservan en memoria para evaluar imputaciones posteriores.

## Ejecución

Se requiere Python, Jupyter y las bibliotecas `pandas`, `numpy`, `scipy`, `matplotlib` y `seaborn`.

### 1. Descargar el repositorio

```bash
git clone https://github.com/hernansaavedrag/estadisticaComputacional.git
cd estadisticaComputacional
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activarlo según el sistema o la terminal utilizada:

**Windows — PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows — Git Bash**

```bash
source .venv/Scripts/activate
```

**macOS o Linux**

```bash
source .venv/bin/activate
```

### 3. Instalar las dependencias

```bash
python -m pip install pandas numpy scipy matplotlib seaborn jupyter ipykernel
```

### 4. Abrir y ejecutar el notebook

```bash
jupyter notebook
```

Abrir `Formativa1/Formativa1_IBM_HR_Paso_a_Paso.ipynb` y ejecutar las celdas en orden, desde la primera hasta la última.

También puede abrirse en **Visual Studio Code** con las extensiones Python y Jupyter. Seleccionar **Python Environments** y elegir el intérprete del entorno `.venv` como kernel.

El notebook localiza el CSV mediante una ruta relativa cuando se ejecuta desde la raíz del repositorio o desde la carpeta `Formativa1`. Mantener la estructura de carpetas indicada.

## Reproducibilidad e interpretación

La primera celda registra las versiones de Python y de las bibliotecas utilizadas. La simulación de faltantes usa una semilla fija y parte de una copia nueva del dataset en cada ejecución completa.

Los datos son sintéticos. Las asociaciones encontradas no establecen causalidad ni representan resultados de una empresa real. Los intervalos de confianza y las pruebas de hipótesis se utilizan con fines didácticos, bajo un modelo hipotético de observaciones independientes del proceso sintético.

## Equipo

- Hernán Saavedra.
- Lenin Salas.
- Rodrigo Gómez.
