<p align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/python/python.png" width="90">
</p>

<h1 align="center">
  Modelo de Regresión sobre IPM Continuo a Nivel de Hogar
</h1>

<p align="center">
  <b>Machine Learning · Econometría · Pobreza Multidimensional</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-CC342D?style=flat&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/SHAP-FF6F00?style=flat&logo=python&logoColor=white" alt="SHAP">
</p>

---

## 📑 Tabla de Contenidos
- [Acerca del Proyecto](#acerca-del-proyecto)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Metodología](#metodología)
- [Modelo Implementado](#modelo-implementado)
- [Requisitos](#requisitos)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Resultados y Métricas](#resultados-y-métricas)
- [Interpretabilidad](#interpretabilidad)
- [Validación](#validación)
- [Aplicaciones](#aplicaciones)
- [Licencia](#licencia)
- [Autor](#autor)

---

## Acerca del Proyecto

Este repositorio contiene un **modelo de regresión supervisada** basado en **XGBoost** para predecir el **Índice de Pobreza Multidimensional (IPM) continuo** a nivel de hogar, utilizando únicamente variables socioeconómicas y territoriales del hogar, **sin información sobre privaciones**.

### Motivación

El IPM oficial del DANE es una medida dicotómica (pobre/no pobre) basada en un umbral de 33.33%. Sin embargo, el valor continuo del IPM contiene información valiosa sobre la **intensidad de la pobreza** que se pierde en la clasificación binaria.

Este proyecto desarrolla un modelo predictivo que:

1. **Predice el valor continuo del IPM** (rango 0-1) usando solo características del hogar
2. **No utiliza información sobre privaciones** para evitar data leakage
3. **Permite identificar hogares vulnerables** antes de que crucen el umbral de pobreza
4. **Facilita la focalización proactiva** de políticas sociales

### Importancia

- **Predicción temprana**: Identifica hogares en riesgo antes de que caigan en pobreza multidimensional
- **Focalización eficiente**: Permite priorizar recursos hacia los hogares más vulnerables
- **Comprensión profunda**: Revela qué características del hogar tienen mayor impacto en la pobreza
- **Validación metodológica**: Confirma que el IPM está bien especificado y es predecible

---

## Estructura del Repositorio
```
IPM_regresion_ipm_continuo/
│
├── ipm_regresion_ipm_continuo (2).ipynb    # Notebook principal
├── hogares_ML.csv                          # Dataset de 53,103 hogares
├── Null                                    # Archivo auxiliar
└── README.md                               # Este archivo
```

### Componentes principales:

- **Notebook Jupyter**: Contiene todo el pipeline desde carga de datos hasta interpretación SHAP
- **Dataset**: Base de hogares con variables socioeconómicas y territoriales
- **README**: Documentación completa del proyecto

---

## Metodología

### 1. Variables utilizadas (SIN privaciones)

**Variables del hogar (5 variables):**
- `TAMANO_HOGAR`: Número de integrantes del hogar
- `EDAD_PROMEDIO`: Edad promedio de los miembros
- `EDU_PROMEDIO`: Años de educación promedio
- `EDU_MAX`: Máximo nivel educativo alcanzado en el hogar
- `PROP_MUJERES`: Proporción de mujeres en el hogar

**Variables territoriales (2 variables):**
- `ZONA_RURAL`: Indicador binario de zona rural
- `ZONA_CENTRO_POBLADO`: Indicador binario de centro poblado

**Variables departamentales:**
- **32 variables dummy** para departamentos de Colombia (excepto categoría de referencia)

**Total: 39 variables predictoras**

### 2. Variable objetivo

- **IPM continuo**: Valor entre 0 y 1 que representa la intensidad de privaciones del hogar
  - 0 = Sin privaciones
  - 0.333 = Umbral oficial de pobreza multidimensional
  - 1 = Máxima intensidad de privaciones

### 3. Pipeline de modelado
```
1. Carga y preparación de datos
   ├── Creación de variables territoriales
   ├── Generación de dummies departamentales
   └── Verificación de integridad

2. Análisis exploratorio
   ├── Distribución del IPM
   ├── Correlaciones con variables explicativas
   └── Detección de valores atípicos

3. División train/test (80/20)

4. Entrenamiento XGBoost
   ├── RandomizedSearchCV para hiperparámetros
   ├── Validación cruzada (5-fold)
   └── Selección del mejor modelo

5. Evaluación
   ├── Métricas: R², RMSE, MAE
   ├── Análisis residual
   └── Análisis por deciles

6. Interpretabilidad
   ├── Importancia de variables
   ├── SHAP values (global y local)
   └── Dependence plots
```

---

## Modelo Implementado

### XGBoost (Extreme Gradient Boosting)

**¿Por qué XGBoost?**

1. **Alto rendimiento**: Algoritmo state-of-the-art para problemas de regresión
2. **Manejo robusto**: Gestiona bien missing values y relaciones no lineales
3. **Regularización**: Previene overfitting mediante L1 y L2 regularization
4. **Eficiencia computacional**: Optimizado para velocidad y uso de memoria

**Hiperparámetros clave optimizados:**
```python
{
    'n_estimators': [100, 300, 500],        # Número de árboles
    'max_depth': [3, 5, 7, 9],              # Profundidad máxima
    'learning_rate': [0.01, 0.05, 0.1],    # Tasa de aprendizaje
    'subsample': [0.6, 0.8, 1.0],          # Fracción de muestras
    'colsample_bytree': [0.6, 0.8, 1.0],   # Fracción de features
    'min_child_weight': [1, 3, 5]          # Peso mínimo hijo
}
```

**Optimización mediante RandomizedSearchCV:**
- Búsqueda aleatoria de 100 combinaciones
- Validación cruzada 5-fold
- Métrica: Negative Mean Squared Error

---

## Requisitos

### Software

- **Python** ≥ 3.7
- **Jupyter Notebook** o **Google Colab**

### Librerías principales
```python
# Análisis de datos
import pandas as pd
import numpy as np

# Modelo y evaluación
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Interpretabilidad
import shap

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
```

### Instalación
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

---

## Cómo Ejecutar

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/jmeza-data/IPM_regresion_ipm_continuo.git
cd IPM_regresion_ipm_continuo
```

### Paso 2: Preparar el entorno
```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 3: Ejecutar el notebook

**Opción A: Jupyter Notebook local**
```bash
jupyter notebook
# Abrir: ipm_regresion_ipm_continuo (2).ipynb
```

**Opción B: Google Colab (recomendado)**
1. Subir notebook a Google Colab
2. Subir `hogares_ML.csv` a la sesión
3. Ejecutar: Runtime → Run all

### Paso 4: Estructura de ejecución

El notebook está diseñado para ejecutarse **celda por celda**:

1. **Sección 0**: Imports y configuración
2. **Sección 1**: Carga y preparación de datos
3. **Sección 1B**: Análisis exploratorio
4. **Sección 2**: División train/test y entrenamiento
5. **Sección 3**: Evaluación y métricas
6. **Sección 4**: SHAP y explicabilidad
7. **Sección 5**: Visualizaciones avanzadas

---

## Resultados y Métricas

### Métricas de rendimiento (conjunto de prueba)
```
R² (Coeficiente de determinación): ~0.XX
RMSE (Root Mean Squared Error): ~0.XX
MAE (Mean Absolute Error): ~0.XX
```

*Nota: Los valores específicos se generan al ejecutar el notebook*

### Interpretación de métricas

- **R²**: Proporción de varianza del IPM explicada por el modelo (0-1)
  - Valores cercanos a 1 indican excelente ajuste
  
- **RMSE**: Error promedio en unidades de IPM
  - Valores bajos indican predicciones precisas
  
- **MAE**: Error absoluto medio
  - Robusto ante outliers, interpretable directamente

### Análisis por deciles

El modelo mantiene estabilidad a lo largo de toda la distribución del IPM:
- **Deciles bajos** (IPM cercano a 0): Alta precisión
- **Deciles medios** (IPM 0.2-0.4): Predicción robusta
- **Deciles altos** (IPM > 0.5): Captura bien casos extremos

---

## Interpretabilidad

### 1. Importancia de variables (Feature Importance)

Ranking de las variables más influyentes en la predicción del IPM:

**Top 5 variables esperadas:**
1. Educación promedio del hogar
2. Educación máxima alcanzada
3. Zona rural/urbana
4. Tamaño del hogar
5. Departamento de residencia

### 2. SHAP Values (SHapley Additive exPlanations)

**SHAP proporciona:**

- **Importancia global**: Qué variables son más importantes en promedio
- **Dirección del impacto**: Si cada variable aumenta o disminuye el IPM
- **Interpretación local**: Explicación de predicciones individuales

**Visualizaciones SHAP generadas:**

1. **Summary Plot**: Distribución de impactos por variable
2. **Dependence Plot**: Relación no lineal entre features y predicción
3. **Force Plot**: Explicación detallada de casos individuales

### 3. Insights clave

- **Educación**: Mayor educación reduce significativamente el IPM predicho
- **Ubicación**: Zona rural aumenta considerablemente la predicción de IPM
- **Composición del hogar**: Hogares más grandes tienden a mayor IPM
- **Heterogeneidad regional**: Diferencias marcadas entre departamentos

---

## Validación

### 1. Validación interna

- **Validación cruzada 5-fold**: Estabilidad del modelo en diferentes particiones
- **Análisis residual**: Verificación de supuestos de regresión
- **Bootstrapping**: Intervalos de confianza para métricas

### 2. Validación externa

- **Comparación con IPM oficial**: Correlación con clasificación binaria del DANE
- **Consistencia territorial**: Patrones geográficos coherentes
- **Robustez temporal**: Aplicabilidad a diferentes años de ECV

### 3. Prevención de data leakage

**Crítico**: El modelo NO utiliza ninguna de las 15 privaciones del IPM:
- ✅ Solo usa características del hogar previas
- ✅ Variables territoriales exógenas
- ✅ Información demográfica no derivada del IPM

---

## Aplicaciones

### 1. Focalización proactiva de políticas

- Identificar hogares en riesgo antes del umbral de pobreza
- Priorizar intervenciones preventivas
- Optimizar asignación de recursos limitados

### 2. Monitoreo y evaluación

- Seguimiento continuo de vulnerabilidad
- Evaluación de impacto de programas sociales
- Early warning system para deterioro de condiciones

### 3. Investigación académica

- Validación de construcción del IPM
- Estudios de determinantes de pobreza
- Comparaciones metodológicas

### 4. Diseño de intervenciones

- Política educativa focalizada
- Programas de desarrollo rural
- Estrategias departamentales diferenciadas

---

## Licencia

Proyecto de **uso académico libre** desarrollado como parte de la tesis de grado en Economía de la Universidad Nacional de Colombia.

### Datos

Los microdatos provienen de la **Encuesta de Calidad de Vida (ECV) 2024** del DANE, sujetos a políticas de uso establecidas por esta entidad.

### Citación sugerida
```
Meza García, J. S. (2024). Modelo de Regresión sobre IPM Continuo a Nivel de Hogar.
GitHub. https://github.com/jmeza-data/IPM_regresion_ipm_continuo
```

---

## Autor

**Jhoan Sebastián Meza García**  
Estudiante de Economía  
Universidad Nacional de Colombia

**Áreas de especialización:**
- Machine Learning aplicado a economía
- Pobreza multidimensional y desigualdad
- Modelado predictivo y econometría
- Interpretabilidad de modelos

**Contacto:**  
📧 GitHub: [jmeza-data](https://github.com/jmeza-data)

---

<p align="center">
  <i>Modelado predictivo al servicio de la lucha contra la pobreza<br>
  Universidad Nacional de Colombia · 2024</i>
</p>
