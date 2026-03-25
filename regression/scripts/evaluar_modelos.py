"""
evaluar_modelos.py - Evaluación modular multi-target con feature selection
=========================================================================
12 modelos organizados en funciones individuales para facilitar modificaciones.
Incluye selección de features (VarianceThreshold, correlación, LassoCV),
log-transform para volumen, Pearson r como métrica adicional,
deduplicación automática por Paciente_ID y visualización demográfica.

Pipeline por modelo: StandardScaler + Estimador + GridSearchCV/RandomizedSearchCV
Targets: Volumen (mm³), Eje Corto (mm), Eje Largo (mm)
Validación: 5-Fold CV × 2 repeticiones + tuning interno 3-Fold

Funciones de evaluación:
  - evaluar_reg_simple()       Regresión lineal simple (1 feature)
  - evaluar_reg_multiple()     Regresión lineal múltiple
  - evaluar_polinomica()       Regresión polinómica (Ridge + PolynomialFeatures)
  - evaluar_ridge()            Ridge (L2)
  - evaluar_lasso()            Lasso (L1)
  - evaluar_elasticnet()       ElasticNet (L1 + L2)
  - evaluar_knn()              K-Nearest Neighbors
  - evaluar_svr()              Support Vector Regression
  - evaluar_arbol()            Árbol de decisión
  - evaluar_random_forest()    Random Forest
  - evaluar_gradient_boost()   Gradient Boosting
  - evaluar_xgboost()          XGBoost
"""

import os
import time
import warnings as w
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.model_selection import (
    KFold, GridSearchCV, RandomizedSearchCV, learning_curve,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LassoCV,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, max_error,
    explained_variance_score, mean_absolute_percentage_error,
)
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from scipy.stats import pearsonr
from xgboost import XGBRegressor

w.filterwarnings("ignore")

# -- Rutas --
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_master.csv")
CARPETA_METRICAS = os.path.join(base_dir, "regression", "metrics")
os.makedirs(CARPETA_METRICAS, exist_ok=True)

# -- Configuración de validación --
N_SPLITS = 5          # 5-Fold -> 80% train, 20% test
N_REPEATS = 2         # 2 repeticiones para estimar varianza
INNER_CV = 3          # Folds internos para tuning de hiperparámetros
OVERFITTING_UMBRAL = 15.0   # Gap% > 15% indica overfitting

# -- Feature selection --
CORR_UMBRAL = 0.05         # Mínimo |correlación| con el target
INTER_CORR_UMBRAL = 0.95   # Máx inter-correlación entre features

# -- Targets --
TARGETS = [
    {"nombre": "Volumen Tumoral",    "col": "target_regresion",
     "origen": "shape_VoxelVolume",  "u": "mm3", "u2": "mm6", "slug": "volumen"},
    {"nombre": "Diámetro Eje Corto", "col": "target_eje_corto",
     "origen": "shape_MinorAxisLength", "u": "mm", "u2": "mm2", "slug": "eje_corto"},
    {"nombre": "Diámetro Eje Largo", "col": "target_eje_largo",
     "origen": "shape_MajorAxisLength", "u": "mm", "u2": "mm2", "slug": "eje_largo"},
]
COLS_TARGET = [t["col"] for t in TARGETS]

# -- Columnas clínicas (solo para gráficas, NO entran al pipeline ML) --
COLS_CLINICAS = ["Body Part Examined", "PatientSex", "PrimaryCondition"]


# ==============================================================
#  PREPARACIÓN DE DATOS
# ==============================================================

def preparar_datos_regresion(df):
    """
    Crea targets de regresión desde shape_* y elimina features de forma
    para prevenir data leakage.

    Transformaciones:
      1. target_regresion  = shape_VoxelVolume      (mm³)
      2. target_eje_corto  = shape_MinorAxisLength   (mm)
      3. target_eje_largo  = shape_MajorAxisLength   (mm)
      4. X = 102 features radiómicos seleccionados automáticamente:
         - firstorder (18): estadísticas de intensidad
         - glcm (24): texturas Gray Level Co-occurrence
         - glrlm (16): patrones de runs (homogeneidad/complejidad)
         - glszm (16): patrones de zonas (estructuras locales)
         - gldm (14): dependencia de niveles de gris
      
      Feature selection pipeline (4-pasos) elige subset óptimo por target:
      Varianza → Correlación → Inter-correlación → LassoCV

    Returns
    -------
    df_clean : DataFrame con targets creados y shape_* eliminadas.
    info     : dict con estadísticas de transformación.
    """
    df_clean = df.copy()

    # Validar que existen las columnas origen de los targets
    required = [t["origen"] for t in TARGETS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas requeridas faltantes en CSV: {missing}")

    # Crear targets ANTES de eliminar shape_*
    for t in TARGETS:
        df_clean[t["col"]] = df[t["origen"]]

    # Eliminar TODAS las columnas shape_* -> prevención data leakage
    shape_cols = [c for c in df_clean.columns if c.startswith("shape_")]
    df_clean = df_clean.drop(columns=shape_cols)

    # Contar features finales - TODAS las familias radiómicas
    feature_cols = [c for c in df_clean.columns
                    if c.startswith(("firstorder_", "glcm_", "glrlm_", 
                                      "glszm_", "gldm_"))]

    info = {
        "n_muestras":       len(df_clean),
        "n_features":       len(feature_cols),
        "shape_eliminadas": len(shape_cols),
        "features":         feature_cols,
        "targets_creados":  COLS_TARGET,
    }
    return df_clean, info


# ==============================================================
#  SELECCIÓN DE FEATURES
# ==============================================================

def seleccionar_features(X, y, nombre_target):
    """
    Pipeline de selección de features en 4 pasos:
      1. VarianceThreshold(0)   → elimina features constantes
      2. Correlación con target → mantiene |r| > CORR_UMBRAL
      3. Inter-correlación      → elimina redundantes |r| > INTER_CORR_UMBRAL
      4. LassoCV + SelectFromModel → selección automática final

    Parámetros
    ----------
    X : DataFrame con features candidatas (hasta 102 radiómicos: firstorder, glcm, glrlm, glszm, gldm)
    y : array con valores del target
    nombre_target : str para identificar en consola

    Pipeline robusto: cada paso preserva información relevante mientras elimina ruido.
    El subset final es específico y óptimo para cada target.

    Retorna
    -------
    cols_final : lista de nombres de columnas seleccionadas (óptimas para target)
    """
    n_original = X.shape[1]

    # Paso 1: eliminar features con varianza cero
    vt = VarianceThreshold(threshold=0)
    vt.fit(X)
    cols_var = X.columns[vt.get_support()].tolist()

    # Paso 2: correlación con target (|r| > CORR_UMBRAL)
    corrs = X[cols_var].corrwith(pd.Series(y, index=X.index)).abs()
    cols_corr = corrs[corrs > CORR_UMBRAL].index.tolist()

    if len(cols_corr) < 3:
        cols_corr = cols_var

    # Paso 3: eliminar features redundantes por inter-correlación
    X_tmp = X[cols_corr]
    corr_matrix = X_tmp.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        redundantes = upper.index[upper[col] > INTER_CORR_UMBRAL].tolist()
        for red in redundantes:
            # Eliminar la que tiene menor correlación con el target
            if corrs.get(red, 0) < corrs.get(col, 0):
                to_drop.add(red)
            else:
                to_drop.add(col)
    cols_inter = [c for c in cols_corr if c not in to_drop]

    if len(cols_inter) < 3:
        cols_inter = cols_corr

    # Paso 4: LassoCV + SelectFromModel
    X_scaled = StandardScaler().fit_transform(X[cols_inter])
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    selector = SelectFromModel(lasso, prefit=True)
    mask = selector.get_support()
    cols_final = [c for c, keep in zip(cols_inter, mask) if keep]

    # Fallback: si LassoCV descarta demasiadas, usar las del paso 3
    if len(cols_final) < 3:
        cols_final = cols_inter

    # Analizar composición de features finales por familia
    familias_final = {
        "firstorder": len([c for c in cols_final if "firstorder" in c]),
        "glcm": len([c for c in cols_final if "glcm" in c]),
        "glrlm": len([c for c in cols_final if "glrlm" in c]),
        "glszm": len([c for c in cols_final if "glszm" in c]),
        "gldm": len([c for c in cols_final if "gldm" in c]),
    }
    familias_entrada = {
        "firstorder": len([c for c in X.columns if "firstorder" in c]),
        "glcm": len([c for c in X.columns if "glcm" in c]),
        "glrlm": len([c for c in X.columns if "glrlm" in c]),
        "glszm": len([c for c in X.columns if "glszm" in c]),
        "gldm": len([c for c in X.columns if "gldm" in c]),
    }

    print(f"    Features: {n_original} → {len(cols_var)} (var) → "
          f"{len(cols_corr)} (corr) → {len(cols_inter)} (inter) → "
          f"{len(cols_final)} (lasso) para {nombre_target}")
    print(f"    Entrada: {dict(sorted(familias_entrada.items(), key=lambda x: -x[1]))} → "
          f"Salida: {dict(sorted(familias_final.items(), key=lambda x: -x[1]))}")

    return cols_final


# ==============================================================
#  MÉTRICAS
# ==============================================================

def calcular_metricas(y_real, y_pred):
    """Calcula 9 métricas de regresión incluyendo Pearson r."""
    mse = mean_squared_error(y_real, y_pred)
    r, _ = pearsonr(y_real, y_pred)
    return {
        "MAE":       round(mean_absolute_error(y_real, y_pred), 2),
        "MSE":       round(mse, 2),
        "RMSE":      round(np.sqrt(mse), 2),
        "R2":        round(r2_score(y_real, y_pred), 4),
        "Pearson_r": round(r, 4),
        "MedAE":     round(median_absolute_error(y_real, y_pred), 2),
        "MaxErr":    round(max_error(y_real, y_pred), 2),
        "MAPE %":    round(mean_absolute_percentage_error(y_real, y_pred) * 100, 2),
        "EVS":       round(explained_variance_score(y_real, y_pred), 4),
    }


# ==============================================================
#  DETECCIÓN DE OVERFITTING
# ==============================================================

def analizar_overfitting(train_maes, test_maes, nombre_modelo):
    """
    Compara MAE de train vs test promediando sobre todos los folds/repeticiones.

    Gap% = (test_mae - train_mae) / test_mae × 100
      > 15%  -> overfitting (modelo memoriza train)
      < -10% -> underfitting (modelo muy simple)
      else   -> balance adecuado

    Returns
    -------
    dict con train_mae, test_mae, gap_pct, diagnostico, recomendacion
    """
    train_mae = np.mean(train_maes)
    test_mae = np.mean(test_maes)

    if test_mae == 0:
        gap = 0.0
    else:
        gap = ((test_mae - train_mae) / test_mae) * 100

    # Diagnóstico
    if gap > OVERFITTING_UMBRAL:
        diagnostico = "OVERFITTING"
        recomendaciones = {
            "Random Forest":  "Reducir max_depth o aumentar min_samples_leaf",
            "XGBoost":        "Reducir max_depth / n_estimators o subir learning_rate",
            "Gradient Boost": "Reducir max_depth / n_estimators o subir learning_rate",
            "Arbol Decision": "Reducir max_depth o aumentar min_samples_leaf",
            "Polinomica":     "Usar degree=2 o aplicar regularización (Ridge)",
            "KNN":            "Aumentar n_neighbors",
            "Reg. Multiple":  "Reducir features (feature selection) o regularizar",
        }
        rec = recomendaciones.get(nombre_modelo,
                                  "Aumentar regularización o reducir complejidad")
    elif gap < -10:
        diagnostico = "UNDERFITTING"
        rec = "Aumentar complejidad: más features, degree, o profundidad"
    else:
        diagnostico = "OK"
        rec = "Balance adecuado train/test"

    return {
        "Train_MAE":     round(train_mae, 2),
        "Test_MAE":      round(test_mae, 2),
        "Gap_%":         round(gap, 2),
        "Diagnostico":   diagnostico,
        "Recomendacion": rec,
    }


def _recomendacion_color(diagnostico):
    """Retorna símbolo según diagnóstico."""
    return {"OK": "[OK]", "OVERFITTING": "[!]", "UNDERFITTING": "[v]"}.get(
        diagnostico, "?")


# ==============================================================
#  PARAMS MÁS FRECUENTES
# ==============================================================

def _params_mas_frecuentes(lista_params):
    """Devuelve los hiperparámetros más elegidos entre los folds."""
    if not lista_params:
        return ""
    tuplas = [tuple(sorted(d.items())) for d in lista_params]
    mas_comun = Counter(tuplas).most_common(1)[0][0]
    return ", ".join(f"{k.split('__')[1]}={v}" for k, v in mas_comun)


# ==============================================================
#  CORE: EVALUACIÓN CON VALIDACIÓN CRUZADA
# ==============================================================

def _evaluar_modelo_cv(nombre, pipe, params, X, y,
                       search_type="grid", n_iter=10, usar_log=False):
    """
    Evaluación core: KFold × N_REPEATS con tuning interno.

    Si usar_log=True, entrena en log-space (log1p) y reporta métricas
    en escala original (expm1). Ideal para targets con distribución
    sesgada como Volumen Tumoral.

    Parámetros
    ----------
    nombre      : str, nombre del modelo para reportes
    pipe        : Pipeline de sklearn (template, se clona internamente)
    params      : dict, grid de hiperparámetros para tuning
    X           : DataFrame con features
    y           : array con target en escala original
    search_type : 'grid' o 'random'
    n_iter      : int, iteraciones para RandomizedSearchCV
    usar_log    : bool, aplicar log1p/expm1 al target

    Retorna
    -------
    resultado : dict con métricas + metadata interna (_pipe, _y_pred, etc.)
    ovf_dict  : dict con diagnóstico de overfitting
    """
    N = len(y)
    y_pred_acum = np.zeros(N)
    conteo_acum = np.zeros(N)
    train_maes = []
    test_maes = []
    mejores_params = []

    for rep in range(N_REPEATS):
        kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42 + rep)

        for train_idx, test_idx in kfold.split(X):
            X_tr = X.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_tr = y[train_idx]
            y_te = y[test_idx]

            # Log transform para entrenamiento
            y_tr_fit = np.log1p(y_tr) if usar_log else y_tr

            pipe_clone = clone(pipe)

            if params:
                inner_cv = KFold(n_splits=INNER_CV, shuffle=True,
                                 random_state=42)

                if search_type == "random":
                    searcher = RandomizedSearchCV(
                        pipe_clone, params, n_iter=n_iter,
                        cv=inner_cv, scoring="neg_mean_absolute_error",
                        n_jobs=1, random_state=42)
                else:
                    searcher = GridSearchCV(
                        pipe_clone, params, cv=inner_cv,
                        scoring="neg_mean_absolute_error", n_jobs=1)
                searcher.fit(X_tr, y_tr_fit)
                modelo_fit = searcher.best_estimator_
                mejores_params.append(searcher.best_params_)
            else:
                pipe_clone.fit(X_tr, y_tr_fit)
                modelo_fit = pipe_clone

            # Predicciones
            pred_te = modelo_fit.predict(X_te)
            pred_tr = modelo_fit.predict(X_tr)

            # Invertir log transform
            if usar_log:
                pred_te = np.expm1(pred_te)
                pred_tr = np.expm1(pred_tr)

            # Acumular predicciones test (promedio entre repeticiones)
            y_pred_acum[test_idx] += pred_te
            conteo_acum[test_idx] += 1

            # MAE por fold para análisis de overfitting (escala original)
            train_maes.append(mean_absolute_error(y_tr, pred_tr))
            test_maes.append(mean_absolute_error(y_te, pred_te))

    # Promediar predicciones sobre repeticiones
    mask = conteo_acum > 0
    y_pred = np.where(mask, y_pred_acum / conteo_acum, 0)

    # Métricas globales (escala original)
    m = calcular_metricas(y, y_pred)

    # Análisis de overfitting
    ovf = analizar_overfitting(train_maes, test_maes, nombre)
    m["Train_MAE"] = ovf["Train_MAE"]
    m["Gap_%"]     = ovf["Gap_%"]

    # Params más frecuentes
    m["Best_Params"] = _params_mas_frecuentes(mejores_params)

    resultado = {
        "Modelo": nombre, **m,
        # Metadata interna (no se escribe a CSV, prefijo _ )
        "_pipe": pipe, "_params": params,
        "_search_type": search_type, "_n_iter": n_iter,
        "_feature_cols": list(X.columns),
        "_y_pred": y_pred.copy(),
    }
    ovf_dict = {"Modelo": nombre, **ovf}

    return resultado, ovf_dict


# ==============================================================
#  FUNCIONES DE EVALUACIÓN POR MODELO
# ==============================================================
# Cada función define su pipeline + hiperparámetros y delega a
# _evaluar_modelo_cv(). Para modificar un modelo específico, editar
# solo su función. Para agregar un modelo nuevo, crear una función
# y añadirla a la lista EVALUADORES al final de esta sección.
# ==============================================================

def evaluar_reg_simple(X, y, mejor_feat, usar_log=False):
    """Regresión lineal simple con la feature de mayor correlación."""
    pipe = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
    return _evaluar_modelo_cv("Reg. Simple", pipe, {}, X[[mejor_feat]], y,
                              usar_log=usar_log)


def evaluar_reg_multiple(X, y, usar_log=False):
    """Regresión lineal múltiple con todas las features seleccionadas."""
    pipe = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
    return _evaluar_modelo_cv("Reg. Multiple", pipe, {}, X, y,
                              usar_log=usar_log)


def evaluar_polinomica(X, y, usar_log=False):
    """Regresión polinómica (grado 2) con regularización Ridge."""
    pipe = Pipeline([
        ("s", StandardScaler()),
        ("p", PolynomialFeatures(include_bias=False)),
        ("m", Ridge()),
    ])
    params = {"p__degree": [2], "m__alpha": [10.0, 100.0, 1000.0]}
    return _evaluar_modelo_cv("Polinomica", pipe, params, X, y,
                              usar_log=usar_log)


def evaluar_ridge(X, y, usar_log=False):
    """Regresión Ridge (regularización L2)."""
    pipe = Pipeline([("s", StandardScaler()), ("m", Ridge())])
    params = {"m__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    return _evaluar_modelo_cv("Ridge", pipe, params, X, y,
                              usar_log=usar_log)


def evaluar_lasso(X, y, usar_log=False):
    """Regresión Lasso (regularización L1)."""
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", Lasso(max_iter=10000))])
    params = {"m__alpha": [0.01, 0.1, 1.0, 10.0]}
    return _evaluar_modelo_cv("Lasso", pipe, params, X, y,
                              usar_log=usar_log)


def evaluar_elasticnet(X, y, usar_log=False):
    """Regresión ElasticNet (L1 + L2)."""
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", ElasticNet(max_iter=10000))])
    params = {
        "m__alpha":    [0.01, 0.1, 1.0],
        "m__l1_ratio": [0.2, 0.5, 0.8],
    }
    return _evaluar_modelo_cv("ElasticNet", pipe, params, X, y,
                              usar_log=usar_log)


def evaluar_knn(X, y, usar_log=False):
    """K-Nearest Neighbors Regressor."""
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", KNeighborsRegressor())])
    params = {
        "m__n_neighbors": [15, 20, 30, 50],
        "m__weights":     ["uniform", "distance"],
        "m__p":           [1, 2],
    }
    return _evaluar_modelo_cv("KNN", pipe, params, X, y,
                              usar_log=usar_log)


def evaluar_svr(X, y, usar_log=False):
    """Support Vector Regression."""
    pipe = Pipeline([("s", StandardScaler()), ("m", SVR())])
    params = {
        "m__C":       [0.001, 0.01, 0.1, 1],
        "m__epsilon": [0.1, 0.5, 1.0],
        "m__kernel":  ["linear", "rbf"],
    }
    return _evaluar_modelo_cv("SVR", pipe, params, X, y,
                              usar_log=usar_log)


def evaluar_arbol(X, y, usar_log=False):
    """Árbol de decisión."""
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", DecisionTreeRegressor(random_state=42))])
    params = {
        "m__max_depth":        [1, 2, 3],
        "m__min_samples_leaf": [4, 8, 16],
    }
    return _evaluar_modelo_cv("Arbol Decision", pipe, params, X, y,
                              usar_log=usar_log)


def evaluar_random_forest(X, y, usar_log=False):
    """Random Forest Regressor."""
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", RandomForestRegressor(random_state=42))])
    params = {
        "m__n_estimators":     [50, 100],
        "m__max_depth":        [2, 3],
        "m__min_samples_leaf": [8, 16],
    }
    return _evaluar_modelo_cv("Random Forest", pipe, params, X, y,
                              usar_log=usar_log)


def evaluar_gradient_boost(X, y, usar_log=False):
    """Gradient Boosting Regressor."""
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", GradientBoostingRegressor(
                         random_state=42, subsample=0.7, max_features=0.7))])
    params = {
        "m__n_estimators":      [10, 15, 25],
        "m__max_depth":         [1],
        "m__learning_rate":     [0.1, 0.2, 0.3],
        "m__min_samples_leaf":  [16, 32, 48],
        "m__min_samples_split": [15, 25],
    }
    return _evaluar_modelo_cv("Gradient Boost", pipe, params, X, y,
                              search_type="random", n_iter=20,
                              usar_log=usar_log)


def evaluar_xgboost(X, y, usar_log=False):
    """XGBoost Regressor."""
    pipe = Pipeline([("s", StandardScaler()),
                     ("m", XGBRegressor(
                         random_state=42, verbosity=0,
                         subsample=0.8, colsample_bytree=0.8))])
    params = {
        "m__n_estimators":     [15, 25, 50],
        "m__max_depth":        [1],
        "m__learning_rate":    [0.1, 0.2, 0.3],
        "m__reg_alpha":        [5, 10],
        "m__reg_lambda":       [10, 20],
        "m__min_child_weight": [10, 20, 30],
    }
    return _evaluar_modelo_cv("XGBoost", pipe, params, X, y,
                              search_type="random", n_iter=25,
                              usar_log=usar_log)


# -- Registro de evaluadores --
# Para agregar un modelo nuevo: crear función evaluar_xxx() y añadirla aquí.
EVALUADORES = [
    ("Reg. Simple",     evaluar_reg_simple),
    ("Reg. Multiple",   evaluar_reg_multiple),
    ("Polinomica",      evaluar_polinomica),
    ("Ridge",           evaluar_ridge),
    ("Lasso",           evaluar_lasso),
    ("ElasticNet",      evaluar_elasticnet),
    ("KNN",             evaluar_knn),
    ("SVR",             evaluar_svr),
    ("Arbol Decision",  evaluar_arbol),
    ("Random Forest",   evaluar_random_forest),
    ("Gradient Boost",  evaluar_gradient_boost),
    ("XGBoost",         evaluar_xgboost),
]


# ==============================================================
#  ORQUESTADOR: EVALUACIÓN MULTI-TARGET
# ==============================================================

def ejecutar_evaluacion():
    """
    Evaluación completa: feature selection + 12 modelos × 3 targets.
    Log-transform automático para Volumen Tumoral.

    Returns
    -------
    todos : dict con resultados por target para generar_graficas()
    """
    # -- Cargar y validar --
    df_raw = pd.read_csv(RUTA_CSV)

    # Deduplicar por Paciente_ID (cada caso puede aparecer >1 vez en el CSV)
    n_antes = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset="Paciente_ID").reset_index(drop=True)
    n_despues = len(df_raw)

    print("\n" + "=" * 80)
    print("  VALIDACIÓN DE DATOS DE ENTRADA")
    print("=" * 80)
    if n_antes != n_despues:
        print(f"  [OK] Deduplicado: {n_antes} → {n_despues} filas únicas")

    required = ["Paciente_ID", "target_riesgo"] + [t["origen"] for t in TARGETS]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"[X] Columnas requeridas faltantes: {missing}")

    shape_n = sum(1 for c in df_raw.columns if c.startswith("shape_"))
    fo_n    = sum(1 for c in df_raw.columns if c.startswith("firstorder_"))
    glcm_n  = sum(1 for c in df_raw.columns if c.startswith("glcm_"))
    print(f"  [OK] CSV cargado: {len(df_raw)} filas x {len(df_raw.columns)} columnas")
    print(f"  [OK] shape: {shape_n} | firstorder: {fo_n} | glcm: {glcm_n}")

    # -- Preparar datos (crear targets + eliminar shape_*) --
    df, prep_info = preparar_datos_regresion(df_raw)

    print(f"\n  PREPARACIÓN:")
    print(f"  [OK] Targets creados: {', '.join(prep_info['targets_creados'])}")
    print(f"  [OK] shape_* eliminadas: {prep_info['shape_eliminadas']} columnas")
    print(f"  [OK] Features iniciales: {prep_info['n_features']} (firstorder + glcm + glrlm + glszm + gldm)")

    # -- Separar X (features) de targets y metadatos --
    cols_drop = ["Paciente_ID", "target_riesgo"] + COLS_TARGET + COLS_CLINICAS
    X_all = df.drop(columns=[c for c in cols_drop if c in df.columns])
    N = len(df)

    print(f"\n" + "=" * 80)
    print(f"  EVALUACIÓN DE MODELOS DE REGRESIÓN")
    print("=" * 80)
    print(f"  Dataset: {N} muestras x {X_all.shape[1]} features iniciales")
    print(f"  Validación: {N_SPLITS}-Fold CV x {N_REPEATS} repeticiones")
    print(f"  Tuning interno: {INNER_CV}-Fold CV")
    print(f"  Umbral overfitting: Gap% > {OVERFITTING_UMBRAL}%")
    print(f"  Feature selection: corr>{CORR_UMBRAL} + inter<{INTER_CORR_UMBRAL} + LassoCV")
    print()

    todos = {}
    t_inicio = time.time()

    for t in TARGETS:
        y = df[t["col"]].values
        u = t["u"]
        usar_log = (t["slug"] == "volumen")

        print(f"\n{'-' * 80}")
        print(f"> {t['nombre']} ({u})")

        # Feature selection por target
        cols_sel = seleccionar_features(X_all, y, t["nombre"])
        X_sel = X_all[cols_sel]

        # Mejor feature para regresión simple (entre las seleccionadas)
        corr = X_sel.corrwith(pd.Series(y, index=X_sel.index)).abs()
        mejor_feat = corr.idxmax()

        print(f"  Rango: {y.min():.2f} - {y.max():.2f} {u} | "
              f"Media: {y.mean():.2f} {u} | "
              f"Mejor feature: {mejor_feat} (r={corr[mejor_feat]:.3f})")
        if usar_log:
            print(f"  [LOG] Log-transform activado para {t['nombre']}")
        print()

        resultados = []
        overfitting_info = []

        for nombre, fn in EVALUADORES:
            try:
                if nombre == "Reg. Simple":
                    res, ovf = fn(X_sel, y, mejor_feat, usar_log)
                else:
                    res, ovf = fn(X_sel, y, usar_log)
                resultados.append(res)
                overfitting_info.append(ovf)
            except Exception as e:
                print(f"  {nombre:18s} | ERROR: {e}")
                nan_row = {
                    "Modelo": nombre, "MAE": np.nan, "MSE": np.nan,
                    "RMSE": np.nan, "R2": np.nan, "Pearson_r": np.nan,
                    "MedAE": np.nan, "MaxErr": np.nan, "MAPE %": np.nan,
                    "EVS": np.nan, "Train_MAE": np.nan, "Gap_%": np.nan,
                    "Best_Params": "",
                }
                resultados.append(nan_row)
                overfitting_info.append({
                    "Modelo": nombre, "Train_MAE": np.nan,
                    "Test_MAE": np.nan, "Gap_%": np.nan,
                    "Diagnostico": "ERROR", "Recomendacion": str(e),
                })

        # -- Tabla compacta de resultados (sin columnas internas) --
        clean_results = [{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in resultados]
        df_res = pd.DataFrame(clean_results).sort_values("MAE")

        # -- Reporte detallado por modelo (formato bloque) --
        print(f"\n  {'='*72}")
        print(f"  REPORTE DETALLADO — {t['nombre']} ({u})")
        print(f"  {'='*72}")
        pos = 0
        for _, row in df_res.iterrows():
            if pd.notna(row["MAE"]):
                pos += 1
                ovf_dx = next(
                    (o for o in overfitting_info if o["Modelo"] == row["Modelo"]),
                    {"Diagnostico": "?"})
                sym = _recomendacion_color(ovf_dx["Diagnostico"])
                print(f"\n  #{pos} {row['Modelo']}  {sym} {ovf_dx['Diagnostico']}")
                print(f"  {'─'*50}")
                print(f"  │ R²={row['R2']:.4f}   Pearson={row['Pearson_r']:.4f}   "
                      f"MAPE={row['MAPE %']:.1f}%   EVS={row['EVS']:.4f}")
                print(f"  │ MAE={row['MAE']:.2f}   RMSE={row['RMSE']:.2f}   "
                      f"MedAE={row['MedAE']:.2f}   MaxErr={row['MaxErr']:.2f}")
                print(f"  │ MSE={row['MSE']:.2f}   Train_MAE={row['Train_MAE']:.2f}   "
                      f"Gap={row['Gap_%']:.1f}%")
                if row.get("Best_Params"):
                    print(f"  │ Params: {row['Best_Params']}")

        # Mejor modelo + justificación
        validos = [r for r in resultados if not np.isnan(r["MAE"])]
        if validos:
            ranking = sorted(validos, key=lambda r: r["MAE"])
            mejor = ranking[0]
            print(f"\n  {'─'*72}")
            print(f"  MEJOR MODELO: {mejor['Modelo']}")
            print(f"  {'─'*72}")
            print(f"  R²={mejor['R2']:.4f}  MAPE={mejor['MAPE %']:.1f}%  "
                  f"MAE={mejor['MAE']:.2f} {u}  RMSE={mejor['RMSE']:.2f}  "
                  f"Gap={mejor['Gap_%']:.1f}%")

            # Justificación vs segundo y tercero
            if len(ranking) >= 2:
                seg = ranking[1]
                print(f"\n  ¿POR QUÉ ESTE MODELO?")
                r2_diff = mejor["R2"] - seg["R2"]
                mape_diff = seg["MAPE %"] - mejor["MAPE %"]
                mae_diff = seg["MAE"] - mejor["MAE"]
                print(f"  vs #{2} {seg['Modelo']}:")
                print(f"    R² {'+' if r2_diff>=0 else ''}{r2_diff:.4f}  |  "
                      f"MAPE {'+' if mape_diff>=0 else ''}{mape_diff:.1f}pp  |  "
                      f"MAE {'+' if mae_diff>=0 else ''}{mae_diff:.2f} {u}")
            if len(ranking) >= 3:
                ter = ranking[2]
                r2_diff = mejor["R2"] - ter["R2"]
                mape_diff = ter["MAPE %"] - mejor["MAPE %"]
                mae_diff = ter["MAE"] - mejor["MAE"]
                print(f"  vs #{3} {ter['Modelo']}:")
                print(f"    R² {'+' if r2_diff>=0 else ''}{r2_diff:.4f}  |  "
                      f"MAPE {'+' if mape_diff>=0 else ''}{mape_diff:.1f}pp  |  "
                      f"MAE {'+' if mae_diff>=0 else ''}{mae_diff:.2f} {u}")

        # Modelos con overfitting
        ovf_modelos = [o for o in overfitting_info
                       if o["Diagnostico"] == "OVERFITTING"]
        if ovf_modelos:
            print(f"\n  [!] Modelos con overfitting ({len(ovf_modelos)}):")
            for o in ovf_modelos:
                print(f"    • {o['Modelo']:18s} Gap={o['Gap_%']:>5.1f}% "
                      f"-> {o['Recomendacion']}")

        todos[t["slug"]] = {
            "cfg": t,
            "resultados": resultados,
            "overfitting": overfitting_info,
            "df": df_res,
            "X_sel": X_sel,
        }
        print()

    # -- CSV consolidado --
    frames = []
    for slug, data in todos.items():
        df_tmp = data["df"].copy()
        df_tmp.insert(0, "Target", data["cfg"]["nombre"])
        ovf_map = {o["Modelo"]: o["Diagnostico"]
                   for o in data["overfitting"]}
        df_tmp["Overfitting"] = df_tmp["Modelo"].map(ovf_map)
        frames.append(df_tmp)
    df_all = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(CARPETA_METRICAS, "comparativa_general.csv")
    df_all.to_csv(csv_path, index=False)

    # -- Ranking multi-target --
    print(f"\n{'=' * 80}")
    print(f"  RANKING MULTI-TARGET")
    print(f"{'=' * 80}")
    print(f"  {'Target':<22} {'Mejor Modelo':<18} {'R²':>7} {'MAPE%':>7} {'RMSE':>9} {'MAE':>9} {'Gap%':>7}")
    print(f"  {'─'*80}")
    for slug in ["volumen", "eje_corto", "eje_largo"]:
        if slug not in todos:
            continue
        data = todos[slug]
        cfg = data["cfg"]
        validos = [r for r in data["resultados"] if not np.isnan(r.get("MAE", np.nan))]
        if validos:
            mejor = min(validos, key=lambda r: r["MAE"])
            print(f"  {cfg['nombre']:<22} {mejor['Modelo']:<18} "
                  f"{mejor['R2']:>7.4f} {mejor['MAPE %']:>6.1f}% "
                  f"{mejor['RMSE']:>9.2f} {mejor['MAE']:>9.2f} "
                  f"{mejor['Gap_%']:>6.1f}%")

    # Modelo más consistente
    conteo_ganador = {}
    for slug in ["volumen", "eje_corto", "eje_largo"]:
        if slug not in todos:
            continue
        validos = [r for r in todos[slug]["resultados"] if not np.isnan(r.get("MAE", np.nan))]
        if validos:
            mejor = min(validos, key=lambda r: r["MAE"])
            nombre = mejor["Modelo"]
            conteo_ganador[nombre] = conteo_ganador.get(nombre, 0) + 1
    if conteo_ganador:
        mas_consistente = max(conteo_ganador, key=conteo_ganador.get)
        n_wins = conteo_ganador[mas_consistente]
        print(f"\n  Modelo más consistente: {mas_consistente} (mejor en {n_wins}/3 targets)")

    t_total = time.time() - t_inicio
    mins = int(t_total // 60)
    segs = t_total % 60
    print(f"\n  Tiempo total: {mins}m {segs:.0f}s")
    print(f"  Resultados: metrics/comparativa_general.csv")
    print(f"{'=' * 80}")

    # Guardar X y df para uso en visualizaciones
    todos["_X"] = X_all
    todos["_df"] = df

    # Datos clínicos para gráfica demográfica
    cols_clin_presentes = [c for c in COLS_CLINICAS if c in df.columns]
    if cols_clin_presentes:
        todos["_df_clinico"] = df[["Paciente_ID"] + cols_clin_presentes].copy()

    return todos


# ==============================================================
#  FASE 2: VISUALIZACIÓN
# ==============================================================

def generar_graficas(todos):
    """
    Crea 7 figuras resumen:
      1. R² comparativo (barras agrupadas)
      2. Heatmap mejor modelo por target
      3. Análisis de overfitting (Gap% por modelo)
      4. Feature importance (top 15)
      5. Curvas de aprendizaje
      6. Scatter: predicción vs valor real (mejor modelo por target)
      7. Distribución demográfica (sexo, región anatómica, diagnóstico)
    """
    plt.close("all")

    X_all = todos.pop("_X")
    df = todos.pop("_df")
    df_clinico = todos.pop("_df_clinico", None)

    slugs = [s for s in todos.keys()]
    n_tgt = len(slugs)

    # ==========================================================
    #  GRÁFICA 1: R² COMPARATIVO
    # ==========================================================
    modelos_set = []
    for slug in slugs:
        for r in todos[slug]["resultados"]:
            if (r["Modelo"] not in modelos_set
                    and not np.isnan(r.get("R2", np.nan))):
                modelos_set.append(r["Modelo"])

    n_mod = len(modelos_set)
    y_pos = np.arange(n_mod)
    altura = 0.8 / n_tgt
    paleta = ["#3498db", "#e74c3c", "#2ecc71"]

    fig1, ax1 = plt.subplots(figsize=(13, max(6, n_mod * 0.55)))
    for i, slug in enumerate(slugs):
        cfg = todos[slug]["cfg"]
        r2_map = {r["Modelo"]: r["R2"] for r in todos[slug]["resultados"]
                  if not np.isnan(r.get("R2", np.nan))}
        vals = [r2_map.get(m, np.nan) for m in modelos_set]
        offset = (i - n_tgt / 2 + 0.5) * altura
        bars = ax1.barh(y_pos + offset, vals, height=altura * 0.9,
                        label=f"{cfg['nombre']} ({cfg['u']})",
                        color=paleta[i % len(paleta)],
                        edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                off = 0.02 if val >= 0 else -0.02
                ax1.text(val + off, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center",
                         ha="left" if val >= 0 else "right",
                         fontsize=7.5, fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(modelos_set, fontsize=9)
    ax1.axvline(x=0, color="#2c3e50", linewidth=1, alpha=0.8)
    ax1.set_xlabel("R²", fontsize=11)
    ax1.set_title(f"R² por Modelo y Target - {N_SPLITS}-Fold CV x "
                  f"{N_REPEATS} rep + Feature Selection + Log-Transform\n"
                  "1.0 = perfecto  |  0 = media  |  < 0 = peor que la media",
                  fontweight="bold", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9)
    fig1.tight_layout()

    # ==========================================================
    #  GRÁFICA 2: HEATMAP MEJORES MODELOS
    # ==========================================================
    filas = []
    row_labels = []
    cols_heat = ["MAE", "RMSE", "R2", "Pearson_r", "MedAE", "MaxErr",
                 "MAPE %", "EVS", "Gap_%"]

    for slug in slugs:
        cfg = todos[slug]["cfg"]
        validos = [r for r in todos[slug]["resultados"]
                   if not np.isnan(r.get("MAE", np.nan))]
        if not validos:
            continue
        mejor = min(validos, key=lambda r: r["MAE"])
        row_labels.append(f"{cfg['nombre']}\n({mejor['Modelo']})")
        filas.append([mejor.get(c, np.nan) for c in cols_heat])

    df_h = pd.DataFrame(filas, columns=cols_heat, index=row_labels)

    mayor_mejor = ["R2", "EVS", "Pearson_r"]
    df_n = df_h.copy()
    for col in df_n.columns:
        d = df_n[col].dropna()
        if d.empty:
            continue
        r = d.max() - d.min()
        if r == 0:
            df_n[col] = 0.5
        elif col in mayor_mejor:
            df_n[col] = 1 - (d - d.min()) / r
        else:
            df_n[col] = (d - d.min()) / r

    fig2, ax2 = plt.subplots(figsize=(15, 4))
    sns.heatmap(df_n, annot=df_h.values, fmt=".2f", cmap="RdYlGn_r",
                linewidths=2, linecolor="white", ax=ax2,
                cbar_kws={"label": "← Mejor          Peor ->", "shrink": 0.8},
                annot_kws={"size": 10, "fontweight": "bold"})
    ax2.set_title("Mejor Modelo por Target - Métricas + Overfitting\n"
                  "(verde = mejor, rojo = peor)", fontweight="bold",
                  fontsize=12)
    ax2.set_ylabel("")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right",
                        fontsize=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)
    fig2.tight_layout()

    # ==========================================================
    #  GRÁFICA 3: ANÁLISIS DE OVERFITTING (Gap% por modelo)
    # ==========================================================
    fig3, axes3 = plt.subplots(1, n_tgt, figsize=(6 * n_tgt, 6),
                               sharey=False)
    if n_tgt == 1:
        axes3 = [axes3]

    for i, slug in enumerate(slugs):
        ax = axes3[i]
        cfg = todos[slug]["cfg"]
        resultados = todos[slug]["resultados"]

        nombres = []
        train_vals = []
        test_vals = []
        gaps = []

        for r in sorted(resultados, key=lambda x: x.get("MAE", np.inf)):
            if np.isnan(r.get("MAE", np.nan)):
                continue
            nombres.append(r["Modelo"])
            train_vals.append(r.get("Train_MAE", 0))
            test_vals.append(r["MAE"])
            gaps.append(r.get("Gap_%", 0))

        y_bar = np.arange(len(nombres))
        h = 0.35

        ax.barh(y_bar - h / 2, train_vals, h, label="Train MAE",
                color="#2ecc71", alpha=0.8, edgecolor="white")
        ax.barh(y_bar + h / 2, test_vals, h, label="Test MAE",
                color="#e74c3c", alpha=0.8, edgecolor="white")

        # Anotar Gap%
        for j, (tr, te, g) in enumerate(zip(train_vals, test_vals, gaps)):
            x_pos = max(tr, te) + max(test_vals) * 0.02
            color = "#c0392b" if g > OVERFITTING_UMBRAL else (
                "#f39c12" if g > 10 else "#27ae60")
            ax.text(x_pos, j, f"Gap {g:.0f}%", va="center",
                    fontsize=8, fontweight="bold", color=color)

        ax.axvline(x=0, color="gray", linewidth=0.5)
        ax.set_yticks(y_bar)
        ax.set_yticklabels(nombres, fontsize=9)
        ax.set_xlabel(f"MAE ({cfg['u']})", fontsize=10)
        ax.set_title(f"{cfg['nombre']}\n(verde=train, rojo=test)",
                     fontweight="bold", fontsize=11)
        ax.legend(loc="lower right", fontsize=8)

    fig3.suptitle(f"Análisis de Overfitting - "
                  f"Umbral: Gap% > {OVERFITTING_UMBRAL}%",
                  fontweight="bold", fontsize=13, y=1.02)
    fig3.tight_layout()

    # ==========================================================
    #  GRÁFICA 4: FEATURE IMPORTANCE (modelos tree-based)
    # ==========================================================
    tree_names = ["Random Forest", "Gradient Boost", "XGBoost",
                  "Arbol Decision"]

    fig4, axes4 = plt.subplots(1, n_tgt, figsize=(6 * n_tgt, 7))
    if n_tgt == 1:
        axes4 = [axes4]

    for i, slug in enumerate(slugs):
        ax = axes4[i]
        cfg_t = todos[slug]["cfg"]
        X_sel = todos[slug]["X_sel"]
        y = df[cfg_t["col"]].values
        y_fit = np.log1p(y) if slug == "volumen" else y

        # Buscar mejor modelo tree-based
        validos_tree = [r for r in todos[slug]["resultados"]
                        if r["Modelo"] in tree_names
                        and not np.isnan(r.get("MAE", np.nan))]

        if not validos_tree:
            ax.text(0.5, 0.5, "No hay modelos\ntree-based válidos",
                    ha="center", va="center", fontsize=12,
                    transform=ax.transAxes)
            ax.set_title(f"{cfg_t['nombre']}")
            continue

        mejor_tree = min(validos_tree, key=lambda r: r["MAE"])

        # Re-entrenar con pipe/params almacenados
        pipe_tree = clone(mejor_tree["_pipe"])
        params = mejor_tree["_params"]
        feature_cols = mejor_tree.get("_feature_cols",
                                      X_sel.columns.tolist())
        X_usar = (X_sel[feature_cols]
                  if set(feature_cols).issubset(X_sel.columns)
                  else X_sel)

        if params:
            st = mejor_tree.get("_search_type", "grid")
            inner_cv = KFold(n_splits=INNER_CV, shuffle=True,
                             random_state=42)
            if st == "random":
                gs = RandomizedSearchCV(
                    pipe_tree, params,
                    n_iter=mejor_tree.get("_n_iter", 10),
                    cv=inner_cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1, random_state=42)
            else:
                gs = GridSearchCV(
                    pipe_tree, params, cv=inner_cv,
                    scoring="neg_mean_absolute_error", n_jobs=1)
            gs.fit(X_usar, y_fit)
            final_model = gs.best_estimator_
        else:
            pipe_tree.fit(X_usar, y_fit)
            final_model = pipe_tree

        modelo_interno = final_model.named_steps["m"]
        if hasattr(modelo_interno, "feature_importances_"):
            importancias = pd.Series(
                modelo_interno.feature_importances_,
                index=X_usar.columns
            ).sort_values(ascending=True)

            top15 = importancias.tail(15)
            colores = ["#3498db" if f.startswith("firstorder_") else "#e67e22"
                       for f in top15.index]
            labels = [f.replace("firstorder_", "fo_").replace("glcm_", "")
                      for f in top15.index]
            top15.index = labels
            top15.plot.barh(ax=ax, color=colores, edgecolor="white")
            ax.set_xlabel("Importancia", fontsize=10)
            ax.set_title(f"{cfg_t['nombre']}\n{mejor_tree['Modelo']} "
                         f"(R²={mejor_tree['R2']:.3f})",
                         fontweight="bold", fontsize=11)

            legend_elements = [
                Patch(facecolor="#3498db", label="First Order"),
                Patch(facecolor="#e67e22", label="GLCM"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
        else:
            ax.text(0.5, 0.5,
                    f"{mejor_tree['Modelo']}\nno tiene\nfeature_importances_",
                    ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            ax.set_title(f"{cfg_t['nombre']}")

    fig4.suptitle("Feature Importance - Top 15 Features por Target",
                  fontweight="bold", fontsize=13, y=1.02)
    fig4.tight_layout()

    # ==========================================================
    #  GRÁFICA 5: CURVAS DE APRENDIZAJE
    # ==========================================================
    fig5, axes5 = plt.subplots(1, n_tgt, figsize=(6 * n_tgt, 5))
    if n_tgt == 1:
        axes5 = [axes5]

    for i, slug in enumerate(slugs):
        ax = axes5[i]
        cfg_t = todos[slug]["cfg"]
        X_sel = todos[slug]["X_sel"]
        y = df[cfg_t["col"]].values
        y_lc = np.log1p(y) if slug == "volumen" else y

        validos = [r for r in todos[slug]["resultados"]
                   if not np.isnan(r.get("MAE", np.nan))]
        if not validos:
            continue
        mejor = min(validos, key=lambda r: r["MAE"])

        feature_cols = mejor.get("_feature_cols", X_sel.columns.tolist())
        X_lc = (X_sel[feature_cols]
                if set(feature_cols).issubset(X_sel.columns)
                else X_sel)
        pipe_lc = clone(mejor["_pipe"])

        try:
            train_sizes = np.linspace(0.2, 1.0, 6)
            train_sizes_abs, train_scores, test_scores = learning_curve(
                pipe_lc, X_lc, y_lc,
                train_sizes=train_sizes,
                cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=42),
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            )

            train_mae = -train_scores.mean(axis=1)
            test_mae = -test_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            test_std = test_scores.std(axis=1)

            ax.plot(train_sizes_abs, train_mae, "o-", label="Train MAE",
                    linewidth=2, color="#2ecc71", markersize=5)
            ax.plot(train_sizes_abs, test_mae, "s-", label="Test MAE",
                    linewidth=2, color="#e74c3c", markersize=5)

            ax.fill_between(train_sizes_abs,
                            train_mae - train_std, train_mae + train_std,
                            alpha=0.1, color="#2ecc71")
            ax.fill_between(train_sizes_abs,
                            test_mae - test_std, test_mae + test_std,
                            alpha=0.1, color="#e74c3c")

            y_label = ("MAE (log-mm³)" if slug == "volumen"
                       else f"MAE ({cfg_t['u']})")
            ax.set_xlabel("Tamaño del Dataset de Entrenamiento", fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f"{cfg_t['nombre']}\n{mejor['Modelo']}",
                         fontweight="bold", fontsize=11)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}",
                    ha="center", va="center", fontsize=9,
                    transform=ax.transAxes)
            ax.set_title(f"{cfg_t['nombre']}\n{mejor['Modelo']}")

    fig5.suptitle("Curvas de Aprendizaje - ¿Más datos mejorarían el modelo?",
                  fontweight="bold", fontsize=13, y=1.02)
    fig5.tight_layout()

    # ==========================================================
    #  GRÁFICA 6: SCATTER PREDICCIÓN vs VALOR REAL
    # ==========================================================
    fig6, axes6 = plt.subplots(1, n_tgt, figsize=(6 * n_tgt, 5))
    if n_tgt == 1:
        axes6 = [axes6]

    for i, slug in enumerate(slugs):
        ax = axes6[i]
        cfg_t = todos[slug]["cfg"]
        y_true = df[cfg_t["col"]].values

        validos = [r for r in todos[slug]["resultados"]
                   if not np.isnan(r.get("MAE", np.nan))]
        if not validos:
            continue
        mejor = min(validos, key=lambda r: r["MAE"])
        y_pred = mejor.get("_y_pred")
        if y_pred is None:
            continue

        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="white",
                   linewidth=0.5, s=40, color="#3498db")

        # Línea identidad
        all_vals = np.concatenate([y_true, y_pred])
        min_val, max_val = all_vals.min(), all_vals.max()
        margin = (max_val - min_val) * 0.05
        lims = [min_val - margin, max_val + margin]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="Identidad")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Anotaciones
        r2_val = mejor.get("R2", np.nan)
        pr_val = mejor.get("Pearson_r", np.nan)
        ax.text(0.05, 0.92,
                f"R²={r2_val:.3f}\nr={pr_val:.3f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8))

        ax.set_xlabel(f"Valor Real ({cfg_t['u']})", fontsize=10)
        ax.set_ylabel(f"Predicción ({cfg_t['u']})", fontsize=10)
        ax.set_title(f"{cfg_t['nombre']}\n{mejor['Modelo']}",
                     fontweight="bold", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    fig6.suptitle("Predicción vs Valor Real - Mejor Modelo por Target",
                  fontweight="bold", fontsize=13, y=1.02)
    fig6.tight_layout()

    # ==========================================================
    #  GRÁFICA 7: DISTRIBUCIÓN DEMOGRÁFICA
    # ==========================================================
    if df_clinico is not None and not df_clinico.empty:
        n_paneles = sum([
            "patient_sex" in df_clinico.columns,
            "body_part_examined" in df_clinico.columns,
            "primary_condition" in df_clinico.columns,
        ])
        if n_paneles > 0:
            fig7, axes7 = plt.subplots(1, n_paneles,
                                       figsize=(6 * n_paneles, 5))
            if n_paneles == 1:
                axes7 = [axes7]

            idx_ax = 0
            paleta_demo = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
                           "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
                           "#d35400", "#c0392b", "#7f8c8d"]

            # (a) Sexo del paciente
            if "patient_sex" in df_clinico.columns:
                ax = axes7[idx_ax]
                conteo = df_clinico["patient_sex"].value_counts()
                bars = ax.bar(conteo.index, conteo.values,
                              color=paleta_demo[:len(conteo)],
                              edgecolor="white", width=0.5)
                for bar, val in zip(bars, conteo.values):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                            str(val), ha="center", fontweight="bold",
                            fontsize=11)
                ax.set_title("Sexo del Paciente", fontweight="bold",
                             fontsize=11)
                ax.set_ylabel("Cantidad", fontsize=10)
                ax.set_xlabel("")
                idx_ax += 1

            # (b) Región anatómica
            if "body_part_examined" in df_clinico.columns:
                ax = axes7[idx_ax]
                conteo = df_clinico["body_part_examined"].value_counts()
                bars = ax.bar(conteo.index, conteo.values,
                              color=paleta_demo[:len(conteo)],
                              edgecolor="white", width=0.5)
                for bar, val in zip(bars, conteo.values):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                            str(val), ha="center", fontweight="bold",
                            fontsize=11)
                ax.set_title("Región Anatómica", fontweight="bold",
                             fontsize=11)
                ax.set_ylabel("Cantidad", fontsize=10)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=15)
                idx_ax += 1

            # (c) Diagnóstico primario (top 10 + Otros)
            if "primary_condition" in df_clinico.columns:
                ax = axes7[idx_ax]
                conteo_full = df_clinico["primary_condition"].value_counts()
                if len(conteo_full) > 10:
                    top10 = conteo_full.head(10)
                    otros = conteo_full.iloc[10:].sum()
                    conteo = pd.concat([top10, pd.Series({"Otros": otros})])
                else:
                    conteo = conteo_full
                bars = ax.barh(conteo.index[::-1], conteo.values[::-1],
                               color=paleta_demo[:len(conteo)],
                               edgecolor="white", height=0.6)
                for bar, val in zip(bars, conteo.values[::-1]):
                    ax.text(val + 0.2,
                            bar.get_y() + bar.get_height() / 2,
                            str(val), va="center", fontweight="bold",
                            fontsize=9)
                ax.set_title("Diagnóstico Primario", fontweight="bold",
                             fontsize=11)
                ax.set_xlabel("Cantidad", fontsize=10)
                idx_ax += 1

            n_total = len(df_clinico)
            fig7.suptitle(
                f"Distribución Demográfica del Dataset (n={n_total})",
                fontweight="bold", fontsize=13, y=1.02)
            fig7.tight_layout()

    # Guardar todas las figuras en regression/metrics/
    metrics_dir = CARPETA_METRICAS
    os.makedirs(metrics_dir, exist_ok=True)
    for i, fig in enumerate(plt.get_fignums(), 1):
        f = plt.figure(fig)
        ruta = os.path.join(metrics_dir, f"grafica_{i}.png")
        f.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close("all")
    n_figs = len([f for f in os.listdir(metrics_dir)
                  if f.startswith("grafica_") and f.endswith(".png")])
    print(f"\n  [OK] {n_figs} gráficas guardadas en metrics/grafica_1..{n_figs}.png")


# ==============================================================
#  GLOSARIO
# ==============================================================

def imprimir_glosario():
    print("\nGLOSARIO DE MÉTRICAS:")
    print("-" * 80)
    metricas = [
        ("R²",        "Varianza explicada: 1=perfecto, >0=bueno, <=0=malo"),
        ("Pearson_r", "Correlación lineal Pearson: 1=perfecto, 0=sin relación"),
        ("MAE",       "Error absoluto medio (mismas unidades que el target)"),
        ("RMSE",      "Raíz del error cuadrático medio (penaliza outliers)"),
        ("MAPE",      "Error porcentual medio (%)"),
        ("Gap%",      f"(Test-Train)/Testx100: >{OVERFITTING_UMBRAL}%=overfitting"),
        ("Train_MAE", "MAE sobre datos de entrenamiento"),
        ("MedAE",     "Error absoluto mediano (robusto a outliers)"),
        ("MaxErr",    "Error máximo (peor caso)"),
        ("EVS",       "Explained Variance Score (similar a R²)"),
    ]
    for metrica, desc in metricas:
        print(f"  {metrica:10s} -> {desc}")
    print()
    print("DIAGNÓSTICOS DE OVERFITTING:")
    print("-" * 80)
    print(f"  [OK] OK           -> Gap% <= {OVERFITTING_UMBRAL}% y >= -10%")
    print(f"  [!] OVERFITTING  -> Gap% > {OVERFITTING_UMBRAL}% (modelo memoriza train)")
    print(f"  [v] UNDERFITTING -> Gap% < -10% (modelo muy simple)")
    print()


# ==============================================================
if __name__ == "__main__":
    resultados = ejecutar_evaluacion()
    imprimir_glosario()
    generar_graficas(resultados)
