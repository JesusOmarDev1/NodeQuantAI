"""
entrenar_modelo.py - Modelado predictivo con Stacking Ensemble
==============================================================
Estimacion de volumen tumoral y diametros axiales de ganglios
linfaticos mediastinicos mediante features radiomicas.

Modelo: StackingRegressor (XGBoost + Random Forest + Gradient Boosting)
        Meta-learner: Ridge | StandardScaler + log1p transform
Features: radiomicas (firstorder + glcm) + derivadas - shape_* eliminadas

Validacion:
  - 10-Fold CV (out-of-fold predictions)
  - Deteccion de overfitting (Gap% train vs test)
  - Verificacion con casos de prueba independientes

Graficas (8 PNGs en regression/metrics/):
  1. Panel de rendimiento (R2 como gauge)
  2. Tasa de acierto por umbral de error
  3. Scatter real vs predicho con bandas de error
  4. Distribucion del error (histogramas)
  5. Top 20 pacientes con mayor error
  6. Metricas de validacion y Gap%
  7. Feature importance (top 10)
  8. Verificacion en casos de prueba
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, max_error,
    explained_variance_score, mean_absolute_percentage_error,
)
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor

from regression.scripts.optimizacion_regression import detectar_overfitting, seleccionar_features_rfecv

warnings.filterwarnings("ignore")

# Generar joblib
import json
import joblib

# ---------------------------------------------------------------------------
#  Rutas
# ---------------------------------------------------------------------------

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_master.csv")
RUTA_PRUEBA = os.path.join(base_dir, "db", "casos_prueba.csv")
CARPETA_METRICAS = os.path.join(base_dir, "regression", "metrics")
CARPETA_NIFIT = os.path.join(base_dir, "Dataset_NIFIT")
os.makedirs(CARPETA_METRICAS, exist_ok=True)


# ---------------------------------------------------------------------------
#  Configuracion
# ---------------------------------------------------------------------------
N_SPLITS = 10
N_REPEATS = 1
OVERFITTING_UMBRAL = 15.0
CORR_UMBRAL = 0.03
INTER_CORR_UMBRAL = 0.95

COLS_CLINICAS = ["Body Part Examined", "PatientSex", "PrimaryCondition"]

TARGETS = [
    {"nombre": "Volumen Tumoral",    "col": "target_regresion",
     "origen": "shape_VoxelVolume",  "u": "mm\u00b3", "slug": "volumen"},
    {"nombre": "Diametro Eje Corto", "col": "target_eje_corto",
     "origen": "shape_MinorAxisLength", "u": "mm", "slug": "eje_corto"},
    {"nombre": "Diametro Eje Largo", "col": "target_eje_largo",
     "origen": "shape_MajorAxisLength", "u": "mm", "slug": "eje_largo"},
]
COLS_TARGET = [t["col"] for t in TARGETS]

# Colores globales para graficas
_COLORES_TARGET = ["#3498db", "#e74c3c", "#2ecc71"]
_NOMBRES_CORTOS = {
    "Volumen Tumoral": "Volumen",
    "Diametro Eje Corto": "Eje Corto",
    "Diametro Eje Largo": "Eje Largo",
}

# ---------------------------------------------------------------------------
#  Estilo global matplotlib (tipo poster)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "grid.color": "#D0D0D0",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ===========================================================================
#  Stacking Pipeline
# ===========================================================================
def _crear_stacking_pipeline():
    """
    Retorna (Pipeline, nombre_modelo) con StackingRegressor.
    Base estimators: XGBoost + Random Forest + Gradient Boosting.
    Meta-learner: Ridge(alpha=10).
    Filtro previo: LASSO para eliminar colinealidad.
    """
    estimators = [
        ("xgb", XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            reg_alpha=5, reg_lambda=10, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0)),
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=7,
            min_samples_leaf=4, max_features=0.8,
            random_state=42)),
        ("gb", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5,
            random_state=42)),
    ]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # --- NUEVO: Filtro LASSO ---
        # alpha=0.01 es un buen punto de partida para que no elimine demasiadas
        ("lasso_filter", SelectFromModel(Lasso(alpha=0.01, random_state=42, max_iter=10000))), 
        ("model", StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=10.0),
            cv=5,
            n_jobs=None, # Probar None en lugar de -1 si la compu se congela o tarda mucho
        )),
    ])
    return pipe, "Stacking (LASSO + XGB+RF+GB)"


def _extraer_importances_stacking(pipe, feature_names):
    """Importances ponderadas: base estimators x coef Ridge."""
    
    # 1. Identificar qué variables sobrevivieron al filtro LASSO
    lasso_step = pipe.named_steps["lasso_filter"]
    surviving_features = np.array(feature_names)[lasso_step.get_support()]

    # 2. Extraer los coeficientes del Stacking solo para las variables sobrevivientes
    stacking = pipe.named_steps["model"]
    coefs = stacking.final_estimator_.coef_
    importances = np.zeros(len(surviving_features))

    for est, coef in zip(stacking.estimators_, coefs):
        if hasattr(est, "feature_importances_"):
            importances += est.feature_importances_ * abs(coef)

    total = importances.sum()
    if total > 0:
        importances /= total

    return dict(zip(surviving_features, importances))


# ===========================================================================
#  Feature engineering
# ===========================================================================
def crear_features_derivadas(X):
    """
    Genera ~25 features derivadas a partir de las 42 features originales:
      - Ratios entre pares informativos
      - Transformaciones log, cuadrado, raiz cubica
      - Diferencias e interacciones cruzadas
      - Coeficiente de variacion y rango inter-percentil
    """
    Xd = X.copy()

    # Ratios informativos (denominador protegido contra division por cero)
    _eps = 1e-9
    if "firstorder_Energy" in X.columns and "firstorder_Entropy" in X.columns:
        Xd["ratio_Energy_Entropy"] = X["firstorder_Energy"] / (X["firstorder_Entropy"].abs() + _eps)
    if "firstorder_Mean" in X.columns and "firstorder_Variance" in X.columns:
        Xd["ratio_Mean_Variance"] = X["firstorder_Mean"] / (X["firstorder_Variance"].abs() + _eps)
    if "firstorder_90Percentile" in X.columns and "firstorder_10Percentile" in X.columns:
        Xd["ratio_90p_10p"] = X["firstorder_90Percentile"] / (X["firstorder_10Percentile"].abs() + _eps)
    if "firstorder_Median" in X.columns and "firstorder_Mean" in X.columns:
        Xd["ratio_Median_Mean"] = X["firstorder_Median"] / (X["firstorder_Mean"].abs() + _eps)
    if "glcm_Correlation" in X.columns and "glcm_Contrast" in X.columns:
        Xd["ratio_Corr_Contrast"] = X["glcm_Correlation"] / (X["glcm_Contrast"].abs() + _eps)
    if "glcm_Homogeneity1" in X.columns and "glcm_Contrast" in X.columns:
        Xd["ratio_Homog_Contrast"] = X["glcm_Homogeneity1"] / (X["glcm_Contrast"].abs() + _eps)
    if "glcm_JointEnergy" in X.columns and "glcm_JointEntropy" in X.columns:
        Xd["ratio_JE_JEntropy"] = X["glcm_JointEnergy"] / (X["glcm_JointEntropy"].abs() + _eps)
    if "glcm_SumAverage" in X.columns and "glcm_SumEntropy" in X.columns:
        Xd["ratio_SumAvg_SumEnt"] = X["glcm_SumAverage"] / (X["glcm_SumEntropy"].abs() + _eps)

    # Coeficiente de variacion
    if "firstorder_Variance" in X.columns and "firstorder_Mean" in X.columns:
        Xd["cv_intensidad"] = np.sqrt(X["firstorder_Variance"].abs()) / (X["firstorder_Mean"].abs() + _eps)

    # Rango inter-percentil
    if "firstorder_90Percentile" in X.columns and "firstorder_10Percentile" in X.columns:
        Xd["rango_interpercentil"] = X["firstorder_90Percentile"] - X["firstorder_10Percentile"]

    # Log transforms
    if "firstorder_Energy" in X.columns:
        Xd["log_Energy"] = np.log1p(X["firstorder_Energy"].abs())
    if "firstorder_TotalEnergy" in X.columns:
        Xd["log_TotalEnergy"] = np.log1p(X["firstorder_TotalEnergy"].abs())
    if "firstorder_Variance" in X.columns:
        Xd["log_Variance"] = np.log1p(X["firstorder_Variance"].abs())
    if "firstorder_Range" in X.columns:
        Xd["log_Range"] = np.log1p(X["firstorder_Range"].abs())

    # Squared
    if "firstorder_Energy" in X.columns:
        Xd["sq_Energy"] = X["firstorder_Energy"] ** 2
    if "firstorder_RootMeanSquared" in X.columns:
        Xd["sq_RMS"] = X["firstorder_RootMeanSquared"] ** 2
    if "glcm_Autocorrelation" in X.columns:
        Xd["sq_Autocorrelation"] = X["glcm_Autocorrelation"] ** 2
    if "glcm_SumAverage" in X.columns:
        Xd["sq_SumAverage"] = X["glcm_SumAverage"] ** 2

    # Cube roots
    if "firstorder_TotalEnergy" in X.columns:
        Xd["cbrt_TotalEnergy"] = np.cbrt(X["firstorder_TotalEnergy"])
    if "firstorder_Energy" in X.columns:
        Xd["cbrt_Energy"] = np.cbrt(X["firstorder_Energy"])

    # Diferencias
    if "firstorder_Median" in X.columns and "firstorder_10Percentile" in X.columns:
        Xd["diff_Median_10p"] = X["firstorder_Median"] - X["firstorder_10Percentile"]
    if "firstorder_Maximum" in X.columns and "firstorder_Minimum" in X.columns:
        Xd["diff_Max_Min"] = X["firstorder_Maximum"] - X["firstorder_Minimum"]

    # Interacciones cruzadas
    if "firstorder_Energy" in X.columns and "glcm_JointEnergy" in X.columns:
        Xd["inter_Energy_JointEnergy"] = X["firstorder_Energy"] * X["glcm_JointEnergy"]
    if "firstorder_TotalEnergy" in X.columns and "glcm_Autocorrelation" in X.columns:
        Xd["inter_TotalEnergy_Autocorr"] = X["firstorder_TotalEnergy"] * X["glcm_Autocorrelation"]
    if "firstorder_RootMeanSquared" in X.columns and "glcm_SumAverage" in X.columns:
        Xd["inter_RMS_SumAvg"] = X["firstorder_RootMeanSquared"] * X["glcm_SumAverage"]
    if "firstorder_Entropy" in X.columns and "glcm_JointEntropy" in X.columns:
        Xd["inter_Entropy_JEntropy"] = X["firstorder_Entropy"] * X["glcm_JointEntropy"]

    # Reemplazar inf/NaN
    Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0)

    return Xd


# ===========================================================================
#  Feature selection (por target)
# ===========================================================================
def seleccionar_features(X, y_target, usar_rfecv=True):
    """
    Selecciona features en tres pasos:
      1. Filtro grueso (Pearson + MI) para descartar ruido total.
      2. Poda de Colinealidad (INTER_CORR_UMBRAL) para eliminar variables gemelas.
      3. RFECV para encontrar el subset óptimo.
    """
    # --- PASO 1: FILTRO GRUESO (Ruido) ---
    corrs = X.corrwith(pd.Series(y_target, index=X.index)).abs().fillna(0)
    cols_corr = set(corrs[corrs > CORR_UMBRAL].index)

    mi = mutual_info_regression(X, y_target, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=X.columns)
    mi_umbral = np.percentile(mi_series.values, 20)
    cols_mi = set(mi_series[mi_series > mi_umbral].index)

    cols_ok = list(cols_corr | cols_mi)
    if not cols_ok:
        cols_ok = list(X.columns)
        
    X_filt = X[cols_ok].copy()

    # --- PASO 2: PODA DE COLINEALIDAD (El paso perdido) ---
    # Calculamos la matriz de correlación de las características sobrevivientes
    corr_matrix = X_filt.corr().abs()
    # Tomamos solo el triángulo superior de la matriz para no borrar ambas variables
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Encontramos las columnas con correlación mayor al 95%
    to_drop = [column for column in upper.columns if any(upper[column] > INTER_CORR_UMBRAL)]
    X_filt = X_filt.drop(columns=to_drop)
    
    cols_ok = list(X_filt.columns)

    # --- PASO 3: RFECV (EL QUIRÓFANO) ---
    if usar_rfecv and len(cols_ok) > 5:
        resultado_rfecv = seleccionar_features_rfecv(X_filt, y_target, cv=3, min_features=5)
        cols_finales = resultado_rfecv["columnas_seleccionadas"]
        return cols_finales
    else:
        return cols_ok


# ===========================================================================
#  Preparacion de datos
# ===========================================================================
def preparar_datos(df_raw):
    """
    Deduplica, crea targets desde shape_*, elimina shape_* (data leakage).
    Returns: X, dict_targets, ids, info
    """
    n_raw = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset="Paciente_ID")
    n_dedup = len(df_raw)
    df_raw = df_raw.reset_index(drop=True)

    df = df_raw.copy()
    for t in TARGETS:
        df[t["col"]] = df_raw[t["origen"]]

    shape_cols = [c for c in df.columns if c.startswith("shape_")]
    df = df.drop(columns=shape_cols)

    cols_drop = ["Paciente_ID", "target_riesgo"] + COLS_TARGET + COLS_CLINICAS
    X = df.drop(columns=[c for c in cols_drop if c in df.columns])

    ids = df_raw["Paciente_ID"].values
    targets = {t["col"]: df[t["col"]].values for t in TARGETS}

    # Desglose por familia radiómica
    familias = {
        "firstorder": len([c for c in X.columns if c.startswith("firstorder_")]),
        "glcm": len([c for c in X.columns if c.startswith("glcm_")]),
        "glrlm": len([c for c in X.columns if c.startswith("glrlm_")]),
        "glszm": len([c for c in X.columns if c.startswith("glszm_")]),
        "gldm": len([c for c in X.columns if c.startswith("gldm_")]),
    }

    info = {
        "n_raw": n_raw,
        "n_muestras": n_dedup,
        "n_features": X.shape[1],
        "shape_eliminadas": len(shape_cols),
        "familias": familias,
    }
    return X, targets, ids, info


def calcular_metricas(y_real, y_pred):
    """8 metricas de regresion."""
    mse = mean_squared_error(y_real, y_pred)
    return {
        "MAE":    round(mean_absolute_error(y_real, y_pred), 4),
        "MSE":    round(mse, 4),
        "RMSE":   round(np.sqrt(mse), 4),
        "R2":     round(r2_score(y_real, y_pred), 4),
        "MedAE":  round(median_absolute_error(y_real, y_pred), 4),
        "MaxErr": round(max_error(y_real, y_pred), 4),
        "MAPE %": round(mean_absolute_percentage_error(y_real, y_pred) * 100, 2),
        "EVS":    round(explained_variance_score(y_real, y_pred), 4),
    }


def categorizar_nivel(valor, y_total):
    """Clasifica en Bajo/Moderado/Medio-Alto/Alto por cuartiles."""
    q1, q2, q3 = np.percentile(y_total, [25, 50, 75])
    if valor <= q1:
        return "Bajo"
    if valor <= q2:
        return "Moderado"
    if valor <= q3:
        return "Medio-Alto"
    return "Crítico"


# ===========================================================================
#  Entrenamiento y evaluacion
# ===========================================================================
def entrenar_y_evaluar():
    """
    Entrena StackingRegressor por target con 10-Fold CV.
    Returns: df_pred, resultados_globales, informacion_modelos
    """
    df_raw = pd.read_csv(RUTA_CSV)
    X_all, targets_dict, ids, info = preparar_datos(df_raw)

    # Feature engineering: agregar ratios e interacciones
    n_orig = X_all.shape[1]
    X_all = crear_features_derivadas(X_all)
    n_derivadas = X_all.shape[1] - n_orig

    N = info["n_muestras"]

    fam = info.get("familias", {})
    fam_str = " + ".join(f"{v} {k}" for k, v in fam.items() if v > 0)

    print(f"\nMODELADO PREDICTIVO")
    print(f"  {info['n_raw']} filas -> {N} unicas")
    print(f"  {info['n_features']} features base ({fam_str})")
    print(f"  + {n_derivadas} derivadas = {X_all.shape[1]} total")
    print(f"  shape eliminadas: {info['shape_eliminadas']}")
    print(f"  Validacion: {N_SPLITS}-Fold CV x{N_REPEATS}")

    resultados_globales = []
    predicciones_totales = []
    informacion_modelos = {}
    t_inicio = time.time()

    for t in TARGETS:
        slug = t["slug"]
        y_orig = targets_dict[t["col"]]
        y_train_space = np.log1p(y_orig)
        u = t["u"]

        pipe_tmpl, nombre_modelo = _crear_stacking_pipeline()

        print(f"\n  {t['nombre']} ({u}) | {nombre_modelo}")
        print(f"    Rango: [{y_orig.min():.1f}, {y_orig.max():.1f}]  Media: {y_orig.mean():.1f}")

        # ---------------------------------------------------------------
        #  KFold CV (out-of-fold predictions) SIN DATA LEAKAGE
        # ---------------------------------------------------------------
        all_y_pred_kf = np.zeros(N)
        all_y_count_kf = np.zeros(N)
        kf_train_maes = []
        kf_test_maes = []
        features_seleccionadas_por_fold = [] # Para rastrear cuántas selecciona

        for rep in range(N_REPEATS):
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42 + rep)
            # Fíjate que ahora dividimos X_all, no la X filtrada
            for train_idx, test_idx in kf.split(X_all): 
                X_tr_completo, X_te_completo = X_all.iloc[train_idx], X_all.iloc[test_idx]
                y_tr_space = y_train_space[train_idx]
                #y_tr_orig = y_orig[train_idx]
                y_te_orig = y_orig[test_idx]

                # 1. SELECCIÓN ESTRICTA: Solo vemos los datos de entrenamiento
                cols_sel_fold = seleccionar_features(X_tr_completo, y_tr_space)
                features_seleccionadas_por_fold.append(len(cols_sel_fold))

                # 2. FILTRADO: Aplicamos las columnas ganadoras al train y al test
                X_tr = X_tr_completo[cols_sel_fold]
                X_te = X_te_completo[cols_sel_fold]

                # 3. ENTRENAMIENTO Y AFINACIÓN: Buscamos los mejores parámetros para el ensamble
                fold_pipe = clone(pipe_tmpl)
                
                # Definimos la red de hiperparámetros a explorar
                param_grid = {
                    "lasso_filter__estimator__alpha": [0.001, 0.01, 0.1], # Decidir cuánta colinealidad limpiar
                    "model__final_estimator__alpha": [0.1, 1.0, 10.0, 50.0], # Ajuste del juez (Ridge)
                    "model__rf__max_depth": [2,3,4], #[3, 5, 7]              # Ajuste de Random Forest
                    "model__xgb__learning_rate": [0.01, 0.05, 0.1],          # Ajuste de XGBoost
                    "model__xgb__max_depth": [2,3], #[3, 5]
                    "model__gb__learning_rate": [0.01, 0.05, 0.1]            # Ajuste de Gradient Boost
                }
                
                # Usamos RandomizedSearchCV para buscar rápido (solo prueba 5 combinaciones al azar)
                # cv=2 es suficiente aquí porque ya estamos dentro de un K-Fold externo
                search = RandomizedSearchCV(
                    fold_pipe, 
                    param_distributions=param_grid,
                    n_iter=5, 
                    cv=2, 
                    scoring="neg_mean_absolute_error",
                    random_state=42,
                    n_jobs=None
                )
                
                # Entrenamos la búsqueda
                search.fit(X_tr, y_tr_space)
                
                # Nos quedamos con el mejor pipeline encontrado
                fold_pipe = search.best_estimator_

                # 4. PREDICCIÓN CON LÍMITE FÍSICO
                # Evitamos que la inversa del logaritmo genere volúmenes negativos
                pred_tr = np.maximum(0, np.expm1(fold_pipe.predict(X_tr)))
                pred_te = np.maximum(0, np.expm1(fold_pipe.predict(X_te)))

                kf_train_maes.append(mean_absolute_error(np.expm1(y_tr_space), pred_tr))
                kf_test_maes.append(mean_absolute_error(y_te_orig, pred_te))

                all_y_pred_kf[test_idx] += pred_te
                all_y_count_kf[test_idx] += 1

        y_pred_kf = all_y_pred_kf / np.maximum(all_y_count_kf, 1)
        metricas_kf = calcular_metricas(y_orig, y_pred_kf)

        kf_train_mae = np.mean(kf_train_maes)
        kf_test_mae = np.mean(kf_test_maes)
        kf_gap = ((kf_test_mae - kf_train_mae) / kf_test_mae * 100
                   ) if kf_test_mae != 0 else 0
                   
        avg_features = np.mean(features_seleccionadas_por_fold)

        print(f"    KFold  R2={metricas_kf['R2']:.4f}  MAE={metricas_kf['MAE']:.1f}  "
              f"Gap={kf_gap:.1f}%  (Promedio de features: {avg_features:.0f})")

        # ---------------------------------------------------------------
        #  Modelo final (todos los datos) - SOLO PARA PRODUCCIÓN
        # ---------------------------------------------------------------
        # Aquí sí usamos todo X_all porque el modelo final va al dashboard
        cols_sel_final = seleccionar_features(X_all, y_train_space)
        X_final = X_all[cols_sel_final]
        
        # ENTRENAMIENTO Y AFINACIÓN DEL MODELO DE PRODUCCIÓN
        pipe_base_final = clone(pipe_tmpl)
        
        search_final = RandomizedSearchCV(
            pipe_base_final, 
            param_distributions=param_grid,
            n_iter=10, # Usamos 10 iteraciones aquí porque es el modelo definitivo
            cv=3,      # CV de 3 para mayor robustez
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=None
        )
        search_final.fit(X_final, y_train_space)
        pipe_final = search_final.best_estimator_
        
        print(f"    Mejores parámetros finales: {search_final.best_params_}")

        imp_dict = _extraer_importances_stacking(pipe_final, list(X_final.columns))
        top_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]

        ovf = {
            "Train_MAE": round(kf_train_mae, 4),
            "Test_MAE": round(kf_test_mae, 4),
            "Gap_%": round(kf_gap, 2),
            "Diagnostico": "OVERFITTING" if kf_gap > OVERFITTING_UMBRAL else ("UNDERFITTING" if kf_gap < -10 else "OK")
        }

        informacion_modelos[t["nombre"]] = {
            "nombre_modelo": nombre_modelo,
            "slug": slug,
            "features": [f for f, _ in top_features],
            "importances": {f: v for f, v in top_features},
            "X": X_final, "y": y_orig,
            "y_pred_kf": y_pred_kf,
            "metricas_kf": metricas_kf,
            "mejor_pipe": pipe_final,
            "overfitting": ovf,
            "unidad": u,
            "use_log": True,
            "cols_selected": cols_sel_final,
            "kf_gap": round(kf_gap, 2),
        }

        resultados_globales.append({
            "Target": t["nombre"], "Modelo": nombre_modelo,
            "Validacion": "KFold", **metricas_kf,
            "Gap_%": round(kf_gap, 2),
        })

        # Predicciones KFold por paciente
        for i in range(N):
            err_abs = abs(y_orig[i] - y_pred_kf[i])
            err_pct = err_abs / max(abs(y_orig[i]), 1e-9) * 100
            predicciones_totales.append({
                "Paciente_ID": ids[i],
                "Target": t["nombre"],
                "Unidad": u,
                "Real": round(y_orig[i], 2),
                "Nivel_Real": categorizar_nivel(y_orig[i], y_orig),
                "Predicho": round(y_pred_kf[i], 2),
                "Nivel_Predicho": categorizar_nivel(y_pred_kf[i], y_orig),
                "Error": round(err_abs, 2),
                "Error %": round(err_pct, 2),
            })
            
    # CSVs
    df_pred = pd.DataFrame(predicciones_totales)
    df_pred.to_csv(os.path.join(CARPETA_METRICAS, "predicciones_kfold.csv"),
                   index=False)
    df_res = pd.DataFrame(resultados_globales)
    df_res.to_csv(os.path.join(CARPETA_METRICAS, "metricas_modelos_finales.csv"),
                  index=False)

    # Overfitting
    print(f"\n  {'Target':<22} {'Modelo':<22} {'TrainMAE':>9} {'TestMAE':>9} {'Gap%':>7}")
    print(f"  {'-'*71}")
    for t in TARGETS:
        im = informacion_modelos[t["nombre"]]
        ovf = im["overfitting"]
        print(f"  {t['nombre']:<22} {im['nombre_modelo']:<22} "
              f"{ovf['Train_MAE']:>9.1f} {ovf['Test_MAE']:>9.1f} "
              f"{ovf['Gap_%']:>6.1f}%  {ovf['Diagnostico']}")

    t_total = time.time() - t_inicio
    print(f"\n  Tiempo: {t_total:.1f}s | CSVs en regression/metrics/")

    return df_pred, resultados_globales, informacion_modelos


# ===========================================================================
#  GRAFICAS PROFESIONALES (8 PNGs)
# ===========================================================================

def _guardar(fig, nombre):
    """Guarda figura en metrics/ como PNG 200 dpi."""
    ruta = os.path.join(CARPETA_METRICAS, nombre)
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    {nombre}")


def _dibujar_gauge(ax, valor, titulo, subtitulo, color_val):
    """Dibuja un arco semicircular tipo gauge con valor porcentual."""
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Fondo del arco
    fondo = Wedge((0, 0), 1.0, 0, 180, width=0.25,
                  facecolor="#E8E8E8", edgecolor="white", linewidth=2)
    ax.add_patch(fondo)

    # Arco de valor (0-100 mapeado a 0-180 grados)
    angulo = max(0, min(180, valor * 1.8))
    if angulo > 0:
        arco = Wedge((0, 0), 1.0, 180 - angulo, 180, width=0.25,
                      facecolor=color_val, edgecolor="white", linewidth=2)
        ax.add_patch(arco)

    # Valor central
    ax.text(0, 0.35, f"{valor:.1f}%", ha="center", va="center",
            fontsize=28, fontweight="bold", color=color_val)
    # Titulo
    ax.text(0, -0.05, titulo, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#333")
    # Subtitulo
    ax.text(0, -0.25, subtitulo, ha="center", va="center",
            fontsize=8.5, color="#777", style="italic")


def generar_graficas(df_pred, informacion_modelos):
    """Genera 7 graficas profesionales y las guarda como PNG."""
    plt.close("all")

    targets = [t for t in TARGETS]
    n = len(targets)
    print("\n  Generando graficas:")

    # ===================================================================
    #  GRAFICA 1: Panel de rendimiento (gauges R2)
    # ===================================================================
    fig1, axes1 = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes1 = [axes1]

    for i, t in enumerate(targets):
        info = informacion_modelos.get(t["nombre"], {})
        r2 = info.get("metricas_kf", {}).get("R2", 0)
        mae = info.get("metricas_kf", {}).get("MAE", 0)
        modelo = info.get("nombre_modelo", "?")
        u = t["u"]
        N_m = len(info.get("y", []))

        r2_pct = max(0, r2 * 100)
        if r2_pct >= 60:
            color = "#27ae60"
        elif r2_pct >= 30:
            color = "#f39c12"
        else:
            color = "#e74c3c"

        sub = f"{modelo} | MAE={mae:.1f} {u} | n={N_m}"
        _dibujar_gauge(axes1[i], r2_pct,
                       _NOMBRES_CORTOS.get(t["nombre"], t["nombre"]),
                       sub, color)

    fig1.suptitle("Capacidad Predictiva del Modelo",
                  fontsize=16, fontweight="bold", y=1.02, color="#222")
    fig1.text(0.5, -0.02,
              "Porcentaje de la variabilidad explicada por el modelo (R2 x 100)",
              ha="center", fontsize=9, color="#888")
    fig1.tight_layout()
    _guardar(fig1, "entrenamiento_01_panel_rendimiento.png")

    # ===================================================================
    #  GRAFICA 2: Tasa de acierto
    # ===================================================================
    umbrales = [10, 15, 25, 50]
    fig2, ax2 = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(umbrales))
    ancho = 0.8 / n

    for i, t in enumerate(targets):
        sub = df_pred[df_pred["Target"] == t["nombre"]]
        errores = sub["Error %"].values
        tasas = [np.sum(errores < u) / len(errores) * 100 for u in umbrales]

        offset = (i - n / 2 + 0.5) * ancho
        bars = ax2.bar(x + offset, tasas, ancho * 0.9,
                       label=_NOMBRES_CORTOS.get(t["nombre"], t["nombre"]),
                       color=_COLORES_TARGET[i], alpha=0.85,
                       edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, tasas):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=_COLORES_TARGET[i])

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Error < {u}%" for u in umbrales], fontsize=10)
    ax2.set_ylabel("Pacientes predichos correctamente (%)", fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.set_title("Tasa de Acierto por Umbral de Error",
                  fontsize=14, fontweight="bold", pad=12)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.text(0.5, -0.1,
             "Porcentaje de pacientes cuyo error de prediccion "
             "esta por debajo de cada umbral",
             ha="center", fontsize=8, color="#888", transform=ax2.transAxes)
    fig2.tight_layout()
    _guardar(fig2, "entrenamiento_02_tasa_acierto.png")

    # ===================================================================
    #  GRAFICA 3: Scatter real vs predicho con bandas
    # ===================================================================
    fig3, axes3 = plt.subplots(1, n, figsize=(6.5 * n, 6))
    if n == 1:
        axes3 = [axes3]

    for i, t in enumerate(targets):
        ax = axes3[i]
        sub = df_pred[df_pred["Target"] == t["nombre"]]
        real = sub["Real"].values
        pred = sub["Predicho"].values
        errs = sub["Error %"].values
        u = t["u"]
        info = informacion_modelos.get(t["nombre"], {})
        r2 = info.get("metricas_kf", {}).get("R2", 0)
        mae = info.get("metricas_kf", {}).get("MAE", 0)
        modelo = info.get("nombre_modelo", "?")

        # Linea identidad y bandas
        all_v = np.concatenate([real, pred])
        vmin, vmax = all_v.min(), all_v.max()
        margin = (vmax - vmin) * 0.08
        lims = [max(0, vmin - margin), vmax + margin]
        xs = np.linspace(lims[0], lims[1], 200)

        ax.fill_between(xs, xs * 0.70, xs * 1.30,
                        alpha=0.06, color="#f39c12", label="Banda +/-30%")
        ax.fill_between(xs, xs * 0.85, xs * 1.15,
                        alpha=0.10, color="#27ae60", label="Banda +/-15%")
        ax.plot(xs, xs, "--", color="#888", linewidth=1.5, alpha=0.7,
                label="Prediccion perfecta")

        # Puntos coloreados por error
        colores_pts = np.where(errs < 15, "#27ae60",
                      np.where(errs < 30, "#f39c12", "#e74c3c"))
        ax.scatter(real, pred, c=colores_pts, s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.4, zorder=3)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Valor Real ({u})", fontsize=10)
        ax.set_ylabel(f"Prediccion ({u})", fontsize=10)
        ax.set_title(f"{_NOMBRES_CORTOS.get(t['nombre'], t['nombre'])}\n"
                     f"{modelo}  |  R2={r2:.3f}  |  MAE={mae:.1f} {u}",
                     fontsize=11, pad=8)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    fig3.suptitle("Prediccion vs Valor Real",
                  fontsize=14, fontweight="bold", y=1.01)
    fig3.text(0.5, -0.02,
              "Cada punto = 1 paciente. "
              "Mas cerca de la linea diagonal = mejor prediccion. "
              "Verde: error<15%  Naranja: 15-30%  Rojo: >30%",
              ha="center", fontsize=8, color="#888")
    fig3.tight_layout()
    _guardar(fig3, "entrenamiento_03_scatter_prediccion.png")

    # ===================================================================
    #  GRAFICA 4: Distribucion del error (histogramas)
    # ===================================================================
    fig4, axes4 = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes4 = [axes4]

    for i, t in enumerate(targets):
        ax = axes4[i]
        sub = df_pred[df_pred["Target"] == t["nombre"]]
        errs = sub["Error %"].values

        bins = np.arange(0, max(errs.max() + 5, 55), 5)
        colores_bins = []
        for b in bins[:-1]:
            if b < 15:
                colores_bins.append("#27ae60")
            elif b < 30:
                colores_bins.append("#f39c12")
            else:
                colores_bins.append("#e74c3c")

        _, _, patches = ax.hist(errs, bins=bins, edgecolor="white",
                                linewidth=0.8, alpha=0.85)
        for patch, color in zip(patches, colores_bins):
            patch.set_facecolor(color)

        # Percentiles
        p50 = np.median(errs)
        p75 = np.percentile(errs, 75)
        p90 = np.percentile(errs, 90)
        for pval, label, ls in [(p50, "P50", "-"), (p75, "P75", "--"), (p90, "P90", ":")]:
            ax.axvline(pval, color="#2c3e50", linestyle=ls, linewidth=1.5,
                       alpha=0.7)
            ax.text(pval + 0.5, ax.get_ylim()[1] * 0.92, f"{label}={pval:.0f}%",
                    fontsize=8, fontweight="bold", color="#2c3e50", rotation=0)

        n_bajo15 = np.sum(errs < 15)
        pct15 = n_bajo15 / len(errs) * 100
        ax.text(0.95, 0.85,
                f"{pct15:.0f}% de pacientes\ncon error < 15%",
                transform=ax.transAxes, ha="right", fontsize=9,
                fontweight="bold", color="#27ae60",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="#27ae60"))

        ax.set_xlabel("Error de Prediccion (%)", fontsize=10)
        ax.set_ylabel("Cantidad de Pacientes", fontsize=10)
        ax.set_title(_NOMBRES_CORTOS.get(t["nombre"], t["nombre"]),
                     fontsize=11, pad=8)

    fig4.suptitle("Distribucion del Error de Prediccion",
                  fontsize=14, fontweight="bold", y=1.01)
    fig4.text(0.5, -0.02,
              "Verde: error<15%  Naranja: 15-30%  Rojo: >30%  |  "
              "Lineas: percentiles P50, P75, P90",
              ha="center", fontsize=8, color="#888")
    fig4.tight_layout()
    _guardar(fig4, "entrenamiento_04_distribucion_error.png")

    # ===================================================================
    #  GRAFICA 5: Top 20 pacientes con mayor error
    # ===================================================================
    fig5, axes5 = plt.subplots(1, n, figsize=(6 * n, 7))
    if n == 1:
        axes5 = [axes5]

    for i, t in enumerate(targets):
        ax = axes5[i]
        sub = df_pred[df_pred["Target"] == t["nombre"]].copy()
        top20 = sub.nlargest(20, "Error %")

        pacs = [p.replace("case_", "") for p in top20["Paciente_ID"].values]
        errs = top20["Error %"].values

        colores_bar = ["#27ae60" if e < 15 else "#f39c12" if e < 30 else "#e74c3c"
                       for e in errs]

        y_pos = np.arange(len(pacs))
        ax.barh(y_pos, errs[::-1], color=colores_bar[::-1],
                edgecolor="white", height=0.7)

        for j, (pac, err) in enumerate(zip(pacs[::-1], errs[::-1])):
            ax.text(err + 0.5, j, f"{err:.0f}%", va="center",
                    fontsize=8, fontweight="bold", color="#555")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pacs[::-1], fontsize=8)
        ax.set_xlabel("Error (%)", fontsize=10)
        ax.set_title(_NOMBRES_CORTOS.get(t["nombre"], t["nombre"]),
                     fontsize=11, pad=8)

    fig5.suptitle("Top 20 Pacientes con Mayor Error de Prediccion",
                  fontsize=14, fontweight="bold", y=1.01)
    fig5.tight_layout()
    _guardar(fig5, "entrenamiento_05_top_errores.png")

    # ===================================================================
    #  GRAFICA 6: Comparacion KFold vs LOO + overfitting
    # ===================================================================
    metricas_keys = ["R2", "MAE", "RMSE"]
    fig6, axes6 = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes6 = [axes6]

    for i, t in enumerate(targets):
        ax = axes6[i]
        info = informacion_modelos.get(t["nombre"], {})
        m_kf = info.get("metricas_kf", {})
        ovf = info.get("overfitting", {})
        gap = info.get("kf_gap", 0)
        dx = ovf.get("Diagnostico", "OK")

        x_mk = np.arange(len(metricas_keys))
        vals_kf = [m_kf.get(k, 0) for k in metricas_keys]

        bars = ax.bar(x_mk, vals_kf, 0.5, color=_COLORES_TARGET[i],
                      alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, vals_kf):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals_kf) * 0.02,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x_mk)
        ax.set_xticklabels(metricas_keys, fontsize=10)

        # Semaforo
        if dx == "OVERFITTING":
            sem_color = "#e74c3c"
        elif dx == "UNDERFITTING":
            sem_color = "#f39c12"
        else:
            sem_color = "#27ae60"

        ax.text(0.95, 0.92,
                f"Gap: {gap:.1f}%\n{dx}",
                transform=ax.transAxes, ha="right", fontsize=9,
                fontweight="bold", color=sem_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor=sem_color))

        ax.set_title(f"{_NOMBRES_CORTOS.get(t['nombre'], t['nombre'])}"
                     f"  ({info.get('nombre_modelo', '')})",
                     fontsize=11, pad=8)

    fig6.suptitle("Metricas de Validacion Cruzada (10-Fold)",
                  fontsize=14, fontweight="bold", y=1.01)
    fig6.text(0.5, -0.02,
              "Gap% bajo indica estabilidad del modelo (sin sobreajuste).",
              ha="center", fontsize=8, color="#888")
    fig6.tight_layout()
    _guardar(fig6, "entrenamiento_06_validacion_cruzada.png")

    # ===================================================================
    #  GRAFICA 7: Feature importance (top 10)
    # ===================================================================
    fig7, axes7 = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes7 = [axes7]

    for i, t in enumerate(targets):
        ax = axes7[i]
        info = informacion_modelos.get(t["nombre"], {})
        imp_dict = info.get("importances", {})
        if not imp_dict:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#999")
            continue

        imp_series = pd.Series(imp_dict).sort_values()
        top10 = imp_series.tail(10)

        def _color_familia(f):
            if f.startswith("firstorder_"): return "#3498db"
            if f.startswith("glcm_"): return "#e67e22"
            if f.startswith("glrlm_"): return "#2ecc71"
            if f.startswith("glszm_"): return "#e74c3c"
            if f.startswith("gldm_"): return "#9b59b6"
            return "#95a5a6"
        colores_feat = [_color_familia(f) for f in top10.index]
        labels = [f.replace("firstorder_", "fo_")
                   .replace("glcm_", "gl_")
                   .replace("glrlm_", "rl_")
                   .replace("glszm_", "sz_")
                   .replace("gldm_", "dm_")
                  for f in top10.index]

        bars = ax.barh(range(len(top10)), top10.values, color=colores_feat,
                       edgecolor="white", height=0.7)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Importancia Ponderada", fontsize=10)

        for bar, val in zip(bars, top10.values):
            ax.text(val + top10.max() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7.5, color="#555")

        ax.set_title(f"{_NOMBRES_CORTOS.get(t['nombre'], t['nombre'])}"
                     f"  ({info.get('nombre_modelo', '')})",
                     fontsize=11, pad=8)

    fig7.suptitle("Caracteristicas mas Importantes para la Prediccion",
                  fontsize=14, fontweight="bold", y=1.01)
    fig7.text(0.5, -0.02,
              "Azul: FirstOrder  |  Naranja: GLCM  |  Verde: GLRLM  |  "
              "Rojo: GLSZM  |  Morado: GLDM  |  Gris: Derivadas  |  Ponderado por Ridge",
              ha="center", fontsize=8, color="#888")
    fig7.tight_layout()
    _guardar(fig7, "entrenamiento_07_feature_importance.png")

    print(f"  7 graficas guardadas en regression/metrics/")


# ===========================================================================
#  Prediccion sobre casos de prueba
# ===========================================================================
def predecir_casos_prueba(informacion_modelos):
    """Predice sobre casos_prueba.csv y guarda verificacion."""
    if not os.path.exists(RUTA_PRUEBA):
        print("\n  casos_prueba.csv no encontrado, omitiendo verificacion")
        return None

    df_prueba = pd.read_csv(RUTA_PRUEBA)
    print(f"\n  VERIFICACION CASOS DE PRUEBA ({len(df_prueba)} casos)")

    shape_cols = [c for c in df_prueba.columns if c.startswith("shape_")]
    cols_drop = ["Paciente_ID", "target_riesgo"] + shape_cols + COLS_CLINICAS
    X_prueba = df_prueba.drop(
        columns=[c for c in cols_drop if c in df_prueba.columns])

    # Aplicar las mismas features derivadas que en entrenamiento
    X_prueba = crear_features_derivadas(X_prueba)

    resultados_prueba = []

    for t in TARGETS:
        nombre = t["nombre"]
        u = t["u"]
        info = informacion_modelos.get(nombre)
        if not info:
            continue

        pipe = info["mejor_pipe"]
        y_train = info["y"]
        cols_sel = info.get("cols_selected")

        if cols_sel:
            # Garantiza que X_pred tenga exactamente las columnas que espera el modelo
            X_pred = X_prueba.reindex(columns=cols_sel, fill_value=0)
        else:
            X_pred = X_prueba

        y_real = df_prueba[t["origen"]].values
        
        # Aplicamos el límite a las predicciones finales
        y_pred_raw = pipe.predict(X_pred)
        if info.get("use_log"):
            y_pred = np.maximum(0, np.expm1(y_pred_raw))
        else:
            y_pred = np.maximum(0, y_pred_raw)

        metricas = calcular_metricas(y_real, y_pred)
        print(f"    {nombre}: R2={metricas['R2']:.4f}  "
              f"MAE={metricas['MAE']:.1f} {u}  "
              f"MAPE={metricas['MAPE %']:.1f}%")

        for i in range(len(df_prueba)):
            pac = df_prueba["Paciente_ID"].iloc[i]
            real_v = y_real[i]
            pred_v = y_pred[i]
            err = abs(real_v - pred_v)
            err_pct = err / max(abs(real_v), 1e-9) * 100
            resultados_prueba.append({
                "Paciente_ID": pac,
                "Target": nombre,
                "Unidad": u,
                "Real": round(real_v, 2),
                "Predicho": round(pred_v, 2),
                "Error": round(err, 2),
                "Error %": round(err_pct, 2),
                "Nivel_Real": categorizar_nivel(real_v, y_train),
                "Nivel_Predicho": categorizar_nivel(pred_v, y_train),
            })

    df_verif = pd.DataFrame(resultados_prueba)
    verif_path = os.path.join(CARPETA_METRICAS, "verificacion_casos_prueba.csv")
    df_verif.to_csv(verif_path, index=False)
    print(f"    Guardado: regression/metrics/verificacion_casos_prueba.csv")

    return df_verif


def generar_grafica_casos_prueba(df_verif, informacion_modelos):
    """Grafica 8: scatter + tabla visual de casos de prueba."""
    if df_verif is None or df_verif.empty:
        return

    targets = [t for t in TARGETS]
    n = len(targets)

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9),
                             gridspec_kw={"height_ratios": [2, 1]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, t in enumerate(targets):
        sub = df_verif[df_verif["Target"] == t["nombre"]]
        if sub.empty:
            continue

        real = sub["Real"].values
        pred = sub["Predicho"].values
        errs = sub["Error %"].values
        u = t["u"]
        info = informacion_modelos.get(t["nombre"], {})
        modelo = info.get("nombre_modelo", "?")

        # -- Scatter (fila superior) --
        ax_s = axes[0, i]
        all_v = np.concatenate([real, pred])
        vmin, vmax = all_v.min(), all_v.max()
        margin = (vmax - vmin) * 0.15
        lims = [max(0, vmin - margin), vmax + margin]
        xs = np.linspace(lims[0], lims[1], 200)

        ax_s.fill_between(xs, xs * 0.85, xs * 1.15,
                          alpha=0.12, color="#27ae60")
        ax_s.plot(xs, xs, "--", color="#888", linewidth=1.5, alpha=0.7)

        colores_pts = ["#27ae60" if e < 15 else "#f39c12" if e < 30
                       else "#e74c3c" for e in errs]
        ax_s.scatter(real, pred, c=colores_pts, s=80, edgecolors="white",
                     linewidth=1, zorder=3)

        # Etiquetas
        for _, row in sub.iterrows():
            pac = row["Paciente_ID"].replace("case_", "")
            ax_s.annotate(pac, (row["Real"], row["Predicho"]),
                          fontsize=7.5, fontweight="bold",
                          xytext=(4, 4), textcoords="offset points",
                          color="#555")

        ax_s.set_xlim(lims)
        ax_s.set_ylim(lims)
        ax_s.set_aspect("equal")
        ax_s.set_xlabel(f"Real ({u})", fontsize=9)
        ax_s.set_ylabel(f"Prediccion ({u})", fontsize=9)
        ax_s.set_title(f"{_NOMBRES_CORTOS.get(t['nombre'], t['nombre'])}"
                       f"  ({modelo})", fontsize=11, pad=8)

        # -- Tabla (fila inferior) --
        ax_t = axes[1, i]
        ax_t.axis("off")
        tabla_data = []
        colores_celda = []
        for _, row in sub.iterrows():
            pac = row["Paciente_ID"].replace("case_", "")
            err_v = row["Error %"]
            tabla_data.append([
                pac,
                f"{row['Real']:.1f}",
                f"{row['Predicho']:.1f}",
                f"{err_v:.0f}%",
            ])
            if err_v < 15:
                c = "#d5f5e3"
            elif err_v < 30:
                c = "#fdebd0"
            else:
                c = "#fadbd8"
            colores_celda.append([c] * 4)

        if tabla_data:
            tabla = ax_t.table(
                cellText=tabla_data,
                colLabels=["Paciente", "Real", "Predicho", "Error"],
                cellColours=colores_celda,
                colColours=["#d6eaf8"] * 4,
                loc="center", cellLoc="center",
            )
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(8.5)
            tabla.scale(1, 1.4)

    fig.suptitle("Rendimiento en Casos Nunca Vistos por el Modelo",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _guardar(fig, "entrenamiento_08_casos_prueba.png")


# ===========================================================================
#  Exportación a Producción (Joblib)
# ===========================================================================
def exportar_modelos_produccion(informacion_modelos):
    """
    Guarda los pipelines definitivos en formato .joblib y genera un archivo .json
    con los metadatos necesarios para predecir nuevos pacientes (features exactas, log_transform).
    """
    carpeta_produccion = os.path.join(base_dir, "regression", "joblib")
    os.makedirs(carpeta_produccion, exist_ok=True)

    print(f"\n  EXPORTANDO MODELOS A PRODUCCIÓN ({carpeta_produccion}):")

    for target_nombre, info in informacion_modelos.items():
        slug = info["slug"]
        modelo_final = info["mejor_pipe"]
        features_requeridas = info["cols_selected"]  # Las columnas que sobrevivieron al RFECV

        # 1. Guardar el modelo compilado (Pipeline completo)
        ruta_joblib = os.path.join(carpeta_produccion, f"modelo_{slug}.joblib")
        joblib.dump(modelo_final, ruta_joblib)

        # 2. Guardar el "Manual de Instrucciones" (Metadatos)
        metadata = {
            "target": target_nombre,
            "unidad": info["unidad"],
            "use_log": info["use_log"],
            "n_features_esperadas": len(features_requeridas),
            "features_entrada": features_requeridas
        }

        ruta_json = os.path.join(carpeta_produccion, f"metadata_{slug}.json")
        with open(ruta_json, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"    [OK] {target_nombre}: {len(features_requeridas)} features requeridas.")
        print(f"         -> {ruta_joblib}")
        print(f"         -> {ruta_json}")


# ===========================================================================
#  Main
# ===========================================================================
if __name__ == "__main__":
    df_pred, _, info_modelos = entrenar_y_evaluar()
    generar_graficas(df_pred, info_modelos)
    df_verif = predecir_casos_prueba(info_modelos)
    generar_grafica_casos_prueba(df_verif, info_modelos)
    exportar_modelos_produccion(info_modelos)
