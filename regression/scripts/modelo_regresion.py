"""
modelo_regresion.py - Modelado predictivo con Stacking Ensemble
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor,
)
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, max_error,
    explained_variance_score, mean_absolute_percentage_error,
)
from sklearn.feature_selection import mutual_info_regression

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor

import optuna
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor

from regression.scripts.optimizar_regresion import detectar_overfitting, seleccionar_features_rfecv

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
    #{"nombre": "Diametro Eje Corto", "col": "target_eje_corto",
     #"origen": "shape_MinorAxisLength", "u": "mm", "slug": "eje_corto"},
    #{"nombre": "Diametro Eje Largo", "col": "target_eje_largo",
     #"origen": "shape_MajorAxisLength", "u": "mm", "slug": "eje_largo"},
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
#  OBTENER LA CONFIGURACIÓN DE LOS MODELOS
# ===========================================================================
def _obtener_config_modelo(target_slug):
    if target_slug == "volumen":
        estimators = [
            # Bajamos a 80 estimadores y usamos el 80% de los datos para forzar generalización
            ("lgbm", LGBMRegressor(n_estimators=80, subsample=0.8, random_state=42, verbose=-1, n_jobs=None)),
            ("gb", GradientBoostingRegressor(n_estimators=80, subsample=0.8, random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=80, max_samples=0.8, random_state=42, n_jobs=None))
        ]
        nombre = "Dream Team (LGBM + GB + RF - Estrictos)"

        # [EL CAMBIO MAESTRO]
        # positive=True: Prohíbe restar predicciones (evita caos)
        # fit_intercept=False: Prohíbe sumar la base fantasma que está inflando el 80%
        # Volvemos a Ridge para tener penalización L2 (evita que un modelo se vuelva loco como en el 0545)
        # Mantenemos positive=True y fit_intercept=False para que sea un promedio estricto.
        meta_learner = Ridge(alpha=10.0, positive=True, fit_intercept=False)

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=None,
    )

    # Volvemos al log1p: Matemáticamente estable para errores porcentuales
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=stacking,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    return pipe, nombre, {}


def _extraer_importances_stacking(pipe, feature_names):
    """Extrae las importancias promediadas del pipeline directo."""
    # [CORRECCIÓN] Como volvió el envoltorio, el Stacking está escondido en .regressor_
    stacking = pipe.named_steps["model"].regressor_

    # Obtenemos los coeficientes del meta-learner (Ridge/LinearRegression)
    coefs = stacking.final_estimator_.coef_

    # Si hay PCA se ajustan los nombres, si no, se usan las variables crudas
    if "pca" in pipe.named_steps:
        n_components = pipe.named_steps["pca"].n_components_
        nombres_reales = [f"PCA_Componente_{i + 1}" for i in range(n_components)]
    else:
        nombres_reales = feature_names

    importances = np.zeros(len(nombres_reales))

    for est, coef in zip(stacking.estimators_, coefs):
        if hasattr(est, "feature_importances_"):
            # Ponderamos la importancia del árbol por el peso que le dio el meta-learner
            importances += est.feature_importances_ * abs(coef)

    total = importances.sum()
    if total > 0:
        importances /= total
    else:
        importances = np.ones(len(nombres_reales)) / len(nombres_reales)

    return dict(zip(nombres_reales, importances))


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

    # Flag Clínico: Detección de Necrosis / Conglomerado Masivo
    # Usando percentiles relativos a la distribución del dataset
    if "firstorder_Minimum" in X.columns and "glszm_ZoneVariance" in X.columns:
        # [CORRECCIÓN] Valores estáticos (Usa los de tu dataset o estos aproximados)
        min_umbral = -30.0
        var_umbral = 15000.0

        Xd["flag_conglomerado_necrotico"] = np.where(
            (X["firstorder_Minimum"] <= min_umbral) & (X["glszm_ZoneVariance"] >= var_umbral),1, 0
        )

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
    # ELIMINADO: y_1d = np.cbrt(y_target)
    # Ahora usamos y_target directamente en todo el bloque.

    # ¡CRÍTICO: Transformar el target para estabilizar la varianza!
    y_transformado = np.log1p(y_target)

    # --- PASO 1: FILTRO GRUESO (Ruido) ---
    corrs = X.corrwith(pd.Series(y_transformado, index=X.index)).abs().fillna(0)
    cols_corr = set(corrs[corrs > CORR_UMBRAL].index)

    mi = mutual_info_regression(X, y_transformado, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=X.columns)
    mi_umbral = np.percentile(mi_series.values, 20)
    cols_mi = set(mi_series[mi_series > mi_umbral].index)

    cols_ok = list(cols_corr | cols_mi)

    # --- EL SALVAVIDAS DE LA BANDERA ---
    if "flag_conglomerado_necrotico" in X.columns and "flag_conglomerado_necrotico" not in cols_ok:
        cols_ok.append("flag_conglomerado_necrotico")

    if not cols_ok:
        cols_ok = list(X.columns)

    X_filt = X[cols_ok].copy()

    # --- PASO 2: PODA DE COLINEALIDAD ---
    corr_matrix = X_filt.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > INTER_CORR_UMBRAL)]
    X_filt = X_filt.drop(columns=to_drop)

    cols_ok = list(X_filt.columns)

    # --- PASO 3: RFECV (EL QUIRÓFANO) ---
    if usar_rfecv and len(cols_ok) > 12:
        resultado_rfecv = seleccionar_features_rfecv(X_filt, y_transformado, cv=3, min_features=12)
        cols_finales = resultado_rfecv["columnas_seleccionadas"]
        return cols_finales
    else:
        return cols_ok


# ===========================================================================
#  Preparacion de datos
# ===========================================================================
def preparar_datos(df_raw):
    """
    Deduplica, crea targets desde shape_*, elimina shape_* y
    BLOQUEA variables radiómicas con Fuga de Datos (Volume-Confounded).
    """
    n_raw = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset="Paciente_ID")
    n_dedup = len(df_raw)
    df_raw = df_raw.reset_index(drop=True)

    df = df_raw.copy()
    for t in TARGETS:
        df[t["col"]] = df_raw[t["origen"]]

    # --- TODO ESTO SALIÓ DEL BUCLE FOR ---
    # 1. Escudo contra tamaño, pero SALVAMOS las proporciones morfológicas (No tienen fuga de datos)
    shape_cols = [c for c in df.columns if
                  c.startswith("shape_") and c not in ["shape_Sphericity", "shape_Elongation", "shape_Flatness"]]

    # 2. ESCUDO DINÁMICO DE SPEARMAN (Anti-Leakage)
    y_temp_vol = df_raw["shape_VoxelVolume"].values
    fuga_cols = []

    # Excluimos las columnas que ya sabemos que son targets o shape
    cols_a_evaluar = [c for c in df.columns if
                      c not in COLS_TARGET + COLS_CLINICAS + shape_cols and c != "Paciente_ID"]

    for col in cols_a_evaluar:
        coef, _ = spearmanr(df[col], y_temp_vol)
        # Tu regla de oro: Si Spearman > 0.75, a la basura
        if abs(coef) > 0.75:
            fuga_cols.append(col)

    # Juntamos todas las columnas prohibidas
    cols_drop = ["Paciente_ID", "target_riesgo"] + COLS_TARGET + COLS_CLINICAS + shape_cols + fuga_cols
    X = df.drop(columns=[c for c in cols_drop if c in df.columns])
    # -------------------------------------

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
        "fugas_eliminadas": len(fuga_cols),  # Para tener registro
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


def optimizar_con_optuna_regresion(X_train, y_train, pipe_tmpl, n_trials=30):
    """Utiliza Optimización Bayesiana para encontrar los hiperparámetros perfectos."""

    def objective(trial):
        pipe = clone(pipe_tmpl)

        # RANGOS ANTI-MEMORIZACIÓN (AÚN MÁS ESTRICTOS)
        rf_depth = trial.suggest_int("rf_depth", 2, 4)  # Máximo 4 niveles
        rf_min_samples = trial.suggest_int("rf_min_samples", 12, 25)  # Más muestras requeridas

        lgbm_lr = trial.suggest_float("lgbm_lr", 0.01, 0.04, log=True)
        lgbm_depth = trial.suggest_int("lgbm_depth", 2, 3)
        lgbm_min_child = trial.suggest_int("lgbm_min_child_samples", 12, 25)
        lgbm_alpha = trial.suggest_float("lgbm_reg_alpha", 0.1, 10.0, log=True)  # NUEVO: Regularización L1

        gb_lr = trial.suggest_float("gb_lr", 0.01, 0.04, log=True)
        gb_depth = trial.suggest_int("gb_depth", 2, 3)
        gb_min_samples = trial.suggest_int("gb_min_samples_leaf", 12, 25)

        # [CORRECCIÓN CRÍTICA] ¡Volvió el envoltorio logarítmico! Debe llevar "regressor__"
        try:
            pipe.set_params(
                model__regressor__rf__max_depth=rf_depth,
                model__regressor__rf__min_samples_leaf=rf_min_samples,
                model__regressor__lgbm__learning_rate=lgbm_lr,
                model__regressor__lgbm__max_depth=lgbm_depth,
                model__regressor__lgbm__min_child_samples=lgbm_min_child,
                model__regressor__gb__learning_rate=gb_lr,
                model__regressor__gb__max_depth=gb_depth,
                model__regressor__gb__min_samples_leaf=gb_min_samples,
                model__regressor__lgbm__reg_alpha=lgbm_alpha
            )
        except ValueError as e:
            print(f"\n[!] ERROR DE RUTAS EN PIPELINE: {e}")
            raise e

        try:
            score = cross_val_score(
                pipe, X_train, y_train, cv=3,
                scoring="neg_mean_absolute_percentage_error",
                n_jobs=None  # Vital mantenerlo en None para no saturar tus 12GB de RAM
            ).mean()

            if np.isnan(score):
                return -999999.0
            return score

        except Exception as e:
            print(f"\n[!] ERROR EN CROSS VAL OPTUNA: {e}")
            return -999999.0

    # Limpiamos los fantasmas duplicados de Optuna aquí afuera
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="maximize")

    print(f"        [Optuna] Explorando {n_trials} combinaciones inteligentes...")
    estudio.optimize(objective, n_trials=n_trials)

    mejor_pipe = clone(pipe_tmpl)

    # [CORRECCIÓN CRÍTICA] Rutas corregidas también aquí al asignar al pipe final
    mejor_pipe.set_params(
        model__regressor__rf__max_depth=estudio.best_params["rf_depth"],
        model__regressor__rf__min_samples_leaf=estudio.best_params["rf_min_samples"],
        model__regressor__lgbm__learning_rate=estudio.best_params["lgbm_lr"],
        model__regressor__lgbm__max_depth=estudio.best_params["lgbm_depth"],
        model__regressor__lgbm__min_child_samples=estudio.best_params["lgbm_min_child_samples"],
        model__regressor__gb__learning_rate=estudio.best_params["gb_lr"],
        model__regressor__gb__max_depth=estudio.best_params["gb_depth"],
        model__regressor__gb__min_samples_leaf=estudio.best_params["gb_min_samples_leaf"],
        model__regressor__lgbm__reg_alpha=estudio.best_params["lgbm_reg_alpha"]
    )
    return mejor_pipe, estudio.best_params


# ===========================================================================
#  Entrenamiento y evaluacion
# ===========================================================================
def entrenar_y_evaluar():
    """
    Entrena StackingRegressor por target con 10-Fold CV.
    Returns: df_pred, resultados_globales, informacion_modelos
    """
    df_raw = pd.read_csv(RUTA_CSV)

    # =======================================================================
    # ELIMINACIÓN DE OUTLIERS SUAVIZADA (Percentil 98 en lugar de IQR)
    # =======================================================================
    col_volumen = "shape_VoxelVolume"
    limite_superior = df_raw[col_volumen].quantile(0.98)  # Conserva el 98% de los datos

    df_raw = df_raw[df_raw[col_volumen] <= limite_superior].copy()
    # =======================================================================

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
        y_train_space = y_orig  # Ya no transformamos aquí
        u = t["u"]

        # [CAMBIO AQUI]: Llamamos a la nueva función que devuelve TODO adaptado al target
        pipe_tmpl, nombre_modelo, param_grid_dinamico = _obtener_config_modelo(slug)

        print(f"\n  {t['nombre']} ({u}) | {nombre_modelo}")
        print(f"    Rango: [{y_orig.min():.1f}, {y_orig.max():.1f}]  Media: {y_orig.mean():.1f}")

        all_y_pred_kf = np.zeros(N)
        all_y_count_kf = np.zeros(N)
        kf_train_maes = []
        kf_test_maes = []
        features_seleccionadas_por_fold = []

        # -------------------------------------------------------------------
        # 1. ENCONTRAR LAS FEATURES MAESTRAS (RFECV GLOBAL)
        # -------------------------------------------------------------------
        print(f"    [1/3] Extrayendo la crema y nata de las features (Filtro + RFECV)...")
        cols_sel_final = seleccionar_features(X_all, y_train_space, usar_rfecv=True)
        X_final = X_all[cols_sel_final]

        # -------------------------------------------------------------------
        # 2. ENCONTRAR LOS HIPERPARÁMETROS MAESTROS
        # -------------------------------------------------------------------
        print(f"    [2/3] Buscando hiperparámetros óptimos...")
        # [CORRECCIÓN] Pasamos la variable cruda. El TransformedTargetRegressor del pipeline hace el resto.
        pipe_optimo, mejores_params = optimizar_con_optuna_regresion(
            X_final, y_train_space, clone(pipe_tmpl), n_trials=20
        )
        print(f"    [Optuna] Listo. Validando modelo definitivo...")

        # -------------------------------------------------------------------
        # 3. VALIDACIÓN K-FOLD LIGERA (El examen final de la arquitectura)
        # -------------------------------------------------------------------
        for rep in range(N_REPEATS):
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42 + rep)
            for train_idx, test_idx in kf.split(X_final):
                X_tr, X_te = X_final.iloc[train_idx], X_final.iloc[test_idx]
                y_tr_space_fold = y_train_space[train_idx]
                y_te_orig = y_orig[test_idx]

                # Usamos el pipeline podado e inteligente con las features exactas
                fold_pipe = clone(pipe_optimo)
                fold_pipe.fit(X_tr, y_tr_space_fold)

                pred_te = fold_pipe.predict(X_te)
                pred_tr = fold_pipe.predict(X_tr)

                all_y_pred_kf[test_idx] += pred_te
                all_y_count_kf[test_idx] += 1

                kf_train_maes.append(mean_absolute_error(y_tr_space_fold, pred_tr))
                kf_test_maes.append(mean_absolute_error(y_te_orig, pred_te))
                # ========================================================

        y_pred_kf = all_y_pred_kf / np.maximum(all_y_count_kf, 1)
        metricas_kf = calcular_metricas(y_orig, y_pred_kf)

        kf_train_mae = np.mean(kf_train_maes)
        kf_test_mae = np.mean(kf_test_maes)
        kf_gap = ((kf_test_mae - kf_train_mae) / kf_test_mae * 100) if kf_test_mae != 0 else 0

        print(f"    KFold  R2={metricas_kf['R2']:.4f}  MAE={metricas_kf['MAE']:.1f}  "
              f"Gap={kf_gap:.1f}%  (Features finales: {len(cols_sel_final)})")

        # ---------------------------------------------------------------
        # 4. ENTRENAMIENTO PARA PRODUCCIÓN
        # ---------------------------------------------------------------
        pipe_final = clone(pipe_optimo)
        pipe_final.fit(X_final, y_train_space)

        print(f"    Mejores parámetros finales: {mejores_params}")

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

    # [CORRECCIÓN CRÍTICA] Salvar las proporciones morfológicas de ser borradas
    shape_cols = [c for c in df_prueba.columns if
                  c.startswith("shape_") and c not in ["shape_Sphericity", "shape_Elongation", "shape_Flatness"]]

    cols_drop = ["Paciente_ID", "target_riesgo"] + shape_cols + COLS_CLINICAS
    X_prueba = df_prueba.drop(
        columns=[c for c in cols_drop if c in df_prueba.columns])

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

        # El modelo ya devuelve el volumen en mm3 reales automáticamente
        y_pred = pipe.predict(X_pred)

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
#  Torneo de Algoritmos
# ===========================================================================
def gran_torneo_volumen():
    print("\n" + "=" * 80)
    print("GRAN TORNEO DE ALGORITMOS: PREDICCIÓN DE VOLUMEN TUMORAL")
    print("=" * 80)

    # 1. Preparar datos
    df_raw = pd.read_csv(RUTA_CSV)
    X_all_raw, targets_dict, ids, info = preparar_datos(df_raw)
    X_all = crear_features_derivadas(X_all_raw)

    # Nos concentramos SOLO en el volumen
    y_orig = targets_dict["target_regresion"]
    y_train_space = y_orig

    # 2. Selección de Características
    print("\n  [1/3] Seleccionando las mejores características (RFECV)...")
    cols_sel = seleccionar_features(X_all, y_train_space, usar_rfecv=True)
    X_filt = X_all[cols_sel]
    p = X_filt.shape[1]
    print(f"        -> Se seleccionaron {p} variables clave.")

    # 3. Catálogo Masivo de Modelos de Regresión
    from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                      PoissonRegressor, HuberRegressor, BayesianRidge)
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "Huber (Robusto)": HuberRegressor(max_iter=2000),
        "Bayesian Ridge": BayesianRidge(),
        "Poisson Regressor": PoissonRegressor(max_iter=2000),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "SVR (Radial)": SVR(kernel='rbf', C=10.0, epsilon=0.1),
        "Árbol de Decisión": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42,
                                               n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
                                n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42,
                                  verbose=-1),
        "Red Neuronal (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1500, early_stopping=True,
                                           random_state=42)
    }

    resultados = []

    print("\n  [2/3] Iniciando combates en la Arena (5-Fold CV)...")
    for nombre_mod, modelo in modelos.items():
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        all_y_real = []
        all_y_pred = []
        train_maes = []

        for train_idx, test_idx in kf.split(X_filt):
            X_tr, X_te = X_filt.iloc[train_idx], X_filt.iloc[test_idx]
            y_tr_space, y_te_orig = y_train_space[train_idx], y_orig[test_idx]

            # Estandarizamos para modelos sensibles (SVM, KNN, Redes Neuronales)
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            clon = clone(modelo)

            try:
                # Entrenamiento
                clon.fit(X_tr_sc, y_tr_space)

                # Predicciones
                pred_tr_space = clon.predict(X_tr_sc)
                pred_te_space = clon.predict(X_te_sc)

                # Revertir raíz cúbica para calcular métricas en mm³ reales
                pred_tr = np.maximum(0, pred_tr_space)
                pred_te = np.maximum(0, pred_te_space)
                y_tr_orig = y_tr_space

                all_y_real.extend(y_te_orig)
                all_y_pred.extend(pred_te)
                train_maes.append(mean_absolute_error(y_tr_orig, pred_tr))
            except Exception as e:
                print(f"      [!] {nombre_mod} falló internamente: {e}")
                continue

        if len(all_y_real) == 0: continue

        all_y_real = np.array(all_y_real)
        all_y_pred = np.array(all_y_pred)

        # Calcular todas tus métricas solicitadas
        mae = mean_absolute_error(all_y_real, all_y_pred)
        rmse = np.sqrt(mean_squared_error(all_y_real, all_y_pred))
        mape = mean_absolute_percentage_error(all_y_real, all_y_pred) * 100
        r2 = r2_score(all_y_real, all_y_pred)

        n_samples = len(all_y_real)
        r2_adj = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - p - 1)) if n_samples > p + 1 else 0

        t_mae = np.mean(train_maes)
        gap = ((mae - t_mae) / mae * 100) if mae > 0 else 0.0

        resultados.append({
            "Modelo": nombre_mod, "MAE": mae, "RMSE": rmse, "MAPE": mape,
            "R2": r2, "R2_Adj": r2_adj, "Train_MAE": t_mae, "Gap_Overfitting": gap
        })
        print(f"    • {nombre_mod:<20} -> MAE: {mae:>8.1f} | R2: {r2:>6.3f} | Gap: {gap:>5.1f}%")

    # 4. Generar Reporte Visual
    print("\n  [3/3] Generando gráficas del torneo...")
    df_res = pd.DataFrame(resultados).sort_values(by="MAE", ascending=True)

    # Gráfica 1: MAE
    plt.figure(figsize=(12, 8))
    bars = plt.barh(df_res["Modelo"][::-1], df_res["MAE"][::-1], color='#3498db', edgecolor='white')
    plt.xlabel('Error Medio Absoluto (MAE) en mm³')
    plt.title('Torneo: ¿Qué modelo predice mejor el Volumen Tumoral?', fontweight='bold')
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {bar.get_width():.1f}', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_METRICAS, "torneo_masivo_mae_volumen.png"), dpi=150)
    plt.close()

    # Gráfica 2: Overfitting Gap
    df_res_gap = df_res.sort_values(by="Gap_Overfitting", ascending=True)
    plt.figure(figsize=(12, 8))
    colores_gap = ['#27ae60' if g < 15 else '#f39c12' if g < 30 else '#e74c3c' for g in
                   df_res_gap["Gap_Overfitting"][::-1]]
    bars2 = plt.barh(df_res_gap["Modelo"][::-1], df_res_gap["Gap_Overfitting"][::-1], color=colores_gap,
                     edgecolor='white')
    plt.xlabel('Brecha de Sobreajuste (Gap %)')
    plt.title('Torneo: Estabilidad frente al Sobreajuste (Menos es mejor)', fontweight='bold')
    for bar in bars2:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {bar.get_width():.1f}%', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_METRICAS, "torneo_masivo_overfitting_volumen.png"), dpi=150)
    plt.close()

    print(f"\n[!] Torneo finalizado. Revisa las nuevas gráficas en la carpeta de metrics.")
    print("=" * 80)


import seaborn as sns
from scipy.stats import spearmanr


def auditar_fuga_de_datos():
    print("\n" + "=" * 70)
    print(" 🕵️ AUDITORÍA DE FUGA DE DATOS (SPEARMAN / PEARSON) 🕵️ ")
    print("=" * 70)

    df_raw = pd.read_csv(RUTA_CSV)
    X_all_raw, targets_dict, _, _ = preparar_datos(df_raw)
    X_all = crear_features_derivadas(X_all_raw)
    y_vol = targets_dict["target_regresion"]

    # 1. Calcular Pearson (Relaciones lineales)
    corrs_pearson = X_all.corrwith(pd.Series(y_vol, index=X_all.index)).abs()

    # 2. Calcular Spearman (Relaciones No-Lineales)
    corrs_spearman = {}
    for col in X_all.columns:
        coef, _ = spearmanr(X_all[col], y_vol)
        corrs_spearman[col] = abs(coef)
    corrs_spearman = pd.Series(corrs_spearman)

    # Unir resultados
    df_fuga = pd.DataFrame({
        "Pearson": corrs_pearson,
        "Spearman": corrs_spearman
    }).sort_values(by="Spearman", ascending=False)

    # Imprimir a los mayores sospechosos
    print("\n  [TOP 10 VARIABLES MÁS CORRELACIONADAS CON EL VOLUMEN]")
    for idx, row in df_fuga.head(10).iterrows():
        alerta = " 🚨 ¡POSIBLE FUGA!" if row['Spearman'] > 0.85 else ""
        print(f"  • {idx:<30} | Spearman: {row['Spearman']:.3f} | Pearson: {row['Pearson']:.3f}{alerta}")

    # Generar Heatmap de los sospechosos
    top_cols = df_fuga.head(15).index.tolist()
    df_plot = X_all[top_cols].copy()
    df_plot["TARGET_VOLUMEN"] = y_vol

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_plot.corr(method='spearman'), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matriz de Correlación de Spearman (Top 15 Sospechosos vs Target)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_METRICAS, "auditoria_fuga_datos.png"), dpi=150)
    plt.close()

    print("\n  [!] Auditoría completada. Heatmap guardado en metrics/auditoria_fuga_datos.png")


def analizar_residuos(df_pred, X_raw):
    """Genera gráfico de residuos y extrae un CSV con los 20 peores casos y sus texturas."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = df_pred[df_pred["Target"] == "Volumen Tumoral"].copy()
    sub["Residuo"] = sub["Predicho"] - sub["Real"]

    # 1. Gráfico de Residuos
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Real", y="Residuo", data=sub, alpha=0.7, color="#3498db")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Valores Reales vs Residuos", fontweight="bold")
    plt.xlabel("Volumen Real (mm³)")
    plt.ylabel("Residuo (Predicción - Real)")
    plt.savefig(os.path.join(CARPETA_METRICAS, "residuos_volumen.png"), dpi=150)
    plt.close()

    # 2. Extracción de los 20 peores casos
    peores_20 = sub.assign(Error_Abs=sub["Residuo"].abs()).nlargest(20, "Error_Abs")

    # Cruzar con las features crudas originales
    peores_20_features = peores_20.merge(X_raw, on="Paciente_ID", how="left")

    ruta_csv = os.path.join(CARPETA_METRICAS, "auditoria_peores_20_casos.csv")
    peores_20_features.to_csv(ruta_csv, index=False)
    print(f"\n  [🕵️] ¡Análisis listo! Residuos graficados y auditoría guardada en: {ruta_csv}")

# ===========================================================================
#  Main
# ===========================================================================
if __name__ == "__main__":
    df_pred, _, info_modelos = entrenar_y_evaluar()

    df_raw = pd.read_csv(RUTA_CSV)
    analizar_residuos(df_pred, df_raw)

    # 1. Generamos TODAS las gráficas de entrenamiento
    generar_graficas(df_pred, info_modelos)

    # 2. Predecimos los casos de prueba y generamos su gráfica
    df_verif = predecir_casos_prueba(info_modelos)
    generar_grafica_casos_prueba(df_verif, info_modelos)

    # 3. EXPORTAMOS A PRODUCCIÓN (Joblib + JSON)
    exportar_modelos_produccion(info_modelos)