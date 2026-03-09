"""
evaluar_modelos.py — Evaluación multi-target con 9-Fold CV optimizado
=====================================================================
12 modelos con Pipeline(StandardScaler) + GridSearchCV/RandomizedSearchCV.
Targets: Volumen (mm³), Eje Corto (mm), Eje Largo (mm)
Features: 42 radiómicas (firstorder + glcm) de ganglios_master.csv
Validación: 9-Fold CV × 3 repeticiones (90% train, 10% test)

Mejoras:
  - Creación de targets desde shape_* y eliminación posterior (data leakage)
  - 9-Fold CV en lugar de LOO-CV (más rápido, resultados comparables)
  - Detección de overfitting con Gap% train vs test
  - RandomizedSearchCV para modelos con espacio de búsqueda grande
  - Feature importance de modelos tree-based
  - Curvas de aprendizaje para evaluar necesidad de más datos
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

from sklearn.model_selection import (
    KFold, GridSearchCV, RandomizedSearchCV, learning_curve,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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
from xgboost import XGBRegressor

w.filterwarnings("ignore")

# ── Rutas ──
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_master.csv")
CARPETA_METRICAS = os.path.join(base_dir, "metrics")
os.makedirs(CARPETA_METRICAS, exist_ok=True)

# ── Configuración de validación ──
N_SPLITS = 5          # 5-Fold → 80% train, 20% test (más rápido)
N_REPEATS = 2         # 2 repeticiones para estimar varianza (más rápido)
INNER_CV = 3          # Folds internos para tuning de hiperparámetros (más rápido)
OVERFITTING_UMBRAL = 15.0   # Gap% > 15% indica overfitting

# ── Targets ──
TARGETS = [
    {"nombre": "Volumen Tumoral",    "col": "target_regresion",
     "origen": "shape_VoxelVolume",  "u": "mm³", "u2": "mm⁶", "slug": "volumen"},
    {"nombre": "Diámetro Eje Corto", "col": "target_eje_corto",
     "origen": "shape_MinorAxisLength", "u": "mm", "u2": "mm²", "slug": "eje_corto"},
    {"nombre": "Diámetro Eje Largo", "col": "target_eje_largo",
     "origen": "shape_MajorAxisLength", "u": "mm", "u2": "mm²", "slug": "eje_largo"},
]
COLS_TARGET = [t["col"] for t in TARGETS]


# ══════════════════════════════════════════════════════════════
#  PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════

def preparar_datos_regresion(df):
    """
    Crea targets de regresión desde shape_* y elimina features de forma
    para prevenir data leakage (columnas_regresion.txt).

    Transformaciones:
      1. target_regresion  = shape_VoxelVolume      (mm³)
      2. target_eje_corto  = shape_MinorAxisLength   (mm)
      3. target_eje_largo  = shape_MajorAxisLength   (mm)
      4. X = firstorder (19) + glcm (24) = 42 features

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

    # Eliminar TODAS las columnas shape_* → prevención data leakage
    shape_cols = [c for c in df_clean.columns if c.startswith("shape_")]
    df_clean = df_clean.drop(columns=shape_cols)

    # Contar features finales
    feature_cols = [c for c in df_clean.columns
                    if c.startswith(("firstorder_", "glcm_"))]

    info = {
        "n_muestras":       len(df_clean),
        "n_features":       len(feature_cols),
        "shape_eliminadas": len(shape_cols),
        "features":         feature_cols,
        "targets_creados":  COLS_TARGET,
    }
    return df_clean, info


# ══════════════════════════════════════════════════════════════
#  DEFINICIÓN DE MODELOS
# ══════════════════════════════════════════════════════════════

# ── Modelos + grids de hiperparámetros ──
def definir_modelos(mejor_feat):
    """
    12 modelos con Pipeline(StandardScaler) y param_grid.
    - search_type='grid'   → GridSearchCV        (pocas combinaciones)
    - search_type='random' → RandomizedSearchCV   (muchas combinaciones)
    """
    return {
        # ── Lineales ──
        "Reg. Simple": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
            "params": {},
            "search_type": "grid",
            "cols": [mejor_feat],
        },
        "Reg. Multiple": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
            "params": {},
            "search_type": "grid",
            "cols": None,
        },
        "Polinomica": {
            "pipe": Pipeline([
                ("s", StandardScaler()),
                ("p", PolynomialFeatures(include_bias=False)),
                ("m", Ridge()),
            ]),
            "params": {
                "p__degree": [2],
                "m__alpha": [10.0, 100.0, 1000.0],
            },
            "search_type": "grid",
            "cols": None,
        },
        # ── Regularizados ──
        "Ridge": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", Ridge())]),
            "params": {"m__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "search_type": "grid",
            "cols": None,
        },
        "Lasso": {
            "pipe": Pipeline([("s", StandardScaler()),
                              ("m", Lasso(max_iter=10000))]),
            "params": {"m__alpha": [0.01, 0.1, 1.0, 10.0]},
            "search_type": "grid",
            "cols": None,
        },
        "ElasticNet": {
            "pipe": Pipeline([("s", StandardScaler()),
                              ("m", ElasticNet(max_iter=10000))]),
            "params": {
                "m__alpha":    [0.01, 0.1, 1.0],
                "m__l1_ratio": [0.2, 0.5, 0.8],
            },
            "search_type": "grid",
            "cols": None,
        },
        # ── KNN ──
        "KNN": {
            "pipe": Pipeline([("s", StandardScaler()),
                              ("m", KNeighborsRegressor())]),
            "params": {
                "m__n_neighbors": [15, 20, 30, 50],
                "m__weights":     ["uniform", "distance"],
                "m__p":           [1, 2],
            },
            "search_type": "grid",    # 4×2×2 = 16 combinaciones → Grid OK
            "cols": None,
        },
        # ── SVM ──
        "SVR": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", SVR())]),
            "params": {
                "m__C":       [0.001, 0.01, 0.1, 1],
                "m__epsilon": [0.1, 0.5, 1.0],
                "m__kernel":  ["linear", "rbf"],
            },
            "search_type": "grid",    # 4×3×2 = 24 → Grid OK
            "cols": None,
        },
        # ── Árboles ──
        "Arbol Decision": {
            "pipe": Pipeline([("s", StandardScaler()),
                              ("m", DecisionTreeRegressor(random_state=42))]),
            "params": {
                "m__max_depth":        [1, 2, 3],
                "m__min_samples_leaf": [4, 8, 16],
            },
            "search_type": "grid",    # 3×3 = 9 → Grid OK
            "cols": None,
        },
        # ── Random Forest ──
        "Random Forest": {
            "pipe": Pipeline([("s", StandardScaler()),
                              ("m", RandomForestRegressor(random_state=42))]),
            "params": {
                "m__n_estimators":     [50, 100],
                "m__max_depth":        [2, 3],
                "m__min_samples_leaf": [8, 16],
            },
            "search_type": "grid",    # 2×2×2 = 8 → Grid OK
            "cols": None,
        },
        # ── Gradient Boosting ──
        "Gradient Boost": {
            "pipe": Pipeline([("s", StandardScaler()),
                              ("m", GradientBoostingRegressor(
                                  random_state=42,
                                  subsample=0.7,
                                  max_features=0.7))]),
            "params": {
                "m__n_estimators":      [10, 15, 25],
                "m__max_depth":         [1],
                "m__learning_rate":     [0.1, 0.2, 0.3],
                "m__min_samples_leaf":  [16, 32, 48],
                "m__min_samples_split": [15, 25],
            },
            "search_type": "random",  # 3×1×3×3×2 = 54 → Random
            "n_iter": 20,
            "cols": None,
        },
        # ── XGBoost ──
        "XGBoost": {
            "pipe": Pipeline([("s", StandardScaler()),
                              ("m", XGBRegressor(
                                  random_state=42,
                                  verbosity=0,
                                  subsample=0.8,
                                  colsample_bytree=0.8))]),
            "params": {
                "m__n_estimators":    [15, 25, 50],
                "m__max_depth":       [1],
                "m__learning_rate":   [0.1, 0.2, 0.3],
                "m__reg_alpha":       [5, 10],
                "m__reg_lambda":      [10, 20],
                "m__min_child_weight": [10, 20, 30],
            },
            "search_type": "random",  # 3×1×3×2×2×3 = 108 → Random
            "n_iter": 25,
            "cols": None,
        },
    }


# ══════════════════════════════════════════════════════════════
#  MÉTRICAS
# ══════════════════════════════════════════════════════════════

def calcular_metricas(y_real, y_pred):
    """Calcula 8 métricas de regresión."""
    mse = mean_squared_error(y_real, y_pred)
    return {
        "MAE":    round(mean_absolute_error(y_real, y_pred), 2),
        "MSE":    round(mse, 2),
        "RMSE":   round(np.sqrt(mse), 2),
        "R²":     round(r2_score(y_real, y_pred), 4),
        "MedAE":  round(median_absolute_error(y_real, y_pred), 2),
        "MaxErr": round(max_error(y_real, y_pred), 2),
        "MAPE %": round(mean_absolute_percentage_error(y_real, y_pred) * 100, 2),
        "EVS":    round(explained_variance_score(y_real, y_pred), 4),
    }


# ══════════════════════════════════════════════════════════════
#  DETECCIÓN DE OVERFITTING
# ══════════════════════════════════════════════════════════════

def analizar_overfitting(train_maes, test_maes, nombre_modelo):
    """
    Compara MAE de train vs test promediando sobre todos los folds/repeticiones.

    Gap% = (test_mae − train_mae) / test_mae × 100
      > 15%  → overfitting (modelo memoriza train)
      < −10% → underfitting (modelo muy simple)
      else   → balance adecuado

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
    return {"OK": "✓", "OVERFITTING": "⚠", "UNDERFITTING": "▽"}.get(
        diagnostico, "?")


# ══════════════════════════════════════════════════════════════
#  PARAMS MÁS FRECUENTES
# ══════════════════════════════════════════════════════════════

def _params_mas_frecuentes(lista_params):
    """Devuelve los hiperparámetros más elegidos entre los folds."""
    if not lista_params:
        return ""
    tuplas = [tuple(sorted(d.items())) for d in lista_params]
    mas_comun = Counter(tuplas).most_common(1)[0][0]
    return ", ".join(f"{k.split('__')[1]}={v}" for k, v in mas_comun)


# ══════════════════════════════════════════════════════════════
#  FASE 1: EVALUACIÓN (cómputo sin gráficas)
# ══════════════════════════════════════════════════════════════

def ejecutar_evaluacion():
    """
    9-Fold CV × 3 repeticiones con análisis de overfitting:
      • Outer loop  → KFold(9) para evaluación imparcial (90/10).
      • Inner loop  → GridSearchCV / RandomizedSearchCV con 5-Fold para tuning.
      • Overfitting → Compara MAE train vs test por fold.

    Returns
    -------
    todos : dict con resultados por target (slug → {cfg, resultados, df})
    """
    # ── Cargar y validar ──
    df_raw = pd.read_csv(RUTA_CSV)

    print("\n" + "═" * 80)
    print("  VALIDACIÓN DE DATOS DE ENTRADA")
    print("═" * 80)

    required = ["Paciente_ID", "target_riesgo"] + [t["origen"] for t in TARGETS]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"❌ Columnas requeridas faltantes: {missing}")

    shape_n = sum(1 for c in df_raw.columns if c.startswith("shape_"))
    fo_n    = sum(1 for c in df_raw.columns if c.startswith("firstorder_"))
    glcm_n  = sum(1 for c in df_raw.columns if c.startswith("glcm_"))
    print(f"  ✓ CSV cargado: {len(df_raw)} filas × {len(df_raw.columns)} columnas")
    print(f"  ✓ shape: {shape_n} | firstorder: {fo_n} | glcm: {glcm_n}")

    # ── Preparar datos (crear targets + eliminar shape_*) ──
    df, prep_info = preparar_datos_regresion(df_raw)

    print(f"\n  PREPARACIÓN:")
    print(f"  ✓ Targets creados: {', '.join(prep_info['targets_creados'])}")
    print(f"  ✓ shape_* eliminadas: {prep_info['shape_eliminadas']} columnas")
    print(f"  ✓ Features finales: {prep_info['n_features']} (firstorder + glcm)")

    # ── Separar X (features) de targets y metadatos ──
    cols_drop = ["Paciente_ID", "target_riesgo"] + COLS_TARGET
    X = df.drop(columns=[c for c in cols_drop if c in df.columns])
    N = len(df)

    print(f"\n" + "═" * 80)
    print(f"  EVALUACIÓN DE MODELOS DE REGRESIÓN")
    print("═" * 80)
    print(f"  Dataset: {N} muestras × {X.shape[1]} features")
    print(f"  Validación: {N_SPLITS}-Fold CV × {N_REPEATS} repeticiones")
    print(f"  Tuning interno: {INNER_CV}-Fold CV")
    print(f"  Umbral overfitting: Gap% > {OVERFITTING_UMBRAL}%")
    print()

    todos = {}
    t_inicio = time.time()

    for t in TARGETS:
        y = df[t["col"]].values
        u = t["u"]

        # Mejor feature para regresión simple (correlación absoluta)
        corr = X.corrwith(pd.Series(y, index=X.index)).abs()
        mejor_feat = corr.idxmax()

        modelos = definir_modelos(mejor_feat)

        print(f"\n{'─' * 80}")
        print(f"▶ {t['nombre']} ({u})")
        print(f"  Rango: {y.min():.2f} – {y.max():.2f} {u} | "
              f"Media: {y.mean():.2f} {u} | "
              f"Mejor feature: {mejor_feat} (r={corr[mejor_feat]:.3f})")
        print()

        resultados = []
        overfitting_info = []

        for nombre, cfg in modelos.items():
            X_usar = X[cfg["cols"]] if cfg["cols"] else X
            params = cfg["params"]
            search_type = cfg.get("search_type", "grid")
            n_iter = cfg.get("n_iter", 10)

            try:
                # Acumuladores para todas las repeticiones
                y_pred_acum = np.zeros(N)
                conteo_acum = np.zeros(N)
                train_maes_folds = []
                test_maes_folds = []
                mejores_params = []

                for rep in range(N_REPEATS):
                    kfold = KFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=42 + rep)

                    for train_idx, test_idx in kfold.split(X_usar):
                        X_tr = X_usar.iloc[train_idx]
                        X_te = X_usar.iloc[test_idx]
                        y_tr = y[train_idx]
                        y_te = y[test_idx]

                        pipe = clone(cfg["pipe"])

                        if params:
                            inner_cv = KFold(n_splits=INNER_CV, shuffle=True,
                                             random_state=42)

                            if search_type == "random":
                                searcher = RandomizedSearchCV(
                                    pipe, params,
                                    n_iter=n_iter,
                                    cv=inner_cv,
                                    scoring="neg_mean_absolute_error",
                                    n_jobs=1,
                                    random_state=42,
                                )
                            else:
                                searcher = GridSearchCV(
                                    pipe, params,
                                    cv=inner_cv,
                                    scoring="neg_mean_absolute_error",
                                    n_jobs=1,
                                )
                            searcher.fit(X_tr, y_tr)
                            modelo_fit = searcher.best_estimator_
                            mejores_params.append(searcher.best_params_)
                        else:
                            pipe.fit(X_tr, y_tr)
                            modelo_fit = pipe

                        # Predicciones
                        pred_te = modelo_fit.predict(X_te)
                        pred_tr = modelo_fit.predict(X_tr)

                        # Acumular predicciones test (promedio entre repeticiones)
                        y_pred_acum[test_idx] += pred_te
                        conteo_acum[test_idx] += 1

                        # MAE por fold para análisis de overfitting
                        train_maes_folds.append(
                            mean_absolute_error(y_tr, pred_tr))
                        test_maes_folds.append(
                            mean_absolute_error(y_te, pred_te))

                # Promediar predicciones sobre repeticiones
                mask = conteo_acum > 0
                y_pred = np.where(mask, y_pred_acum / conteo_acum, 0)

                # Métricas globales
                m = calcular_metricas(y, y_pred)

                # Análisis de overfitting
                ovf = analizar_overfitting(train_maes_folds, test_maes_folds,
                                           nombre)
                m["Train_MAE"] = ovf["Train_MAE"]
                m["Gap_%"]     = ovf["Gap_%"]

                # Params más frecuentes
                m["Best_Params"] = _params_mas_frecuentes(mejores_params)

                resultados.append({"Modelo": nombre, **m})
                overfitting_info.append({"Modelo": nombre, **ovf})

            except Exception as e:
                print(f"  {nombre:18s} | ERROR: {e}")
                nan_row = {
                    "Modelo": nombre, "MAE": np.nan, "MSE": np.nan,
                    "RMSE": np.nan, "R²": np.nan, "MedAE": np.nan,
                    "MaxErr": np.nan, "MAPE %": np.nan, "EVS": np.nan,
                    "Train_MAE": np.nan, "Gap_%": np.nan, "Best_Params": "",
                }
                resultados.append(nan_row)
                overfitting_info.append({
                    "Modelo": nombre, "Train_MAE": np.nan,
                    "Test_MAE": np.nan, "Gap_%": np.nan,
                    "Diagnostico": "ERROR", "Recomendacion": str(e),
                })

        # ── Tabla compacta de resultados ──
        df_res = pd.DataFrame(resultados).sort_values("MAE")
        print(f"\n  {'Modelo':<18} {'R²':>7} {'MAE':>10} {'Train':>8} "
              f"{'Gap%':>7} {'MAPE':>7}  Dx")
        print("  " + "═" * 70)
        for _, row in df_res.iterrows():
            if pd.notna(row["MAE"]):
                ovf_dx = next(
                    (o for o in overfitting_info if o["Modelo"] == row["Modelo"]),
                    {"Diagnostico": "?"})
                sym = _recomendacion_color(ovf_dx["Diagnostico"])
                print(f"  {row['Modelo']:<18} {row['R²']:>7.3f} "
                      f"{row['MAE']:>8.1f}{u:>2} {row['Train_MAE']:>7.1f} "
                      f"{row['Gap_%']:>6.1f}% {row['MAPE %']:>6.1f}%  "
                      f"{sym} {ovf_dx['Diagnostico']}")

        # Mejor modelo
        validos = [r for r in resultados if not np.isnan(r["MAE"])]
        if validos:
            mejor = min(validos, key=lambda r: r["MAE"])
            print(f"\n  ✓ Mejor: {mejor['Modelo']} | "
                  f"MAE={mejor['MAE']:.1f} {u} | R²={mejor['R²']:.3f} | "
                  f"MAPE={mejor['MAPE %']:.1f}%")
            if mejor.get("Best_Params"):
                print(f"    Params: {mejor['Best_Params']}")

        # Modelos con overfitting
        ovf_modelos = [o for o in overfitting_info
                       if o["Diagnostico"] == "OVERFITTING"]
        if ovf_modelos:
            print(f"\n  ⚠ Modelos con overfitting ({len(ovf_modelos)}):")
            for o in ovf_modelos:
                print(f"    • {o['Modelo']:18s} Gap={o['Gap_%']:>5.1f}% "
                      f"→ {o['Recomendacion']}")

        todos[t["slug"]] = {
            "cfg": t,
            "resultados": resultados,
            "overfitting": overfitting_info,
            "df": df_res,
        }
        print()

    # ── CSV consolidado ──
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

    t_total = time.time() - t_inicio
    print(f"\n{'═' * 80}")
    print(f"  Tiempo total: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  Resultados: metrics/comparativa_general.csv")
    print(f"{'═' * 80}")

    # Guardar X y df para uso en visualizaciones
    todos["_X"] = X
    todos["_df"] = df

    return todos



# ══════════════════════════════════════════════════════════════
#  FASE 2: VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════

def generar_graficas(todos):
    """
    Crea 5 figuras resumen:
      1. R² comparativo (barras agrupadas)
      2. Heatmap mejor modelo por target
      3. Análisis de overfitting (Gap% por modelo)
      4. Feature importance (top 15)
      5. Curvas de aprendizaje
    """
    plt.close("all")

    X = todos.pop("_X")
    df = todos.pop("_df")

    slugs = [s for s in todos.keys()]
    n_tgt = len(slugs)

    # ══════════════════════════════════════════════════════════
    #  GRÁFICA 1: R² COMPARATIVO
    # ══════════════════════════════════════════════════════════
    modelos_set = []
    for slug in slugs:
        for r in todos[slug]["resultados"]:
            if (r["Modelo"] not in modelos_set
                    and not np.isnan(r.get("R²", np.nan))):
                modelos_set.append(r["Modelo"])

    n_mod = len(modelos_set)
    y_pos = np.arange(n_mod)
    altura = 0.8 / n_tgt
    paleta = ["#3498db", "#e74c3c", "#2ecc71"]

    fig1, ax1 = plt.subplots(figsize=(13, max(6, n_mod * 0.55)))
    for i, slug in enumerate(slugs):
        cfg = todos[slug]["cfg"]
        r2_map = {r["Modelo"]: r["R²"] for r in todos[slug]["resultados"]
                  if not np.isnan(r.get("R²", np.nan))}
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
    ax1.set_title(f"R² por Modelo y Target — {N_SPLITS}-Fold CV × "
                  f"{N_REPEATS} rep + GridSearchCV/RandomizedSearchCV\n"
                  "1.0 = perfecto  |  0 = media  |  < 0 = peor que la media",
                  fontweight="bold", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9)
    fig1.tight_layout()

    # ══════════════════════════════════════════════════════════
    #  GRÁFICA 2: HEATMAP MEJORES MODELOS
    # ══════════════════════════════════════════════════════════
    filas = []
    row_labels = []
    cols_heat = ["MAE", "RMSE", "R²", "MedAE", "MaxErr", "MAPE %", "EVS",
                 "Gap_%"]

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

    mayor_mejor = ["R²", "EVS"]
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

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    sns.heatmap(df_n, annot=df_h.values, fmt=".2f", cmap="RdYlGn_r",
                linewidths=2, linecolor="white", ax=ax2,
                cbar_kws={"label": "← Mejor          Peor →", "shrink": 0.8},
                annot_kws={"size": 10, "fontweight": "bold"})
    ax2.set_title("Mejor Modelo por Target — Métricas + Overfitting\n"
                  "(verde = mejor, rojo = peor)", fontweight="bold",
                  fontsize=12)
    ax2.set_ylabel("")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right",
                        fontsize=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)
    fig2.tight_layout()

    # ══════════════════════════════════════════════════════════
    #  GRÁFICA 3: ANÁLISIS DE OVERFITTING (Gap% por modelo)
    # ══════════════════════════════════════════════════════════
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

    fig3.suptitle(f"Análisis de Overfitting — "
                  f"Umbral: Gap% > {OVERFITTING_UMBRAL}%",
                  fontweight="bold", fontsize=13, y=1.02)
    fig3.tight_layout()

    # ══════════════════════════════════════════════════════════
    #  GRÁFICA 4: FEATURE IMPORTANCE (modelos tree-based)
    # ══════════════════════════════════════════════════════════
    tree_names = ["Random Forest", "Gradient Boost", "XGBoost",
                  "Arbol Decision"]

    fig4, axes4 = plt.subplots(1, n_tgt, figsize=(6 * n_tgt, 7),
)
    if n_tgt == 1:
        axes4 = [axes4]

    for i, slug in enumerate(slugs):
        ax = axes4[i]
        cfg_t = todos[slug]["cfg"]
        y = df[cfg_t["col"]].values

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
        nombre_tree = mejor_tree["Modelo"]

        # Re-entrenar para extraer feature importance
        corr = X.corrwith(pd.Series(y, index=X.index)).abs()
        mejor_f = corr.idxmax()
        modelos_def = definir_modelos(mejor_f)
        cfg_m = modelos_def[nombre_tree]
        pipe_tree = clone(cfg_m["pipe"])

        params = cfg_m["params"]
        if params:
            inner_cv = KFold(n_splits=INNER_CV, shuffle=True, random_state=42)
            st = cfg_m.get("search_type", "grid")
            if st == "random":
                gs = RandomizedSearchCV(
                    pipe_tree, params,
                    n_iter=cfg_m.get("n_iter", 10),
                    cv=inner_cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1, random_state=42)
            else:
                gs = GridSearchCV(
                    pipe_tree, params, cv=inner_cv,
                    scoring="neg_mean_absolute_error", n_jobs=1)
            gs.fit(X, y)
            final_model = gs.best_estimator_
        else:
            pipe_tree.fit(X, y)
            final_model = pipe_tree

        modelo_interno = final_model.named_steps["m"]
        if hasattr(modelo_interno, "feature_importances_"):
            importancias = pd.Series(
                modelo_interno.feature_importances_,
                index=X.columns
            ).sort_values(ascending=True)

            top15 = importancias.tail(15)
            colores = ["#3498db" if f.startswith("firstorder_") else "#e67e22"
                       for f in top15.index]
            labels = [f.replace("firstorder_", "fo_").replace("glcm_", "")
                      for f in top15.index]
            top15.index = labels
            top15.plot.barh(ax=ax, color=colores, edgecolor="white")
            ax.set_xlabel("Importancia", fontsize=10)
            ax.set_title(f"{cfg_t['nombre']}\n{nombre_tree} "
                         f"(R²={mejor_tree['R²']:.3f})",
                         fontweight="bold", fontsize=11)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#3498db", label="First Order"),
                Patch(facecolor="#e67e22", label="GLCM"),
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{nombre_tree}\nno tiene\nfeature_importances_",
                    ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            ax.set_title(f"{cfg_t['nombre']}")

    fig4.suptitle("Feature Importance — Top 15 Features por Target",
                  fontweight="bold", fontsize=13, y=1.02)
    fig4.tight_layout()

    # ══════════════════════════════════════════════════════════
    #  GRÁFICA 5: CURVAS DE APRENDIZAJE
    # ══════════════════════════════════════════════════════════
    fig5, axes5 = plt.subplots(1, n_tgt, figsize=(6 * n_tgt, 5),
)
    if n_tgt == 1:
        axes5 = [axes5]

    for i, slug in enumerate(slugs):
        ax = axes5[i]
        cfg_t = todos[slug]["cfg"]
        y = df[cfg_t["col"]].values

        validos = [r for r in todos[slug]["resultados"]
                   if not np.isnan(r.get("MAE", np.nan))]
        if not validos:
            continue
        mejor = min(validos, key=lambda r: r["MAE"])
        nombre_mejor = mejor["Modelo"]

        corr = X.corrwith(pd.Series(y, index=X.index)).abs()
        mejor_f = corr.idxmax()
        modelos_def = definir_modelos(mejor_f)
        cfg_m = modelos_def[nombre_mejor]
        X_lc = X[cfg_m["cols"]] if cfg_m["cols"] else X
        pipe_lc = clone(cfg_m["pipe"])

        try:
            train_sizes = np.linspace(0.2, 1.0, 6)
            train_sizes_abs, train_scores, test_scores = learning_curve(
                pipe_lc, X_lc, y,
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

            ax.set_xlabel("Tamaño del Dataset de Entrenamiento", fontsize=10)
            ax.set_ylabel(f"MAE ({cfg_t['u']})", fontsize=10)
            ax.set_title(f"{cfg_t['nombre']}\n{nombre_mejor}",
                         fontweight="bold", fontsize=11)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}",
                    ha="center", va="center", fontsize=9,
                    transform=ax.transAxes)
            ax.set_title(f"{cfg_t['nombre']}\n{nombre_mejor}")

    fig5.suptitle("Curvas de Aprendizaje — ¿Más datos mejorarían el modelo?",
                  fontweight="bold", fontsize=13, y=1.02)
    fig5.tight_layout()

    # Guardar todas las figuras en metrics/
    metrics_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    for i, fig in enumerate(plt.get_fignums(), 1):
        f = plt.figure(fig)
        ruta = os.path.join(metrics_dir, f"grafica_{i}.png")
        f.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\n  ✓ 5 gráficas guardadas en metrics/grafica_1..5.png")


# ══════════════════════════════════════════════════════════════
#  GLOSARIO
# ══════════════════════════════════════════════════════════════

def imprimir_glosario():
    print("\nGLOSARIO DE MÉTRICAS:")
    print("━" * 80)
    metricas = [
        ("R²",        "Varianza explicada: 1=perfecto, >0=bueno, ≤0=malo"),
        ("MAE",       "Error absoluto medio (mismas unidades que el target)"),
        ("RMSE",      "Raíz del error cuadrático medio (penaliza outliers)"),
        ("MAPE",      "Error porcentual medio (%)"),
        ("Gap%",      f"(Test−Train)/Test×100: >{OVERFITTING_UMBRAL}%=overfitting"),
        ("Train_MAE", "MAE sobre datos de entrenamiento"),
        ("MedAE",     "Error absoluto mediano (robusto a outliers)"),
        ("MaxErr",    "Error máximo (peor caso)"),
        ("EVS",       "Explained Variance Score (similar a R²)"),
    ]
    for metrica, desc in metricas:
        print(f"  {metrica:10s} → {desc}")
    print()
    print("DIAGNÓSTICOS DE OVERFITTING:")
    print("━" * 80)
    print(f"  ✓ OK           → Gap% ≤ {OVERFITTING_UMBRAL}% y ≥ −10%")
    print(f"  ⚠ OVERFITTING  → Gap% > {OVERFITTING_UMBRAL}% (modelo memoriza train)")
    print(f"  ▽ UNDERFITTING → Gap% < −10% (modelo muy simple)")
    print()


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    resultados = ejecutar_evaluacion()
    imprimir_glosario()
    generar_graficas(resultados)
