"""
entrenar_modelo.py — Gradient Boosting con validación dual (10-Fold + LOO)
========================================================================
Modelado Predictivo de Regresión: Estimación precisa del volumen tumoral
y diámetros axiales para seguimiento del progreso de la enfermedad.

Pipeline: StandardScaler + GradientBoostingRegressor (log-transform)
Targets:  Volumen (mm³), Eje Corto (mm), Eje Largo (mm)
Features: radiómicas (firstorder + glcm) — shape_* eliminadas (data leakage)

Optimizaciones:
  - Log-transform de targets → reduce MAPE (errores proporcionales)
  - Feature selection por correlación → elimina ruido
  - LOO-CV con params fijos del KFold → ~6x más rápido

Validación:
  - 10-Fold CV × 2 rep (90/10 split) → tuning + estimación rápida
  - LOO-CV con params fijos → evaluación exhaustiva imparcial
  - Detección de overfitting: Gap% train vs test
  - Verificación con casos de prueba independientes (casos_prueba.csv)
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import LeaveOneOut, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, max_error,
    explained_variance_score, mean_absolute_percentage_error,
)

sys.path.insert(0, os.path.dirname(__file__))
from optimizacion import evaluar_kfold, detectar_overfitting

warnings.filterwarnings("ignore")

# ── Rutas ──
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_master.csv")
RUTA_PRUEBA = os.path.join(base_dir, "db", "casos_prueba.csv")
CARPETA_METRICAS = os.path.join(base_dir, "metrics")
CARPETA_NIFIT = os.path.join(base_dir, "Dataset_NIFIT")
os.makedirs(CARPETA_METRICAS, exist_ok=True)

# ── Configuración de validación ──
N_SPLITS = 10         # 10-Fold → 90% train, 10% test
N_REPEATS = 2         # Repeticiones (2×10 = 20 evals, suficiente)
INNER_CV = 5          # Folds internos para tuning
OVERFITTING_UMBRAL = 15.0
CORR_UMBRAL = 0.05    # Mínimo |correlación| con algún target
INTER_CORR_UMBRAL = 0.95  # Máx inter-correlación entre features

# ── Targets (origen: shape_*, eliminadas post-creación) ──
TARGETS = [
    {"nombre": "Volumen Tumoral",    "col": "target_regresion",
     "origen": "shape_VoxelVolume",  "u": "mm³", "slug": "volumen"},
    {"nombre": "Diámetro Eje Corto", "col": "target_eje_corto",
     "origen": "shape_MinorAxisLength", "u": "mm", "slug": "eje_corto"},
    {"nombre": "Diámetro Eje Largo", "col": "target_eje_largo",
     "origen": "shape_MajorAxisLength", "u": "mm", "slug": "eje_largo"},
]
COLS_TARGET = [t["col"] for t in TARGETS]


def _crear_pipeline():
    """Pipeline: StandardScaler + GradientBoostingRegressor."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            random_state=42, subsample=0.7, max_features=0.7))
    ])
    params = {
        "model__n_estimators":      [50, 100, 200],
        "model__max_depth":         [1, 2],
        "model__learning_rate":     [0.05, 0.1],
        "model__min_samples_leaf":  [16, 32, 48],
        "model__min_samples_split": [15, 25],
    }
    return pipe, params


def seleccionar_features(X, targets_dict):
    """
    Filtra features por correlación con targets y elimina redundantes.
    1. Mantiene features con |corr| > CORR_UMBRAL con al menos un target
    2. Elimina features con inter-correlación > INTER_CORR_UMBRAL
    """
    # Correlación de cada feature con cada target
    max_corr = pd.Series(0.0, index=X.columns)
    for col_target, y_arr in targets_dict.items():
        corrs = X.corrwith(pd.Series(y_arr, index=X.index)).abs()
        max_corr = np.maximum(max_corr, corrs.fillna(0))

    # Paso 1: mantener features con |corr| > umbral
    cols_ok = max_corr[max_corr > CORR_UMBRAL].index.tolist()
    X_filtrado = X[cols_ok]

    # Paso 2: eliminar features redundantes (inter-corr > 0.95)
    corr_matrix = X_filtrado.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        redundantes = upper.index[upper[col] > INTER_CORR_UMBRAL].tolist()
        for red in redundantes:
            # Eliminar la que tiene menor correlación con targets
            if max_corr[red] < max_corr[col]:
                to_drop.add(red)
            else:
                to_drop.add(col)

    cols_final = [c for c in cols_ok if c not in to_drop]
    return cols_final


def preparar_datos(df_raw):
    """
    Crea targets desde shape_* y elimina shape_* para prevenir data leakage.
    Returns: X (42 features), dict de targets {col: array}, ids, info
    """
    df = df_raw.copy()

    # Crear targets ANTES de eliminar shape_*
    for t in TARGETS:
        df[t["col"]] = df_raw[t["origen"]]

    # Eliminar shape_* → prevención data leakage
    shape_cols = [c for c in df.columns if c.startswith("shape_")]
    df = df.drop(columns=shape_cols)

    # Separar features de metadatos y targets
    cols_drop = ["Paciente_ID", "target_riesgo"] + COLS_TARGET
    X = df.drop(columns=[c for c in cols_drop if c in df.columns])
    ids = df_raw["Paciente_ID"].values
    targets = {t["col"]: df[t["col"]].values for t in TARGETS}

    info = {
        "n_muestras": len(df),
        "n_features": X.shape[1],
        "shape_eliminadas": len(shape_cols),
        "features": list(X.columns),
    }
    return X, targets, ids, info


def calcular_metricas(y_real, y_pred):
    """Calcula 8 métricas de regresión."""
    mse = mean_squared_error(y_real, y_pred)
    return {
        "MAE":    round(mean_absolute_error(y_real, y_pred), 4),
        "MSE":    round(mse, 4),
        "RMSE":   round(np.sqrt(mse), 4),
        "R²":     round(r2_score(y_real, y_pred), 4),
        "MedAE":  round(median_absolute_error(y_real, y_pred), 4),
        "MaxErr": round(max_error(y_real, y_pred), 4),
        "MAPE %": round(mean_absolute_percentage_error(y_real, y_pred) * 100, 2),
        "EVS":    round(explained_variance_score(y_real, y_pred), 4),
    }


def categorizar_nivel(valor, y_total):
    """Clasifica en Bajo/Moderado/Medio-Alto/Alto por cuartiles."""
    q1 = np.percentile(y_total, 25)
    q2 = np.percentile(y_total, 50)
    q3 = np.percentile(y_total, 75)
    
    if valor <= q1:
        return "Bajo"
    elif valor <= q2:
        return "Moderado"
    elif valor <= q3:
        return "Medio-Alto"
    else:
        return "Alto"


def entrenar_y_evaluar():
    """
    Validación dual por target:
      1) 10-Fold CV × 3 rep → estimación rápida con tuning interno (5-Fold)
      2) Nested LOO-CV → evaluación exhaustiva imparcial
      3) Detección de overfitting (Gap% train vs test)
      4) Modelo final con hiperparámetros más frecuentes del LOO

    Returns: df_pred, resultados_globales, informacion_modelos
    """
    df_raw = pd.read_csv(RUTA_CSV)
    X, targets_dict, ids, info = preparar_datos(df_raw)
    N = info["n_muestras"]

    # ── Feature Selection ──
    cols_selected = seleccionar_features(X, targets_dict)
    n_orig = X.shape[1]
    X = X[cols_selected]
    info["n_features_orig"] = n_orig
    info["n_features"] = len(cols_selected)
    info["features"] = cols_selected

    print("\n" + "═" * 70)
    print("  GRADIENT BOOSTING (log-transform) — MODELADO PREDICTIVO")
    print("═" * 70)
    print(f"  Dataset: {N} muestras × {info['n_features']} features (de {n_orig} originales)")
    print(f"  shape_* eliminadas: {info['shape_eliminadas']} (data leakage)")
    print(f"  Feature selection: |corr|>{CORR_UMBRAL} + inter-corr<{INTER_CORR_UMBRAL}")
    print(f"  Log-transform: np.log1p(y) → entrena en log-space, métricas en escala original")
    print(f"  Validación: {N_SPLITS}-Fold CV ×{N_REPEATS} rep (tuning) + LOO-CV (params fijos)")
    print(f"  Tuning interno: {INNER_CV}-Fold CV | Umbral overfitting: {OVERFITTING_UMBRAL}%")

    resultados_globales = []
    predicciones_totales = []
    informacion_modelos = {}
    t_inicio = time.time()

    for t in TARGETS:
        y_orig = targets_dict[t["col"]]
        y_log = np.log1p(y_orig)  # Log-transform
        u = t["u"]
        pipe_tmpl, params = _crear_pipeline()

        print(f"\n{'─' * 70}")
        print(f"▶ {t['nombre']} ({u})  [{y_orig.min():.1f} – {y_orig.max():.1f}]  Media: {y_orig.mean():.1f}")
        print(f"  Log-space: [{y_log.min():.2f} – {y_log.max():.2f}]")

        # ════════════════════════════════════════════════════
        # FASE 1: 10-Fold CV × 2 rep (tuning en log-space)
        # ════════════════════════════════════════════════════
        print(f"\n  ── {N_SPLITS}-Fold CV ×{N_REPEATS} rep ──")

        all_y_pred_kf = np.zeros(N)  # acumula en escala original
        all_y_count_kf = np.zeros(N)
        kf_best_params = []
        kf_train_maes = []
        kf_test_maes = []

        for rep in range(N_REPEATS):
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42 + rep)

            for train_idx, test_idx in kf.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr_log = y_log[train_idx]
                y_te_orig = y_orig[test_idx]

                inner_search = RandomizedSearchCV(
                    clone(pipe_tmpl), params,
                    n_iter=30, cv=INNER_CV,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1, random_state=42,
                )
                inner_search.fit(X_tr, y_tr_log)

                pred_tr_log = inner_search.predict(X_tr)
                pred_te_log = inner_search.predict(X_te)
                pred_tr_orig = np.expm1(pred_tr_log)
                pred_te_orig = np.expm1(pred_te_log)

                kf_best_params.append(inner_search.best_params_)
                kf_train_maes.append(mean_absolute_error(
                    np.expm1(y_tr_log), pred_tr_orig))
                kf_test_maes.append(mean_absolute_error(
                    y_te_orig, pred_te_orig))

                all_y_pred_kf[test_idx] += pred_te_orig
                all_y_count_kf[test_idx] += 1

        # Promediar predicciones sobre repeticiones (escala original)
        y_pred_kf = all_y_pred_kf / np.maximum(all_y_count_kf, 1)
        metricas_kf = calcular_metricas(y_orig, y_pred_kf)

        kf_train_mae = np.mean(kf_train_maes)
        kf_test_mae = np.mean(kf_test_maes)
        kf_gap = ((kf_test_mae - kf_train_mae) / kf_test_mae * 100) if kf_test_mae != 0 else 0

        dx_kf = "OVERFITTING" if kf_gap > OVERFITTING_UMBRAL else ("UNDERFITTING" if kf_gap < -10 else "OK")
        dx_sym = {"OK": "✓", "OVERFITTING": "⚠", "UNDERFITTING": "▽"}[dx_kf]

        # Mejores params por votación del KFold
        mejores_params = {}
        for key in kf_best_params[0]:
            values = [p[key] for p in kf_best_params]
            mejores_params[key] = Counter(values).most_common(1)[0][0]

        params_str = ", ".join(f"{k.split('__')[1]}={v}" for k, v in mejores_params.items())
        print(f"  R²={metricas_kf['R²']:.4f} | MAE={metricas_kf['MAE']:.1f} {u} | "
              f"RMSE={metricas_kf['RMSE']:.1f} {u} | MAPE={metricas_kf['MAPE %']:.1f}%")
        print(f"  Train MAE={kf_train_mae:.1f} | Test MAE={kf_test_mae:.1f} | "
              f"Gap={kf_gap:.1f}% {dx_sym} {dx_kf}")
        print(f"  Mejores params (votación): {params_str}")

        # ════════════════════════════════════════════════════
        # FASE 2: LOO-CV con params fijos (sin re-tuning)
        # ════════════════════════════════════════════════════
        print(f"\n  ── LOO-CV ({N} folds, params fijos) ──")

        loo = LeaveOneOut()
        y_pred_loo = np.zeros(N)  # escala original

        for train_idx, test_idx in loo.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr_log = y_log[train_idx]

            pipe_loo = clone(pipe_tmpl)
            pipe_loo.set_params(**mejores_params)
            pipe_loo.fit(X_tr, y_tr_log)
            y_pred_loo[test_idx] = np.expm1(pipe_loo.predict(X_te))

        metricas_loo = calcular_metricas(y_orig, y_pred_loo)

        print(f"  R²={metricas_loo['R²']:.4f} | MAE={metricas_loo['MAE']:.1f} {u} | "
              f"RMSE={metricas_loo['RMSE']:.1f} {u} | MAPE={metricas_loo['MAPE %']:.1f}%")

        # ════════════════════════════════════════════════════
        # COMPARACIÓN KFold vs LOO
        # ════════════════════════════════════════════════════
        print(f"\n  ── Comparación ──")
        print(f"  {'Métrica':<10} {'KFold':>10} {'LOO':>10}")
        print(f"  {'─'*32}")
        for m in ["R²", "MAE", "RMSE", "MAPE %"]:
            print(f"  {m:<10} {metricas_kf[m]:>10} {metricas_loo[m]:>10}")

        # ════════════════════════════════════════════════════
        # MODELO FINAL (entrenado con todos los datos en log-space)
        # ════════════════════════════════════════════════════
        pipe_final = clone(pipe_tmpl)
        pipe_final.set_params(**mejores_params)
        pipe_final.fit(X, y_log)

        importances = pipe_final.named_steps["model"].feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = [(X.columns[i], importances[i]) for i in top_idx]
        print(f"\n  Top 10 features (importancia):")
        for feat, imp in top_features:
            print(f"    {feat}: {imp:.4f}")

        # Overfitting check con modelo final (en log-space)
        ovf = detectar_overfitting(pipe_final, X, y_log, n_splits=N_SPLITS,
                                   umbral=OVERFITTING_UMBRAL)
        kfold_eval = evaluar_kfold(pipe_final, X, y_log, n_splits=N_SPLITS,
                                   n_repeats=N_REPEATS)

        informacion_modelos[t["nombre"]] = {
            "features": [f for f, _ in top_features],
            "X": X, "y": y_orig,
            "y_pred_kf": y_pred_kf,
            "y_pred_loo": y_pred_loo,
            "metricas_kf": metricas_kf,
            "metricas_loo": metricas_loo,
            "mejor_pipe": pipe_final,
            "best_params": mejores_params,
            "overfitting": ovf,
            "kfold_eval": kfold_eval,
            "unidad": u,
            "use_log": True,
            "cols_selected": list(X.columns),
        }

        resultados_globales.append({
            "Target": t["nombre"],
            "Modelo": "GradientBoosting (log)",
            "Validacion": "KFold",
            **metricas_kf,
            "Gap_%": round(kf_gap, 2),
            "Dx": dx_kf,
        })
        resultados_globales.append({
            "Target": t["nombre"],
            "Modelo": "GradientBoosting (log)",
            "Validacion": "LOO",
            **metricas_loo,
        })

        # ═══ Predicciones por paciente (usando LOO como referencia) ═══
        datos_pacientes = []
        for i in range(N):
            nivel_real = categorizar_nivel(y_orig[i], y_orig)
            nivel_pred = categorizar_nivel(y_pred_loo[i], y_orig)
            fila = {
                "Paciente_ID": ids[i],
                "Target": t["nombre"],
                "Unidad": u,
                "Real": round(y_orig[i], 2),
                "Nivel_Real": nivel_real,
                "Predicho": round(y_pred_loo[i], 2),
                "Nivel_Predicho": nivel_pred,
                "Error": round(abs(y_orig[i] - y_pred_loo[i]), 2),
                "Error %": round(abs(y_orig[i] - y_pred_loo[i]) / max(abs(y_orig[i]), 1e-9) * 100, 2),
            }
            predicciones_totales.append(fila)
            datos_pacientes.append(fila)

        # Resumen de errores (compacto en vez de 508 filas)
        errores_pct = [f["Error %"] for f in datos_pacientes]
        datos_sorted = sorted(datos_pacientes, key=lambda x: x["Error %"], reverse=True)

        print(f"\n  Top 5 mayores errores:")
        print(f"  {'Paciente':<12} {'Real':>8} {'Pred':>8} {'Err':>8} {'Err%':>7} {'Nivel':>12} {'Pred':>12}")
        print(f"  {'─'*70}")
        for f in datos_sorted[:5]:
            pac = f["Paciente_ID"].replace("case_", "")
            cambio = " ⚠" if f["Nivel_Real"] != f["Nivel_Predicho"] else ""
            print(f"  {pac:<12} {f['Real']:>8.1f} {f['Predicho']:>8.1f} "
                  f"{f['Error']:>8.1f} {f['Error %']:>6.1f}% "
                  f"{f['Nivel_Real']:>12} {(f['Nivel_Predicho']+cambio):>12}")

        n_bajo15 = sum(1 for e in errores_pct if e < 15)
        n_15_30 = sum(1 for e in errores_pct if 15 <= e < 30)
        n_30 = sum(1 for e in errores_pct if e >= 30)
        print(f"\n  Distribución errores: median={np.median(errores_pct):.1f}% | "
              f"P75={np.percentile(errores_pct, 75):.1f}% | P90={np.percentile(errores_pct, 90):.1f}%")
        print(f"  <15%: {n_bajo15}/{N} ({n_bajo15/N*100:.0f}%) | "
              f"15-30%: {n_15_30}/{N} ({n_15_30/N*100:.0f}%) | "
              f"≥30%: {n_30}/{N} ({n_30/N*100:.0f}%)")

    # ── CSVs ──
    df_pred = pd.DataFrame(predicciones_totales)
    df_pred.to_csv(os.path.join(CARPETA_METRICAS, "predicciones_loo.csv"), index=False)

    df_res = pd.DataFrame(resultados_globales)
    df_res.to_csv(os.path.join(CARPETA_METRICAS, "metricas_modelos_finales.csv"), index=False)

    # ═══ REPORTE DE OVERFITTING ═══
    print(f"\n{'═' * 70}")
    print(f"  DIAGNÓSTICO DE OVERFITTING")
    print(f"{'═' * 70}")
    print(f"  {'Target':<25} {'Train MAE':>10} {'Test MAE':>10} {'Gap%':>8} {'Diagnóstico':>15}")
    print(f"  {'─'*68}")
    for t in TARGETS:
        info = informacion_modelos[t["nombre"]]
        ovf = info["overfitting"]
        u = t["u"]
        dx_sym = {"OK": "✓", "OVERFITTING": "⚠", "UNDERFITTING": "▽"}.get(ovf["Diagnostico"], "?")
        print(f"  {t['nombre']:<25} {ovf['Train_MAE']:>9.1f}{u} {ovf['Test_MAE']:>9.1f}{u} "
              f"{ovf['Gap_%']:>7.1f}% {dx_sym} {ovf['Diagnostico']:>12}")
    print(f"\n  Umbral: Gap > {OVERFITTING_UMBRAL}% → OVERFITTING | Gap < -10% → UNDERFITTING")
    print(f"  Fórmula: Gap% = (Test_MAE − Train_MAE) / Test_MAE × 100")
    print(f"{'═' * 70}")

    t_total = time.time() - t_inicio
    print(f"\n{'═' * 70}")
    print(f"  Tiempo total: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  Resultados: metrics/predicciones_loo.csv, metrics/metricas_modelos_finales.csv")
    print(f"{'═' * 70}")

    return df_pred, resultados_globales, informacion_modelos


# -- Visualizacion --
def generar_graficas(df_pred, informacion_modelos):
    """Scatter Real vs Predicho por target (LOO y KFold)."""
    plt.close("all")

    targets_unicos = df_pred["Target"].unique()
    n = len(targets_unicos)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), num="Real vs Predicho (LOO)")
    if n == 1:
        axes = [axes]

    colores = ["#5B8DBE", "#D97560", "#70AD9F"]

    for i, target in enumerate(targets_unicos):
        ax = axes[i]
        sub = df_pred[df_pred["Target"] == target]
        real = sub["Real"].values
        pred = sub["Predicho"].values
        u = sub["Unidad"].iloc[0]

        r2 = r2_score(real, pred)
        mae = mean_absolute_error(real, pred)
        rmse = np.sqrt(mean_squared_error(real, pred))

        ax.scatter(real, pred, c=colores[i], s=110, edgecolors="white",
                   linewidth=1.5, zorder=3, alpha=0.85, label="Predicciones")

        lims = [min(real.min(), pred.min()), max(real.max(), pred.max())]
        m = (lims[1] - lims[0]) * 0.08
        lims = [lims[0] - m, lims[1] + m]
        ax.plot(lims, lims, "--", color="#A8A8A8", linewidth=2, alpha=0.6,
                label="Predicción perfecta", zorder=2)

        for _, row in sub.iterrows():
            pac = row["Paciente_ID"].replace("case_", "")
            err = row["Error %"]
            cambio = "*" if row.get("Nivel_Real", "") != row.get("Nivel_Predicho", "") else ""
            c_lbl = "#2D7F3F" if err < 15 else ("#D89D3D" if err < 30 else "#A63C37")
            ax.annotate(f"{pac}{cambio}", (row["Real"], row["Predicho"]),
                        fontsize=8, fontweight="bold", ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points", color=c_lbl)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Valor Real ({u})", fontsize=10, color="#333")
        ax.set_ylabel(f"Valor Predicho ({u})", fontsize=10, color="#333")

        # Añadir info de overfitting si existe
        info_t = informacion_modelos.get(target, {})
        ovf = info_t.get("overfitting", {})
        dx = ovf.get("Diagnostico", "")
        gap_str = f" | Gap={ovf.get('Gap_%', 0):.1f}%" if dx else ""

        ax.set_title(f"{target}\nR²={r2:.3f} | MAE={mae:.1f} {u} | RMSE={rmse:.1f} {u}{gap_str}",
                     fontweight="bold", fontsize=11, pad=10, color="#222")

        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#999")
        ax.spines["bottom"].set_color("#999")
        ax.grid(True, alpha=0.2, linestyle=":", color="#D0D0D0")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors="#666")

    fig.suptitle(f"Gradient Boosting Nested LOO-CV + {N_SPLITS}-Fold CV ×{N_REPEATS} — Real vs Predicho",
                 fontweight="bold", fontsize=13, y=0.98, color="#222")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, 0.005,
             "Verde: err<15% | Naranja: 15-30% | Rojo: >30% | * cambio de categoría",
             ha="center", fontsize=8, style="italic", color="#888")


def visualizar_casos(df_pred):
    """Grid CT + mascara por paciente con anotaciones de prediccion."""
    pacientes = df_pred["Paciente_ID"].unique()
    N = len(pacientes)

    fig, axes = plt.subplots(N, 2, figsize=(10, 4.5 * N), num="Verificacion CT + Mascara")
    if N == 1:
        axes = axes[np.newaxis, :]

    for row, pac_id in enumerate(pacientes):
        ruta_ct = os.path.join(CARPETA_NIFIT, pac_id, "image.nii.gz")
        ruta_seg = os.path.join(CARPETA_NIFIT, pac_id, "mask.nii.gz")

        if not os.path.exists(ruta_ct) or not os.path.exists(ruta_seg):
            for c in range(2):
                axes[row, c].text(0.5, 0.5, f"{pac_id}\nArchivos no encontrados",
                                  ha="center", va="center", fontsize=11, color="#999")
                axes[row, c].axis("off")
            continue

        # Leer con SimpleITK
        ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ruta_ct))
        seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(ruta_seg))

        # Slice con mayor area de mascara
        area_por_slice = np.sum(seg_arr, axis=(1, 2))
        idx_slice = int(np.argmax(area_por_slice))
        if area_por_slice[idx_slice] == 0:
            idx_slice = ct_arr.shape[0] // 2

        ct_slice = ct_arr[idx_slice, :, :]
        seg_slice = seg_arr[idx_slice, :, :]

        # Col 0: CT original
        ax0 = axes[row, 0]
        ax0.imshow(ct_slice, cmap="gray", vmin=-160, vmax=240)
        ax0.set_title(f"{pac_id} - CT (slice {idx_slice})", fontsize=10, fontweight="bold", color="#222")
        ax0.axis("off")

        # Col 1: CT + mascara overlay
        ax1 = axes[row, 1]
        ax1.imshow(ct_slice, cmap="gray", vmin=-160, vmax=240)
        ax1.imshow(seg_slice, cmap="Reds", alpha=0.5, interpolation="none")
        ax1.set_title(f"{pac_id} - CT + Mascara", fontsize=10, fontweight="bold", color="#222")
        ax1.axis("off")

        # Anotacion con predicciones de los 3 targets
        sub = df_pred[df_pred["Paciente_ID"] == pac_id]
        partes = []
        max_err = 0.0
        for _, r in sub.iterrows():
            nombre_corto = r["Target"].replace("Volumen Tumoral", "Vol") \
                                       .replace("Diametro Eje Corto", "Corto") \
                                       .replace("Diametro Eje Largo", "Largo") \
                                       .replace("Di\u00e1metro Eje Corto", "Corto") \
                                       .replace("Di\u00e1metro Eje Largo", "Largo")
            partes.append(f"{nombre_corto}: {r['Real']:.0f}->{r['Predicho']:.0f} {r['Unidad']} ({r['Error %']:.0f}%)")
            max_err = max(max_err, r["Error %"])

        texto = "  |  ".join(partes)
        color_txt = "#2D7F3F" if max_err < 15 else ("#D89D3D" if max_err < 30 else "#A63C37")

        # Texto debajo de la fila
        fig.text(0.5, 1.0 - (row + 0.97) / N,
                 texto, ha="center", fontsize=8.5, color=color_txt, fontweight="bold")

    fig.suptitle("Verificacion de Casos - CT + Mascara de Segmentacion",
                 fontsize=13, fontweight="bold", y=0.995, color="#222")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.25)


def predecir_casos_prueba(informacion_modelos):
    """
    Predice sobre casos de prueba independientes (casos_prueba.csv).
    Usa los modelos finales entrenados para verificar generalización.
    Compara predicción vs valor real (shape_*) para cada target.
    """
    if not os.path.exists(RUTA_PRUEBA):
        print("\n  ⚠ No se encontró casos_prueba.csv — omitiendo verificación")
        return None

    df_prueba = pd.read_csv(RUTA_PRUEBA)
    print(f"\n{'═' * 70}")
    print(f"  VERIFICACIÓN CON CASOS DE PRUEBA ({len(df_prueba)} casos)")
    print(f"{'═' * 70}")

    # Preparar features (misma transformación que entrenamiento)
    shape_cols = [c for c in df_prueba.columns if c.startswith("shape_")]
    cols_drop = ["Paciente_ID", "target_riesgo"] + shape_cols
    X_prueba = df_prueba.drop(columns=[c for c in cols_drop if c in df_prueba.columns])

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

        # Usar mismas features seleccionadas que entrenamiento
        X_pred = X_prueba[cols_sel] if cols_sel else X_prueba

        # Valores reales desde shape_* del CSV de prueba
        y_real = df_prueba[t["origen"]].values
        y_pred_raw = pipe.predict(X_pred)
        y_pred = np.expm1(y_pred_raw) if info.get("use_log") else y_pred_raw

        metricas = calcular_metricas(y_real, y_pred)

        print(f"\n  ▶ {nombre} ({u})")
        print(f"    R²={metricas['R²']:.4f} | MAE={metricas['MAE']:.1f} {u} | "
              f"RMSE={metricas['RMSE']:.1f} {u} | MAPE={metricas['MAPE %']:.1f}%")
        print(f"\n    {'Paciente':<12} {'Real':>10} {'Predicho':>10} {'Error':>8} {'Error%':>8} {'Nivel Real':>12} {'Nivel Pred':>12}")
        print(f"    {'─'*75}")

        for i in range(len(df_prueba)):
            pac = df_prueba["Paciente_ID"].iloc[i]
            real = y_real[i]
            pred = y_pred[i]
            err = abs(real - pred)
            err_pct = err / max(abs(real), 1e-9) * 100
            nivel_r = categorizar_nivel(real, y_train)
            nivel_p = categorizar_nivel(pred, y_train)
            cambio = " ⚠" if nivel_r != nivel_p else ""

            print(f"    {pac:<12} {real:>10.1f} {pred:>10.1f} {err:>8.1f} {err_pct:>7.1f}% {nivel_r:>12} {(nivel_p+cambio):>12}")

            resultados_prueba.append({
                "Paciente_ID": pac,
                "Target": nombre,
                "Unidad": u,
                "Real": round(real, 2),
                "Predicho": round(pred, 2),
                "Error": round(err, 2),
                "Error %": round(err_pct, 2),
                "Nivel_Real": nivel_r,
                "Nivel_Predicho": nivel_p,
            })

    # Guardar CSV de verificación
    df_verif = pd.DataFrame(resultados_prueba)
    verif_path = os.path.join(CARPETA_METRICAS, "verificacion_casos_prueba.csv")
    df_verif.to_csv(verif_path, index=False)
    print(f"\n  ✓ Verificación guardada: metrics/verificacion_casos_prueba.csv")
    print(f"{'═' * 70}")

    return df_verif


def visualizar_casos_prueba(df_verif):
    """Grid CT + máscara para casos de prueba con predicciones y margen de error."""
    if df_verif is None or df_verif.empty:
        return

    pacientes = df_verif["Paciente_ID"].unique()
    N = len(pacientes)

    fig, axes = plt.subplots(N, 2, figsize=(10, 4.5 * N),
                             num="Verificación Casos de Prueba")
    if N == 1:
        axes = axes[np.newaxis, :]

    for row, pac_id in enumerate(pacientes):
        ruta_ct = os.path.join(CARPETA_NIFIT, pac_id, "image.nii.gz")
        ruta_seg = os.path.join(CARPETA_NIFIT, pac_id, "mask.nii.gz")

        if not os.path.exists(ruta_ct) or not os.path.exists(ruta_seg):
            for c in range(2):
                axes[row, c].text(0.5, 0.5, f"{pac_id}\nArchivos no encontrados",
                                  ha="center", va="center", fontsize=11, color="#999")
                axes[row, c].axis("off")
            continue

        ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ruta_ct))
        seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(ruta_seg))

        area_por_slice = np.sum(seg_arr, axis=(1, 2))
        idx_slice = int(np.argmax(area_por_slice))
        if area_por_slice[idx_slice] == 0:
            idx_slice = ct_arr.shape[0] // 2

        ct_slice = ct_arr[idx_slice, :, :]
        seg_slice = seg_arr[idx_slice, :, :]

        # Col 0: CT original
        ax0 = axes[row, 0]
        ax0.imshow(ct_slice, cmap="gray", vmin=-160, vmax=240)
        ax0.set_title(f"{pac_id} - CT (slice {idx_slice})",
                      fontsize=10, fontweight="bold", color="#222")
        ax0.axis("off")

        # Col 1: CT + máscara overlay
        ax1 = axes[row, 1]
        ax1.imshow(ct_slice, cmap="gray", vmin=-160, vmax=240)
        ax1.imshow(seg_slice, cmap="Reds", alpha=0.5, interpolation="none")
        ax1.set_title(f"{pac_id} - CT + Máscara",
                      fontsize=10, fontweight="bold", color="#222")
        ax1.axis("off")

        # Anotación con predicciones y errores
        sub = df_verif[df_verif["Paciente_ID"] == pac_id]
        partes = []
        max_err = 0.0
        for _, r in sub.iterrows():
            nombre_corto = r["Target"].replace("Volumen Tumoral", "Vol") \
                                       .replace("Diámetro Eje Corto", "Corto") \
                                       .replace("Diámetro Eje Largo", "Largo")
            partes.append(f"{nombre_corto}: {r['Real']:.0f}→{r['Predicho']:.0f} "
                          f"{r['Unidad']} (err {r['Error %']:.0f}%)")
            max_err = max(max_err, r["Error %"])

        texto = "  |  ".join(partes)
        color_txt = "#2D7F3F" if max_err < 15 else ("#D89D3D" if max_err < 30 else "#A63C37")

        fig.text(0.5, 1.0 - (row + 0.97) / N,
                 texto, ha="center", fontsize=8.5, color=color_txt, fontweight="bold")

    fig.suptitle("Verificación Casos de Prueba — CT + Máscara + Predicciones",
                 fontsize=13, fontweight="bold", y=0.995, color="#222")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.25)


if __name__ == "__main__":
    df_pred, _, info_modelos = entrenar_y_evaluar()
    df_verif = predecir_casos_prueba(info_modelos)
    generar_graficas(df_verif, info_modelos)
    visualizar_casos_prueba(df_verif)
    plt.show()
