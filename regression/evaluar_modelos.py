"""
entrenar_modelo.py — Evaluación multi-target con GridSearchCV optimizado
========================================================================
12 modelos con Pipeline(StandardScaler) + GridSearchCV nested LOO-CV.
Targets: Volumen (mm³), Eje Corto (mm), Eje Largo (mm)
Features: 42 radiómicas (firstorder + glcm) de ganglios_regresion.csv
Validación: Nested LOO-CV (outer LOO para eval, inner LOO para tuning)
"""

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg") # Para evitar errores de backend en algunos sistemas
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneOut, GridSearchCV
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

warnings.filterwarnings("ignore")

# ── Rutas ──
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_regresion.csv")
CARPETA_METRICAS = os.path.join(base_dir, "metrics")
os.makedirs(CARPETA_METRICAS, exist_ok=True)

# ── Targets ──
TARGETS = [
    {"nombre": "Volumen Tumoral",    "col": "target_regresion",  "u": "mm³", "u2": "mm⁶", "slug": "volumen"},
    {"nombre": "Diámetro Eje Corto", "col": "target_eje_corto",  "u": "mm",  "u2": "mm²", "slug": "eje_corto"},
    {"nombre": "Diámetro Eje Largo", "col": "target_eje_largo",  "u": "mm",  "u2": "mm²", "slug": "eje_largo"},
]
COLS_TARGET = [t["col"] for t in TARGETS]


# ── Modelos + grids de hiperparámetros ──
def definir_modelos(mejor_feat):
    """
    12 modelos con Pipeline(StandardScaler) y param_grid para GridSearchCV.
    Modelos sin params → ajuste directo (sin tuning).
    Modelos con params → GridSearchCV nested LOO para seleccionar hiperparámetros.
    """
    return {
        # ── Lineales ──
        "Reg. Simple": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
            "params": {},
            "cols": [mejor_feat],
        },
        "Reg. Multiple": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
            "params": {},
            "cols": None,
        },
        "Polinomica": {
            "pipe": Pipeline([
                ("s", StandardScaler()),
                ("p", PolynomialFeatures(include_bias=False)),
                ("m", LinearRegression()),
            ]),
            "params": {"p__degree": [2, 3]},
            "cols": None,
        },
        # ── Regularizados ──
        "Ridge": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", Ridge())]),
            "params": {"m__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "cols": None,
        },
        "Lasso": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", Lasso(max_iter=10000))]),
            "params": {"m__alpha": [0.01, 0.1, 1.0, 10.0]},
            "cols": None,
        },
        "ElasticNet": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", ElasticNet(max_iter=10000))]),
            "params": {
                "m__alpha": [0.01, 0.1, 1.0],
                "m__l1_ratio": [0.2, 0.5, 0.8],
            },
            "cols": None,
        },
        # ── KNN Vecinos ──
        "KNN": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor())]),
            "params": {
                "m__n_neighbors": [2, 3],
                "m__weights": ["uniform", "distance"],
                "m__p": [1, 2],
            },
            "cols": None,
        },
        # ── SVM ──
        "SVR": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", SVR())]),
            "params": {
                "m__C": [0.1, 1, 10],
                "m__epsilon": [0.01, 0.1, 0.5],
                "m__kernel": ["linear", "rbf"],
            },
            "cols": None,
        },
        # ── Árboles ──
        "Arbol Decision": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", DecisionTreeRegressor(random_state=42))]),
            "params": {
                "m__max_depth": [1, 2, 3],
                "m__min_samples_leaf": [1, 2],
            },
            "cols": None,
        },
        # ── Random Forest ──
        "Random Forest": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", RandomForestRegressor(random_state=42))]),
            "params": {
                "m__n_estimators": [50, 100],
                "m__max_depth": [1, 2, 3],
            },
            "cols": None,
        },
        # ── Boosting ──
        "Gradient Boost": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", GradientBoostingRegressor(random_state=42))]),
            "params": {
                "m__n_estimators": [50, 100],
                "m__max_depth": [1, 2],
                "m__learning_rate": [0.01, 0.1, 0.5],
            },
            "cols": None,
        },
        # ── XGBoost ──
        "XGBoost": {
            "pipe": Pipeline([("s", StandardScaler()), ("m", XGBRegressor(random_state=42, verbosity=0))]),
            "params": {
                "m__n_estimators": [50, 100],
                "m__max_depth": [1, 2, 3],
                "m__learning_rate": [0.01, 0.1, 0.5],
            },
            "cols": None,
        },
    }


def calcular_metricas(y_real, y_pred):
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


def _params_mas_frecuentes(lista_params):
    """Devuelve los hiperparámetros más elegidos entre los folds como string."""
    if not lista_params:
        return ""
    tuplas = [tuple(sorted(d.items())) for d in lista_params]
    mas_comun = Counter(tuplas).most_common(1)[0][0]
    return ", ".join(f"{k.split('__')[1]}={v}" for k, v in mas_comun)


# ── Fase 1: CÓMPUTO (sin gráficas) ──
def ejecutar_evaluacion():
    """
    Nested LOO-CV:
      • Outer loop  → LOO (N folds, train=N-1, test=1) para evaluación imparcial.
      • Inner loop  → GridSearchCV con LOO sobre el train set para tuning.
    Retorna dict con resultados por target.
    """
    df = pd.read_csv(RUTA_CSV)
    X = df.drop(columns=["Paciente_ID"] + COLS_TARGET)
    N = df.shape[0]
    print(f"Dataset: {N} muestras × {X.shape[1]} features")
    print(f"Fuente: ganglios_regresion.csv")
    print(f"Validación: Nested LOO-CV (outer LOO={N} folds, inner GridSearchCV LOO={N-1} folds)\n")

    loo = LeaveOneOut()
    todos = {}

    for t in TARGETS:
        y = df[t["col"]].values
        u = t["u"]

        # Mejor feature para regresión simple
        corr = X.corrwith(pd.Series(y, index=X.index)).abs()
        mejor_feat = corr.idxmax()

        modelos = definir_modelos(mejor_feat)

        print("=" * 80)
        print(f"TARGET: {t['nombre']} ({u})")
        print(f"  Rango: {y.min():.2f} — {y.max():.2f}  |  Media: {y.mean():.2f} {u}")
        print(f"  Mejor feature simple: '{mejor_feat}' (|r| = {corr[mejor_feat]:.4f})")
        print(f"  Modelos: {len(modelos)} | Grids: {sum(1 for v in modelos.values() if v['params'])} con tuning")
        print("-" * 80)

        resultados = []
        for nombre, cfg in modelos.items():
            X_usar = X[cfg["cols"]] if cfg["cols"] else X
            params = cfg["params"]

            try:
                y_pred = np.zeros(N)
                mejores_params = []

                for train_idx, test_idx in loo.split(X_usar):
                    X_tr = X_usar.iloc[train_idx]
                    X_te = X_usar.iloc[test_idx]
                    y_tr = y[train_idx]

                    pipe = clone(cfg["pipe"])

                    if params:
                        grid = GridSearchCV(
                            pipe, params,
                            cv=LeaveOneOut(),
                            scoring="neg_mean_absolute_error",
                            n_jobs=-1,
                        )
                        grid.fit(X_tr, y_tr)
                        modelo_fit = grid.best_estimator_
                        mejores_params.append(grid.best_params_)
                    else:
                        pipe.fit(X_tr, y_tr)
                        modelo_fit = pipe

                    y_pred[test_idx] = modelo_fit.predict(X_te)

                m = calcular_metricas(y, y_pred)
                resultados.append({"Modelo": nombre, **m})

                if mejores_params:
                    freq = _params_mas_frecuentes(mejores_params)
                    print(f"  {nombre:18s} | R²={m['R²']:>8.4f} | MAE={m['MAE']:>10.2f} {u}"
                          f" | MAPE={m['MAPE %']:>7.2f}% | Best: {freq}")
                else:
                    print(f"  {nombre:18s} | R²={m['R²']:>8.4f} | MAE={m['MAE']:>10.2f} {u}"
                          f" | MAPE={m['MAPE %']:>7.2f}%")

            except Exception as e:
                print(f"  {nombre:18s} | ERROR: {e}")
                resultados.append({
                    "Modelo": nombre, "MAE": np.nan, "MSE": np.nan,
                    "RMSE": np.nan, "R²": np.nan, "MedAE": np.nan,
                    "MaxErr": np.nan, "MAPE %": np.nan, "EVS": np.nan,
                })

        # Tabla
        df_res = pd.DataFrame(resultados).sort_values("MAE")
        print(f"\n  Tabla ({t['nombre']}, ordenada por MAE):")
        print(df_res.to_string(index=False))

        validos = [r for r in resultados if not np.isnan(r["MAE"])]
        if validos:
            mejor = min(validos, key=lambda r: r["MAE"])
            print(f"\n  MEJOR: {mejor['Modelo']} — MAE={mejor['MAE']:.2f} {u}, "
                  f"R²={mejor['R²']:.4f}, MAPE={mejor['MAPE %']:.2f}%")

        todos[t["slug"]] = {"cfg": t, "resultados": resultados, "df": df_res}
        print()

    # CSV único consolidado
    frames = []
    for slug, data in todos.items():
        df_tmp = data["df"].copy()
        df_tmp.insert(0, "Target", data["cfg"]["nombre"])
        frames.append(df_tmp)
    df_all = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(CARPETA_METRICAS, "comparativa_general.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"Guardado: metrics/comparativa_general.csv ({len(df_all)} filas)")

    return todos


# ── Fase 2: VISUALIZACIÓN (2 figuras consolidadas) ──
def generar_graficas(todos):
    """Crea 2 figuras resumen y las muestra con un solo plt.show()."""
    plt.close("all")

    # ── Gráfica 1: R² comparativo (barras agrupadas, 3 targets) ──
    slugs = list(todos.keys())
    modelos_set = []
    for slug in slugs:
        for r in todos[slug]["resultados"]:
            if r["Modelo"] not in modelos_set and not np.isnan(r.get("R²", np.nan)):
                modelos_set.append(r["Modelo"])

    n_mod = len(modelos_set)
    n_tgt = len(slugs)
    y_pos = np.arange(n_mod)
    altura = 0.8 / n_tgt
    paleta = ["#3498db", "#e74c3c", "#2ecc71"]

    fig1, ax1 = plt.subplots(figsize=(13, max(6, n_mod * 0.55)), num="R² Comparativo")
    for i, slug in enumerate(slugs):
        cfg = todos[slug]["cfg"]
        r2_map = {r["Modelo"]: r["R²"] for r in todos[slug]["resultados"]
                  if not np.isnan(r.get("R²", np.nan))}
        vals = [r2_map.get(m, np.nan) for m in modelos_set]
        offset = (i - n_tgt / 2 + 0.5) * altura
        bars = ax1.barh(y_pos + offset, vals, height=altura * 0.9,
                        label=f"{cfg['nombre']} ({cfg['u']})",
                        color=paleta[i % len(paleta)], edgecolor="white", alpha=0.85)
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
    ax1.set_title("R² por Modelo y Target — Nested LOO-CV + GridSearchCV\n"
                   "1.0 = perfecto  |  0 = media  |  < 0 = peor que la media",
                   fontweight="bold", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9)
    fig1.tight_layout()

    # ── Gráfica 2: Heatmap resumen (mejor modelo por target) ──
    filas = []
    filas_normalizadas = []
    row_labels = []
    cols = ["MAE", "RMSE", "R²", "MedAE", "MaxErr", "MAPE %", "EVS"]

    for slug in slugs:
        cfg = todos[slug]["cfg"]
        validos = [r for r in todos[slug]["resultados"]
                   if not np.isnan(r.get("MAE", np.nan))]
        if not validos:
            continue
        mejor = min(validos, key=lambda r: r["MAE"])
        row_labels.append(f"{cfg['nombre']}\n({mejor['Modelo']})")
        filas.append([mejor.get(c, np.nan) for c in cols])

    df_h = pd.DataFrame(filas, columns=cols, index=row_labels)

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

    fig2, ax2 = plt.subplots(figsize=(12, 4), num="Heatmap Mejores Modelos")
    sns.heatmap(df_n, annot=df_h.values, fmt=".2f", cmap="RdYlGn_r",
                linewidths=2, linecolor="white", ax=ax2,
                cbar_kws={"label": "← Mejor          Peor →", "shrink": 0.8},
                annot_kws={"size": 11, "fontweight": "bold"})
    ax2.set_title("Mejor Modelo por Target — Métricas Completas\n"
                   "(verde = mejor, rojo = peor)", fontweight="bold", fontsize=12)
    ax2.set_ylabel("")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)
    fig2.tight_layout()

    print("\nMostrando 2 gráficas. Cierra las ventanas para terminar.")
    plt.show()


# ── Glosario ──
def imprimir_glosario():
    print("\n" + "=" * 80)
    print("GLOSARIO DE MÉTRICAS")
    print("=" * 80)
    print("""
  MAE    Error Absoluto Medio. Promedio de |real - pred|. [0,∞) Menor=mejor.
  MSE    Error Cuadrático Medio. Promedio de (real-pred)². [0,∞) Menor=mejor.
  RMSE   √MSE. Misma unidad que el target. [0,∞) Menor=mejor.
  R²     Varianza explicada. 1=perfecto, 0=media, <0=peor. (-∞,1] Mayor=mejor.
  MedAE  Mediana del error absoluto. Robusto a outliers. [0,∞) Menor=mejor.
  MaxErr Peor predicción individual. [0,∞) Menor=mejor.
  MAPE   Error porcentual medio. [0,∞) Menor=mejor.
  EVS    Varianza explicada (sin penalizar bias). (-∞,1] Mayor=mejor.
  GridSearchCV  Búsqueda exhaustiva de hiperparámetros por validación cruzada.
  Nested CV     CV externo (evaluación) + CV interno (tuning) → sin data leakage.
  Pipeline      StandardScaler + Modelo encapsulados para evitar leakage en CV.
""")


# ============================================================
if __name__ == "__main__":
    resultados = ejecutar_evaluacion()
    imprimir_glosario()
    generar_graficas(resultados)
