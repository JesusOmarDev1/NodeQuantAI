"""
entrenar_modelo.py — Modelado Predictivo de Regresión
=====================================================
Selecciona el mejor modelo por target (según evaluar_modelos.py),
entrena con GridSearchCV + LOO sobre TODOS los datos y genera
predicciones para seguimiento clínico.

Mejores modelos (evaluación Nested LOO-CV):
  • Volumen Tumoral  → Reg. Simple  (R²=0.8755)
  • Diámetro Eje Corto → Polinómica (R²=0.8327)
  • Diámetro Eje Largo  → Reg. Simple  (R²=0.9886)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg") # Para evitar errores de backend en MacOS al mostrar gráficas
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, max_error,
    explained_variance_score, mean_absolute_percentage_error,
)

warnings.filterwarnings("ignore")

# ── Rutas ──
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_regresion.csv")
CARPETA_METRICAS = os.path.join(base_dir, "metrics")
os.makedirs(CARPETA_METRICAS, exist_ok=True)

# ── Targets y sus mejores modelos (resultado de evaluar_modelos.py) ──
TARGETS = [
    {
        "nombre": "Volumen Tumoral",
        "col": "target_regresion",
        "u": "mm³",
        "slug": "volumen",
        "modelo": "reg_simple",        # LinearRegression sobre mejor feature
    },
    {
        "nombre": "Diámetro Eje Corto",
        "col": "target_eje_corto",
        "u": "mm",
        "slug": "eje_corto",
        "modelo": "polinomica",         # PolynomialFeatures + LinearRegression
    },
    {
        "nombre": "Diámetro Eje Largo",
        "col": "target_eje_largo",
        "u": "mm",
        "slug": "eje_largo",
        "modelo": "reg_simple",         # LinearRegression sobre mejor feature
    },
]
COLS_TARGET = [t["col"] for t in TARGETS]


def _crear_pipeline(tipo, mejor_feat):
    """Crea el Pipeline + param_grid según el tipo de modelo ganador."""
    if tipo == "reg_simple":
        pipe = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
        params = {}
        cols = [mejor_feat]
    elif tipo == "polinomica":
        pipe = Pipeline([
            ("s", StandardScaler()),
            ("p", PolynomialFeatures(include_bias=False)),
            ("m", LinearRegression()),
        ])
        params = {"p__degree": [2, 3]}
        cols = None  # todas las features
    else:
        raise ValueError(f"Tipo de modelo no soportado: {tipo}")
    return pipe, params, cols


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


def entrenar_y_evaluar():
    """
    Para cada target:
      1. Selecciona el mejor modelo (según evaluación previa)
      2. GridSearchCV + LOO para encontrar mejores hiperparámetros
      3. Genera predicciones LOO (real vs pred) para validación
    """
    df = pd.read_csv(RUTA_CSV)
    X_all = df.drop(columns=["Paciente_ID"] + COLS_TARGET)
    ids = df["Paciente_ID"].values
    N = df.shape[0]

    print("=" * 70)
    print("MODELADO PREDICTIVO DE REGRESIÓN")
    print("Estimación de volumen tumoral y diámetros axiales")
    print("=" * 70)
    print(f"Dataset: {N} pacientes × {X_all.shape[1]} features radiómicas")
    print(f"Fuente: ganglios_regresion.csv\n")

    resultados_globales = []
    predicciones_totales = []

    for t in TARGETS:
        y = df[t["col"]].values
        u = t["u"]

        # Mejor feature correlacionada (para reg_simple)
        corr = X_all.corrwith(pd.Series(y, index=X_all.index)).abs()
        mejor_feat = corr.idxmax()

        pipe_tmpl, params, cols = _crear_pipeline(t["modelo"], mejor_feat)
        X_usar = X_all[cols] if cols else X_all

        print("-" * 70)
        print(f"TARGET: {t['nombre']} ({u})")
        print(f"  Modelo seleccionado: {t['modelo']}")
        if cols:
            print(f"  Feature: '{mejor_feat}' (|r| = {corr[mejor_feat]:.4f})")
        else:
            print(f"  Features: {X_usar.shape[1]} (todas)")
        print(f"  Rango y: [{y.min():.2f}, {y.max():.2f}]  Media: {y.mean():.2f} {u}")

        # ── Paso 1: Predicciones LOO (evaluación imparcial) ──
        loo = LeaveOneOut()
        y_pred_loo = np.zeros(N)

        for train_idx, test_idx in loo.split(X_usar):
            X_tr, X_te = X_usar.iloc[train_idx], X_usar.iloc[test_idx]
            y_tr = y[train_idx]

            pipe = Pipeline([(name, type(step)(**step.get_params()))
                             for name, step in pipe_tmpl.steps])

            if params:
                grid = GridSearchCV(
                    pipe, params,
                    cv=LeaveOneOut(),
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                grid.fit(X_tr, y_tr)
                modelo_fold = grid.best_estimator_
            else:
                pipe.fit(X_tr, y_tr)
                modelo_fold = pipe

            y_pred_loo[test_idx] = modelo_fold.predict(X_te)

        metricas = calcular_metricas(y, y_pred_loo)
        print(f"\n  Métricas LOO-CV:")
        print(f"    R²    = {metricas['R²']:.4f}")
        print(f"    MAE   = {metricas['MAE']:.2f} {u}")
        print(f"    RMSE  = {metricas['RMSE']:.2f} {u}")
        print(f"    MAPE  = {metricas['MAPE %']:.2f}%")
        print(f"    MaxErr= {metricas['MaxErr']:.2f} {u}")

        resultados_globales.append({
            "Target": t["nombre"],
            "Modelo": t["modelo"],
            **metricas,
        })

        # Guardar predicciones por paciente
        for i in range(N):
            predicciones_totales.append({
                "Paciente_ID": ids[i],
                "Target": t["nombre"],
                "Unidad": u,
                "Real": round(y[i], 2),
                "Predicho": round(y_pred_loo[i], 2),
                "Error": round(abs(y[i] - y_pred_loo[i]), 2),
                "Error %": round(abs(y[i] - y_pred_loo[i]) / max(abs(y[i]), 1e-9) * 100, 2),
            })



    # ── CSV de predicciones ──
    df_pred = pd.DataFrame(predicciones_totales)
    pred_path = os.path.join(CARPETA_METRICAS, "predicciones_loo.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"\nPredicciones guardadas: metrics/predicciones_loo.csv ({len(df_pred)} filas)")

    # ── CSV resumen de métricas finales ──
    df_res = pd.DataFrame(resultados_globales)
    res_path = os.path.join(CARPETA_METRICAS, "metricas_modelos_finales.csv")
    df_res.to_csv(res_path, index=False)
    print(f"Métricas guardadas: metrics/metricas_modelos_finales.csv")

    return df_pred, resultados_globales


# ── Visualización ──
def generar_graficas(df_pred):
    """Gráfica Real vs Predicho por target con línea de identidad."""
    plt.close("all")

    targets_unicos = df_pred["Target"].unique()
    n = len(targets_unicos)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), num="Real vs Predicho")
    if n == 1:
        axes = [axes]

    paleta = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, target in enumerate(targets_unicos):
        ax = axes[i]
        sub = df_pred[df_pred["Target"] == target]
        real = sub["Real"].values
        pred = sub["Predicho"].values
        u = sub["Unidad"].iloc[0]

        # R² para subtítulo
        r2 = r2_score(real, pred)

        ax.scatter(real, pred, c=paleta[i], s=80, edgecolors="white",
                   linewidth=1.2, zorder=3, alpha=0.9)

        # Etiquetas de paciente
        for _, row in sub.iterrows():
            ax.annotate(row["Paciente_ID"].replace("case_", ""),
                        (row["Real"], row["Predicho"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points")

        # Línea de identidad
        lims = [min(real.min(), pred.min()), max(real.max(), pred.max())]
        margen = (lims[1] - lims[0]) * 0.1
        lims = [lims[0] - margen, lims[1] + margen]
        ax.plot(lims, lims, "--", color="#7f8c8d", linewidth=1.2, alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f"Real ({u})", fontsize=10)
        ax.set_ylabel(f"Predicho ({u})", fontsize=10)
        ax.set_title(f"{target}\nR² = {r2:.4f}", fontweight="bold", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Modelado Predictivo — Real vs Predicho (LOO-CV)\n"
                 "Seguimiento del progreso de la enfermedad",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()

    print("\nMostrando gráfica. Cierra la ventana para terminar.")
    plt.show()


# ============================================================
if __name__ == "__main__":
    df_pred, _ = entrenar_y_evaluar()
    generar_graficas(df_pred)
