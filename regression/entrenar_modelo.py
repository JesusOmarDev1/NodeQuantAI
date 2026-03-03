"""Ridge regularizado con Nested LOO-CV y SelectKBest interno."""
"""Este modelo sirve para predecir los 3 targets (volumen, eje corto, eje largo) con un pipeline de Ridge regularizado (L2) y selección de features interna (SelectKBest con f_regression). El tuning del hiperparámetro alpha se realiza dentro de cada fold del LOO-CV para evitar data leakage. Se calculan 8 métricas de regresión y se categoriza cada predicción en niveles (Bajo/Moderado/Medio-Alto/Alto) según los cuartiles del target. Además, se genera un CSV con las predicciones por paciente y un resumen de métricas finales. Finalmente, se visualizan gráficos de Real vs Predicho y CTs con máscaras para verificar casos individuales."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("TkAgg") # Para evitar problemas con las graficas en algunos entornos
import matplotlib.pyplot as plt

from scipy.stats import loguniform # Para búsqueda de alpha en espacio logarítmico
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, max_error,
    explained_variance_score, mean_absolute_percentage_error,
)

# Importar utilidades de optimización
sys.path.insert(0, os.path.dirname(__file__))
from optimizacion import evaluar_varianza

warnings.filterwarnings("ignore")

# Rutas
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_regresion.csv")
CARPETA_METRICAS = os.path.join(base_dir, "metrics")
CARPETA_NIFIT = os.path.join(base_dir, "Dataset_NIFIT")
os.makedirs(CARPETA_METRICAS, exist_ok=True)

# ── Targets y sus mejores modelos (resultado de evaluar_modelos.py) ──
TARGETS = [
    {
        "nombre": "Volumen Tumoral",
        "col": "target_regresion",
        "u": "mm³",
        "slug": "volumen",
    },
    {
        "nombre": "Diámetro Eje Corto",
        "col": "target_eje_corto",
        "u": "mm",
        "slug": "eje_corto",
    },
    {
        "nombre": "Diámetro Eje Largo",
        "col": "target_eje_largo",
        "u": "mm",
        "slug": "eje_largo",
    },
]
COLS_TARGET = [t["col"] for t in TARGETS]


def _crear_pipeline(k_features=5):
    """Crea Pipeline: StandardScaler + SelectKBest(f_regression) + Ridge(L2)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_regression, k=k_features)),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])
    params = {"model__alpha": loguniform(0.001, 1000)} # Espacio de búsqueda logarítmico para alpha
    return pipe, params


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
    """Nested LOO-CV por target. Retorna predicciones, metricas e info de modelos."""
    df = pd.read_csv(RUTA_CSV)
    X_all = df.drop(columns=["Paciente_ID"] + COLS_TARGET)
    ids = df["Paciente_ID"].values
    N = df.shape[0]

    print("\n" + "═" * 60)
    print(f"  RIDGE (L2) | {N} pac x {X_all.shape[1]} feat | Nested LOO-CV")
    print("═" * 60)

    resultados_globales = []
    predicciones_totales = []
    informacion_modelos = {}

    for t in TARGETS:
        y = df[t["col"]].values
        u = t["u"]
        
        pipe_tmpl, params = _crear_pipeline(k_features=5)
        
        print(f"\n▶ {t['nombre']} ({u})  [{y.min():.1f}–{y.max():.1f}]")

        # ── Nested LOO-CV (tuning dentro de cada fold — sin data leakage) ──
        loo = LeaveOneOut()
        y_pred_loo = np.zeros(N)
        alphas_por_fold = []
        mae_inner_por_fold = []
        
        for train_idx, test_idx in loo.split(X_all):
            X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
            y_tr = y[train_idx]
            
            inner_search = RandomizedSearchCV(
                clone(pipe_tmpl), params,
                n_iter=20,
                cv=LeaveOneOut(),
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                random_state=42,
            )
            inner_search.fit(X_tr, y_tr)
            
            alphas_por_fold.append(inner_search.best_params_["model__alpha"])
            mae_inner_por_fold.append(-inner_search.best_score_)
            y_pred_loo[test_idx] = inner_search.predict(X_te)

        metricas = calcular_metricas(y, y_pred_loo)
        
        # ── cv_results_ resumen: alpha y MAE por fold ──
        alphas_arr = np.array(alphas_por_fold)
        mae_arr = np.array(mae_inner_por_fold)
        alpha_mediana = float(np.median(alphas_arr))
        
        print(f"  alpha={alpha_mediana:.4f}")
        print(f"  R²={metricas['R²']:.4f} | EVS={metricas['EVS']:.4f}")
        print(f"  MAE={metricas['MAE']:.1f} {u} | MSE={metricas['MSE']:.1f} | RMSE={metricas['RMSE']:.1f} {u}")
        print(f"  MedAE={metricas['MedAE']:.1f} {u} | MaxErr={metricas['MaxErr']:.1f} {u} | MAPE={metricas['MAPE %']:.1f}%")

        # ── Modelo final para visualización (todos los datos) ──
        pipe_final = clone(pipe_tmpl)
        pipe_final.set_params(model__alpha=alpha_mediana)
        pipe_final.fit(X_all, y)
        
        # Extraer features seleccionadas por el modelo final
        selector_final = pipe_final.named_steps["selector"]
        selected_mask = selector_final.get_support()
        selected_features = X_all.columns[selected_mask].tolist()
        
        print(f"  Features: {', '.join(selected_features)}")
        
        # Análisis de varianza con el pipeline completo
        analisis_var = evaluar_varianza(pipe_final, X_all, y, cv=N)

        informacion_modelos[t["nombre"]] = {
            "features": selected_features,
            "X_usar": X_all,
            "y": y,
            "y_pred": y_pred_loo,
            "mejor_pipe": pipe_final,
            "alpha": alpha_mediana,
            "metricas": metricas,
            "varianza": analisis_var,
            "unidad": u,
            "alphas_por_fold": alphas_por_fold,
        }

        resultados_globales.append({
            "Target": t["nombre"],
            "Modelo": f"Ridge (alpha_med={alpha_mediana:.4f})",
            **metricas,
        })

        # Guardar predicciones por paciente con categoría ganglionar
        datos_pacientes = []
        for i in range(N):
            nivel_real = categorizar_nivel(y[i], y)
            nivel_pred = categorizar_nivel(y_pred_loo[i], y)
            fila = {
                "Paciente_ID": ids[i],
                "Target": t["nombre"],
                "Unidad": u,
                "Real": round(y[i], 2),
                "Nivel_Real": nivel_real,
                "Predicho": round(y_pred_loo[i], 2),
                "Nivel_Predicho": nivel_pred,
                "Error": round(abs(y[i] - y_pred_loo[i]), 2),
                "Error %": round(abs(y[i] - y_pred_loo[i]) / max(abs(y[i]), 1e-9) * 100, 2),
            }
            predicciones_totales.append(fila)
            datos_pacientes.append(fila)
        
        # Imprimir tabla de predicciones por paciente
        print(f"  {'Paciente':<12} {'Real':<10} {'Nivel Real':<15} {'Predicho':<10} {'Nivel Pred':<15} {'Error':<8} {'Error %':<8}")
        print(f"  {'-'*92}")
        for fila in datos_pacientes:
            paciente = fila["Paciente_ID"].replace("case_", "")
            real = fila["Real"]
            nivel_real = fila["Nivel_Real"]
            predicho = fila["Predicho"]
            nivel_pred = fila["Nivel_Predicho"]
            error = fila["Error"]
            error_pct = fila["Error %"]
            cambio = " ⚠" if nivel_real != nivel_pred else ""
            print(f"  {paciente:<12} {real:<10.1f} {(nivel_real+cambio):<15} {predicho:<10.1f} {nivel_pred:<15} {error:<8.1f} {error_pct:<8.1f}%")
        print()



    # ── CSV de predicciones ──
    df_pred = pd.DataFrame(predicciones_totales)
    pred_path = os.path.join(CARPETA_METRICAS, "predicciones_loo.csv")
    df_pred.to_csv(pred_path, index=False)

    # ── CSV resumen de métricas finales ──
    df_res = pd.DataFrame(resultados_globales)
    res_path = os.path.join(CARPETA_METRICAS, "metricas_modelos_finales.csv")
    df_res.to_csv(res_path, index=False)
    
    print(f"\n  Guardado en metrics/")
    print("═" * 60)

    return df_pred, resultados_globales, informacion_modelos


# -- Visualizacion --
def generar_graficas(df_pred):
    """Scatter Real vs Predicho por target."""
    plt.close("all")

    targets_unicos = df_pred["Target"].unique()
    n = len(targets_unicos)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), num="Real vs Predicho")
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

        # Linea identidad
        lims = [min(real.min(), pred.min()), max(real.max(), pred.max())]
        m = (lims[1] - lims[0]) * 0.08
        lims = [lims[0] - m, lims[1] + m]
        ax.plot(lims, lims, "--", color="#A8A8A8", linewidth=2, alpha=0.6,
                label="Prediccion perfecta", zorder=2)

        # Etiquetas paciente coloreadas por error
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
        ax.set_title(f"{target}\nR2={r2:.3f} | MAE={mae:.1f} {u} | RMSE={rmse:.1f} {u}",
                     fontweight="bold", fontsize=11, pad=10, color="#222")

        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#999")
        ax.spines["bottom"].set_color("#999")
        ax.grid(True, alpha=0.2, linestyle=":", color="#D0D0D0")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors="#666")

    fig.suptitle("Ridge (Nested LOO-CV) - Real vs Predicho",
                 fontweight="bold", fontsize=13, y=0.98, color="#222")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.text(0.5, 0.005,
             "Verde: err<15% | Naranja: 15-30% | Rojo: >30% | * cambio de categoria",
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


if __name__ == "__main__":
    df_pred, _, _ = entrenar_y_evaluar()
    generar_graficas(df_pred)
    visualizar_casos(df_pred)
    plt.show()
