"""
modelo_clasificacion.py - Clasificación de Riesgo Ganglionar (Multiclase)
======================================================================
Modelo: StackingClassifier (XGBoost + Random Forest + Gradient Boosting)
        Meta-learner: LogisticRegression | StandardScaler + L1 Filter
Features: radiómicas + derivadas - (shape_* y targets continuos eliminados)

Validación: Stratified 5-Fold CV con optimización de F1-Score Ponderado.
"""

import os
import sys
import time
import warnings
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

from classification.scripts.optimizar_classification import seleccionar_features_rfecv

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  Rutas
# ---------------------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUTA_CSV = os.path.join(base_dir, "db", "ganglios_master.csv")
CARPETA_METRICAS = os.path.join(base_dir, "classification", "metrics")
CARPETA_PRODUCCION = os.path.join(base_dir, "classification", "joblib")
os.makedirs(CARPETA_METRICAS, exist_ok=True)
os.makedirs(CARPETA_PRODUCCION, exist_ok=True)

# ---------------------------------------------------------------------------
#  Configuracion
# ---------------------------------------------------------------------------
N_SPLITS = 5 # Usamos 5 en clasificación para asegurar suficientes casos por clase
CORR_UMBRAL = 0.03
INTER_CORR_UMBRAL = 0.95

COLS_CLINICAS = ["Body Part Examined", "PatientSex", "PrimaryCondition"]
NOMBRES_CLASES = ["Sin riesgo", "Bajo riesgo", "Notorio", "Crítico"]

# Estilo global matplotlib
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ===========================================================================
#  Stacking Pipeline (Clasificación)
# ===========================================================================
def _crear_stacking_pipeline():
    estimators = [
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='mlogloss')),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=7,
            class_weight='balanced',
            random_state=42)),
        ("gb", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42)),
    ]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # Filtro L1 (LASSO para clasificación) usando el motor 'saga'
        ("lasso_filter", SelectFromModel(
            LogisticRegression(penalty='l1', solver='saga', C=0.1, class_weight='balanced',
                               random_state=42, max_iter=2000))),
        ("model", StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
            cv=5,
            n_jobs=None,
        )),
    ])
    return pipe, "Stacking Classifier (L1 + XGB+RF+GB)"

def _extraer_importances_stacking(pipe, feature_names):
    lasso_step = pipe.named_steps["lasso_filter"]
    surviving_features = np.array(feature_names)[lasso_step.get_support()]
    stacking = pipe.named_steps["model"]

    # En multiclase, coef_ tiene forma (n_classes, n_features). Promediamos el impacto absoluto.
    coefs = np.mean(np.abs(stacking.final_estimator_.coef_), axis=0)
    importances = np.zeros(len(surviving_features))

    for est, coef in zip(stacking.estimators_, coefs):
        if hasattr(est, "feature_importances_"):
            importances += est.feature_importances_ * coef

    total = importances.sum()
    if total > 0:
        importances /= total

    return dict(zip(surviving_features, importances))

# ===========================================================================
#  Feature engineering
# ===========================================================================
def crear_features_derivadas(X):
    # (Mantengo tu función exacta, ya que las derivadas sirven para ambos modelos)
    Xd = X.copy()
    _eps = 1e-9
    if "firstorder_Energy" in X.columns and "firstorder_Entropy" in X.columns:
        Xd["ratio_Energy_Entropy"] = X["firstorder_Energy"] / (X["firstorder_Entropy"].abs() + _eps)
    if "firstorder_Mean" in X.columns and "firstorder_Variance" in X.columns:
        Xd["ratio_Mean_Variance"] = X["firstorder_Mean"] / (X["firstorder_Variance"].abs() + _eps)
    if "firstorder_90Percentile" in X.columns and "firstorder_10Percentile" in X.columns:
        Xd["ratio_90p_10p"] = X["firstorder_90Percentile"] / (X["firstorder_10Percentile"].abs() + _eps)
    if "glcm_Correlation" in X.columns and "glcm_Contrast" in X.columns:
        Xd["ratio_Corr_Contrast"] = X["glcm_Correlation"] / (X["glcm_Contrast"].abs() + _eps)

    if "firstorder_Variance" in X.columns and "firstorder_Mean" in X.columns:
        Xd["cv_intensidad"] = np.sqrt(X["firstorder_Variance"].abs()) / (X["firstorder_Mean"].abs() + _eps)
    if "firstorder_90Percentile" in X.columns and "firstorder_10Percentile" in X.columns:
        Xd["rango_interpercentil"] = X["firstorder_90Percentile"] - X["firstorder_10Percentile"]

    if "firstorder_Energy" in X.columns:
        Xd["log_Energy"] = np.log1p(X["firstorder_Energy"].abs())
    if "firstorder_Variance" in X.columns:
        Xd["log_Variance"] = np.log1p(X["firstorder_Variance"].abs())

    Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Xd

# ===========================================================================
#  Feature selection (Clasificación)
# ===========================================================================
def seleccionar_features(X, y_target, usar_rfecv=True):
    # 1. Filtro grueso (Mutual Information para Clasificación)
    mi = mutual_info_classif(X, y_target, random_state=42)
    mi_series = pd.Series(mi, index=X.columns)
    mi_umbral = np.percentile(mi_series.values, 20) # Mantiene el top 80%
    cols_mi = set(mi_series[mi_series > mi_umbral].index)

    cols_ok = list(cols_mi)
    if not cols_ok: cols_ok = list(X.columns)
    X_filt = X[cols_ok].copy()

    # 2. Poda de Colinealidad
    corr_matrix = X_filt.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > INTER_CORR_UMBRAL)]
    X_filt = X_filt.drop(columns=to_drop)

    cols_ok = list(X_filt.columns)

    # 3. RFECV
    if usar_rfecv and len(cols_ok) > 5:
        resultado_rfecv = seleccionar_features_rfecv(X_filt, y_target, cv=3, min_features=5)
        return resultado_rfecv["columnas_seleccionadas"]
    else:
        return cols_ok

# ===========================================================================
#  Preparacion de datos
# ===========================================================================
def preparar_datos(df_raw):
    df_raw = df_raw.drop_duplicates(subset="Paciente_ID").reset_index(drop=True)
    y = df_raw["target_riesgo"].values
    ids = df_raw["Paciente_ID"].values

    # ANTI-DATA LEAKAGE: Eliminar features de forma y targets de regresión continuos
    shape_cols = [c for c in df_raw.columns if c.startswith("shape_")]
    cols_drop = ["Paciente_ID", "target_riesgo", "target_regresion", "target_eje_corto", "target_eje_largo"] + shape_cols + COLS_CLINICAS

    X = df_raw.drop(columns=[c for c in cols_drop if c in df_raw.columns])

    return X, y, ids

# ===========================================================================
#  Entrenamiento y evaluacion
# ===========================================================================
def entrenar_y_evaluar():
    df_raw = pd.read_csv(RUTA_CSV)
    X_all, y_orig, ids = preparar_datos(df_raw)
    X_all = crear_features_derivadas(X_all)

    print(f"\nMODELADO PREDICTIVO (CLASIFICACIÓN)")
    print(f"  Dataset: {len(X_all)} pacientes | {X_all.shape[1]} features iniciales")

    pipe_tmpl, nombre_modelo = _crear_stacking_pipeline()

    # ---------------------------------------------------------------
    #  Stratified KFold CV
    # ---------------------------------------------------------------
    y_pred_kf = np.zeros(len(y_orig))
    train_f1s = []
    test_f1s = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(X_all, y_orig):
        X_tr_completo, X_te_completo = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_tr, y_te = y_orig[train_idx], y_orig[test_idx]

        cols_sel_fold = seleccionar_features(X_tr_completo, y_tr)
        X_tr = X_tr_completo[cols_sel_fold]
        X_te = X_te_completo[cols_sel_fold]

        fold_pipe = clone(pipe_tmpl)

        param_grid = {
            "lasso_filter__estimator__C": [0.01, 0.1, 1.0],
            "model__final_estimator__C": [0.1, 1.0, 10.0],
            "model__rf__max_depth": [3, 5],
            "model__xgb__learning_rate": [0.01, 0.05],
            "model__xgb__max_depth": [2, 3]
        }

        search = RandomizedSearchCV(
            fold_pipe, param_distributions=param_grid,
            n_iter=5, cv=2, scoring="f1_weighted", random_state=42, n_jobs=None
        )
        search.fit(X_tr, y_tr)
        fold_pipe = search.best_estimator_

        pred_tr = fold_pipe.predict(X_tr)
        pred_te = fold_pipe.predict(X_te)

        train_f1s.append(f1_score(y_tr, pred_tr, average='weighted'))
        test_f1s.append(f1_score(y_te, pred_te, average='weighted'))

        y_pred_kf[test_idx] = pred_te # Guardamos la predicción de la clase

    f1_global = f1_score(y_orig, y_pred_kf, average='weighted')
    acc_global = accuracy_score(y_orig, y_pred_kf)
    gap = ((np.mean(train_f1s) - np.mean(test_f1s)) / np.mean(train_f1s)) * 100

    print(f"\nRESULTADOS VALIDACIÓN CRUZADA:")
    print(f"  F1-Score Ponderado: {f1_global:.4f}")
    print(f"  Accuracy Global:    {acc_global:.4f}")
    print(f"  Gap (Overfitting):  {gap:.1f}%")

    print("\nREPORTE DE CLASIFICACIÓN (K-Fold):")
    print(classification_report(y_orig, y_pred_kf, target_names=NOMBRES_CLASES))

    # ---------------------------------------------------------------
    #  Modelo final (Producción)
    # ---------------------------------------------------------------
    cols_sel_final = seleccionar_features(X_all, y_orig)
    X_final = X_all[cols_sel_final]

    pipe_base_final = clone(pipe_tmpl)
    search_final = RandomizedSearchCV(
        pipe_base_final, param_distributions=param_grid,
        n_iter=10, cv=3, scoring="f1_weighted", random_state=42, n_jobs=None
    )
    search_final.fit(X_final, y_orig)
    pipe_final = search_final.best_estimator_

    # Extraer Importancias para gráficas
    imp_dict = _extraer_importances_stacking(pipe_final, list(X_final.columns))
    top_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    # Generar Matriz de Confusión Visual
    generar_matriz_confusion(y_orig, y_pred_kf)
    generar_grafica_importancias(top_features)

    # ---------------------------------------------------------------
    #  Exportación Joblib
    # ---------------------------------------------------------------
    print(f"\nEXPORTANDO MODELO A PRODUCCIÓN ({CARPETA_PRODUCCION}):")
    ruta_joblib = os.path.join(CARPETA_PRODUCCION, "modelo_riesgo.joblib")
    joblib.dump(pipe_final, ruta_joblib)

    metadata = {
        "target": "riesgo",
        "unidad": "Categoria",
        "use_log": False,
        "n_features_esperadas": len(cols_sel_final),
        "features_entrada": cols_sel_final
    }
    ruta_json = os.path.join(CARPETA_PRODUCCION, "metadata_riesgo.json")
    with open(ruta_json, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"  [OK] modelo_riesgo.joblib guardado ({len(cols_sel_final)} features)")

def generar_matriz_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=NOMBRES_CLASES, yticklabels=NOMBRES_CLASES)
    plt.title("Matriz de Confusión - Predicción de Riesgo Ganglionar", fontweight="bold")
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_METRICAS, "matriz_confusion.png"), dpi=150)
    plt.close()

def generar_grafica_importancias(top_features):
    nombres = [f for f, v in top_features]
    valores = [v for f, v in top_features]
    plt.figure(figsize=(10, 6))
    plt.barh(nombres[::-1], valores[::-1], color='#3498db', edgecolor='white')
    plt.title("Top 10 Características Radiómicas para predecir Riesgo", fontweight="bold")
    plt.xlabel("Importancia Ponderada")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_METRICAS, "feature_importance_class.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    entrenar_y_evaluar()