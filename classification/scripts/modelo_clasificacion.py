"""
modelo_clasificacion.py - Clasificación de Riesgo Ganglionar (Multiclase)
======================================================================
Modelo: StackingClassifier (XGBoost + Random Forest + Gradient Boosting)
        Meta-learner: LogisticRegression | StandardScaler + L1 Filter
Features: radiómicas + derivadas - (shape_* y targets continuos eliminados)

Validación: Stratified 5-Fold CV con optimización de F1-Score Ponderado.
"""

import os
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
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score, cohen_kappa_score,
    mean_absolute_error, matthews_corrcoef, log_loss,
    roc_curve, auc, precision_recall_curve, average_precision_score
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
RUTA_PRUEBA = os.path.join(base_dir, "db", "casos_prueba.csv")
os.makedirs(CARPETA_METRICAS, exist_ok=True)
os.makedirs(CARPETA_PRODUCCION, exist_ok=True)

# ---------------------------------------------------------------------------
#  Configuracion
# ---------------------------------------------------------------------------
N_SPLITS = 5 # Usamos 5 en clasificación para asegurar suficientes casos por clase
CORR_UMBRAL = 0.03
INTER_CORR_UMBRAL = 0.98

COLS_CLINICAS = ["Body Part Examined", "PatientSex", "PrimaryCondition"]
NOMBRES_CLASES = ["Bajo", "Moderado", "Medio-Alto", "Crítico"]

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
            random_state=42, eval_metric='mlogloss', n_jobs=2)),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=7,
            class_weight='balanced',
            random_state=42, n_jobs=2)),
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
            n_jobs=-1,
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

    print(f"\nMODELADO PREDICTIVO (CLASIFICACIÓN MULTICLASE)")
    print(f"  Dataset: {len(X_all)} pacientes | {X_all.shape[1]} features iniciales")

    pipe_tmpl, nombre_modelo = _crear_stacking_pipeline()

    # Variables para guardar predicciones de todo el K-Fold
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    train_f1s = []
    test_f1s = []
    train_accs = []

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
            n_iter=5, cv=2, scoring="f1_weighted", random_state=42, n_jobs=1
        )
        search.fit(X_tr, y_tr)
        fold_pipe = search.best_estimator_

        # Predicciones de clase y probabilidades
        pred_tr = fold_pipe.predict(X_tr)
        pred_te = fold_pipe.predict(X_te)
        proba_te = fold_pipe.predict_proba(X_te)

        # Guardamos para el reporte final global
        all_y_true.extend(y_te)
        all_y_pred.extend(pred_te)
        all_y_proba.extend(proba_te)
        train_f1s.append(f1_score(y_tr, pred_tr, average='weighted'))
        test_f1s.append(f1_score(y_te, pred_te, average='weighted'))
        train_accs.append(accuracy_score(y_tr, pred_tr))

    # ==========================================================
    # CÁLCULO DE MÉTRICAS CLÍNICAS GLOBALES (Sobre los 508 casos)
    # ==========================================================
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    # 1. Métricas Estándar Multiclase
    acc_global = accuracy_score(all_y_true, all_y_pred)
    prec_global = precision_score(all_y_true, all_y_pred, average='weighted')
    rec_global = recall_score(all_y_true, all_y_pred, average='weighted')
    f1_global = f1_score(all_y_true, all_y_pred, average='weighted')

    # 2. Especificidad y NPV (Métricas Clínicas derivadas de Matriz Confusión)
    cm = confusion_matrix(all_y_true, all_y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    especificidad = np.mean(TN / (TN + FP))
    npv = np.mean(TN / (TN + FN))

    # 3. Métricas Probabilísticas y de Calibración
    auc_global = roc_auc_score(all_y_true, all_y_proba, multi_class='ovr', average='weighted')
    mcc_global = matthews_corrcoef(all_y_true, all_y_pred)
    log_loss_global = log_loss(all_y_true, all_y_proba)

    # 4. Métricas de Orden / Severidad
    kappa_global = cohen_kappa_score(all_y_true, all_y_pred, weights='quadratic')
    mae_ordinal = mean_absolute_error(all_y_true, all_y_pred)

    # 5. Overfitting y Estabilidad
    train_f1_mean = np.mean(train_f1s)
    train_acc_mean = np.mean(train_accs)
    gap_f1 = ((train_f1_mean - f1_global) / train_f1_mean) * 100

    print(f"\n" + "=" * 65)
    print(f"       PANEL EXHAUSTIVO DE MÉTRICAS CLÍNICAS (10-Fold)")
    print(f"=" * 65)
    print(f" [Métricas de Clasificación Médica]")
    print(f"  • Exactitud (Accuracy)      : {acc_global:.4f}  (Aciertos totales)")
    print(f"  • Precisión Ponderada       : {prec_global:.4f}  (Fiabilidad cuando detecta algo)")
    print(f"  • Sensibilidad (Recall)     : {rec_global:.4f}  (Capacidad de no dejar ir positivos)")
    print(f"  • Especificidad Macro       : {especificidad:.4f}  (Capacidad de descartar sanos)")
    print(f"  • Valor Predictivo Neg. NPV : {npv:.4f}  (Seguridad de un diagnóstico negativo)")
    print(f"  • F1-Score Ponderado        : {f1_global:.4f}  (Balance Precisión/Sensibilidad)")
    print(f"")
    print(f" [Métricas de Probabilidad y Robustez Estadistica]")
    print(f"  • ROC-AUC (OVR)             : {auc_global:.4f}  (Calidad probabilística)")
    print(f"  • Coeficiente Matthews (MCC): {mcc_global:.4f}  (Correlación global de aciertos)")
    print(f"  • Log Loss (Cross-Entropy)  : {log_loss_global:.4f}  (Incertidumbre en las probabilidades)")
    print(f"")
    print(f" [Métricas de Gravedad Ordinal]")
    print(f"  • Índice Kappa Cuadrático   : {kappa_global:.4f}  (Concordancia penalizando errores graves)")
    print(f"  • MAE Categórico (Escalones): {mae_ordinal:.4f}  (Desviación promedio en niveles de riesgo)")
    print(f"")
    print(f" [Diagnóstico de Aprendizaje y Sobreajuste]")
    print(f"  • Train F1-Score            : {train_f1_mean:.4f}")
    print(f"  • Test F1-Score             : {f1_global:.4f}")
    print(f"  • Train Accuracy            : {train_acc_mean:.4f}")
    print(f"  • Test Accuracy             : {acc_global:.4f}")
    print(f"  • Gap de Sobreajuste (F1)   : {gap_f1:.1f}%")
    print(f"=" * 65)

    print("\nREPORTE DETALLADO POR CATEGORÍA:")
    print(classification_report(all_y_true, all_y_pred, target_names=NOMBRES_CLASES))

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

    imp_dict = _extraer_importances_stacking(pipe_final, list(X_final.columns))
    top_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    generar_matriz_confusion(all_y_true, all_y_pred)
    generar_grafica_importancias(top_features)
    generar_curvas_roc(all_y_true, all_y_proba)
    generar_curvas_pr(all_y_true, all_y_proba)
    generar_grafica_overfitting(train_f1s, test_f1s)


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
    predecir_casos_prueba(pipe_final, cols_sel_final)



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

def generar_curvas_roc(y_true, y_proba):
    # Binarizamos las etiquetas para el enfoque Uno-vs-Resto (OVR)
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    colores = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    for i, color in zip(range(4), colores):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{NOMBRES_CLASES[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC Multiclase (OVR)', fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_METRICAS, "curvas_roc.png"), dpi=150)
    plt.close()


def generar_curvas_pr(y_true, y_proba):
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    colores = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    for i, color in zip(range(4), colores):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        avg_pr = average_precision_score(y_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{NOMBRES_CLASES[i]} (Avg PR = {avg_pr:.2f})')

    plt.xlabel('Recall (Sensibilidad)')
    plt.ylabel('Precisión')
    plt.title('Curvas Precision-Recall', fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_METRICAS, "curvas_pr.png"), dpi=150)
    plt.close()


def generar_grafica_overfitting(train_f1s, test_f1s):
    folds = np.arange(1, len(train_f1s) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(folds, train_f1s, 'o-', color='#2ecc71', linewidth=2, label='Train F1-Score')
    plt.plot(folds, test_f1s, 's-', color='#e74c3c', linewidth=2, label='Test F1-Score')

    plt.fill_between(folds, train_f1s, test_f1s, color='#e74c3c', alpha=0.1, label='Gap (Sobreajuste)')
    plt.title('Análisis de Sobreajuste por K-Fold', fontweight="bold")
    plt.xlabel('Número de Fold')
    plt.ylabel('F1-Score Ponderado')
    plt.xticks(folds)
    plt.ylim([0.4, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_METRICAS, "analisis_overfitting.png"), dpi=150)
    plt.close()

def predecir_casos_prueba(modelo, cols_sel):
    if not os.path.exists(RUTA_PRUEBA):
        print(f"\n[!] Archivo de casos de prueba no encontrado en: {RUTA_PRUEBA}")
        return

    df_prueba = pd.read_csv(RUTA_PRUEBA)
    print(f"\n" + "="*50)
    print(f" VERIFICACIÓN EN CASOS DE PRUEBA ({len(df_prueba)} pacientes)")
    print(f"==================================================")

    y_real = df_prueba["target_riesgo"].values
    ids = df_prueba["Paciente_ID"].values

    # 1. Limpieza anti-data leakage
    shape_cols = [c for c in df_prueba.columns if c.startswith("shape_")]
    cols_drop = ["Paciente_ID", "target_riesgo", "target_regresion", "target_eje_corto", "target_eje_largo"] + shape_cols + COLS_CLINICAS
    X_prueba = df_prueba.drop(columns=[c for c in cols_drop if c in df_prueba.columns])

    # 2. Magia matemática (Derivadas) y Filtro de Columnas exactas
    X_prueba = crear_features_derivadas(X_prueba)
    X_pred = X_prueba.reindex(columns=cols_sel, fill_value=0)

    # 3. Predicción
    y_pred = modelo.predict(X_pred)

    acc = accuracy_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred, average='weighted')

    print(f"  • Exactitud (Accuracy) : {acc:.4f}")
    print(f"  • F1-Score Ponderado   : {f1:.4f}\n")

    print("  [Detalle del Veredicto por Paciente]")
    for pac, real, pred in zip(ids, y_real, y_pred):
        r_nom = NOMBRES_CLASES[int(real)]
        p_nom = NOMBRES_CLASES[int(pred)]
        marca = "✅ ACIERTO" if real == pred else "❌ FALLO"
        print(f"   - {pac}: Real = {r_nom:<10} | Predicho = {p_nom:<10} -> {marca}")

if __name__ == "__main__":
    entrenar_y_evaluar()