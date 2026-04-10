"""
optimizar_clasificacion.py — Utilidades de optimización para Clasificación
================================================================
Evaluación de métricas con StratifiedKFold, detección de overfitting,
y selección de características avanzada (RFECV) optimizada para F1-Score.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def evaluar_varianza(pipe, X, y, cv=5):
    # cross-validation f1 stats
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_weighted")
    acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")

    return {
        "F1_media": round(f1.mean(), 4),
        "F1_std": round(f1.std(), 4),
        "Acc_media": round(acc.mean(), 4),
        "Acc_std": round(acc.std(), 4),
        "Rango_F1": round(f1.max() - f1.min(), 4),
        "Estabilidad": "Alta" if f1.std() < 0.05 else ("Media" if f1.std() < 0.15 else "Baja"),
    }


def detectar_overfitting(pipe, X, y, n_splits=5, umbral=15.0):
    # Compara f1-score train vs test en stratified k-fold para detectar overfitting
    # gap% = (train_f1 - test_f1) / train_f1 * 100
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_f1s = []
    test_f1s = []

    # configuración para dataframes
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        from sklearn.base import clone as sk_clone
        p = sk_clone(pipe)
        p.fit(X_tr, y_tr)

        pred_tr = p.predict(X_tr)
        pred_te = p.predict(X_te)

        # f1-score
        train_f1s.append(f1_score(y_tr, pred_tr, average='weighted'))
        test_f1s.append(f1_score(y_te, pred_te, average='weighted'))

    train_f1 = np.mean(train_f1s)
    test_f1 = np.mean(test_f1s)

    # caída del rendimiento de train a test
    gap = ((train_f1 - test_f1) / train_f1 * 100) if train_f1 != 0 else 0.0

    if gap > umbral:
        dx = "OVERFITTING"
    elif gap < -10:
        dx = "UNDERFITTING"
    else:
        dx = "OK"

    return {
        "Train_F1": round(train_f1, 4),
        "Test_F1": round(test_f1, 4),
        "Gap_%": round(gap, 2),
        "Diagnostico": dx,
    }

# =====================================================================
#  FEATURE SELECTION (WRAPPER METHOD)
# =====================================================================
def seleccionar_features_rfecv(X, y, cv=3, min_features=5):
    # usa RFECV con un RF para encontrar el subconjunto
    # para características que maximiza el F1-Score
    print(f"        [RFECV] Evaluando {X.shape[1]} features para encontrar el subset óptimo (F1-Score)...")

    # class_weight='balanced' para que no ignore las clases críticas
    estimador_base = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42, n_jobs=None)

    # configurar el esquema de validación cruzada estratificada
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # configurar el rfecv
    selector = RFECV(
        estimator=estimador_base,
        step=1,
        cv=skf,
        scoring='f1_weighted', # OBJETIVO: MAXIMIZAR F1-SCORE
        min_features_to_select=min_features,
        n_jobs=-1
    )

    # entrenar y seleccionar
    selector.fit(X, y)

    # extraer los nombres de las columnas ganadoras
    if hasattr(X, 'columns'):
        features_optimas = X.columns[selector.support_].tolist()
    else:
        features_optimas = [i for i, val in enumerate(selector.support_) if val]

    print(f"        [RFECV] ¡Completado! Número óptimo de características: {selector.n_features_}")
    
    return {
        "selector_model": selector,
        "n_features": selector.n_features_,
        "columnas_seleccionadas": features_optimas,
    }