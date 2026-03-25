"""
optimizar_classification.py — Utilidades de optimización para Clasificación
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
    """Cross-validation F1 stats con múltiples métricas."""
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
    """
    Compara F1-Score train vs test en Stratified K-Fold para detectar overfitting.
    En clasificación, el Gap% mide cuánto CAE el rendimiento en test.
    Gap% = (train_f1 - test_f1) / train_f1 * 100
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_f1s = []
    test_f1s = []

    # Ajuste para manejar DataFrames y Series de Pandas correctamente
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

        # Calculamos el F1-Score ponderado
        train_f1s.append(f1_score(y_tr, pred_tr, average='weighted'))
        test_f1s.append(f1_score(y_te, pred_te, average='weighted'))

    train_f1 = np.mean(train_f1s)
    test_f1 = np.mean(test_f1s)

    # Gap% en clasificación: % de caída del rendimiento de Train a Test
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
#  Selección de Características Avanzada (Wrapper Method para Clasificación)
# =====================================================================
def seleccionar_features_rfecv(X, y, cv=3, min_features=5):
    """
    Utiliza RFECV con un RandomForestClassifier para encontrar el subconjunto
    óptimo de características que maximiza el F1-Score ponderado.
    """
    print(f"        [RFECV] Evaluando {X.shape[1]} features para encontrar el subset óptimo (F1-Score)...")

    # 1. Definir el "Cirujano" (El estimador base ahora es un CLASIFICADOR)
    # class_weight='balanced' es crucial aquí para que no ignore las clases críticas
    estimador_base = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42, n_jobs=None)

    # 2. Configurar el esquema de validación cruzada ESTRATIFICADA
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # 3. Configurar el RFECV
    selector = RFECV(
        estimator=estimador_base,
        step=1,
        cv=skf,
        scoring='f1_weighted',             # Objetivo: maximizar F1-Score
        min_features_to_select=min_features,
        n_jobs=-1
    )

    # 4. Entrenar y seleccionar
    selector.fit(X, y)

    # 5. Extraer los nombres de las columnas ganadoras
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