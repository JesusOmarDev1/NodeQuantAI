"""
optimizacion_regression.py — Utilidades de optimización para regresión Ridge
================================================================
Evaluación de varianza con KFold y LOO, detección de overfitting,
y análisis de estabilidad del modelo.
"""

import numpy as np
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor


def evaluar_varianza(pipe, X, y, cv=5):
    """Cross-validation R² stats con múltiples métricas."""
    r2 = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
    return {
        "R2_media": round(r2.mean(), 4),
        "R2_std": round(r2.std(), 4),
        "R2_min": round(r2.min(), 4),
        "R2_max": round(r2.max(), 4),
        "MAE_media": round(mae.mean(), 4),
        "MAE_std": round(mae.std(), 4),
        "Rango_R2": round(r2.max() - r2.min(), 4),
        "Estabilidad": "Alta" if r2.std() < 0.05 else ("Media" if r2.std() < 0.15 else "Baja"),
    }


def evaluar_kfold(pipe, X, y, n_splits=9, n_repeats=3):
    """
    9-Fold CV × n repeticiones (90% train, 10% test).
    Retorna métricas promediadas y por fold.
    """
    all_r2 = []
    all_mae = []

    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + rep)
        r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        mae = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
        all_r2.extend(r2)
        all_mae.extend(mae)

    all_r2 = np.array(all_r2)
    all_mae = np.array(all_mae)

    return {
        "metodo": f"{n_splits}-Fold CV x{n_repeats}",
        "n_evals": len(all_r2),
        "R2_media": round(all_r2.mean(), 4),
        "R2_std": round(all_r2.std(), 4),
        "R2_min": round(all_r2.min(), 4),
        "R2_max": round(all_r2.max(), 4),
        "MAE_media": round(all_mae.mean(), 4),
        "MAE_std": round(all_mae.std(), 4),
        "Estabilidad": "Alta" if all_r2.std() < 0.05 else ("Media" if all_r2.std() < 0.15 else "Baja"),
    }


def evaluar_loo(pipe, X, y):
    """LOO-CV exhaustivo (N folds, 1 muestra test)."""
    loo = LeaveOneOut()
    r2 = cross_val_score(pipe, X, y, cv=loo, scoring="r2")
    mae = -cross_val_score(pipe, X, y, cv=loo, scoring="neg_mean_absolute_error")
    return {
        "metodo": "LOO-CV",
        "n_evals": len(r2),
        "R2_media": round(r2.mean(), 4),
        "R2_std": round(r2.std(), 4),
        "MAE_media": round(mae.mean(), 4),
        "MAE_std": round(mae.std(), 4),
    }


def detectar_overfitting(pipe, X, y, n_splits=9, umbral=15.0):
    """
    Compara MAE train vs test en K-Fold para detectar overfitting.
    Gap% = (test_mae - train_mae) / test_mae × 100
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_maes = []
    test_maes = []

    for train_idx, test_idx in kf.split(X):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_te = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        from sklearn.base import clone as sk_clone
        p = sk_clone(pipe)
        p.fit(X_tr, y_tr)

        pred_tr = p.predict(X_tr)
        pred_te = p.predict(X_te)

        train_maes.append(np.mean(np.abs(y_tr - pred_tr)))
        test_maes.append(np.mean(np.abs(y_te - pred_te)))

    train_mae = np.mean(train_maes)
    test_mae = np.mean(test_maes)
    gap = ((test_mae - train_mae) / test_mae * 100) if test_mae != 0 else 0.0

    if gap > umbral:
        dx = "OVERFITTING"
    elif gap < -10:
        dx = "UNDERFITTING"
    else:
        dx = "OK"

    return {
        "Train_MAE": round(train_mae, 4),
        "Test_MAE": round(test_mae, 4),
        "Gap_%": round(gap, 2),
        "Diagnostico": dx,
    }

# =====================================================================
#  Selección de Características Avanzada (Wrapper Method)
# =====================================================================
def seleccionar_features_rfecv(X, y, cv=3, min_features=5):
    """
    Utiliza RFECV con un RandomForestRegressor para encontrar el subconjunto
    óptimo de características que minimiza el error absoluto medio (MAE).
    
    Parámetros:
    - X: DataFrame con las features (después del filtro grueso).
    - y: Array/Series con el target (ej. Volumen Tumoral).
    - cv: Número de particiones para la validación cruzada interna.
    - min_features: Número mínimo de columnas a conservar.
    
    Retorna un diccionario con las columnas seleccionadas y el modelo entrenado.
    """
    print(f"        [RFECV] Evaluando {X.shape[1]} features para encontrar el subset óptimo...")

    # 1. Definir el "Cirujano" (El estimador base)
    # Usamos Random Forest porque maneja muy bien las relaciones no lineales
    estimador_base = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=None) # Probar None en lugar de -1 por si la compu se congela o tarda mucho

    # 2. Configurar el esquema de validación cruzada
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # 3. Configurar el RFECV
    selector = RFECV(
        estimator=estimador_base,
        step=1,                            # Elimina la peor característica 1 a 1
        cv=kf,                             # Evalúa el impacto usando K-Fold
        scoring='neg_mean_absolute_error', # Objetivo: minimizar el MAE
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