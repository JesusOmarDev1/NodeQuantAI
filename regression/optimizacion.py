"""Utilidades de optimizacion: evaluacion de varianza."""

import numpy as np
from sklearn.model_selection import cross_val_score

def evaluar_varianza(pipe, X, y, cv=5):
    """Cross-validation R2 stats."""
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='r2')
    return {
        'R2 media': round(scores.mean(), 4),
        'R2 desv. est.': round(scores.std(), 4),
        'R2 min': round(scores.min(), 4),
        'R2 max': round(scores.max(), 4),
        'Rango': round(scores.max() - scores.min(), 4),
        'Estabilidad': 'Alta' if scores.std() < 0.05 else ('Media' if scores.std() < 0.15 else 'Baja'),
        'Scores por fold': [round(s, 4) for s in scores]
    }
