<<<<<<< HEAD
import joblib
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# convertir el clasificador en un modelo ordinal
class ClasificadorOrdinalFrankHall(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.estimators_ = []
        for i in range(len(self.classes_) - 1):
            y_binario = (y > self.classes_[i]).astype(int)
            clon = clone(self.estimator)
            clon.fit(X, y_binario)
            self.estimators_.append(clon)
        return self

    def predict_proba(self, X):
        probs_mayor_que = [est.predict_proba(X)[:, 1] for est in self.estimators_]
        probs = np.zeros((X.shape[0], len(self.classes_)))
        probs[:, 0] = 1.0 - probs_mayor_que[0]
        for i in range(1, len(self.classes_) - 1):
            probs[:, i] = probs_mayor_que[i-1] - probs_mayor_que[i]
        probs[:, -1] = probs_mayor_que[-1]
        probs = np.clip(probs, 0.0, 1.0)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

ruta_modelo = os.path.join("classification", "joblib", "modelo_clasificacion.joblib")

if os.path.exists(ruta_modelo):
    modelo = joblib.load(ruta_modelo)

    print("\n" + "=" * 80)
    print("EXTRACCIÓN DE PARÁMETROS DEL MODELO CAMPEÓN DE CLASIFICACIÓN (JOBLIB)")
    print("=" * 80)

    # tomamos el primer fold calibrado
    pipeline_interno = modelo.calibrated_classifiers_[0].estimator

    # extraer el preprocesamiento
    escalador = pipeline_interno.named_steps['scaler']
    print(f"\nPREPROCESAMIENTO:")
    print(f"    Escalador        : {escalador.__class__.__name__}")

    # extraer lasso
    lasso_filter = pipeline_interno.named_steps['lasso_filter'].estimator
    print(f"\nFILTRO DE CARACTERÍSTICAS (LASSO):")
    print(f"    Clase            : {lasso_filter.__class__.__name__}")
    print(f"    Parámetro (C)    : {lasso_filter.get_params().get('C')}")

    # romper capa ordinal para llegar al stackng
    ordinal_clf = pipeline_interno.named_steps['model']
    # tomamos el primer clasificador binario del ordinal
    stacking = ordinal_clf.estimators_[0]

    print("\nMODELOS BASE:")
    for nombre, estimador in stacking.named_estimators_.items():
        print(f"\n  --- {nombre.upper()} ({estimador.__class__.__name__}) ---")
        params = estimador.get_params()

        # iteramos todos los parámetros
        for param in sorted(params.keys()):
            # ocultamos variables internas que ensucian la consola
            if param not in ['monotone_constraints', 'interaction_constraints', 'cat_features', 'text_features']:
                valor = params[param]
                # limpiar la salida de catboost
                if not isinstance(valor, (list, tuple, dict)) or len(str(valor)) < 50:
                    print(f"      {param:<25}: {valor}")

    # extraer el meta-leaner
    meta = stacking.final_estimator_
    print(f"\nMETA-LEARNER:")
    print(f"  --- {meta.__class__.__name__} ---")
    params_meta = meta.get_params()
    for param in sorted(params_meta.keys()):
        print(f"      {param:<25}: {params_meta[param]}")

    print("\n" + "=" * 80)
    print("¡Rescate completado!")
    print("=" * 80 + "\n")
else:
=======
import joblib
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# convertir el clasificador en un modelo ordinal
class ClasificadorOrdinalFrankHall(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.estimators_ = []
        for i in range(len(self.classes_) - 1):
            y_binario = (y > self.classes_[i]).astype(int)
            clon = clone(self.estimator)
            clon.fit(X, y_binario)
            self.estimators_.append(clon)
        return self

    def predict_proba(self, X):
        probs_mayor_que = [est.predict_proba(X)[:, 1] for est in self.estimators_]
        probs = np.zeros((X.shape[0], len(self.classes_)))
        probs[:, 0] = 1.0 - probs_mayor_que[0]
        for i in range(1, len(self.classes_) - 1):
            probs[:, i] = probs_mayor_que[i-1] - probs_mayor_que[i]
        probs[:, -1] = probs_mayor_que[-1]
        probs = np.clip(probs, 0.0, 1.0)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

ruta_modelo = os.path.join("classification", "joblib", "modelo_clasificacion.joblib")

if os.path.exists(ruta_modelo):
    modelo = joblib.load(ruta_modelo)

    print("\n" + "=" * 80)
    print("EXTRACCIÓN DE PARÁMETROS DEL MODELO CAMPEÓN DE CLASIFICACIÓN (JOBLIB)")
    print("=" * 80)

    # tomamos el primer fold calibrado
    pipeline_interno = modelo.calibrated_classifiers_[0].estimator

    # extraer el preprocesamiento
    escalador = pipeline_interno.named_steps['scaler']
    print(f"\nPREPROCESAMIENTO:")
    print(f"    Escalador        : {escalador.__class__.__name__}")

    # extraer lasso
    lasso_filter = pipeline_interno.named_steps['lasso_filter'].estimator
    print(f"\nFILTRO DE CARACTERÍSTICAS (LASSO):")
    print(f"    Clase            : {lasso_filter.__class__.__name__}")
    print(f"    Parámetro (C)    : {lasso_filter.get_params().get('C')}")

    # romper capa ordinal para llegar al stackng
    ordinal_clf = pipeline_interno.named_steps['model']
    # tomamos el primer clasificador binario del ordinal
    stacking = ordinal_clf.estimators_[0]

    print("\nMODELOS BASE:")
    for nombre, estimador in stacking.named_estimators_.items():
        print(f"\n  --- {nombre.upper()} ({estimador.__class__.__name__}) ---")
        params = estimador.get_params()

        # iteramos todos los parámetros
        for param in sorted(params.keys()):
            # ocultamos variables internas que ensucian la consola
            if param not in ['monotone_constraints', 'interaction_constraints', 'cat_features', 'text_features']:
                valor = params[param]
                # limpiar la salida de catboost
                if not isinstance(valor, (list, tuple, dict)) or len(str(valor)) < 50:
                    print(f"      {param:<25}: {valor}")

    # extraer el meta-leaner
    meta = stacking.final_estimator_
    print(f"\nMETA-LEARNER:")
    print(f"  --- {meta.__class__.__name__} ---")
    params_meta = meta.get_params()
    for param in sorted(params_meta.keys()):
        print(f"      {param:<25}: {params_meta[param]}")

    print("\n" + "=" * 80)
    print("¡Rescate completado!")
    print("=" * 80 + "\n")
else:
>>>>>>> 9015f7c609fa2b4ea4bfb8b397c19d6d54040751
    print(f"\n[!] No se encontró el archivo en: {ruta_modelo}")