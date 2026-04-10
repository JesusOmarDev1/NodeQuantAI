import joblib
import os

ruta_modelo = os.path.join("regression", "joblib", "modelo_volumen.joblib")

if os.path.exists(ruta_modelo):
    # cargar el modelo
    modelo = joblib.load(ruta_modelo)

    print("\n" + "=" * 80)
    print("EXTRACCIÓN DE PARÁMETROS DEL MODELO CAMPEÓN DE REGRESIÓN (JOBLIB)")
    print("=" * 80)

    # extraer el escalador
    escalador = modelo.named_steps['scaler']
    print(f"\nESCALADOR SELECCIONADO POR OPTUNA:")
    print(f"    Clase exacta : {escalador.__class__.__name__}")
    print(f"    Parámetros   : {escalador.get_params()}")

    # extraemos el StackingRegressor que está dentro del TransformedTargetRegressor
    # TransformedTargetRegressor es el named_step 'model'
    stacking = modelo.named_steps['model'].regressor_

    print("\nMODELOS BASE (EXTRACCIÓN DE CARACTERÍSTICAS):")
    for nombre, estimador in stacking.named_estimators_.items():
        print(f"\n  --- {nombre.upper()} ({estimador.__class__.__name__}) ---")
        params = estimador.get_params()
        # iteramos todos los parámetros
        for param in sorted(params.keys()):
            valor = params[param]
            # filtramos los parámetros muy largos y poco útiles para que la terminal se vea limpia
            if param not in ['monotone_constraints', 'interaction_constraints']:
                print(f"      {param:<25}: {valor}")

    # extraer el neta-learner
    meta = stacking.final_estimator_
    print(f"\nMETA-LEARNER:")
    print(f"  --- RIDGE ({meta.__class__.__name__}) ---")
    params_meta = meta.get_params()
    for param in sorted(params_meta.keys()):
        print(f"      {param:<25}: {params_meta[param]}")

    print("\n" + "=" * 80)
    print("¡Rescate completado!")
    print("=" * 80 + "\n")
else:
    print(f"\n[!] No se encontró el archivo en: {ruta_modelo}")