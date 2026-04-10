import joblib
import os

ruta_modelo = os.path.join("regression", "joblib", "modelo_volumen.joblib")

if os.path.exists(ruta_modelo):
    # Cargamos el modelo congelado
    modelo = joblib.load(ruta_modelo)

    print("\n" + "=" * 80)
    print("EXTRACCIÓN DE PARÁMETROS DEL MODELO CAMPEÓN DE REGRESIÓN (JOBLIB)")
    print("=" * 80)

    # 1. Extraemos el Escalador
    escalador = modelo.named_steps['scaler']
    print(f"\n[1] ESCALADOR SELECCIONADO POR OPTUNA:")
    print(f"    Clase exacta : {escalador.__class__.__name__}")
    print(f"    Parámetros   : {escalador.get_params()}")

    # 2. Extraemos el StackingRegressor (está dentro del TransformedTargetRegressor)
    # TransformedTargetRegressor es el named_step 'model'
    stacking = modelo.named_steps['model'].regressor_

    print("\n[2] MODELOS BASE (EXTRACCIÓN DE CARACTERÍSTICAS):")
    for nombre, estimador in stacking.named_estimators_.items():
        print(f"\n  --- {nombre.upper()} ({estimador.__class__.__name__}) ---")
        params = estimador.get_params()
        # Iteramos e imprimimos todos los parámetros ordenados alfabéticamente
        for param in sorted(params.keys()):
            valor = params[param]
            # Filtramos algunos parámetros muy largos y poco útiles visualmente para que la terminal se vea limpia
            if param not in ['monotone_constraints', 'interaction_constraints']:
                print(f"      {param:<25}: {valor}")

    # 3. Extraemos el Meta-Learner
    meta = stacking.final_estimator_
    print(f"\n[3] META-LEARNER (JUEZ FINAL ESPACIO LOGARÍTMICO):")
    print(f"  --- RIDGE ({meta.__class__.__name__}) ---")
    params_meta = meta.get_params()
    for param in sorted(params_meta.keys()):
        print(f"      {param:<25}: {params_meta[param]}")

    print("\n" + "=" * 80)
    print("¡Rescate completado!")
    print("=" * 80 + "\n")
else:
    print(f"\n[!] No se encontró el archivo en: {ruta_modelo}")