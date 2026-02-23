import pandas as pd


def limpiar_y_preparar_datasets(ruta_csv_crudo):
    print("Cargando base de datos radiómica...")
    df = pd.read_csv(ruta_csv_crudo)

    # ==========================================
    # 1. DATASET DE CLASIFICACIÓN
    # ==========================================
    df_clasificacion = df.copy()

    # Regla clínica: Eje corto > 10 mm es Adenopatía (1), si no, Normal (0)
    umbral_clinico = 10.0
    df_clasificacion['target_clasificacion'] = (df_clasificacion['shape_MinorAxisLength'] > umbral_clinico).astype(int)

    # Guardamos el dataset de clasificación
    # (Mantenemos todas las features, el clasificador puede usar forma y textura)
    df_clasificacion.to_csv(r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\ganglios_clasificacion.csv", index=False)
    print(f"[OK] Dataset de Clasificación creado: {df_clasificacion['target_clasificacion'].value_counts().to_dict()}")

    # ==========================================
    # 2. DATASET DE REGRESIÓN
    # ==========================================
    df_regresion = df.copy()

    # Nuestro objetivo a predecir será el Volumen exacto
    df_regresion['target_regresion'] = df_regresion['shape_VoxelVolume']

    # PREVENCIÓN DE DATA LEAKAGE:
    # Eliminamos TODAS las columnas de forma (shape) de las características predictoras.
    # El modelo deberá deducir el volumen usando solo textura e intensidad radiológica.
    columnas_shape = [col for col in df_regresion.columns if col.startswith('shape_')]
    df_regresion = df_regresion.drop(columns=columnas_shape)

    # Guardamos el dataset de regresión
    df_regresion.to_csv( r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\ganglios_regresion.csv", index=False)
    print(
        f"[OK] Dataset de Regresión creado. Se eliminaron {len(columnas_shape)} columnas 'shape_' para evitar Fuga de Datos.")

    print("\n¡Limpieza terminada! Tienes dos CSVs prístinos listos para Machine Learning.")


# --- EJECUCIÓN ---
if __name__ == "__main__":
    archivo_crudo = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\ganglios_radiomica.csv"
    limpiar_y_preparar_datasets(archivo_crudo)