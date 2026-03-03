import pandas as pd
import os


def limpiar_y_preparar_datasets(ruta_csv_crudo, carpeta_salida):
    print("Cargando base de datos radiómica...")
    df = pd.read_csv(ruta_csv_crudo)

    # ==========================================
    # 1. DATASET DE CLASIFICACIÓN
    # ==========================================
    df_clasificacion = df.copy()

    # Regla clínica: Eje corto > 10 mm es Adenopatía (1), si no, Normal (0)
    umbral_clinico = 10.0
    df_clasificacion['target_clasificacion'] = (df_clasificacion['shape_MinorAxisLength'] > umbral_clinico).astype(int)

    # Guardamos el dataset de clasificación en db/ (todas las features + target binario)
    df_clasificacion.to_csv(os.path.join(carpeta_salida, "ganglios_clasificacion.csv"), index=False)
    print(f"[OK] Dataset de Clasificación creado: {df_clasificacion['target_clasificacion'].value_counts().to_dict()}")

    # ==========================================
    # 2. DATASET DE REGRESIÓN
    # ==========================================
    df_regresion = df.copy()

    # Targets clínicos a predecir
    df_regresion['target_regresion'] = df_regresion['shape_VoxelVolume']          # Volumen (mm³)
    df_regresion['target_eje_corto'] = df_regresion['shape_MinorAxisLength']      # Eje corto (mm)
    df_regresion['target_eje_largo'] = df_regresion['shape_MajorAxisLength']      # Eje largo (mm)

    # PREVENCIÓN DE DATA LEAKAGE:
    # Eliminamos TODAS las columnas de forma (shape) de las características predictoras.
    # El modelo deberá deducir volumen y diámetros usando solo textura e intensidad radiológica.
    columnas_shape = [col for col in df_regresion.columns if col.startswith('shape_')]
    df_regresion = df_regresion.drop(columns=columnas_shape)

    # Guardamos el dataset de regresión en db/ (features firstorder+glcm + target volumen mm³)
    df_regresion.to_csv(os.path.join(carpeta_salida, "ganglios_regresion.csv"), index=False)
    print(
        f"[OK] Dataset de Regresión creado. Se eliminaron {len(columnas_shape)} columnas 'shape_' para evitar Fuga de Datos.")

    print("\n¡Limpieza terminada! Tienes dos CSVs prístinos listos para Machine Learning.")


# --- EJECUCIÓN ---
if __name__ == "__main__":
    # Raíz del proyecto (Lymph-Node/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # CSV crudo con las 57 features radiómicas (generado por extraccion_radiomica.py)
    archivo_crudo = os.path.join(base_dir, "db", "ganglios_radiomica.csv")

    # Carpeta donde se guardan los CSVs de clasificación y regresión
    carpeta_salida = os.path.join(base_dir, "db")

    limpiar_y_preparar_datasets(archivo_crudo, carpeta_salida)