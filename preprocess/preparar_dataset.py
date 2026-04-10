import pandas as pd
import os

def limpiar_y_preparar_datasets(ruta_csv_crudo, carpeta_salida):
    print("Cargando base de datos radiómica...")
    df = pd.read_csv(ruta_csv_crudo)

    # ==========================================
    # DATASET DE CLASIFICACIÓN
    # ==========================================
    df_clasificacion = df.copy()

    # eje corto > 10 mm = adenopatía (1), si no, normal (0)
    umbral_clinico = 10.0
    df_clasificacion['target_clasificacion'] = (df_clasificacion['shape_MinorAxisLength'] > umbral_clinico).astype(int)

    # guadar dataset
    df_clasificacion.to_csv(os.path.join(carpeta_salida, "ganglios_clasificacion.csv"), index=False)
    print(f"[OK] Dataset de Clasificación creado: {df_clasificacion['target_clasificacion'].value_counts().to_dict()}")

    # ==========================================
    # DATASET DE REGRESIÓN
    # ==========================================
    df_regresion = df.copy()

    # targets clínicos a predecir
    df_regresion['target_regresion'] = df_regresion['shape_VoxelVolume']          # Volumen (mm^3)
    df_regresion['target_eje_corto'] = df_regresion['shape_MinorAxisLength']      # Eje corto (mm)
    df_regresion['target_eje_largo'] = df_regresion['shape_MajorAxisLength']      # Eje largo (mm)

    # prevenir dataleak
    # eliminamos TODAS las columnas de shape de las características predictoras
    # usamos solo textura e intensidad
    columnas_shape = [col for col in df_regresion.columns if col.startswith('shape_')]
    df_regresion = df_regresion.drop(columns=columnas_shape)

    # guardar dataset
    df_regresion.to_csv(os.path.join(carpeta_salida, "ganglios_regresion.csv"), index=False)
    print(
        f"[OK] Dataset de Regresión creado. Se eliminaron {len(columnas_shape)} columnas 'shape_' para evitar Fuga de Datos.")

    print("\n¡Proceso Finalizado!.")


# --- EJECUCIÓN ---
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    archivo_crudo = os.path.join(base_dir, "db", "ganglios_radiomica.csv") # csv crudo
    carpeta_salida = os.path.join(base_dir, "db")
    limpiar_y_preparar_datasets(archivo_crudo, carpeta_salida)