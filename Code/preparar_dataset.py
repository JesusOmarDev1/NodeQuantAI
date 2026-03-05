import pandas as pd
from sklearn.model_selection import train_test_split
import os

def crear_datasets_finales(ruta_csv_crudo, n_muestras=5):
    # 1. Cargar datos extraídos de PyRadiomics
    print("Leyendo base de datos radiómica...")
    df = pd.read_csv(ruta_csv_crudo)

    # 2. Definición de Targets
    # Regresión: Usamos el nombre original para el volumen
    # Clasificación: Creamos 'target_riesgo' basado en los cuartiles del eje corto
    df['target_riesgo'] = pd.qcut(
        df['shape_MinorAxisLength'],
        q=4,
        labels=[0, 1, 2, 3]
    ).astype(int)

    print(f"Targets configurados: 'shape_VoxelVolume' (Regresión) y 'target_riesgo' (Clasificación).")

    # 3. Muestreo Estratificado y Separación
    # 'df_master' recibirá el resto de los datos (aprox. 508 registros)
    # 'df_prueba' recibirá exactamente 5 registros para el demo
    df_master, df_prueba = train_test_split(
        df,
        test_size=n_muestras,
        stratify=df['target_riesgo'],
        random_state=42  # Para reproducibilidad en tu presentación
    )

    # 4. Guardar archivos en las rutas locales
    ruta_base = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local"
    ruta_master = os.path.join(ruta_base, "ganglios_master.csv")
    ruta_test = os.path.join(ruta_base, "casos_prueba.csv")

    df_master.to_csv(ruta_master, index=False)
    df_prueba.to_csv(ruta_test, index=False)

    # 5. Resumen de la operación
    print("\n" + "="*50)
    print("PROCESO COMPLETADO CON ÉXITO")
    print("="*50)
    print(f"Base Maestra: {len(df_master)} registros (Casos de prueba eliminados).")
    print(f"Casos de Prueba: {len(df_prueba)} registros (Para el Dashboard).")
    print("-" * 50)
    print("Muestra de Casos de Prueba (Estratificados):")
    print(df_prueba[['Paciente_ID', 'shape_MinorAxisLength', 'target_riesgo', 'shape_VoxelVolume']].to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    archivo_entrada = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\ganglios_radiomica.csv"
    crear_datasets_finales(archivo_entrada)