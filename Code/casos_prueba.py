import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split


def generar_casos_prueba_estratificados(ruta_csv, n_muestras=5):
    print(f"Leyendo base de datos: {ruta_csv}")
    df = pd.read_csv(ruta_csv)

    # 1. Muestreo Estratificado usando Scikit-Learn
    # Esto garantiza que la proporción de clases se mantenga lo mejor posible
    _, df_prueba = train_test_split(
        df,
        test_size=n_muestras,
        stratify=df['target_clasificacion'],
        random_state=42  # Para que siempre obtengas los mismos 5 si lo vuelves a correr
    )

    print(f"\n✅ Se seleccionaron {n_muestras} casos de prueba:")
    print(df_prueba[['Paciente_ID', 'target_clasificacion', 'shape_MinorAxisLength']])

    # 2. Guardar el CSV de casos de prueba
    ruta_base = os.path.dirname(ruta_csv)
    csv_salida = os.path.join(ruta_base, "casos_prueba_demo.csv")
    df_prueba.to_csv(csv_salida, index=False)

    # 3. OPCIONAL: Organizar los archivos NIfTI para el Dashboard
    # Creamos una carpeta 'Demo_Files' para que el Dashboard los cargue rápido
    carpeta_demo = os.path.join(ruta_base, "Demo_Files")
    os.makedirs(carpeta_demo, exist_ok=True)

    carpeta_preprocesada = os.path.join(ruta_base, "Dataset_Preprocesado")

    print(f"\nCopiando archivos NIfTI a {carpeta_demo}...")
    for pac_id in df_prueba['Paciente_ID']:
        origen = os.path.join(carpeta_preprocesada, pac_id)
        destino = os.path.join(carpeta_demo, pac_id)

        if os.path.exists(origen):
            if os.path.exists(destino): shutil.rmtree(destino)
            shutil.copytree(origen, destino)
            print(f" -> [OK] {pac_id}")
        else:
            print(f" -> [!] No se encontró la carpeta para {pac_id}")

    print(f"\n¡Listo! Tienes tu CSV de prueba y los archivos 3D en la carpeta 'Demo_Files'.")


# --- EJECUCIÓN ---
if __name__ == "__main__":
    # Usamos el CSV de clasificación que tiene las etiquetas de riesgo
    ruta_dataset = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\ganglios_radiomica.csv"
    generar_casos_prueba_estratificados(ruta_dataset)