import pandas as pd
import os
from joblib import load

# 1. Configurar rutas
csv_entrada = r"I:\GitHub\NodeQuantAI\db\ganglios_master.csv"
ruta_modelo_rf = r"I:\GitHub\NodeQuantAI\classification\modelo_random_forest.joblib"


def ver_importancia_variables():
    # 2. Cargar el modelo guardado
    if not os.path.exists(ruta_modelo_rf):
        print("No se encontró el modelo Random Forest guardado.")
        return

    modelo_rf = load(ruta_modelo_rf)
    print("Modelo Random Forest cargado exitosamente.")

    # 3. Necesitamos los nombres de las columnas que usó el modelo
    # Para eso, leemos rápido el CSV y hacemos el mismo preprocesamiento (sin el escalador)
    df = pd.read_csv(csv_entrada)
    objetivo_col = "target_riesgo"
    columna_id = "Paciente_ID"

    x_raw = df.drop(columns=[objetivo_col, columna_id], errors='ignore')
    x = pd.get_dummies(x_raw)  # Esto genera los mismos nombres de columnas que usó el modelo

    # 4. Extraer la importancia de las características
    importancias = modelo_rf.feature_importances_

    # Crear un DataFrame para verlo bonito
    df_importancias = pd.DataFrame({
        'Variable': x.columns,
        'Importancia': importancias
    })

    # Ordenar de mayor a menor importancia y sacar el Top 15
    df_importancias = df_importancias.sort_values(by='Importancia', ascending=False)

    print("\n" + "=" * 50)
    print(" TOP 15 VARIABLES MÁS IMPORTANTES PARA EL RIESGO")
    print("=" * 50)

    # Imprimir con formato de porcentaje
    for index, row in df_importancias.head(15).iterrows():
        print(f"{row['Variable']:<35}: {row['Importancia'] * 100:.2f}%")


if __name__ == "__main__":
    ver_importancia_variables()
    