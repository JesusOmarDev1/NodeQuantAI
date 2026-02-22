# Script para preparar los datos limpios a partir de las máscaras originales

# 0 - Importar librerías necesarias
import pandas as pd # Para manejar dataframes y guardar resultados
import numpy as np # Para operaciones numéricas
import os # Para manejar rutas de archivos
from tqdm import tqdm # Para mostrar barras de progreso

# 1 - Configuración de la carpeta de datos y salida
# Asegúrate de que este nombre coincida con tu carpeta de descarga
csv_entrada = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\resultados\caracteristicas_radiomicas.csv"
csv_salida = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\db\dataset_limpio_mascaras.csv"

def preparar_datos_limpios():
	# Verificar CSV de entrada
	if not os.path.exists(csv_entrada):
		print(f"Error: No se encontro el CSV en {csv_entrada}")
		return None

	print("\nCargando CSV de caracteristicas...")
	df = pd.read_csv(csv_entrada)

	if df.empty:
		print("Error: El CSV esta vacio.")
		return None

	# Columnas no numericas a eliminar (si existen)
	cols_descartar = ["case_id", "caja_delimitadora", "centroide"]
	df = df.drop(columns=[c for c in cols_descartar if c in df.columns], errors="ignore")

	# Mantener solo columnas numericas
	df = df.select_dtypes(include=[np.number])

	# Rellenar valores faltantes con la mediana (con barra de progreso)
	columnas_numericas = list(df.columns)
	for col in tqdm(columnas_numericas, desc="Imputando medianas", unit="col"):
		df[col] = df[col].fillna(df[col].median())

	# Crear carpeta de salida si no existe
	carpeta_destino = os.path.dirname(csv_salida)
	if carpeta_destino and not os.path.exists(carpeta_destino):
		os.makedirs(carpeta_destino)

	# Guardar CSV limpio
	df.to_csv(csv_salida, index=False, encoding="utf-8")

	print(f"\nDataset limpio guardado en: {csv_salida}")
	print(f"Filas: {len(df)} | Columnas: {len(df.columns)}")

	return df


if __name__ == "__main__":
	df = preparar_datos_limpios()

	if df is not None:
		print("\nProceso terminado exitosamente\n")
	else:
		print("\nProceso terminado con errores. Revisa los mensajes anteriores para detalles.\n")
