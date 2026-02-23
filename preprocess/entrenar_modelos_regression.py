# Script para entrenar modelos de regresion con datos limpios
# Regresion Lineal Multiple y XGBoost Regressor

# 0 - Importar librerías necesarias
import pandas as pd # Para manejar dataframes
import numpy as np # Para operaciones numéricas
import os # Para manejar rutas de archivos
from sklearn.model_selection import train_test_split # Para dividir datos
from sklearn.preprocessing import StandardScaler # Para escalar características
from sklearn.linear_model import LinearRegression # Modelo de Regresión Lineal
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Para evaluar modelos
import xgboost as xgb # Para el modelo XGBoost Regressor
from joblib import dump # Para guardar modelos
import warnings as w # Para manejar advertencias

w.filterwarnings("ignore") # Ignorar advertencias para una salida más limpia

# 1 - Configuración de la carpeta de datos y salida
# Asegúrate de que este nombre coincida con tu carpeta del dataset y de salida de los modelos
csv_entrada = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\db\dataset_limpio_mascaras.csv"
carpeta_modelos = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\modelos"

# 2 - Función para entrenar modelos de regresión
def entrenar_modelos_regression():
   # Verificar CSV de entrada
   if not os.path.exists(csv_entrada):
      print(f"Error: No se encontro el CSV en {csv_entrada}")
      return None
   
   # Leer dataset limpio
   print("\nCargando dataset limpio...")
   df = pd.read_csv(csv_entrada)

   # Verificar que el dataset no esté vacío
   if df.empty:
      print("Error: El CSV esta vacio.")
      return None
   
   print(f"Dataset cargado con {len(df)} filas y {len(df.columns)} columnas.")

   # Objetivo
   objetivo_col = "volumen_calculado_mm3"

   # Separar X e Y
   y = df[objetivo_col]
   x = df.drop(columns=[objetivo_col])

   print(f"\nCaracteristicas: {x.shape[1]} | Objetivo: {objetivo_col}")

   # Dividir en entrenamiento y prueba
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

   # === Regresion Lineal ===
   scaler = StandardScaler()
   x_train_scaled = scaler.fit_transform(x_train)
   x_test_scaled = scaler.transform(x_test)

   print("\nEntrenando Regresión Lineal Multiple...")

   # Escalar características
   modelo_rl = LinearRegression()
   modelo_rl.fit(x_train_scaled, y_train)
   modelo_rl_pred = modelo_rl.predict(x_test_scaled)

   modelo_rl_r2 = r2_score(y_test, modelo_rl_pred)
   modelo_rl_mae = mean_absolute_error(y_test, modelo_rl_pred)
   modelo_rl_rmse = np.sqrt(mean_squared_error(y_test, modelo_rl_pred))

   print(f"\nRegresión Lineal Multiple - R2:")
   print(f"  R²:   {modelo_rl_r2:.4f}")
   print(f"  MAE:  {modelo_rl_mae:.2f} mm³")
   print(f"  RMSE: {modelo_rl_rmse:.2f} mm³\n")

   # === XGBoost Regressor ===
   modelo_xgb = xgb.XGBRegressor(
      n_estimators=200, # Número de árboles
      max_depth=5, # Profundidad máxima de cada árbol
      learning_rate=0.05, # Tasa de aprendizaje
      subsample=0.8, # Submuestreo de filas para cada árbol
      colsample_bytree=0.8, # Submuestreo de columnas para cada árbol
      random_state=42, # Para reproducibilidad
      verbose=0 # Para suprimir salida de entrenamiento
   )

   print("\nEntrenando XGBoost Regressor...")

   # Escalar caracteristicas
   modelo_xgb.fit(x_train, y_train)
   modelo_xgb_pred = modelo_xgb.predict(x_test)

   modelo_xgb_r2 = r2_score(y_test, modelo_xgb_pred)
   modelo_xgb_mae = mean_absolute_error(y_test, modelo_xgb_pred)
   modelo_xgb_rmse = np.sqrt(mean_squared_error(y_test, modelo_xgb_pred))

   print(f"\nXGBoost Regressor - R2:")
   print(f"  R²:   {modelo_xgb_r2:.4f}")
   print(f"  MAE:  {modelo_xgb_mae:.2f} mm³")
   print(f"  RMSE: {modelo_xgb_rmse:.2f} mm³\n")

   # Comparar modelos por R2
   print("Comparación de Modelos:")
   if modelo_xgb_r2 > modelo_rl_r2:
      mejor = "XGBoost Regressor"
      diferencia = (modelo_xgb_r2 - modelo_rl_r2) * 100
   else:
      mejor = "Regresión Lineal Multiple"
      diferencia = (modelo_rl_r2 - modelo_xgb_r2) * 100

   # ¿Como se comparan los mejores modelos con R2?
   # 0.9 - 1.0	Excelente (el modelo predice muy bien)
   # 0.7 - 0.9	Bueno (el modelo predice bien)
   # 0.5 - 0.7	Aceptable (el modelo tiene un desempeño moderado)
   # 0.3 - 0.5	Pobre (el modelo tiene un desempeño bajo)

   # Mostrar comparación
   print(f"  Mejor modelo: {mejor} con R² = {max(modelo_rl_r2, modelo_xgb_r2):.4f}")
   print(f"  Diferencia de R² entre modelos: {diferencia:.2f}%\n")

   # Guardar modelos con joblib
   print(f"\nGuardando modelos en {carpeta_modelos}...\n")

   os.makedirs(carpeta_modelos, exist_ok=True)

   # Modelo de Regresión Lineal Multiple
   dump(modelo_rl, os.path.join(carpeta_modelos, "modelo_regresion_lineal.joblib"))
   print("Modelo de Regresión Lineal Multiple guardado como 'modelo_regresion_lineal.joblib'\n")

   # Modelo de XGBoost Regressor
   dump(modelo_xgb, os.path.join(carpeta_modelos, "modelo_xgboost.joblib"))
   print("Modelo de XGBoost Regressor guardado como 'modelo_xgboost.joblib'\n")

   # Guardar StandardScaler para predicciones futuras
   dump(scaler, os.path.join(carpeta_modelos, "scaler.joblib"))
   print("StandardScaler guardado como 'scaler.joblib'\n")

   print("Modelos entrenados y guardados exitosamente.\n")

   return True

if __name__ == "__main__":
   resultado = entrenar_modelos_regression()

   if resultado:
      print("\nProceso terminado exitosamente\n")
   else:
      print("\nProceso terminado con errores. Revisa los mensajes anteriores para detalles.\n")