# Script para predecir volumen en nuevos casos usando modelos entrenados
# Predice volumen exacto en mm³ usando Regresión Lineal o XGBoost

# 0 - Importar librerías necesarias
import SimpleITK as sitk # Para manejar imágenes médicas
import pandas as pd # Para manejar dataframes
import numpy as np # Para operaciones numéricas
import os # Para manejar rutas de archivos
from joblib import load # Para cargar modelos entrenados
from tqdm import tqdm # Para mostrar barras de progreso
import warnings as w # Para manejar advertencias

w.filterwarnings("ignore") # Ignorar advertencias para una salida más limpia

# 1 - Configuración de rutas
carpeta_dataset = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\Dataset_Entrenamiento"
carpeta_modelos = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\modelos"
csv_salida_predicciones = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\resultados\predicciones_nuevos_casos.csv"

# Casos a predecir (los que fueron omitidos durante entrenamiento)
casos_a_predecir = ["case_0002", "case_0005", "case_0006", "case_0009", "case_0012"]

# 2 - Función para extraer características de una máscara
def extraer_features_caso(mascara_path, caso_id):
   try:
      # Cargar máscara
      mascara = sitk.ReadImage(mascara_path)

      # Convertir máscara a array numpy
      mascara_vector = sitk.GetArrayFromImage(mascara)

      # Calcular volumen en mm^3
      espaceado = mascara.GetSpacing()
      voxel_volumen = espaceado[0] * espaceado[1] * espaceado[2]
      num_voxeles = np.sum(mascara_vector > 0)
      volumen_mm3 = num_voxeles * voxel_volumen

      # Estadísticas de forma usando SimpleITK
      label_stats = sitk.LabelShapeStatisticsImageFilter()
      label_stats.Execute(mascara)
      label = 255

      caracteristicas_dic = {
         "case_id": caso_id, # Identificador del caso
         "num_voxeles": int(num_voxeles), # Cantidad de voxeles dentro de la máscara
         "diametro_feret": float(label_stats.GetFeretDiameter(label)) if label_stats.HasLabel(label) else 0.0, # Diámetro de Feret
         "elongacion": float(label_stats.GetElongation(label)) if label_stats.HasLabel(label) else 0.0, # Elongación
         "redondez": float(label_stats.GetRoundness(label)) if label_stats.HasLabel(label) else 0.0, # Redondez
         "espaciado_x": float(espaceado[0]), # Espaciado en X
         "espaciado_y": float(espaceado[1]), # Espaciado en Y
         "espaciado_z": float(espaceado[2]), # Espaciado en Z
      }

      return caracteristicas_dic
   
   except Exception as e:
      print(f"[Error] {caso_id}: {str(e)}")
      return None

# 3 - Función principal para predecir volúmenes
def predecir_volumenes_nuevos_casos():

   print("\nPrediciendo volúmenes para nuevos casos usando modelos entrenados...\n")

   # Verificar que existen los modelos
   modelo_rl_path = os.path.join(carpeta_modelos, "modelo_regresion_lineal.joblib")
   modelo_xgb_path = os.path.join(carpeta_modelos, "modelo_xgboost.joblib")
   scaler_path = os.path.join(carpeta_modelos, "scaler.joblib")

   if not os.path.exists(modelo_rl_path):
      print(f"No se encontró modelo en {modelo_rl_path}")
      return None

   if not os.path.exists(modelo_xgb_path):
      print(f"No se encontró modelo en {modelo_xgb_path}")
      return None

   if not os.path.exists(scaler_path):
      print(f"No se encontró scaler en {scaler_path}")
      return None

   # Cargar modelos y scaler
   print("\nCargando modelos entrenados...")
   modelo_rl = load(modelo_rl_path)
   modelo_xgb = load(modelo_xgb_path)
   scaler = load(scaler_path)
   
   print("Modelos y scaler cargados exitosamente")

   # Verificar que dataset existe
   if not os.path.exists(carpeta_dataset):
      print(f"No se encontró la carpeta del dataset en {carpeta_dataset}")
      return None

   # Extraer características de nuevos casos
   print("\nExtrayendo características de nuevos casos...\n")
   lista_features = []
   casos_fallidos = []

   for caso_folder in tqdm(casos_a_predecir, desc="Procesando casos", unit="caso"):
      caso_path = os.path.join(carpeta_dataset, caso_folder)
      mascara_path = os.path.join(caso_path, "mask.nii.gz")

      # Verificar que la máscara existe
      if not os.path.exists(mascara_path):
         print(f"[Advertencia] No se encontró máscara para {caso_folder}")
         casos_fallidos.append(caso_folder)
         continue

      # Extraer características
      caracteristicas = extraer_features_caso(mascara_path, caso_folder)

      if caracteristicas is not None:
         lista_features.append(caracteristicas)
      else:
         casos_fallidos.append(caso_folder)

   if not lista_features:
      print("\nNo se pudieron extraer características de ningún caso.")
      return None

   # Crear DataFrame con features
   df_features = pd.DataFrame(lista_features)
   print(f"{len(df_features)} casos procesados correctamente")
   
   if casos_fallidos:
      print(f"{len(casos_fallidos)} casos con problemas: {', '.join(casos_fallidos)}")

   # Preparar características para predicción (mismo orden que entrenamiento)
   columnas_features = ["num_voxeles", "diametro_feret", "elongacion", "redondez", 
                       "espaciado_x", "espaciado_y", "espaciado_z"]
   
   X_nuevos = df_features[columnas_features].copy()

   # Escalar características con el scaler del entrenamiento
   print("\nEscalando características...")
   X_nuevos_escalados = scaler.transform(X_nuevos)

   # Hacer predicciones con ambos modelos
   print("Realizando predicciones...")
   pred_rl = modelo_rl.predict(X_nuevos_escalados)
   pred_xgb = modelo_xgb.predict(X_nuevos)

   # Crear DataFrame con resultados
   df_resultados = df_features[["case_id"]].copy()
   df_resultados["prediccion_rl_mm3"] = pred_rl
   df_resultados["prediccion_xgb_mm3"] = pred_xgb
   df_resultados["promedio_prediccion_mm3"] = (pred_rl + pred_xgb) / 2
   df_resultados["diferencia_modelos_mm3"] = np.abs(pred_rl - pred_xgb)

   # Guardar resultados
   carpeta_salida = os.path.dirname(csv_salida_predicciones)
   if not os.path.exists(carpeta_salida):
      os.makedirs(carpeta_salida)

   df_resultados.to_csv(csv_salida_predicciones, index=False, encoding="utf-8")

   # Mostrar resultados
   print(f"\nPredicciones para nuevos casos:\n")

   for idx, row in df_resultados.iterrows():
      print(f"\n{row['case_id']}:")
      print(f"   Regresión Lineal:        {row['prediccion_rl_mm3']:.2f} mm³")
      print(f"   XGBoost Regressor:       {row['prediccion_xgb_mm3']:.2f} mm³")
      print(f"   Promedio (recomendado):  {row['promedio_prediccion_mm3']:.2f} mm³")
      print(f"   Diferencia entre modelos: {row['diferencia_modelos_mm3']:.2f} mm³")

   # Estadísticas generales
   print("\nEstadísticas de predicciones:")

   print(f"\nRegresión Lineal Multiple:")
   print(f"   Media:     {pred_rl.mean():.2f} mm³")
   print(f"   Mediana:   {np.median(pred_rl):.2f} mm³")
   print(f"   Mínimo:    {pred_rl.min():.2f} mm³")
   print(f"   Máximo:    {pred_rl.max():.2f} mm³")
   print(f"   Desv. Est: {pred_rl.std():.2f} mm³")

   print(f"\nXGBoost Regressor:")
   print(f"   Media:     {pred_xgb.mean():.2f} mm³")
   print(f"   Mediana:   {np.median(pred_xgb):.2f} mm³")
   print(f"   Mínimo:    {pred_xgb.min():.2f} mm³")
   print(f"   Máximo:    {pred_xgb.max():.2f} mm³")
   print(f"   Desv. Est: {pred_xgb.std():.2f} mm³")

   print(f"\nPredicciones guardadas en: {csv_salida_predicciones}\n")

   return df_resultados


if __name__ == "__main__":
   resultado = predecir_volumenes_nuevos_casos()

   if resultado is not None:
      print("Proceso terminado exitosamente\n")
   else:
      print("Proceso terminado con errores. Revisa los mensajes anteriores para detalles.\n")