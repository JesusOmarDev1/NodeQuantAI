# Script para extraer características radiómicas de las imágenes y máscaras alineadas

# 0 - Importar librerías necesarias
import SimpleITK as sitk # Para manejar imágenes médicas
import pandas as pd # Para manejar dataframes y guardar resultados
import numpy as np # Para operaciones numéricas
import os # Para manejar rutas de archivos
from tqdm import tqdm # Para mostrar barras de progreso
 
# 1 - Configuración de la carpeta de datos y salida
# Asegúrate de que este nombre coincida con tu carpeta del dataset y de salida
carpeta_dataset = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\Dataset_Entrenamiento"
carpeta_salida = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\resultados\caracteristicas_radiomicas.csv"

# 5 Casos a omitir para pruebas a futuro
casos_omitidos = {"case_0002", "case_0005", "case_0006", "case_0009", "case_0012"}

# 2 - Funcion para extraer características SOLO de la máscara
def extraer_features_caso(mascara_path, caso_id):
   try:
      # Cargar máscara
      mascara = sitk.ReadImage(mascara_path)

      # Convertir máscara a array numpy
      mascara_vector = sitk.GetArrayFromImage(mascara)

      # Calcular volumen en mm^3 usando spacing de la máscara
      espaceado = mascara.GetSpacing() # Obtener espacio entre voxeles
      voxel_volumen = espaceado[0] * espaceado[1] * espaceado[2] # Volumen de un voxel en mm^3S
      num_voxeles = np.sum(mascara_vector > 0) # Contar voxeles dentro de la máscara
      volumen_mm3 = num_voxeles * voxel_volumen # Volumen total en mm^3

      # Estadísticas de forma usando SimpleITK
      label_stats = sitk.LabelShapeStatisticsImageFilter()
      label_stats.Execute(mascara)
      label = 255

      caracteristicas_dic = {
         "case_id": caso_id,
         "num_voxeles": int(num_voxeles),
         "volumen_calculado_mm3": float(volumen_mm3),
         "caja_delimitadora": str(label_stats.GetBoundingBox(label)) if label_stats.HasLabel(label) else "",
         "centroide": str(label_stats.GetCentroid(label)) if label_stats.HasLabel(label) else "",
         "diametro_feret": float(label_stats.GetFeretDiameter(label)) if label_stats.HasLabel(label) else 0.0,
         "elongacion": float(label_stats.GetElongation(label)) if label_stats.HasLabel(label) else 0.0,
         "redondez": float(label_stats.GetRoundness(label)) if label_stats.HasLabel(label) else 0.0,
         "espaciado_x": float(espaceado[0]),
         "espaciado_y": float(espaceado[1]),
         "espaciado_z": float(espaceado[2]),
      }

      return caracteristicas_dic
   
   except Exception as e:
      print(f"Error al procesar caso {caso_id}: {str(e)}")
      return None

# 3 - Procesar todos los casos en la carpeta
def procesar_dataset():
   # Verificar que existe el dataset
   if not os.path.exists(carpeta_dataset):
      print(f"Error: No se encontró la carpeta del dataset en {carpeta_dataset}")
      return None

   # Listar todos los casos por "case_id"
   todos_archivos = os.listdir(carpeta_dataset)
   casos = [d for d in todos_archivos
               if os.path.isdir(os.path.join(carpeta_dataset, d)) and d.startswith("case_")
            ]
   casos = sorted(casos) # Ordenar casos por nombre

   # Imprimir número de casos encontrados
   print(f"\nSe encontraron {len(casos)} casos en el dataset.\n")

   if len(casos) == 0:
      print("No se encontraron casos para procesar. Verifica la estructura de tu carpeta.")
      return None
   
   # Extraer características de cada caso y almacenar en una lista
   lista_caracteristicas = []
   omitidos_en_ejecucion = []

   for caso_folder in tqdm(casos, desc="Procesando casos", unit="caso"):
      if caso_folder in casos_omitidos:
         omitidos_en_ejecucion.append(caso_folder)
         continue

      caso_path = os.path.join(carpeta_dataset, caso_folder)
      mascara_path = os.path.join(caso_path, "mask.nii.gz")

      # Extraer caracteristicas solo de la mascara
      caracteristicas = extraer_features_caso(mascara_path, caso_folder)

      if caracteristicas is not None:
         lista_caracteristicas.append(caracteristicas)
      else:
         print(f"[Advertencia] No se pudieron extraer características para {caso_folder}")

   # Crear Dataframe con todas las características
   df = pd.DataFrame(lista_caracteristicas)

   # Ordenar columnas por (case_id y volumen calculado primero)
   prioridad_cols = ["case_id", "volumen_calculado_mm3", "num_voxeles"]
   otras_cols = [c for c in df.columns if c not in prioridad_cols]
   df = df[prioridad_cols + otras_cols]

   # Crear directorio de salida si no existe
   if not os.path.exists(os.path.dirname(carpeta_salida)):
      os.makedirs(os.path.dirname(carpeta_salida))

   # Guardar DataFrame a CSV
   df.to_csv(carpeta_salida, index=False, encoding="utf-8")

   # Reportar casos omitidos
   if omitidos_en_ejecucion:
      omitidos_en_ejecucion = sorted(omitidos_en_ejecucion)
      print("\nCasos omitidos:")
      print(" - " + "\n - ".join(omitidos_en_ejecucion))

   # Mostrar resultados
   print(f"\nTasa de exito: {len(df)}/{len(casos)} casos procesados correctamente.")

   # Estadísticas de volumen
   print(f"\nESTADÍSTICAS DE VOLUMEN:")
   print(f"   Media:    {df['volumen_calculado_mm3'].mean():.2f} mm³")
   print(f"   Mediana:  {df['volumen_calculado_mm3'].median():.2f} mm³")
   print(f"   Mínimo:   {df['volumen_calculado_mm3'].min():.2f} mm³")
   print(f"   Máximo:   {df['volumen_calculado_mm3'].max():.2f} mm³")
   print(f"   Desviación estándar:  {df['volumen_calculado_mm3'].std():.2f} mm³")

   print(f"\n¡Extracción de características radiómicas completada! Resultados guardados en: {carpeta_salida}")

   return df


if __name__ == "__main__":
   df = procesar_dataset()

   if df is not None:
      print(f"\nProceso terminado exitosamente\n")
   else:
      print(f"\nProceso terminado con errores. Revisa los mensajes anteriores para detalles.\n")