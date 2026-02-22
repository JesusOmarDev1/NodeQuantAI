# Script para extraer características radiómicas de las imágenes y máscaras alineadas

# 0 - Importar librerías necesarias
import SimpleITK as sitk # Para manejar imágenes médicas
import pandas as pd # Para manejar dataframes y guardar resultados
import numpy as np # Para operaciones numéricas
from radiomics import featureextractor # Para extraer características radiómicas
import os # Para manejar rutas de archivos
from tqdm import tqdm # Para mostrar barras de progreso
import warnings as w # Para manejar advertencias

# 0.5 - Filtrar advertencias para mejorar la legibilidad de la salida
w.filterwarnings('ignore', message='GLCM is symmetrical')
 
# 1 - Configuración de la carpeta de datos y salida
# Asegúrate de que este nombre coincida con tu carpeta de descarga
carpeta_dataset = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\Dataset_Entrenamiento"
carpeta_salida = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\resultados\caracteristicas_radiomicas.csv"

# 2 - Funcion para extraer características radiómicas de un caso
def extraer_features_caso(imagen_path, mascara_path, caso_id):
   try:
      # Cargar imagen y máscara
      imagen = sitk.ReadImage(imagen_path)
      mascara = sitk.ReadImage(mascara_path)

      # Configurar extractor de características radiómicas
      extractor = featureextractor.RadiomicsFeatureExtractor()

      # Extraer características
      extractor.enableAllFeatures() # Habilitar todas las características disponibles

      # Configuracion de hiperparametros optimizada para CT medicos
      ajustes = {
         "binWidth": 25, # Ancho de bin para discretización
         "resampledPixelSpacing": None, # No re-muestrear
         "interpolator": sitk.sitkBSpline, # Interpolación para re-muestreo
         "normalize": True, # Normalizar intensidades
         "normalizeScale": 100, # Escala de normalización
         "removeOutliers": 3, # Eliminar outliers extremos
         "label": 255, # CRÍTICO: Las máscaras tienen valores 0-255, no 0-1
      }

      # Aplicar ajustes al extractor
      for key, value in ajustes.items():
         extractor.settings[key] = value

      # Extraer características
      caracteristicas = extractor.execute(imagen, mascara)

      # Convertir a diccionario y agregar ID del caso
      caracteristicas_dic = {"case_id": caso_id}

      for key, value in caracteristicas.items():
         # Filtrar solo características (ignorar metadatos)
         if not key.startswith("diagnostics_"):
            try:
               # Intentar convertir a float, si falla dejar como string
               caracteristicas_dic[key] = float(value)
            except:
               # Si falla, guardar como string
               caracteristicas_dic[key] = str(value)

      # ¿Que es un voxel? Es la unidad más pequeña de una imagen 3D, similar a un píxel en 2D pero con volumen.

      espaceado = imagen.GetSpacing() # Obtener espacio entre voxeles
      mascara_vector = sitk.GetArrayFromImage(mascara) # Convertir máscara a array numpy
      voxel_volumen = espaceado[0] * espaceado[1] * espaceado[2] # Volumen de un voxel en mm^3
      num_voxeles = np.sum(mascara_vector > 0) # Contar voxeles dentro de la máscara
      volumen_mm3 = num_voxeles * voxel_volumen # Calcular volumen total en mm^3

      caracteristicas_dic["volumen_calculado_mm3"] = volumen_mm3
      caracteristicas_dic["num_voxeles"] = int(num_voxeles)

      # ¿Que es HU? Hounsfield Units, es una escala de medida de densidad utilizada en imagenes de tomografia computarizada.
      # ¿Que es ROI? Region of Interest, es la zona específica de la imagen que se analiza, en este caso el ganglio linfático.

      # Estadisticas de intensidad HU del ROI
      imagen_vector = sitk.GetArrayFromImage(imagen) # Convertir imagen a array numpy
      roi_intensidades = imagen_vector[mascara_vector > 0] # Extraer intensidades dentro del ROI

      caracteristicas_dic["hu_media"] = float(np.mean(roi_intensidades)) # Intensidad media en HU
      caracteristicas_dic["hu_desviacion"] = float(np.std(roi_intensidades)) # Desviación estándar en HU
      caracteristicas_dic["hu_minimo"] = float(np.min(roi_intensidades)) # Intensidad mínima en HU
      caracteristicas_dic["hu_maximo"] = float(np.max(roi_intensidades)) # Intensidad máxima en HU
      caracteristicas_dic["hu_mediana"] = float(np.median(roi_intensidades)) # Intensidad mediana en HU

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

   for caso_folder in tqdm(casos, desc="Procesando casos", unit="caso"):
      caso_path = os.path.join(carpeta_dataset, caso_folder)
      imagen_path = os.path.join(caso_path, "image.nii.gz")
      mascara_path = os.path.join(caso_path, "mask.nii.gz")

      # Extraer caracteristicas
      caracteristicas = extraer_features_caso(imagen_path, mascara_path, caso_folder)

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

   # Mostrar resultados
   print(f"\nTasa de exito: {len(df)}/{len(casos)} casos procesados correctamente.")

   # Estadísticas de volumen
   print(f"\nESTADÍSTICAS DE VOLUMEN:")
   print(f"   Media:    {df['volumen_calculado_mm3'].mean():.2f} mm³")
   print(f"   Mediana:  {df['volumen_calculado_mm3'].median():.2f} mm³")
   print(f"   Mínimo:   {df['volumen_calculado_mm3'].min():.2f} mm³")
   print(f"   Máximo:   {df['volumen_calculado_mm3'].max():.2f} mm³")
   print(f"   Std Dev:  {df['volumen_calculado_mm3'].std():.2f} mm³")

   print(f"\n¡Extracción de características radiómicas completada! Resultados guardados en: {carpeta_salida}")

   return df


if __name__ == "__main__":
   df = procesar_dataset()

   if df is not None:
      print(f"\nProceso terminado exitosamente\n")
   else:
      print(f"\nProceso terminado con errores. Revisa los mensajes anteriores para detalles.\n")