from tcia_utils import nbia
import time
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# archivo .tcia
archivo_manifiesto = os.path.join(base_dir, "preprocess", "Mediastinal-Lymph-Node-SEG-DA-RAD.tcia")
nombre_coleccion = "Mediastinal-Lymph-Node-SEG"
# DICOM crudos desde TCIA
carpeta_destino = os.path.join(base_dir, "Mediastinal_Data")

# lista total de series
todas_las_series = nbia.getSeries(collection=nombre_coleccion)
total_esperado = len(todas_las_series)

print(f"Objetivo: Descargar {total_esperado} series.")

while True:
    try:
        # salta si el registro ya se había descargado
        nbia.downloadSeries(series_data=todas_las_series, path=carpeta_destino)
        print("\n¡Proceso finalizado! Verificando si falta algo...")
        break
    except Exception as e:
        # vuelve a intentar por si se cerró la conexión
        print(f"\nOcurrió un error de conexión: {e}")
        print("Reintentando en 10 segundos...")
        time.sleep(10)

print("Descarga completa.")

print("\n¡Descarga completada! Revisar la carpeta 'Mediastinal_Data'.")