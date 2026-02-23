from tcia_utils import nbia
import time

# Reemplaza con la ruta real de tu archivo .tcia
archivo_manifiesto = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\Mediastinal-Lymph-Node-SEG-DA-RAD.tcia"
nombre_coleccion = "Mediastinal-Lymph-Node-SEG"
carpeta_destino = './Mediastinal_Data'

# Obtenemos la lista total de series
todas_las_series = nbia.getSeries(collection=nombre_coleccion)
total_esperado = len(todas_las_series)

print(f"Objetivo: Descargar {total_esperado} series.")

while True:
    try:
        # Intenta descargar. La librería saltará automáticamente lo que ya existe.
        nbia.downloadSeries(series_data=todas_las_series, path=carpeta_destino)

        # Si llega aquí sin lanzar una excepción fatal, terminamos el ciclo
        print("\n¡Proceso finalizado! Verificando si falta algo...")
        break

    except Exception as e:
        print(f"\nOcurrió un error de conexión: {e}")
        print("Reintentando en 10 segundos... (No te preocupes, retomará donde se quedó)")
        time.sleep(10)

print("Descarga completa.")

print(f"\n¡Descarga completada! Revisa la carpeta '{carpeta_destino}'.")