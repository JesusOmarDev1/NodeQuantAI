from tcia_utils import nbia
import pandas as pd

nombre_coleccion = "Mediastinal-Lymph-Node-SEG"
carpeta_destino = './Mediastinal_Data_Sample'

print("Conectando para obtener metadatos de las series...")

# 1. Obtenemos TODAS las series (esto es rápido, solo baja texto)
datos_series = nbia.getSeries(collection=nombre_coleccion)

# 2. Convertimos a DataFrame de Pandas para verlo como una tabla
df = pd.DataFrame(datos_series)

# Mostramos las columnas para que veas qué información hay disponible
print(f"Columnas detectadas: {df.columns.tolist()}")

# 3. Seleccionamos los primeros 5 pacientes ÚNICOS
# Usamos 'PatientID' (que suele venir en getSeries)
try:
    lista_pacientes_unicos = df['PatientID'].unique()
    subset_pacientes = lista_pacientes_unicos[:5]

    print(f"\nSe encontraron {len(lista_pacientes_unicos)} pacientes en total.")
    print(f"Descargando datos para estos 5 pacientes de prueba: {subset_pacientes}")

    # 4. Filtramos las series que pertenecen a esos 5 pacientes
    # El script buscará en el dataframe las filas que coincidan con esos IDs
    series_a_descargar = df[df['PatientID'].isin(subset_pacientes)]

    # Convertimos de nuevo a lista de diccionarios para que nbia lo entienda
    lista_series_final = series_a_descargar.to_dict('records')

    # 5. Descargamos
    nbia.downloadSeries(series_data=lista_series_final, path=carpeta_destino)

    print("\n¡Descarga del subset completada exitosamente!")

except KeyError as e:
    print(f"\n[ERROR] No se encontró la columna esperada. Las disponibles son: {df.columns.tolist()}")
    print("Intenta cambiar 'PatientID' por el nombre correcto que veas en la lista de arriba.")
