import SimpleITK as sitk
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# carpeta con los DICOM crudos descargados desde TCIA
carpeta_origen = os.path.join(base_dir, "Mediastinal_Data")

# carpeta destino donde se guardan los NIfTI convertidos (image.nii.gz + mask.nii.gz por paciente)
carpeta_destino = os.path.join(base_dir, "Dataset_NIFIT")

if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

print(f"Leyendo desde: {carpeta_origen}")
print(f"Escribiendo en: {carpeta_destino}")

# recorrer todas las carpetas buscando series DICOM
for root, dirs, files in os.walk(carpeta_origen):
    if not files:
        continue

    # verificar si hay archivos .dcm en esta carpeta
    if not files[0].endswith(".dcm"):
        continue

    try:
        print(f" -> Procesando carpeta: {root}...")

        # leer la serie completa de .dcm
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(root)
        reader.SetFileNames(dicom_names)
        image_3d = reader.Execute()

        # leer metadatos para saber el nombre del paciente
        img_temp = sitk.ReadImage(dicom_names[0])
        paciente_id = img_temp.GetMetaData("0010|0020").strip()  # ID Paciente
        modalidad = img_temp.GetMetaData("0008|0060").strip()  # CT o SEG

        # creamos la carpeta del paciente en el destino
        salida_paciente = os.path.join(carpeta_destino, paciente_id)
        if not os.path.exists(salida_paciente):
            os.makedirs(salida_paciente)

        # guardamos como NIfTI (.nii.gz)
        nombre_archivo = "image.nii.gz" if modalidad == "CT" else "mask.nii.gz"
        ruta_final = os.path.join(salida_paciente, nombre_archivo)

        sitk.WriteImage(image_3d, ruta_final)
        print(f"    [OK] Guardado: {nombre_archivo} en {salida_paciente}")

    except Exception as e:
        print(f"    [Error] No se pudo convertir: {e}")

print("\n¡Conversión terminada!")