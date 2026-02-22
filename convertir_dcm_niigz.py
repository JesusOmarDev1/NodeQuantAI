import SimpleITK as sitk
import os

# 1. DÓNDE ESTÁN TUS DATOS AHORA (La carpeta con números raros)
# Asegúrate de que este nombre coincida con tu carpeta de descarga
carpeta_origen = ".\Mediastinal_Data"

# 2. DÓNDE QUIERES QUE APAREZCAN LOS NIFTI
carpeta_destino = ".\Dataset_Entrenamiento"

if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

print(f"Leyendo desde: {carpeta_origen}")
print(f"Escribiendo en: {carpeta_destino}")

# Recorremos todas las carpetas buscando series DICOM
for root, dirs, files in os.walk(carpeta_origen):
    if not files:
        continue

    # Verificamos si hay archivos .dcm en esta carpeta
    if not files[0].endswith(".dcm"):
        continue

    try:
        print(f" -> Procesando carpeta: {root}...")

        # Leemos la serie completa de .dcm
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(root)
        reader.SetFileNames(dicom_names)
        image_3d = reader.Execute()

        # Leemos metadatos para saber el nombre del paciente
        # Leemos solo el primer archivo para no perder tiempo
        img_temp = sitk.ReadImage(dicom_names[0])
        paciente_id = img_temp.GetMetaData("0010|0020").strip()  # ID Paciente
        modalidad = img_temp.GetMetaData("0008|0060").strip()  # CT o SEG

        # Creamos la carpeta del paciente en el destino
        salida_paciente = os.path.join(carpeta_destino, paciente_id)
        if not os.path.exists(salida_paciente):
            os.makedirs(salida_paciente)

        # Guardamos como NIfTI (.nii.gz)
        nombre_archivo = "image.nii.gz" if modalidad == "CT" else "mask.nii.gz"
        ruta_final = os.path.join(salida_paciente, nombre_archivo)

        sitk.WriteImage(image_3d, ruta_final)
        print(f"    [OK] Guardado: {nombre_archivo} en {salida_paciente}")

    except Exception as e:
        print(f"    [Error] No se pudo convertir: {e}")

print("\n¡Conversión terminada! AHORA SÍ revisa la carpeta 'Dataset_Entrenamiento'.")