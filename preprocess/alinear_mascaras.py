import SimpleITK as sitk
import os

# carpeta con los tus NIfTIs generados
carpeta_dataset = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\Dataset_NIFIT"

print("Iniciando alineación espacial (Resampling)...")

for paciente_folder in os.listdir(carpeta_dataset):
    ruta_paciente = os.path.join(carpeta_dataset, paciente_folder)

    # ignorar archivos sueltos, solo procesar carpetas
    if not os.path.isdir(ruta_paciente):
        continue

    ruta_ct = os.path.join(ruta_paciente, "image.nii.gz")
    ruta_seg = os.path.join(ruta_paciente, "mask.nii.gz")

    # solo procedemos si existen ambos archivos
    if os.path.exists(ruta_ct) and os.path.exists(ruta_seg):
        print(f" -> Alineando máscara del paciente: {paciente_folder}")

        try:
            # cargar ambas imágenes con su metadato espacial intacto
            ct = sitk.ReadImage(ruta_ct)
            mascara = sitk.ReadImage(ruta_seg)

            # configurar el filtro de resampling
            resampler = sitk.ResampleImageFilter()

            # usamos el CT original como molde físico y de dimensiones
            resampler.SetReferenceImage(ct)

            # usando interpolación nearest neighbor para tener 0 y 1
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)

            # lo que quede fuera de la máscara original, se rellena con 0 (fondo)
            resampler.SetDefaultPixelValue(0)

            # mantenemos la identidad espacial (sin rotaciones extrañas)
            resampler.SetTransform(sitk.Transform())

            # ejecutar la alineación
            mascara_alineada = resampler.Execute(mascara)

            # sobrescribir la máscara vieja con la nueva y corregida
            sitk.WriteImage(mascara_alineada, ruta_seg)
            print("    [OK] Máscara alineada y guardada exitosamente")

        except Exception as e:
            print(f"    [Error] Fallo al alinear {paciente_folder}: {e}")

print("\n¡Alineación completada!")