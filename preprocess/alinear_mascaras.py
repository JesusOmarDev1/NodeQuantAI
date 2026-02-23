import SimpleITK as sitk
import os

# Apuntamos a la carpeta donde ya tienes tus NIfTIs generados
carpeta_dataset = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\Dataset_NIFIT"

print("Iniciando alineación espacial (Resampling)...")

for paciente_folder in os.listdir(carpeta_dataset):
    ruta_paciente = os.path.join(carpeta_dataset, paciente_folder)

    # Ignorar archivos sueltos, solo procesar carpetas
    if not os.path.isdir(ruta_paciente):
        continue

    ruta_ct = os.path.join(ruta_paciente, "image.nii.gz")
    ruta_seg = os.path.join(ruta_paciente, "mask.nii.gz")

    # Solo procedemos si existen ambos archivos
    if os.path.exists(ruta_ct) and os.path.exists(ruta_seg):
        print(f" -> Alineando máscara del paciente: {paciente_folder}")

        try:
            # 1. Cargar ambas imágenes con su metadato espacial intacto
            ct = sitk.ReadImage(ruta_ct)
            mascara = sitk.ReadImage(ruta_seg)

            # 2. Configurar el filtro de Resampling
            resampler = sitk.ResampleImageFilter()

            # Usamos el CT original como nuestro "molde" físico y de dimensiones
            resampler.SetReferenceImage(ct)

            # MUY IMPORTANTE: Interpolación Nearest Neighbor.
            # Si usamos otra (como lineal), los píxeles de la máscara dejarán de ser 0 y 1,
            # y tendrías valores decimales como 0.5, arruinando las etiquetas.
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)

            # Lo que quede fuera de la máscara original, se rellena con 0 (fondo)
            resampler.SetDefaultPixelValue(0)

            # Mantenemos la identidad espacial (sin rotaciones extrañas)
            resampler.SetTransform(sitk.Transform())

            # 3. Ejecutar la alineación
            mascara_alineada = resampler.Execute(mascara)

            # 4. Sobrescribir la máscara vieja con la nueva y corregida
            sitk.WriteImage(mascara_alineada, ruta_seg)
            print("    [OK] Máscara alineada y guardada exitosamente.")

        except Exception as e:
            print(f"    [Error] Fallo al alinear {paciente_folder}: {e}")

print("\n¡Alineación completada! Todas las máscaras ahora comparten la misma matriz que su CT.")