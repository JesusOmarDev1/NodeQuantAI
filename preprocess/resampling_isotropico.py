import SimpleITK as sitk
import os
import numpy as np


def resamplear_isotropico(imagen, es_mascara=False, espaciado_objetivo=(1.0, 1.0, 1.0)):
    """
    Convierte la imagen a una resolución física estándar (ej. 1x1x1 mm).
    """
    espaciado_original = imagen.GetSpacing()
    tamano_original = imagen.GetSize()

    # Calculamos el nuevo tamaño de la matriz en base al nuevo espaciado
    nuevo_tamano = [
        int(np.round(tamano_original[0] * (espaciado_original[0] / espaciado_objetivo[0]))),
        int(np.round(tamano_original[1] * (espaciado_original[1] / espaciado_objetivo[1]))),
        int(np.round(tamano_original[2] * (espaciado_original[2] / espaciado_objetivo[2])))
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(nuevo_tamano)
    resampler.SetOutputSpacing(espaciado_objetivo)
    resampler.SetOutputOrigin(imagen.GetOrigin())
    resampler.SetOutputDirection(imagen.GetDirection())

    if es_mascara:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        # Interpolación lineal para el CT (tejido blando)
        resampler.SetInterpolator(sitk.sitkLinear)
        # Rellenar con aire (-1000 HU) lo que quede fuera de límites
        resampler.SetDefaultPixelValue(-1000)

    return resampler.Execute(imagen)



# --- EJECUCIÓN DEL PIPELINE ---
carpeta_origen = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\Dataset_NIFIT"
carpeta_salida = r"C:\Users\omar\OneDrive\Escritorio\Lymph-Node\Dataset_Preprocesado" # Carpeta lista para PyTorch

if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

print("Iniciando Pipeline de Preprocesamiento (Cropping + Resampling)...")

for paciente in os.listdir(carpeta_origen):
    ruta_paciente_in = os.path.join(carpeta_origen, paciente)

    if not os.path.isdir(ruta_paciente_in):
        continue

    ruta_ct = os.path.join(ruta_paciente_in, "image.nii.gz")
    ruta_seg = os.path.join(ruta_paciente_in, "mask.nii.gz")

    if os.path.exists(ruta_ct) and os.path.exists(ruta_seg):
        print(f"\nProcesando: {paciente}")

        # 1. Cargar imágenes
        ct = sitk.ReadImage(ruta_ct)
        mask = sitk.ReadImage(ruta_seg)

        # 3. Re-muestreo Isotrópico (1x1x1 mm)
        print("   -> Convirtiendo a resolución 1x1x1 mm...")
        ct_final = resamplear_isotropico(ct, es_mascara=False)
        mask_final = resamplear_isotropico(mask, es_mascara=True)

        # 4. Guardar resultados
        ruta_paciente_out = os.path.join(carpeta_salida, paciente)
        if not os.path.exists(ruta_paciente_out):
            os.makedirs(ruta_paciente_out)

        sitk.WriteImage(ct_final, os.path.join(ruta_paciente_out, "image.nii.gz"))
        sitk.WriteImage(mask_final, os.path.join(ruta_paciente_out, "mask.nii.gz"))
        print(f"   [OK] Guardado en {carpeta_salida}")

print("\n¡Pipeline finalizado! Tu data está lista para la red neuronal.")