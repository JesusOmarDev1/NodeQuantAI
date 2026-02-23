import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os

def visualizar_conversion(ruta_paciente):
    """
    Carga el CT y la máscara de una carpeta, busca el corte con mayor área
    de segmentación y lo muestra.
    """
    ruta_ct = os.path.join(ruta_paciente, "image.nii.gz")
    ruta_seg = os.path.join(ruta_paciente, "mask.nii.gz")

    if not os.path.exists(ruta_ct) or not os.path.exists(ruta_seg):
        print(f"Faltan archivos en {ruta_paciente}")
        return

    # Leer imágenes
    ct = sitk.ReadImage(ruta_ct)
    seg = sitk.ReadImage(ruta_seg)

    # Convertir a array numpy (z, y, x)
    ct_arr = sitk.GetArrayFromImage(ct)
    seg_arr = sitk.GetArrayFromImage(seg)

    # Buscar el corte (slice) con más píxeles de máscara (donde está el ganglio)
    # Sumamos en los ejes X e Y para ver qué slice Z tiene más '1's
    pixeles_por_slice = np.sum(seg_arr, axis=(1, 2))
    indice_mejor_slice = np.argmax(pixeles_por_slice)

    if pixeles_por_slice[indice_mejor_slice] == 0:
        print(f"¡OJO! La máscara en {ruta_paciente} parece estar vacía.")
        indice_mejor_slice = ct_arr.shape[0] // 2  # Mostramos el centro por defecto

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"CT Original (Slice {indice_mejor_slice})")
    # Usamos vmin/vmax para simular ventana de tejido blando (-160, 240)
    plt.imshow(ct_arr[indice_mejor_slice, :, :], cmap="gray", vmin=-160, vmax=240)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Superposición (CT + Máscara)")
    plt.imshow(ct_arr[indice_mejor_slice, :, :], cmap="gray", vmin=-160, vmax=240)
    # Mostramos la máscara en rojo con transparencia
    plt.imshow(seg_arr[indice_mejor_slice, :, :], cmap="Reds", alpha=0.5, interpolation='none')
    plt.axis('off')

    plt.show()

# Carpeta del paciente
carpeta_paciente = r"C:\Users\korev\Documents\Cursos\Samsung Innovation Campus\Proyecto\Local\Dataset_NIFIT\case_0093"

visualizar_conversion(carpeta_paciente)