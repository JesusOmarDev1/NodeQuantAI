import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os

def visualizar_conversion(ruta_paciente):
    # carga la TC y la máscara de una carpeta,
    # busca el corte con mayor área de segmentación y lo muestra
    ruta_ct = os.path.join(ruta_paciente, "image.nii.gz")
    ruta_seg = os.path.join(ruta_paciente, "mask.nii.gz")

    if not os.path.exists(ruta_ct) or not os.path.exists(ruta_seg):
        print(f"Faltan archivos en {ruta_paciente}")
        return

    # leer imágenes
    ct = sitk.ReadImage(ruta_ct)
    seg = sitk.ReadImage(ruta_seg)

    # convertir a array numpy (z, y, x)
    ct_arr = sitk.GetArrayFromImage(ct)
    seg_arr = sitk.GetArrayFromImage(seg)

    # buscar el corte (slice) con más píxeles de máscara (donde está el ganglio)
    # sumar en los ejes X e Y para ver qué slice Z tiene más 1s
    pixeles_por_slice = np.sum(seg_arr, axis=(1, 2))
    indice_mejor_slice = np.argmax(pixeles_por_slice)

    if pixeles_por_slice[indice_mejor_slice] == 0:
        print(f"[!] La máscara en {ruta_paciente} parece estar vacía.")
        indice_mejor_slice = ct_arr.shape[0] // 2  # mostrar el centro por defecto

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"CT Original (Slice {indice_mejor_slice})")
    # usamos vmin/vmax para simular ventana de tejido blando (-160, 240)
    plt.imshow(ct_arr[indice_mejor_slice, :, :], cmap="gray", vmin=-160, vmax=240)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Superposición (CT + Máscara)")
    plt.imshow(ct_arr[indice_mejor_slice, :, :], cmap="gray", vmin=-160, vmax=240)
    # mostramos la máscara en rojo con transparencia
    plt.imshow(seg_arr[indice_mejor_slice, :, :], cmap="Reds", alpha=0.5, interpolation='none')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Carpeta con los NIfTI originales (CT + máscara alineada por paciente)
    carpeta_dataset = os.path.join(base_dir, "Dataset_NIFIT")
    # ID del paciente a visualizar
    paciente = os.environ.get("PACIENTE_ID", "case_0093")
    carpeta_paciente = os.path.join(carpeta_dataset, paciente)

    if not os.path.isdir(carpeta_paciente):
        candidatos = sorted([d for d in os.listdir(carpeta_dataset) if os.path.isdir(os.path.join(carpeta_dataset, d))])
        if not candidatos:
            raise FileNotFoundError(f"No se encontraron pacientes en: {carpeta_dataset}")
        print(f"Paciente {paciente} no encontrado. Usando {candidatos[0]}.")
        carpeta_paciente = os.path.join(carpeta_dataset, candidatos[0])

    visualizar_conversion(carpeta_paciente)