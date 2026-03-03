import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

def auditar_resampling(ruta_original, ruta_preprocesada):
    """
    Compara las propiedades físicas de la imagen antes y después del resampling,
    y visualiza un corte coronal para confirmar proporciones reales.
    """
    ct_orig = sitk.ReadImage(os.path.join(ruta_original, "image.nii.gz"))
    ct_prep = sitk.ReadImage(os.path.join(ruta_preprocesada, "image.nii.gz"))

    # 1. VERIFICACIÓN NUMÉRICA (Lo que la computadora ve)
    print("=== ANTES (Original) ===")
    print(f" Espaciado físico (X, Y, Z): {ct_orig.GetSpacing()}")
    print(f" Tamaño de matriz (X, Y, Z): {ct_orig.GetSize()}")

    print("\n=== DESPUÉS (Preprocesado) ===")
    print(f" Espaciado físico (X, Y, Z): {ct_prep.GetSpacing()}  <-- ¡Debe ser (1.0, 1.0, 1.0)!")
    print(f" Tamaño de matriz (X, Y, Z): {ct_prep.GetSize()}")
    print("-" * 40)

    # 2. VERIFICACIÓN VISUAL (Corte Coronal)
    arr_orig = sitk.GetArrayFromImage(ct_orig)
    arr_prep = sitk.GetArrayFromImage(ct_prep)

    # Tomamos el corte de la mitad en el eje Y para ver al paciente de frente
    mitad_y_orig = arr_orig.shape[1] // 2
    mitad_y_prep = arr_prep.shape[1] // 2

    plt.figure(figsize=(12, 6))

    # Gráfico Original
    plt.subplot(1, 2, 1)
    plt.title("Coronal Original\n(La matriz cruda sin corregir aspecto)")
    # En Python, si graficamos la matriz pura, un Z de 2.5mm se ve aplastado
    plt.imshow(arr_orig[:, mitad_y_orig, :], cmap="gray", vmin=-160, vmax=240, origin='lower')
    plt.axis('off')

    # Gráfico Resampleado
    plt.subplot(1, 2, 2)
    plt.title("Coronal Resampleado a 1x1x1mm\n(Proporción anatómica real)")
    # Aquí el paciente debería verse con sus proporciones normales
    plt.imshow(arr_prep[:, mitad_y_prep, :], cmap="gray", vmin=-160, vmax=240, origin='lower')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Raíz del proyecto (Lymph-Node/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Carpeta con los NIfTI originales sin resampling
    carpeta_nifit = os.path.join(base_dir, "Dataset_NIFIT")

    # Carpeta con los NIfTI resampleados a 1×1×1 mm isotrópico
    carpeta_preprocesado = os.path.join(base_dir, "Dataset_Preprocesado")

    # ID del paciente a auditar (puedes cambiarlo o usar variable de entorno)
    paciente = os.environ.get("PACIENTE_ID", "case_0093")
    carpeta_orig = os.path.join(carpeta_nifit, paciente)
    carpeta_prep = os.path.join(carpeta_preprocesado, paciente)

    if not os.path.isdir(carpeta_orig):
        candidatos = sorted([d for d in os.listdir(carpeta_nifit) if os.path.isdir(os.path.join(carpeta_nifit, d))])
        if not candidatos:
            raise FileNotFoundError(f"No se encontraron pacientes en: {carpeta_nifit}")
        paciente = candidatos[0]
        print(f"Paciente no encontrado. Usando {paciente}.")
        carpeta_orig = os.path.join(carpeta_nifit, paciente)
        carpeta_prep = os.path.join(carpeta_preprocesado, paciente)

    auditar_resampling(carpeta_orig, carpeta_prep)