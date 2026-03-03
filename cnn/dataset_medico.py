import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class GangliosDataset3D(Dataset):
    def __init__(self, ruta_dataset, patch_size=(96, 96, 96)):
        self.ruta_dataset = ruta_dataset
        self.patch_size = patch_size

        # Listar carpetas válidas
        self.pacientes = [p for p in os.listdir(ruta_dataset)
                          if os.path.isdir(os.path.join(ruta_dataset, p))]

    def __len__(self):
        return len(self.pacientes)

    def normalizar_imagen(self, imagen_arr, min_hu=-160, max_hu=240):
        # Windowing de tejido blando y normalización [0, 1]
        imagen_arr = np.clip(imagen_arr, min_hu, max_hu)
        imagen_arr = (imagen_arr - min_hu) / (max_hu - min_hu)
        return imagen_arr.astype(np.float32)

    def extraer_parche_aleatorio(self, img_arr, mask_arr):
        z_max, y_max, x_max = img_arr.shape
        pz, py, px = self.patch_size

        # Elegir coordenadas de inicio aleatorias
        z_start = np.random.randint(0, z_max - pz)
        y_start = np.random.randint(0, y_max - py)
        x_start = np.random.randint(0, x_max - px)

        # Recortar el cubo 3D
        img_patch = img_arr[z_start:z_start + pz, y_start:y_start + py, x_start:x_start + px]
        mask_patch = mask_arr[z_start:z_start + pz, y_start:y_start + py, x_start:x_start + px]

        return img_patch, mask_patch

    def __getitem__(self, idx):
        nombre_paciente = self.pacientes[idx]
        ruta_ct = os.path.join(self.ruta_dataset, nombre_paciente, "image.nii.gz")
        ruta_seg = os.path.join(self.ruta_dataset, nombre_paciente, "mask.nii.gz")

        ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ruta_ct))
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(ruta_seg))

        ct_arr = self.normalizar_imagen(ct_arr)
        img_patch, mask_patch = self.extraer_parche_aleatorio(ct_arr, mask_arr)

        # 5. Convertir a Tensores de PyTorch y añadir la dimensión de canal (C, Z, Y, X)
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0)  # Canal 1 (Escala de grises)

        # BINARIZACIÓN ESTRICTA: Cualquier valor mayor a 0 se convierte en 1.0
        mask_patch_binario = (mask_patch > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_patch_binario).unsqueeze(0)

        return img_tensor, mask_tensor



# ==========================================
# BLOQUE PRINCIPAL (MAIN)
# ==========================================
if __name__ == "__main__":
    # 1. Rutas y configuración
    # Raíz del proyecto (Lymph-Node/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Carpeta con los NIfTI resampleados a 1×1×1 mm (CT + máscara por paciente)
    carpeta_preprocesada = os.path.join(base_dir, "Dataset_Preprocesado")
    tamano_parche = (96, 96, 96)  # Tamaño del cubo 3D
    tamano_batch = 2  # Cuántos cubos procesar al mismo tiempo

    print(f"Iniciando Dataset desde: {carpeta_preprocesada}")

    # 2. Instanciar el Dataset y el DataLoader
    mi_dataset = GangliosDataset3D(carpeta_preprocesada, patch_size=tamano_parche)

    # El DataLoader mezcla a los pacientes y los agrupa en batches
    mi_dataloader = DataLoader(mi_dataset, batch_size=tamano_batch, shuffle=True)

    print(f"Total de pacientes encontrados: {len(mi_dataset)}")

    # 3. Extraer un Batch de prueba
    print("Extrayendo un batch aleatorio...")
    imagenes_batch, mascaras_batch = next(iter(mi_dataloader))

    print(f" -> Forma del Tensor de Imágenes: {imagenes_batch.shape} (Batch, Canal, Z, Y, X)")
    print(f" -> Forma del Tensor de Máscaras: {mascaras_batch.shape} (Batch, Canal, Z, Y, X)")
    print(
        f" -> Rango de valores de la imagen: [{imagenes_batch.min():.2f}, {imagenes_batch.max():.2f}] (Debería ser 0 a 1)")

    # 4. Visualización rápida (Corte central del primer parche del batch)
    # Seleccionamos el paciente 0 del batch, canal 0, y el corte de la mitad en Z
    mitad_z = tamano_parche[0] // 2

    img_mostrar = imagenes_batch[0, 0, mitad_z, :, :].numpy()
    mask_mostrar = mascaras_batch[0, 0, mitad_z, :, :].numpy()

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Corte central del Parche (Z={mitad_z})\nNormalizado 0-1")
    plt.imshow(img_mostrar, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Máscara del Parche\n(Puede estar negra si el parche cayó en fondo)")
    plt.imshow(img_mostrar, cmap="gray")
    plt.imshow(mask_mostrar, cmap="Reds", alpha=0.5, interpolation='none')
    plt.axis('off')

    plt.tight_layout()
    plt.show()