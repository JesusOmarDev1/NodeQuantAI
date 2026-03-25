import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importamos nuestras clases creadas previamente
from dataset_medico import GangliosDataset3D
from modelo_unet3d import AttentionUNet3D


# ==========================================
# 1. DEFINICIÓN DE LA FUNCIÓN DE PÉRDIDA
# ==========================================
class DiceBCELoss(nn.Module):
    """Combinación de Binary Cross Entropy y Dice Loss para clases desbalanceadas."""

    def __init__(self, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, prediccion, objetivo):
        # 1. Calcular BCE normal
        bce_loss = self.bce(prediccion, objetivo)

        # 2. Calcular Dice Loss
        pred_aplanado = prediccion.view(-1)
        obj_aplanado = objetivo.view(-1)

        interseccion = (pred_aplanado * obj_aplanado).sum()
        dice = (2. * interseccion + self.smooth) / (pred_aplanado.sum() + obj_aplanado.sum() + self.smooth)
        dice_loss = 1.0 - dice

        # 3. Sumar ambas
        return bce_loss + dice_loss


# ==========================================
# 2. CONFIGURACIÓN DEL BUCLE DE ENTRENAMIENTO
# ==========================================
def entrenar_modelo():
    # Detectar tarjeta gráfica (GPU) automáticamente
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando entrenamiento en: {dispositivo}")
    if dispositivo.type == 'cpu':
        print("¡ADVERTENCIA! Entrenar 3D en CPU será extremadamente lento.")

    # Hiperparámetros
    # Raíz del proyecto (Lymph-Node/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Carpeta con los NIfTI resampleados a 1×1×1 mm (entrada para el entrenamiento de la CNN)
    CARPETA_DATOS = os.path.join(base_dir, "Dataset_Preprocesado")
    TAMANO_PARCHE = (96, 96, 96)
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    EPOCHS = 10  # Empezamos con pocas épocas para probar

    # Instanciar Dataset y DataLoader
    print("Cargando base de datos...")
    dataset = GangliosDataset3D(CARPETA_DATOS, patch_size=TAMANO_PARCHE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instanciar Modelo, Función de Pérdida y Optimizador
    print("Inicializando U-Net 3D con Attention Gates...")
    modelo = AttentionUNet3D(in_channels=1, out_channels=1).to(dispositivo)

    criterio = DiceBCELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 3. EL CICLO DE APRENDIZAJE (TRAINING LOOP)
    # ==========================================
    print("¡Comenzando el entrenamiento!")

    for epoca in range(EPOCHS):
        modelo.train()  # Modo de entrenamiento activado
        perdida_acumulada = 0.0

        for i, (imagenes, mascaras) in enumerate(dataloader):
            # Mover datos a la memoria de la tarjeta gráfica
            imagenes = imagenes.to(dispositivo)
            mascaras = mascaras.to(dispositivo)

            # Paso 1: Reiniciar los gradientes del optimizador
            optimizador.zero_grad()

            # Paso 2: Forward Pass (Hacer predicciones)
            predicciones = modelo(imagenes)

            # Paso 3: Calcular el error (Loss)
            loss = criterio(predicciones, mascaras)

            # Paso 4: Backward Pass (Calcular gradientes)
            loss.backward()

            # Paso 5: Actualizar los pesos de la red
            optimizador.step()

            perdida_acumulada += loss.item()

            # Imprimir progreso por cada batch
            print(f"  Época [{epoca + 1}/{EPOCHS}] - Batch [{i + 1}/{len(dataloader)}] - Loss: {loss.item():.4f}")

        # Resumen al final de cada época
        perdida_promedio = perdida_acumulada / len(dataloader)
        print(f"=== Fin Época {epoca + 1} | Pérdida Promedio: {perdida_promedio:.4f} ===")

        # Guardar un checkpoint (un "Save State" del modelo)
        torch.save(modelo.state_dict(), f"unet3d_epoca_{epoca + 1}.pth")

    print("¡Entrenamiento completado!")


if __name__ == "__main__":
    entrenar_modelo()