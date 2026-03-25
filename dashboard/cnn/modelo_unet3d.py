import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Bloque básico de convolución: Conv3D -> BatchNorm -> ReLU -> Conv3D -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate3D(nn.Module):
    """Compuerta de Atención para filtrar características irrelevantes en las Skip Connections"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Multiplicamos la máscara de atención por la skip connection


class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(AttentionUNet3D, self).__init__()
        features = init_features

        # ENCODER (Bajada)
        self.encoder1 = ConvBlock3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # BOTTLENECK (El fondo de la U)
        self.bottleneck = ConvBlock3D(features * 4, features * 8)

        # DECODER (Subida) con Attention Gates
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = ConvBlock3D(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = ConvBlock3D(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = ConvBlock3D(features * 2, features)

        # CAPA FINAL (Predicción)
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        # Usamos Sigmoide porque es clasificación binaria (Es ganglio o no es ganglio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder 3
        d3 = self.upconv3(b)
        x3 = self.att3(g=d3, x=e3)  # Aplicar atención
        d3 = self.decoder3(torch.cat((x3, d3), dim=1))  # Concatenar

        # Decoder 2
        d2 = self.upconv2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = self.decoder2(torch.cat((x2, d2), dim=1))

        # Decoder 1
        d1 = self.upconv1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = self.decoder1(torch.cat((x1, d1), dim=1))

        # Salida
        out = self.final_conv(d1)
        return self.sigmoid(out)


# ==========================================
# PRUEBA RÁPIDA (Sanity Check)
# ==========================================
if __name__ == "__main__":
    # 1. Instanciar el modelo
    modelo = AttentionUNet3D(in_channels=1, out_channels=1)

    # 2. Crear un tensor "falso" que simula un batch de tu DataLoader
    # (Batch=2, Canales=1, Z=96, Y=96, X=96)
    tensor_prueba = torch.randn(2, 1, 96, 96, 96)
    print(f"Forma de la entrada: {tensor_prueba.shape}")

    # 3. Pasar el tensor por el modelo (Forward Pass)
    print("Pasando datos por la Attention 3D U-Net (esto puede tardar unos segundos)...")
    salida = modelo(tensor_prueba)

    # 4. Verificar la salida
    print(f"Forma de la salida (Predicción): {salida.shape}")