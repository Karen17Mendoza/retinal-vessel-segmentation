
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════
# BLOQUE CONVOLUCIONAL BASE
# Dos convoluciones 3x3 + BatchNorm + ReLU
# Es el bloque fundamental que se repite en toda la U-Net
# ═══════════════════════════════════════════════════════
class BloqueConv(nn.Module):
    def __init__(self, canales_entrada, canales_salida, dropout=0.0):
        super().__init__()
        self.bloque = nn.Sequential(
            nn.Conv2d(canales_entrada, canales_salida,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(canales_salida, canales_salida,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bloque(x)


# ═══════════════════════════════════════════════════════
# BLOQUE ENCODER (bajada)
# MaxPool 2x2 → BloqueConv
# Reduce resolución espacial a la mitad
# ═══════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self, canales_entrada, canales_salida, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = BloqueConv(canales_entrada, canales_salida, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


# ═══════════════════════════════════════════════════════
# BLOQUE DECODER (subida)
# ConvTranspose2d (upsample) → concatenar skip → BloqueConv
# Las skip connections son la clave de U-Net:
# recuperan detalles espaciales perdidos en el encoder
# ═══════════════════════════════════════════════════════
class Decoder(nn.Module):
    def __init__(self, canales_entrada, canales_salida, dropout=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            canales_entrada, canales_salida,
            kernel_size=2, stride=2
        )
        # Después de concatenar skip: canales_salida * 2
        self.conv = BloqueConv(canales_salida * 2, canales_salida, dropout)

    def forward(self, x, skip):
        x = self.upsample(x)

        # Ajuste de tamaño si hay diferencia por padding
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)

        # Concatenar a lo largo del eje de canales (skip connection)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ═══════════════════════════════════════════════════════
# U-NET ESTÁNDAR
# Arquitectura original de Ronneberger et al. (2015)
# Implementada desde cero sin usar librerías externas
# ═══════════════════════════════════════════════════════
class UNet(nn.Module):
    """
    U-Net para segmentación binaria de vasos retinianos.
    
    Arquitectura:
        Encoder: 3 → 64 → 128 → 256 → 512
        Bottleneck: 512 → 1024
        Decoder: 1024 → 512 → 256 → 128 → 64
        Salida: 64 → 1 (mapa de probabilidades)
    
    Referencia: Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-net: Convolutional networks for biomedical image segmentation.
    MICCAI 2015.
    """
    def __init__(self, canales_entrada=3, canales_salida=1, dropout=0.1):
        super().__init__()

        # ── Encoder (camino de bajada) ──────────────────
        self.enc1 = BloqueConv(canales_entrada, 64,   dropout)
        self.enc2 = Encoder(64,  128, dropout)
        self.enc3 = Encoder(128, 256, dropout)
        self.enc4 = Encoder(256, 512, dropout)

        # ── Bottleneck (parte más profunda) ────────────
        self.bottleneck = Encoder(512, 1024, dropout)

        # ── Decoder (camino de subida) ──────────────────
        self.dec4 = Decoder(1024, 512, dropout)
        self.dec3 = Decoder(512,  256, dropout)
        self.dec2 = Decoder(256,  128, dropout)
        self.dec1 = Decoder(128,  64,  dropout)

        # ── Capa de salida ──────────────────────────────
        # Conv 1x1 → mapa de 1 canal (probabilidad de vaso)
        self.salida = nn.Conv2d(64, canales_salida, kernel_size=1)

    def forward(self, x):
        # Encoder — guardamos cada salida como skip connection
        s1 = self.enc1(x)        # [B,  64, H,   W  ]
        s2 = self.enc2(s1)       # [B, 128, H/2, W/2]
        s3 = self.enc3(s2)       # [B, 256, H/4, W/4]
        s4 = self.enc4(s3)       # [B, 512, H/8, W/8]

        # Bottleneck
        bn = self.bottleneck(s4) # [B,1024, H/16,W/16]

        # Decoder — cada bloque recibe el skip correspondiente
        x = self.dec4(bn, s4)    # [B, 512, H/8, W/8]
        x = self.dec3(x,  s3)    # [B, 256, H/4, W/4]
        x = self.dec2(x,  s2)    # [B, 128, H/2, W/2]
        x = self.dec1(x,  s1)    # [B,  64, H,   W  ]

        return self.salida(x)    # [B,   1, H,   W  ]


# ═══════════════════════════════════════════════════════
# ATTENTION U-NET
# Variante mejorada con compuertas de atención
# en las skip connections (Oktay et al., 2018)
# Permite al decoder enfocarse en regiones relevantes
# Usada en el estudio de ablación (Entregable 2)
# ═══════════════════════════════════════════════════════
class CompuertaAtencion(nn.Module):
    """
    Attention Gate de Oktay et al. (2018).
    Filtra las skip connections para resaltar
    solo las regiones relevantes (vasos).
    
    g: señal del decoder (guía)
    x: skip connection del encoder (a filtrar)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Alinear resoluciones si difieren
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode='bilinear', align_corners=False)

        # Calcular mapa de atención (alfa)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Aplicar atención a la skip connection
        return x * psi


class DecoderAtencion(nn.Module):
    def __init__(self, canales_entrada, canales_salida, dropout=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            canales_entrada, canales_salida,
            kernel_size=2, stride=2
        )
        self.atencion = CompuertaAtencion(
            F_g=canales_salida,
            F_l=canales_salida,
            F_int=canales_salida // 2
        )
        self.conv = BloqueConv(canales_salida * 2, canales_salida, dropout)

    def forward(self, x, skip):
        x    = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        # Filtrar skip con atención antes de concatenar
        skip = self.atencion(g=x, x=skip)
        x    = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net para ablación.
    Igual que U-Net pero con compuertas de atención
    en cada skip connection del decoder.
    
    Referencia: Oktay, O. et al. (2018).
    Attention U-Net: Learning where to look for the pancreas.
    MIDL 2018.
    """
    def __init__(self, canales_entrada=3, canales_salida=1, dropout=0.1):
        super().__init__()

        self.enc1      = BloqueConv(canales_entrada, 64,   dropout)
        self.enc2      = Encoder(64,  128, dropout)
        self.enc3      = Encoder(128, 256, dropout)
        self.enc4      = Encoder(256, 512, dropout)
        self.bottleneck= Encoder(512, 1024, dropout)

        # Decoder con atención
        self.dec4 = DecoderAtencion(1024, 512, dropout)
        self.dec3 = DecoderAtencion(512,  256, dropout)
        self.dec2 = DecoderAtencion(256,  128, dropout)
        self.dec1 = DecoderAtencion(128,  64,  dropout)

        self.salida = nn.Conv2d(64, canales_salida, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bn = self.bottleneck(s4)

        x  = self.dec4(bn, s4)
        x  = self.dec3(x,  s3)
        x  = self.dec2(x,  s2)
        x  = self.dec1(x,  s1)

        return self.salida(x)


# ═══════════════════════════════════════════════════════
# FUNCIONES DE PÉRDIDA (Entregable 2 — ablación)
# ═══════════════════════════════════════════════════════
class PerdidaDice(nn.Module):
    """
    Dice Loss — penaliza directamente el solapamiento
    entre predicción y ground truth.
    Más robusta al desbalance de clases que BCE.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.view(-1)
        target = target.view(-1)

        interseccion = (pred * target).sum()
        dice = (2.0 * interseccion + self.smooth) / (
               pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class PerdidaBCEDice(nn.Module):
    """
    Pérdida combinada BCE + Dice.
    BCE aporta estabilidad en gradientes.
    Dice aporta robustez al desbalance de clases.
    alpha controla el peso relativo.
    """
    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce   = nn.BCEWithLogitsLoss()
        self.dice  = PerdidaDice(smooth)

    def forward(self, pred, target):
        return (self.alpha * self.bce(pred, target) +
                (1 - self.alpha) * self.dice(pred, target))


def obtener_perdida(nombre='bce_dice'):
    """
    Fábrica de funciones de pérdida para el estudio
    de ablación (Entregable 2).
    
    Opciones: 'bce', 'dice', 'bce_dice'
    """
    if nombre == 'bce':
        return nn.BCEWithLogitsLoss()
    elif nombre == 'dice':
        return PerdidaDice()
    elif nombre == 'bce_dice':
        return PerdidaBCEDice(alpha=0.5)
    else:
        raise ValueError(f"Pérdida desconocida: {nombre}")


def contar_parametros(modelo):
    """Cuenta parámetros entrenables del modelo."""
    total = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total
