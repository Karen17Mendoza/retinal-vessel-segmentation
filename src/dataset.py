
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# ─────────────────────────────────────────────
# Función de preprocesamiento CLAHE
# Mejora contraste de vasos finos
# También es nuestra estrategia de adaptación
# de dominio (Entregable 6)
# ─────────────────────────────────────────────
def aplicar_clahe(imagen_bgr):
    """
    Aplica CLAHE al canal verde (el más informativo
    en imágenes de fondo de ojo retiniano).
    Retorna imagen RGB normalizada.
    """
    # Extraer canal verde
    canal_verde = imagen_bgr[:, :, 1]
    
    # Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    canal_clahe = clahe.apply(canal_verde)
    
    # Reconstruir imagen con canal verde mejorado
    imagen_proc = imagen_bgr.copy()
    imagen_proc[:, :, 1] = canal_clahe
    
    # Convertir BGR → RGB
    imagen_rgb = cv2.cvtColor(imagen_proc, cv2.COLOR_BGR2RGB)
    return imagen_rgb


def cargar_imagen(ruta):
    """Carga imagen y aplica preprocesamiento CLAHE."""
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {ruta}")
    return aplicar_clahe(img)


def cargar_mascara(ruta):
    """Carga máscara binaria en escala de grises."""
    mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"No se pudo cargar máscara: {ruta}")
    # Binarizar: cualquier valor > 0 es vaso
    mask = (mask > 0).astype(np.float32)
    return mask


# ─────────────────────────────────────────────
# Aumentaciones de datos
# ─────────────────────────────────────────────
def get_transforms_entrenamiento(tam=256):
    """
    Aumentaciones para entrenamiento.
    Incluye flips, rotaciones y ajustes de brillo
    para simular variabilidad clínica.
    """
    return A.Compose([
        A.Resize(tam, tam),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-30, 30),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_transforms_validacion(tam=256):
    """
    Sin aumentaciones aleatorias para validación/test.
    Solo resize y normalización.
    """
    return A.Compose([
        A.Resize(tam, tam),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# Dataset DRIVE — entrenamiento y validación
# ─────────────────────────────────────────────
class DRIVEDataset(Dataset):
    """
    Dataset DRIVE para segmentación de vasos retinianos.
    División fija: 18 imágenes train, 2 validación (de las 20 de training).
    Las 20 de test se usan solo para evaluación final.
    """
    def __init__(self, base_dir, modo='train', tam=256):
        """
        Args:
            base_dir: ruta base del proyecto
            modo: 'train', 'val', o 'test'
            tam: tamaño de imagen de salida
        """
        self.modo = modo
        self.tam = tam
        
        if modo in ['train', 'val']:
            img_dir  = os.path.join(base_dir, 'data/DRIVE/training/images')
            mask_dir = os.path.join(base_dir, 'data/DRIVE/training/1st_manual')
            
            # Obtener lista ordenada de archivos
            imagenes = sorted([
                f for f in os.listdir(img_dir)
                if f.endswith('.tif') or f.endswith('.png')
            ])
            mascaras = sorted([
                f for f in os.listdir(mask_dir)
                if f.endswith('.gif') or f.endswith('.png') or f.endswith('.tif')
            ])
            
            # División fija: 18 train, 2 val
            if modo == 'train':
                imagenes = imagenes[:18]
                mascaras = mascaras[:18]
            else:  # val
                imagenes = imagenes[18:]
                mascaras = mascaras[18:]
            
            self.pares = [
                (os.path.join(img_dir, i), os.path.join(mask_dir, m))
                for i, m in zip(imagenes, mascaras)
            ]
            
        else:  # test
            img_dir  = os.path.join(base_dir, 'data/DRIVE/test/images')
            mask_dir = os.path.join(base_dir, 'data/DRIVE/test/mask')
            
            imagenes = sorted([
                f for f in os.listdir(img_dir)
                if f.endswith('.tif') or f.endswith('.png')
            ])
            
            # Test no tiene máscaras ground truth públicas
            # usamos las del FOV como proxy
            mascaras = sorted([
                f for f in os.listdir(mask_dir)
                if f.endswith('.gif') or f.endswith('.png') or f.endswith('.tif')
            ])
            
            self.pares = [
                (os.path.join(img_dir, i), os.path.join(mask_dir, m))
                for i, m in zip(imagenes, mascaras)
            ]
        
        # Seleccionar transformaciones según modo
        if modo == 'train':
            self.transform = get_transforms_entrenamiento(tam)
        else:
            self.transform = get_transforms_validacion(tam)
        
        print(f"DRIVEDataset [{modo}]: {len(self.pares)} imágenes cargadas")
    
    def __len__(self):
        return len(self.pares)
    
    def __getitem__(self, idx):
        ruta_img, ruta_mask = self.pares[idx]
        
        imagen  = cargar_imagen(ruta_img)
        mascara = cargar_mascara(ruta_mask)
        
        # Aplicar aumentaciones
        augmented = self.transform(image=imagen, mask=mascara)
        imagen    = augmented["image"]        # Tensor [3, H, W]
        mascara   = augmented["mask"]         # Tensor [H, W]
        mascara   = mascara.unsqueeze(0)      # → [1, H, W]
        
        return imagen, mascara


# ─────────────────────────────────────────────
# Dataset CHASE_DB1 — solo para evaluación
# cross-domain (Entregable 4)
# ─────────────────────────────────────────────
class CHASEDataset(Dataset):
    """
    Dataset CHASE_DB1 para experimento de generalización.
    Se usa SOLO para evaluar el modelo entrenado en DRIVE.
    """
    def __init__(self, base_dir, tam=256, usar_clahe=True):
        """
        Args:
            usar_clahe: True para experimento de adaptación
                        de dominio (Entregable 6)
        """
        self.usar_clahe = usar_clahe
        self.tam = tam
        
        img_dir  = os.path.join(base_dir, 'data/CHASE_DB1/images')
        mask_dir = os.path.join(base_dir, 'data/CHASE_DB1/annotations')
        
        imagenes = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        
        # Usar solo máscaras del primer anotador (_1stHO)
        mascaras = sorted([
            f for f in os.listdir(mask_dir)
            if '1st' in f or '1stHO' in f
        ])
        
        # Si no hay distinción de anotador, tomar todas
        if len(mascaras) == 0:
            mascaras = sorted([
                f for f in os.listdir(mask_dir)
                if f.endswith('.png') or f.endswith('.gif')
            ])
        
        self.pares = [
            (os.path.join(img_dir, i), os.path.join(mask_dir, m))
            for i, m in zip(imagenes, mascaras)
        ]
        
        self.transform = get_transforms_validacion(tam)
        print(f"CHASEDataset: {len(self.pares)} imágenes cargadas")
    
    def __len__(self):
        return len(self.pares)
    
    def __getitem__(self, idx):
        ruta_img, ruta_mask = self.pares[idx]
        
        imagen  = cargar_imagen(ruta_img)   # CLAHE aplicado
        mascara = cargar_mascara(ruta_mask)
        
        augmented = self.transform(image=imagen, mask=mascara)
        imagen    = augmented["image"]
        mascara   = augmented["mask"].unsqueeze(0)
        
        return imagen, mascara


# ─────────────────────────────────────────────
# Función para crear todos los DataLoaders
# ─────────────────────────────────────────────
def crear_dataloaders(base_dir, batch_size=4, tam=256, num_workers=2):
    """
    Crea los 4 DataLoaders del proyecto.
    batch_size=4 es seguro para Colab gratuito (T4).
    """
    train_ds = DRIVEDataset(base_dir, modo='train', tam=tam)
    val_ds   = DRIVEDataset(base_dir, modo='val',   tam=tam)
    test_ds  = DRIVEDataset(base_dir, modo='test',  tam=tam)
    chase_ds = CHASEDataset(base_dir, tam=tam)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1,   # batch=1 para evaluación pixel-a-pixel
        shuffle=False, num_workers=num_workers
    )
    chase_loader = DataLoader(
        chase_ds, batch_size=1,
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, chase_loader
