readme = """# Retinal Vessel Segmentation — U-Net

Segmentación de vasos retinianos para detección de retinopatía diabética
mediante U-Net y Attention U-Net implementadas desde cero en PyTorch.

## Resultados principales

| Modelo | F1 | AUC-ROC | Sensibilidad |
|---|---|---|---|
| UNet + BCE+Dice | 0.7029 | 0.9564 | 0.6720 |
| **UNet + Dice** | **0.7078** | 0.9249 | **0.7391** |
| Attention UNet | 0.7019 | 0.9583 | 0.6605 |

## Datasets

| Dataset | Imágenes | Uso |
|---|---|---|
| DRIVE | 40 | Entrenamiento y evaluación |
| CHASE_DB1 | 28 | Evaluación cross-domain |
| STARE | 19 | Análisis de concordancia |

## Ejecución en un solo comando

### Requisitos
- Google Colab Pro (GPU L4 recomendada)
- Google Drive con los datasets organizados

### Pasos

**1. Clonar el repositorio**
```bash
git clone https://github.com/TU_USUARIO/retinal-vessel-segmentation
```

**2. Organizar datasets en Google Drive**
