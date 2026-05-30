# Retinal Vessel Segmentation — U-Net
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
MyDrive/retinal_segmentation/

**3. Abrir y ejecutar**

Abrir `run_all.ipynb` en Google Colab y ejecutar:
Runtime > Run all
Esto ejecuta automáticamente:
- Instalación de dependencias
- Carga y preprocesamiento de datos (CLAHE)
- Entrenamiento de 3 experimentos (ablación)
- Evaluación completa en DRIVE
- Experimento cross-domain DRIVE→CHASE_DB1
- Generación de todas las figuras

## Estructura del proyecto
retinal_segmentation/
├── src/
│   ├── dataset.py     # DataLoaders DRIVE y CHASE_DB1
│   ├── model.py       # U-Net y Attention U-Net desde cero
│   ├── train.py       # Loop de entrenamiento + métricas
│   └── evaluate.py    # Evaluación, ROC, análisis de fallos
├── checkpoints/       # Mejores modelos guardados (.pth)
├── results/
│   ├── figures/       # Todas las gráficas generadas
│   └── metrics/       # Métricas en JSON
├── run_all.ipynb      # Notebook principal (un solo comando)
├── requirements.txt   # Dependencias
└── README.md

## Entorno

- Python 3.12
- PyTorch 2.0+
- CUDA 12.x (GPU NVIDIA L4)
- Ver `requirements.txt` para lista completa

## Autor

- Karen Melanie Mendoza Ayala
- Maestria en Inteligencia Artificial- POSGRADO-UNI  
- Cusco, Perú — 2026




