
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ═══════════════════════════════════════════════════════
# EVALUACIÓN COMPLETA — ENTREGABLE 3
# Métricas: sensibilidad, especificidad, F1, AUC-ROC
# ═══════════════════════════════════════════════════════
def evaluar_modelo(modelo, loader, device, nombre='modelo', umbral=0.5):
    """
    Evaluación completa sobre un DataLoader.
    Retorna diccionario con todas las métricas.
    """
    modelo.eval()
    modelo.to(device)

    all_preds  = []
    all_probs  = []
    all_targets= []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            logits= modelo(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > umbral).float()

            all_probs.append(probs.cpu().numpy().flatten())
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())

    probs   = np.concatenate(all_probs)
    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Métricas básicas
    tp = np.sum((preds == 1) & (targets == 1))
    tn = np.sum((preds == 0) & (targets == 0))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))

    sensibilidad  = tp / (tp + fn + 1e-8)
    especificidad = tn / (tn + fp + 1e-8)
    precision     = tp / (tp + fp + 1e-8)
    f1            = 2*tp / (2*tp + fp + fn + 1e-8)
    accuracy      = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    auc_roc       = roc_auc_score(targets, probs)

    metricas = {
        'nombre':        nombre,
        'sensibilidad':  float(sensibilidad),
        'especificidad': float(especificidad),
        'precision':     float(precision),
        'f1':            float(f1),
        'accuracy':      float(accuracy),
        'auc_roc':       float(auc_roc),
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }

    print(f"\n{'─'*50}")
    print(f" Modelo       : {nombre}")
    print(f"{'─'*50}")
    print(f" Sensibilidad : {sensibilidad:.4f}")
    print(f" Especificidad: {especificidad:.4f}")
    print(f" Precisión    : {precision:.4f}")
    print(f" F1-score     : {f1:.4f}")
    print(f" Accuracy     : {accuracy:.4f}")
    print(f" AUC-ROC      : {auc_roc:.4f}")
    print(f"{'─'*50}")

    return metricas, probs, targets


# ═══════════════════════════════════════════════════════
# MATRIZ DE CONFUSIÓN VISUAL
# ═══════════════════════════════════════════════════════
def graficar_confusion(metricas, base_dir):
    nombre = metricas['nombre']
    tp = metricas['tp']
    tn = metricas['tn']
    fp = metricas['fp']
    fn = metricas['fn']

    matriz = np.array([[tn, fp], [fn, tp]])
    total  = tp + tn + fp + fn

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matriz, cmap='Blues')

    etiquetas = [['TN\n(fondo correcto)',  'FP\n(fondo→vaso)'],
                 ['FN\n(vaso→fondo)',      'TP\n(vaso correcto)']]

    for i in range(2):
        for j in range(2):
            val = matriz[i, j]
            pct = val / total * 100
            ax.text(j, i, f'{etiquetas[i][j]}\n{val:,}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=10,
                    color='white' if val > matriz.max()*0.6 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Fondo', 'Pred: Vaso'])
    ax.set_yticklabels(['Real: Fondo', 'Real: Vaso'])
    ax.set_title(f'Matriz de confusión — {nombre}', fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    ruta = f'{base_dir}/results/figures/confusion_{nombre}.png'
    plt.savefig(ruta, dpi=150)
    plt.show()
    print(f'✅ Guardada: confusion_{nombre}.png')


# ═══════════════════════════════════════════════════════
# CURVA ROC
# ═══════════════════════════════════════════════════════
def graficar_roc(lista_resultados, base_dir):
    """
    lista_resultados: [(nombre, probs, targets), ...]
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colores = ['#2196F3', '#FF5722', '#4CAF50']
    for i, (nombre, probs, targets) in enumerate(lista_resultados):
        fpr, tpr, _ = roc_curve(targets, probs)
        auc = roc_auc_score(targets, probs)
        ax.plot(fpr, tpr, color=colores[i],
                linewidth=2, label=f'{nombre} (AUC={auc:.4f})')

    ax.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Tasa de Falsos Positivos (1-Especificidad)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    ax.set_title('Curva ROC — Comparación de modelos')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = f'{base_dir}/results/figures/curva_roc_comparacion.png'
    plt.savefig(ruta, dpi=150)
    plt.show()
    print('✅ Guardada: curva_roc_comparacion.png')


# ═══════════════════════════════════════════════════════
# TABLA COMPARATIVA DE ABLACIÓN — ENTREGABLE 2
# ═══════════════════════════════════════════════════════
def tabla_comparativa(lista_metricas, base_dir):
    nombres = [m['nombre'] for m in lista_metricas]
    campos  = ['sensibilidad','especificidad','precision','f1','accuracy','auc_roc']
    etiq    = ['Sensibilidad','Especificidad','Precisión','F1-score','Accuracy','AUC-ROC']

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')

    filas = []
    for m in lista_metricas:
        filas.append([f"{m[c]:.4f}" for c in campos])

    tabla = ax.table(
        cellText=filas,
        rowLabels=nombres,
        colLabels=etiq,
        cellLoc='center',
        loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 2.0)

    # Resaltar mejor F1
    mejor_f1_idx = np.argmax([m['f1'] for m in lista_metricas])
    col_f1 = campos.index('f1')
    tabla[mejor_f1_idx+1, col_f1].set_facecolor('#C8E6C9')

    ax.set_title('Comparación cuantitativa — Estudio de ablación',
                 fontsize=13, pad=20)
    plt.tight_layout()
    ruta = f'{base_dir}/results/figures/tabla_ablacion.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print('✅ Guardada: tabla_ablacion.png')


# ═══════════════════════════════════════════════════════
# VISUALIZACIÓN CUALITATIVA — ENTREGABLE 5
# Análisis de fallos: capilares finos vs arterias
# ═══════════════════════════════════════════════════════
def analizar_fallos(modelo, loader, device, base_dir, n_ejemplos=6):
    """
    Identifica y visualiza ejemplos mal clasificados.
    Analiza dónde falla el modelo: capilares vs arterias.
    """
    modelo.eval()
    modelo.to(device)

    resultados = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            logits= modelo(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(imgs.shape[0]):
                img_np  = imgs[i].cpu().permute(1,2,0).numpy()
                mean = np.array([0.485,0.456,0.406])
                std  = np.array([0.229,0.224,0.225])
                img_np = np.clip(img_np * std + mean, 0, 1)

                mask_np = masks[i,0].cpu().numpy()
                pred_np = preds[i,0].cpu().numpy()
                prob_np = probs[i,0].cpu().numpy()

                # Calcular F1 por imagen
                tp = np.sum((pred_np==1)&(mask_np==1))
                fp = np.sum((pred_np==1)&(mask_np==0))
                fn = np.sum((pred_np==0)&(mask_np==1))
                f1 = 2*tp/(2*tp+fp+fn+1e-8)

                resultados.append({
                    'img': img_np, 'mask': mask_np,
                    'pred': pred_np, 'prob': prob_np,
                    'f1': f1, 'fn': fn, 'fp': fp
                })

    # Ordenar por F1 ascendente (peores primero)
    resultados.sort(key=lambda x: x['f1'])

    fig, axes = plt.subplots(n_ejemplos, 4,
                             figsize=(16, n_ejemplos*3.5))

    titulos_col = ['Imagen original', 'Ground truth',
                   'Predicción', 'Error (FP=rojo, FN=azul)']

    for j, titulo in enumerate(titulos_col):
        axes[0, j].set_title(titulo, fontsize=11, fontweight='bold')

    for i in range(n_ejemplos):
        r = resultados[i]

        # Mapa de errores
        error = np.zeros((*r['mask'].shape, 3))
        error[r['fp'].astype(bool) if isinstance(r['fp'], np.ndarray)
              else ((r['pred']==1)&(r['mask']==0))] = [1, 0, 0]  # FP rojo
        error[((r['pred']==0)&(r['mask']==1))] = [0, 0, 1]       # FN azul

        axes[i,0].imshow(r['img'])
        axes[i,1].imshow(r['mask'],  cmap='gray')
        axes[i,2].imshow(r['pred'],  cmap='gray')
        axes[i,3].imshow(r['img'])
        axes[i,3].imshow(error, alpha=0.6)

        axes[i,0].set_ylabel(f'F1={r["f1"]:.3f}\nFN={r["fn"]}',
                             fontsize=9)

        for j in range(4):
            axes[i,j].axis('off')

    # Leyenda de colores
    rojo  = mpatches.Patch(color='red',  label='Falso Positivo (FP)')
    azul  = mpatches.Patch(color='blue', label='Falso Negativo (FN)')
    fig.legend(handles=[rojo, azul], loc='lower center',
               ncol=2, fontsize=11)

    plt.suptitle('Análisis de fallos — peores predicciones',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    ruta = f'{base_dir}/results/figures/analisis_fallos.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.show()
    print('✅ Guardada: analisis_fallos.png')
    return resultados
