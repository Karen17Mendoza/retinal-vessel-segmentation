
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ═══════════════════════════════════════════════════════
# MÉTRICAS DE EVALUACIÓN
# ═══════════════════════════════════════════════════════
def calcular_metricas(pred, target, umbral=0.5):
    """
    Calcula métricas de segmentación binaria.
    pred  : logits del modelo [B, 1, H, W]
    target: máscaras ground truth [B, 1, H, W]
    """
    pred_bin = (torch.sigmoid(pred) > umbral).float()

    tp = (pred_bin * target).sum().item()
    tn = ((1-pred_bin) * (1-target)).sum().item()
    fp = (pred_bin * (1-target)).sum().item()
    fn = ((1-pred_bin) * target).sum().item()

    sensibilidad = tp / (tp + fn + 1e-8)   # recall
    especificidad = tn / (tn + fp + 1e-8)
    precision    = tp / (tp + fp + 1e-8)
    f1           = 2 * tp / (2*tp + fp + fn + 1e-8)
    accuracy     = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        'sensibilidad': sensibilidad,
        'especificidad': especificidad,
        'precision':     precision,
        'f1':            f1,
        'accuracy':      accuracy,
    }


# ═══════════════════════════════════════════════════════
# LOOP DE ENTRENAMIENTO — UNA ÉPOCA
# ═══════════════════════════════════════════════════════
def entrenar_epoca(modelo, loader, optimizador, perdida_fn, device):
    modelo.train()
    perdida_total = 0.0
    metricas_acum = {
        'sensibilidad': 0, 'especificidad': 0,
        'precision': 0, 'f1': 0, 'accuracy': 0
    }

    for imgs, masks in tqdm(loader, desc='  Entrenando', leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizador.zero_grad()
        preds = modelo(imgs)
        loss  = perdida_fn(preds, masks)
        loss.backward()
        optimizador.step()

        perdida_total += loss.item()
        metricas = calcular_metricas(preds.detach(), masks)
        for k in metricas_acum:
            metricas_acum[k] += metricas[k]

    n = len(loader)
    return perdida_total / n, {k: v/n for k, v in metricas_acum.items()}


# ═══════════════════════════════════════════════════════
# LOOP DE VALIDACIÓN — UNA ÉPOCA
# ═══════════════════════════════════════════════════════
def validar_epoca(modelo, loader, perdida_fn, device):
    modelo.eval()
    perdida_total = 0.0
    metricas_acum = {
        'sensibilidad': 0, 'especificidad': 0,
        'precision': 0, 'f1': 0, 'accuracy': 0
    }

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='  Validando', leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            preds = modelo(imgs)
            loss  = perdida_fn(preds, masks)

            perdida_total += loss.item()
            metricas = calcular_metricas(preds, masks)
            for k in metricas_acum:
                metricas_acum[k] += metricas[k]

    n = len(loader)
    return perdida_total / n, {k: v/n for k, v in metricas_acum.items()}


# ═══════════════════════════════════════════════════════
# ENTRENAMIENTO COMPLETO CON EARLY STOPPING
# ═══════════════════════════════════════════════════════
def entrenar(modelo, train_loader, val_loader, config):
    """
    Entrena el modelo completo con early stopping
    y guardado automático del mejor checkpoint.

    config debe tener:
        epochs, lr, perdida, device,
        checkpoint_dir, nombre_experimento
    """
    device        = config['device']
    epochs        = config['epochs']
    checkpoint_dir= config['checkpoint_dir']
    nombre        = config['nombre_experimento']
    paciencia     = config.get('paciencia', 15)

    modelo = modelo.to(device)

    # Optimizador y scheduler
    optimizador = Adam(modelo.parameters(), lr=config['lr'],
                       weight_decay=1e-5)
    scheduler   = ReduceLROnPlateau(optimizador, mode='min',
                                    patience=5, factor=0.5,
                                    )

    # Función de pérdida
    from model import obtener_perdida
    perdida_fn = obtener_perdida(config['perdida'])

    # Historial para graficar curvas de aprendizaje
    historial = {
        'train_loss': [], 'val_loss': [],
        'train_f1':   [], 'val_f1':   [],
        'train_sens': [], 'val_sens': [],
        'lr': []
    }

    mejor_val_loss = float('inf')
    epocas_sin_mejora = 0
    mejor_epoca = 0

    print(f"\n{'═'*55}")
    print(f" Experimento : {nombre}")
    print(f" Pérdida     : {config['perdida']}")
    print(f" LR inicial  : {config['lr']}")
    print(f" Épocas máx  : {epochs}")
    print(f" Dispositivo : {device}")
    print(f"{'═'*55}\n")

    for epoca in range(1, epochs + 1):
        t0 = time.time()

        # Entrenamiento
        train_loss, train_met = entrenar_epoca(
            modelo, train_loader, optimizador, perdida_fn, device)

        # Validación
        val_loss, val_met = validar_epoca(
            modelo, val_loader, perdida_fn, device)

        # Scheduler paso
        scheduler.step(val_loss)
        lr_actual = optimizador.param_groups[0]['lr']

        # Guardar historial
        historial['train_loss'].append(train_loss)
        historial['val_loss'].append(val_loss)
        historial['train_f1'].append(train_met['f1'])
        historial['val_f1'].append(val_met['f1'])
        historial['train_sens'].append(train_met['sensibilidad'])
        historial['val_sens'].append(val_met['sensibilidad'])
        historial['lr'].append(lr_actual)

        # Guardar mejor modelo
        if val_loss < mejor_val_loss:
            mejor_val_loss    = val_loss
            mejor_epoca       = epoca
            epocas_sin_mejora = 0
            ruta_ckpt = os.path.join(
                checkpoint_dir, f'{nombre}_mejor.pth')
            torch.save({
                'epoca':       epoca,
                'modelo':      modelo.state_dict(),
                'optimizador': optimizador.state_dict(),
                'val_loss':    val_loss,
                'val_f1':      val_met['f1'],
                'config':      config,
            }, ruta_ckpt)
        else:
            epocas_sin_mejora += 1

        # Log por época
        dt = time.time() - t0
        print(
            f"Época {epoca:03d}/{epochs} "
            f"| TLoss {train_loss:.4f} "
            f"| VLoss {val_loss:.4f} "
            f"| TF1 {train_met['f1']:.4f} "
            f"| VF1 {val_met['f1']:.4f} "
            f"| VSens {val_met['sensibilidad']:.4f} "
            f"| LR {lr_actual:.6f} "
            f"| {dt:.1f}s"
            f"{'  ✓MEJOR' if epocas_sin_mejora==0 else ''}"
        )

        # Early stopping
        if epocas_sin_mejora >= paciencia:
            print(f"\nEarly stopping en época {epoca}.")
            print(f"Mejor época: {mejor_epoca} "
                  f"(val_loss={mejor_val_loss:.4f})")
            break

    print(f"\n✅ Entrenamiento completo.")
    print(f"   Mejor época : {mejor_epoca}")
    print(f"   Mejor VLoss : {mejor_val_loss:.4f}")
    print(f"   Checkpoint  : {nombre}_mejor.pth")

    return historial
