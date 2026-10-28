"""
evaluate_test_set.py - MNIST Test Kümesi Kapsamlı Değerlendirmesi ve Karmaşıklık Matrisi (Confusion Matrix)

Ödev Teslimatı için:
- 10.000 test örneği üzerinde sınıflandırma raporu (Precision, Recall, F1-Score)
- Karmaşıklık Matrisi (Confusion Matrix) ısı haritası
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_data_loaders
from src.models import get_model


def evaluate_test_set_full(
    model_checkpoint_path="models/Deney1_BaseMLP_best.pth",
    figures_dir="figures",
    save_fig_name="confusion_matrix.png"
):
    os.makedirs(figures_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli yükle
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    config = checkpoint.get('config', {'model_name': 'base', 'dropout_rate': 0.2})
    model = get_model(config['model_name'], dropout_rate=config['dropout_rate']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Yalnızca Test DataLoader'ı al
    _, _, test_loader, _ = get_data_loaders(batch_size=128)

    all_preds = []
    all_targets = []

    print("\n--- MNIST Test Kümesi (10,000 örnek) Değerlendiriliyor ---")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Metrik Raporu
    report_str = classification_report(all_targets, all_preds, digits=4)
    print("\n[SINIFLANDIRMA RAPORU (Classification Report)]")
    print(report_str)

    # 2. Karmaşıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=True,
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)],
        linewidths=.5
    )
    plt.title("MNIST Test Kümesi - Karmaşıklık Matrisi (Confusion Matrix)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Tahmin Edilen Sınıf (Predicted)", fontsize=12)
    plt.ylabel("Gerçek Sınıf (True)", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(figures_dir, save_fig_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Karmaşıklık Matrisi Grafiği Kaydedildi] {save_path}")

    return {
        "classification_report": report_str,
        "confusion_matrix": cm.tolist(),
        "save_path": save_path
    }


if __name__ == "__main__":
    evaluate_test_set_full()
