"""
experiments.py - Model Yapılandırmalarını ve Hiperparametreleri Karşılaştırma Deneyleri

Ödev Çıktısı:
- "Hyperparameter değişikliklerinin etkisini analiz edebilmesi"
- "Model yapılandırmalarının performansını karşılaştırmaktır"

Karşılaştırılan Deneyler:
1. Deney 1 (Temel MLP - Baseline): 2 Gizli Katman (512, 256), Adam (lr=0.001), Dropout 0.2
2. Deney 2 (SGD Optimizer): 2 Gizli Katman (512, 256), SGD+Momentum (lr=0.01), Dropout 0.2
3. Deney 3 (Daha Derin MLP): 3 Gizli Katman (512, 256, 128), Adam (lr=0.001), Dropout 0.2
"""

import sys
import os
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
from src.train import run_experiment


def run_all_experiments(epochs=10, batch_size=64, figures_dir="figures"):
    """
    3 farklı konfigürasyondaki deneyi sırayla çalıştırır ve karşılaştırma grafikleri üretir.
    """
    configs = [
        {
            "exp_name": "Deney1_BaseMLP",
            "model_name": "base",
            "optimizer_type": "adam",
            "lr": 0.001,
            "dropout_rate": 0.2
        },
        {
            "exp_name": "Deney2_BaseMLP_SGD",
            "model_name": "base",
            "optimizer_type": "sgd",
            "lr": 0.01,
            "dropout_rate": 0.2
        },
        {
            "exp_name": "Deney3_DeeperMLP_Adam",
            "model_name": "deeper",
            "optimizer_type": "adam",
            "lr": 0.001,
            "dropout_rate": 0.2
        }
    ]

    all_results = []

    for cfg in configs:
        json_path = os.path.join(figures_dir, f"{cfg['exp_name']}_results.json")
        if os.path.exists(json_path):
            print(f"[{cfg['exp_name']}] Önceden tamamlanmış sonuç dosyası bulundu ({json_path}), yükleniyor...")
            with open(json_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
            all_results.append(res)
        else:
            res = run_experiment(
                exp_name=cfg["exp_name"],
                model_name=cfg["model_name"],
                optimizer_type=cfg["optimizer_type"],
                lr=cfg["lr"],
                epochs=epochs,
                batch_size=batch_size,
                dropout_rate=cfg["dropout_rate"]
            )
            all_results.append(res)

    # Karşılaştırma Grafiği Oluştur
    plot_experiments_comparison(all_results, save_path="figures/all_experiments_comparison.png")
    
    # Karşılaştırma Tablosunu Konsola ve Markdown formatında yazdır
    print_comparison_table(all_results)

    return all_results


def plot_experiments_comparison(all_results, save_path="figures/all_experiments_comparison.png"):
    """Tüm deneylerin doğrulama eğrilerini tek bir grafikte karşılaştırır."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, res in enumerate(all_results):
        history = res["history"]
        epochs = range(1, len(history["val_loss"]) + 1)
        label = f"{res['experiment_name']} (Test: %{res['test_accuracy']:.2f})"

        ax1.plot(epochs, history["val_loss"], marker='o', color=colors[i % len(colors)], label=label, linewidth=2)
        ax2.plot(epochs, history["val_acc"], marker='s', color=colors[i % len(colors)], label=label, linewidth=2)

    ax1.set_title("Validation Loss Karşılaştırması", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Kayıp (Loss)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.set_title("Validation Accuracy Karşılaştırması", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Doğruluk (%)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.suptitle("MLP Model Yapılandırmaları ve Optimizasyon Karşılaştırması", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Toplu Karşılaştırma Grafiği Kaydedildi] {save_path}")


def print_comparison_table(all_results):
    """Karşılaştırma sonuçlarını Markdown ve konsol tablosu olarak yazdırır."""
    print("\n" + "="*100)
    print("TÜM DENEYLERİN PERFORMANS KARŞILAŞTIRMA TABLOSU")
    print("="*100)
    header = f"{'Deney Adı':<25} | {'Model':<8} | {'Opt (LR)':<13} | {'Param':<10} | {'Süre (s)':<9} | {'Val Acc':<8} | {'Test Acc':<8}"
    print(header)
    print("-" * len(header))
    
    for r in all_results:
        opt_str = f"{r['optimizer'].upper()} ({r['learning_rate']})"
        row = f"{r['experiment_name']:<25} | {r['model_name']:<8} | {opt_str:<13} | {r['total_parameters']:<10,d} | {r['training_time_seconds']:<9.1f} | %{r['best_val_acc']:<7.2f} | %{r['test_accuracy']:<7.2f}"
        print(row)
    print("="*100 + "\n")


if __name__ == "__main__":
    run_all_experiments(epochs=10)
