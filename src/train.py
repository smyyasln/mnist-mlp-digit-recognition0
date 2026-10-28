"""
train.py - MLP Model Eğitimi, Doğrulama ve Test Süreci

Deney 1 (Temel MLP):
- En az 2 gizli katman (Hidden Layers: 512, 256)
- ReLU aktivasyon fonksiyonu
- CrossEntropyLoss (Sınıflandırma kaybı)
- Adam veya SGD Optimizer
- Train, Validation, Test ayrımı
- Eğitim süresi, Loss ve Accuracy takibi
- Loss ve Accuracy grafiklerinin çizilmesi
- Test verisinin kesinlikle eğitim sürecinde kullanılmaması kuralı
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.dataset import get_data_loaders
from src.models import get_model, count_parameters


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Tek bir eğitim epoch'u yürütür."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Modeli doğrulama (validation) veya test kümesinde değerlendirir."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    avg_acc = (correct / total) * 100.0
    return avg_loss, avg_acc


def plot_metrics(history, save_path="figures/training_curves.png", title="MLP Eğitim ve Doğrulama Grafikleri"):
    """
    Eğitim ve Doğrulama Loss & Accuracy eğrilerini profesyonelce çizer ve kaydeder.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Grafiği
    ax1.plot(epochs, history["train_loss"], 'o-', color='#1f77b4', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history["val_loss"], 's--', color='#ff7f0e', label='Validation Loss', linewidth=2)
    ax1.set_title("Kayıp (Loss) Eğrisi", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)

    # Accuracy Grafiği
    ax2.plot(epochs, history["train_acc"], 'o-', color='#2ca02c', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history["val_acc"], 's--', color='#d62728', label='Validation Accuracy', linewidth=2)
    ax2.set_title("Doğruluk (Accuracy) Eğrisi", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Doğruluk (%)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Grafik Kaydedildi] {save_path}")


def run_experiment(
    exp_name="Deney1_BaseMLP",
    model_name="base",
    optimizer_type="adam",
    lr=0.001,
    epochs=15,
    batch_size=64,
    dropout_rate=0.2,
    seed=42,
    save_model_dir="models",
    figures_dir="figures"
):
    """
    Belirli bir konfigürasyon ile tam eğitim ve değerlendirme deneyini yürütür.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Deney Başlatılıyor: {exp_name}")
    print(f"Model: {model_name} | Optimizer: {optimizer_type} | LR: {lr} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Cihaz (Device): {device}")
    print(f"{'='*70}")

    # 1. Veri Yükleyicileri
    train_loader, val_loader, test_loader, _ = get_data_loaders(batch_size=batch_size, seed=seed)

    # 2. Model, Loss ve Optimizer
    model = get_model(model_name, dropout_rate=dropout_rate).to(device)
    total_params = count_parameters(model)
    print(f"Toplam Parametre Sayısı: {total_params:,}")

    criterion = nn.CrossEntropyLoss()

    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Desteklenmeyen optimizer: {optimizer_type}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_times": []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(save_model_dir, f"{exp_name}_best.pth")
    os.makedirs(save_model_dir, exist_ok=True)

    start_total_time = time.time()

    # 3. Eğitim Döngüsü (Train & Validation)
    print("\n--- Eğitim Süreci Başlıyor ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)

        # En iyi validation accuracy'ye sahip modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'model_name': model_name,
                    'optimizer_type': optimizer_type,
                    'lr': lr,
                    'dropout_rate': dropout_rate
                }
            }, best_model_path)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: %{train_acc:.2f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: %{val_acc:.2f} | "
              f"Süre: {epoch_time:.2f}s")

    total_training_time = time.time() - start_total_time
    print(f"\nEğitim Tamamlandı! Toplam Eğitim Süresi: {total_training_time:.2f} saniye")

    # 4. Grafikleri Çiz ve Kaydet
    curve_path = os.path.join(figures_dir, f"{exp_name}_curves.png")
    plot_metrics(history, save_path=curve_path, title=f"{exp_name} - Kayıp ve Doğruluk Grafikleri")

    # 5. TEST KÜMESİNDE NİHAİ DEĞERLENDİRME
    # Önemli Kural: Test kümesi eğitim bittikten sonra yalnızca 1 kez en iyi model üzerinde çalıştırılır.
    print("\n--- Nihai Test Kümesi Değerlendirmesi ---")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"En İyi Model (Val Acc: %{checkpoint['val_acc']:.2f}) yüklendi.")
    print(f"-> Test Loss:     {test_loss:.4f}")
    print(f"-> Test Accuracy: %{test_acc:.2f}")

    # Deney Rapor Özeti
    results_summary = {
        "experiment_name": exp_name,
        "model_name": model_name,
        "total_parameters": total_params,
        "optimizer": optimizer_type,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "training_time_seconds": round(total_training_time, 2),
        "final_train_loss": round(history["train_loss"][-1], 4),
        "final_val_loss": round(history["val_loss"][-1], 4),
        "final_train_acc": round(history["train_acc"][-1], 2),
        "final_val_acc": round(history["val_acc"][-1], 2),
        "best_val_acc": round(best_val_acc, 2),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 2),
        "best_model_path": best_model_path,
        "curve_path": curve_path,
        "history": history
    }

    # Sonuçları JSON olarak kaydet
    json_path = os.path.join(figures_dir, f"{exp_name}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)

    return results_summary


if __name__ == "__main__":
    # Temel MLP (Deney 1) Çalıştır
    run_experiment(
        exp_name="Deney1_BaseMLP",
        model_name="base",
        optimizer_type="adam",
        lr=0.001,
        epochs=15,
        batch_size=64,
        dropout_rate=0.2
    )
