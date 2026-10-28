"""
evaluate_custom.py - Kendi El Yazisi Rakamlarini Degerlendirme ve Dagilim Kaymasi (Distribution Shift) Analizi

Odev Gereksinimleri:
- Ogrencinin el yazisiyla yazdigi okul numarasi rakamlarinin modele verilmesi.
- Egitilen modelin gercek hayattaki el yazilari uzerindeki basarisinin degerlendirilmesi.
- Training data ile gercek dunya verisi arasindaki farkin (Distribution Shift / Generalization) analizi.
"""

import sys
import os
import glob
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset import get_mnist_transforms
from src.models import get_model


def extract_label_from_filename(filename):
    """
    Dosya adindan gercek rakam etiketini cikarir.
    Ornekler:
      digit_0_val2.png -> 2
      custom_7.png     -> 7
      2.png            -> 2
    """
    match = re.search(r'val(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    digits = re.findall(r'\d+', filename)
    if digits:
        return int(digits[-1])
    return 0


def evaluate_custom_digits(
    model_checkpoint_path="models/Deney1_BaseMLP_best.pth",
    custom_dir="data/custom_digits/processed",
    figures_dir="figures",
    save_fig_name="custom_digit_predictions.png"
):
    """
    Islenmis el yazisi rakamlarini yukler, egitilmis MLP ile tahmin yapar
    ve sonuclari gorsellestirir.
    """
    os.makedirs(figures_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Model agirlik dosyasi bulunamadi: {model_checkpoint_path}")

    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    config = checkpoint.get('config', {'model_name': 'base', 'dropout_rate': 0.2})
    model = get_model(config['model_name'], dropout_rate=config['dropout_rate']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image_files = sorted(glob.glob(os.path.join(custom_dir, "*.png")) + 
                         glob.glob(os.path.join(custom_dir, "*.jpg")))

    if not image_files:
        raise FileNotFoundError(f"'{custom_dir}' dizininde gorsel bulunamadi! Once preprocess_custom.py calistirilmalidir.")

    transform = get_mnist_transforms()

    results = []
    correct_count = 0

    print("\n" + "="*70)
    print(f"Kendi El Yazisi Rakamlari Degerlendiriliyor (Toplam {len(image_files)} Rakam)")
    print(f"Kullanilan Model: {model_checkpoint_path}")
    print("="*70)

    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        true_label = extract_label_from_filename(filename)

        img = Image.open(img_path).convert('L')
        tensor_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor_img)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            pred_label = int(np.argmax(probs))
            confidence = float(probs[pred_label]) * 100.0

        is_correct = (pred_label == true_label)
        if is_correct:
            correct_count += 1

        results.append({
            "index": idx,
            "filename": filename,
            "image": img,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "is_correct": is_correct,
            "probs": probs
        })

        status = "[DOGRU]" if is_correct else "[YANLIS]"
        print(f"Rakam {idx+1:02d} ({filename}) | Gercek: {true_label} | Tahmin: {pred_label} | "
              f"Guven: %{confidence:.2f} | {status}")

    total_images = len(results)
    custom_accuracy = (correct_count / total_images) * 100.0 if total_images > 0 else 0.0

    print("\n--- Sonuc Ozeti ---")
    print(f"Toplam Rakam:    {total_images}")
    print(f"Dogru Tahmin:    {correct_count}")
    print(f"Ozel Veri Basarisi (Accuracy): %{custom_accuracy:.2f}")

    # Gorsellestirme
    cols = min(total_images, 10)
    rows = (total_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    if total_images == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        if i < total_images:
            item = results[i]
            axes[i].imshow(item["image"], cmap='gray')
            color = 'green' if item["is_correct"] else 'red'
            axes[i].set_title(
                f"Gercek: {item['true_label']}\nTahmin: {item['pred_label']}\n(%{item['confidence']:.1f})",
                color=color, fontsize=10, fontweight='bold'
            )
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.suptitle(f"Ogrenci El Yazisi Rakam Tanima Sonuclari (Basari: %{custom_accuracy:.1f})",
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    save_path = os.path.join(figures_dir, save_fig_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Tahmin Gorseli Kaydedildi] {save_path}")

    return {
        "total": total_images,
        "correct": correct_count,
        "accuracy": custom_accuracy,
        "results": results,
        "plot_path": save_path
    }


def print_distribution_shift_analysis(custom_eval_results, test_acc):
    """
    Odevin beklenen ogrenme ciktisi olan 'Distribution Shift / Generalization'
    farkini ayrintili olarak konsola yazdirir.
    """
    custom_acc = custom_eval_results["accuracy"]
    diff = test_acc - custom_acc

    print("\n" + "="*80)
    print("GERCEK DUNYA VERISI VE DAGILIM KAYMASI (DISTRIBUTION SHIFT) ANALIZI")
    print("="*80)
    print(f"1. MNIST Test Kumesi Dogrulugu:       %{test_acc:.2f}")
    print(f"2. Ogrenci El Yazisi Kumesi Dogrulugu: %{custom_acc:.2f}")
    print(f"3. Basarim Farki (Genellestirme Boslugu): %{diff:.2f}")
    print("-" * 80)
    print("ANALIZ VE TARTISMA:")
    print("1. Uzamsal Oteleme Hassasiyeti (Lack of Translation Invariance):")
    print("   MLP mimarisi pikselleri duzlestirip (Flatten: 784) tek boyutlu bir vektor olarak")
    print("   isler. Evrisimli (CNN) katmanlarin aksine, MLP'de kayan filtreler (weight sharing)")
    print("   bulunmaz. Bu nedenle, rakam 28x28 karenin merkezinden 1-2 piksel dahi kaysa,")
    print("   agirliklar farkli piksellerle esleserek yanlis siniflandirmaya yol acar.")
    print("\n2. Cizgi Kalinligi ve Kontrast (Stroke Width & Contrast Shift):")
    print("   MNIST veri seti 1990'larda taranmis ve ikili/antialiased edilmis ozel bir yazi")
    print("   grubuna (NIST) aittir. Tukenmez/kursun kalemle kagida yazilan rakamlarin cizgi")
    print("   kalinligi ve kenar yumusatmasi MNIST'in dagilimindan farklidir.")
    print("\n3. Kisisel Yazi Stili ve Sembol Varyasyonlari:")
    print("   Ornegin '7' rakaminin ortasina cizgi cekilmesi (Avrupa/Turkiye stili),")
    print("   MNIST'te cogunlukla cizgisiz olan 7'lerle celisebilir ve model bu rakami")
    print("   '1' veya '2' olarak tahmin edebilir.")
    print("="*80 + "\n")


if __name__ == "__main__":
    res = evaluate_custom_digits()
    print_distribution_shift_analysis(res, test_acc=98.15)
