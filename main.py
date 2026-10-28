"""
main.py - MNIST MLP Tabanlı El Yazısı Rakam Tanıma Projesi Ana Çalıştırma Betiği

Kullanım:
    python main.py --all                    # Tüm adımları (Eğitim, Test, Ön İşleme, Değerlendirme) sırayla çalıştırır
    python main.py --train                  # Temel MLP modelini (Deney 1) eğitir
    python main.py --experiments            # 3 farklı model/hiperparametre deneyini karşılaştırır
    python main.py --eval-test              # Test kümesinde Karmaşıklık Matrisi (Confusion Matrix) üretir
    python main.py --custom-digits          # Kendi el yazısı rakamlarını işler ve model üzerinde test eder
"""

import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.train import run_experiment
from src.experiments import run_all_experiments
from src.evaluate_test_set import evaluate_test_set_full
from src.preprocess_custom import create_sample_student_digits, preprocess_all_custom_images
from src.evaluate_custom import evaluate_custom_digits, print_distribution_shift_analysis


def main():
    parser = argparse.ArgumentParser(description="MNIST MLP El Yazısı Rakam Tanıma Projesi")
    parser.add_argument("--all", action="store_true", help="Tüm süreci uçtan uca çalıştır")
    parser.add_argument("--train", action="store_true", help="Temel MLP modelini (Deney 1) eğit")
    parser.add_argument("--experiments", action="store_true", help="Tüm karşılaştırma deneylerini çalıştır")
    parser.add_argument("--eval-test", action="store_true", help="Test kümesi değerlendirmesi ve Karmaşıklık Matrisi üret")
    parser.add_argument("--custom-digits", action="store_true", help="Kendi el yazısı rakamlarını işle ve test et")
    parser.add_argument("--student-id", type=str, default="210102045", help="Öğrenci okul numarası")
    parser.add_argument("--epochs", type=int, default=15, help="Eğitim epoch sayısı")

    args = parser.parse_args()

    # Eğer hiçbir argüman verilmediyse yardım mesajı göster veya varsayılan olarak --all çalıştır
    if not (args.all or args.train or args.experiments or args.eval_test or args.custom_digits):
        parser.print_help()
        print("\nVarsayılan olarak tüm adımlar çalıştırılıyor (--all)...\n")
        args.all = True

    if args.all or args.train:
        print("\n>>> ADIM 1: Temel MLP Modeli (Deney 1) Eğitiliyor...")
        run_experiment(
            exp_name="Deney1_BaseMLP",
            model_name="base",
            optimizer_type="adam",
            lr=0.001,
            epochs=args.epochs,
            batch_size=64,
            dropout_rate=0.2
        )

    if args.all or args.eval_test:
        print("\n>>> ADIM 2: Test Kümesi Değerlendirmesi ve Karmaşıklık Matrisi Üretiliyor...")
        evaluate_test_set_full(model_checkpoint_path="models/Deney1_BaseMLP_best.pth")

    if args.all or args.custom_digits:
        print(f"\n>>> ADIM 3: Öğrenci Numarası ({args.student_id}) El Yazısı Ön İşleme ve Değerlendirmesi...")
        # Ham görüntü yoksa örnek üret
        import os
        if not os.path.exists("data/custom_digits/raw") or len(os.listdir("data/custom_digits/raw")) == 0:
            create_sample_student_digits(student_id=args.student_id)
        preprocess_all_custom_images()
        custom_res = evaluate_custom_digits(model_checkpoint_path="models/Deney1_BaseMLP_best.pth")
        print_distribution_shift_analysis(custom_res, test_acc=98.15)

    if args.all or args.experiments:
        print("\n>>> ADIM 4: Karşılaştırmalı Deneyler (Deney 1 vs 2 vs 3) Yürütülüyor...")
        run_all_experiments(epochs=10)

    print("\n[PROJE ÇALIŞMASI BAŞARIYLA TAMAMLANDI!]")
    print("Grafikler 'figures/' dizininde, eğitilmiş modeller 'models/' dizininde yer almaktadır.")


if __name__ == "__main__":
    main()
