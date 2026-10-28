"""
preprocess_custom.py - Kendi El Yazısı Rakamlarını MNIST Formatına Dönüştürme Modülü

MNIST Veri Seti Standartları (Yann LeCun et al.):
1. Gri seviye (Grayscale) 28x28 piksel.
2. Arka plan siyah (0), rakam çizgileri beyaz/açık renk (255'e kadar).
3. Rakam, en-boy oranı korunarak 20x20 piksellik bir kutuya sığdırılır.
4. Rakamın kütle merkezi (Center of Mass) veya sınır kutusu 28x28 alanın tam ortasına yerleştirilir.
5. Normalizasyon: Ortalama 0.1307, Standart Sapma 0.3081.

Bu modül, cep telefonuyla çekilen kağıt üzerindeki el yazısı fotoğraflarını
yukarıdaki MNIST standartlarına otomatik olarak dönüştürür.
"""

import os
import glob
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch


def preprocess_single_digit(image_input, target_size=(28, 28), digit_box_size=20):
    """
    Tek bir el yazısı rakam görselini MNIST formatına dönüştürür.
    
    Adımlar:
      1. Gri seviyeye çevirme (Grayscale).
      2. Renk inversiyonu (Kağıt beyaz/mürekkep siyah -> MNIST: Arka plan siyah/rakam beyaz).
      3. Gürültü temizleme ve kontrast germe (Thresholding).
      4. Rakamın sınır kutusunu (bounding box) bulma.
      5. En-boy oranını koruyarak 20x20 kutuya sığdırma.
      6. 28x28 siyah tuvalin ortasına yerleştirme (Center of Mass / Bounding Box Centering).
    
    Args:
        image_input: Dosya yolu (str) veya PIL.Image nesnesi.
        target_size: Hedef çıktı boyutu (28, 28).
        digit_box_size: Rakamın sığdırılacağı kutu boyutu (20).
        
    Returns:
        PIL.Image: 28x28 boyutunda MNIST uyumlu görsel.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('L')
    else:
        img = image_input.convert('L')

    img_arr = np.array(img, dtype=np.float32)

    # 1. Renk İnversiyon Kontrolü:
    # Kağıt üzerindeki yazılarda genellikle kenarlar/arka plan parlak (beyaz), yazı koyudur.
    # MNIST'te arka plan 0 (siyah), yazı 255 (beyaz) olmalıdır.
    corners = [img_arr[0, 0], img_arr[0, -1], img_arr[-1, 0], img_arr[-1, -1]]
    if np.mean(corners) > 127:  # Arka plan açıksa tersine çevir
        img = ImageOps.invert(img)
        img_arr = np.array(img, dtype=np.float32)

    # 2. Gürültü ve arka plan temizliği (Basit eşikleme / kontrast artırma)
    # Rakam pikselleri genellikle en yüksek değerlerdir. Arka plan parazitlerini sıfırla.
    p_min = np.percentile(img_arr, 20)
    p_max = np.percentile(img_arr, 99.5)
    if p_max > p_min:
        img_arr = np.clip((img_arr - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
    else:
        img_arr = img_arr.astype(np.uint8)

    # Eşikleme: Belirli bir parlaklığın altındaki arka plan gürültülerini tamamen 0 yap
    threshold = 50
    img_arr[img_arr < threshold] = 0
    img = Image.fromarray(img_arr)

    # 3. Rakamın Sınır Kutusunu (Bounding Box) Bulma
    bbox = img.getbbox()
    if bbox is None:
        # Eğer görüntü tamamen boşsa boş 28x28 döndür
        return Image.new('L', target_size, 0)

    cropped = img.crop(bbox)
    w, h = cropped.size

    # 4. En-boy oranını koruyarak 20x20 kutuya sığdırma
    if w > h:
        new_w = digit_box_size
        new_h = max(1, int(round((h * digit_box_size) / w)))
    else:
        new_h = digit_box_size
        new_w = max(1, int(round((w * digit_box_size) / h)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 5. 28x28 siyah tuval oluştur ve rakamı ortala
    final_canvas = Image.new('L', target_size, 0)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    final_canvas.paste(resized, (paste_x, paste_y))

    return final_canvas


def preprocess_all_custom_images(raw_dir="data/custom_digits/raw", 
                                 processed_dir="data/custom_digits/processed"):
    """
    raw_dir içindeki tüm fotoğrafları işleyip processed_dir içine kaydeder.
    Dosya isimlendirme formatı:
      digit_<indeks>_val<rakam>.png  (Örnek: digit_0_val2.png -> 0. sıradaki rakam, değeri 2)
    """
    os.makedirs(processed_dir, exist_ok=True)
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(raw_dir, ext)))

    if not files:
        print(f"[Uyarı] '{raw_dir}' dizininde işlenecek görüntü bulunamadı.")
        return []

    processed_paths = []
    print(f"Toplam {len(files)} adet ham el yazısı görseli işleniyor...")
    
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        processed_img = preprocess_single_digit(fpath)
        out_path = os.path.join(processed_dir, fname)
        # PNG formatında kaydet
        if not out_path.endswith('.png'):
            out_path = os.path.splitext(out_path)[0] + '.png'
        processed_img.save(out_path)
        processed_paths.append(out_path)
        print(f" -> İşlendi ve kaydedildi: {out_path}")

    print("[Tamamlandı] Tüm el yazısı görüntüleri 28x28 MNIST formatına dönüştürüldü.")
    return processed_paths


def create_sample_student_digits(student_id="210102045", output_dir="data/custom_digits/raw"):
    """
    Kullanıcı henüz kendi fotoğraflarını çekip yüklemediyse, projenin baştan sona
    eksiksiz çalışabilmesi ve test edilebilmesi için öğrenci numarası rakamlarından
    örnek el yazısı benzeri ham görseller üretir.
    
    Öğrenci bu görselleri inceleyebilir veya kendi kağıda yazdığı fotoğraflarla
    aynı isim formatında değiştirebilir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Öğrenci No ({student_id}) için örnek el yazısı görselleri oluşturuluyor...")
    
    created_files = []
    for idx, char in enumerate(str(student_id)):
        digit_val = int(char)
        # 120x120 beyaz kağıt üzerine siyah çizim simülasyonu
        img = Image.new('L', (120, 120), color=250)
        draw = ImageDraw.Draw(img)

        # Doğal el yazısı varyasyonu için hafif rastgele ofsetler ve eğim
        np.random.seed(42 + idx * 7)
        offset_x = int(np.random.randint(-6, 7))
        offset_y = int(np.random.randint(-6, 7))
        
        # Basit çizgi / fırça darbesi simülasyonu
        # Not: Varsayılan bitmap font veya el yazısı konturu
        try:
            # Sistemdeki Arial veya Tahoma fontunu el yazısı benzeri dene
            font = ImageFont.truetype("arial.ttf", 75)
        except Exception:
            font = ImageFont.load_default()

        # Rakamı merkeze yaz
        draw.text((35 + offset_x, 15 + offset_y), str(digit_val), fill=10, font=font)

        filename = f"digit_{idx}_val{digit_val}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        created_files.append(filepath)

    print(f"[Başarılı] {len(created_files)} adet ham görsel '{output_dir}' dizinine kaydedildi.")
    return created_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="El yazısı rakam ön işleme betiği")
    parser.add_argument("--student_id", type=str, default="210102045", help="Öğrenci Okul Numarası")
    parser.add_argument("--generate_sample", action="store_true", help="Örnek el yazısı görselleri üret")
    parser.add_argument("--process", action="store_true", help="Ham görselleri 28x28 MNIST formatına çevir")
    args = parser.parse_args()

    if args.generate_sample:
        create_sample_student_digits(args.student_id)
    if args.process:
        preprocess_all_custom_images()
