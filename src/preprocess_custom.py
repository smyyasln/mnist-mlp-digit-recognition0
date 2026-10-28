"""
preprocess_custom.py - Kendi El Yazısı Rakamlarını MNIST Formatına Dönüştürme Modülü

MNIST Veri Seti Standartları (Yann LeCun et al.):
1. Gri seviye (Grayscale) 28x28 piksel.
2. Arka plan siyah (0), rakam çizgileri beyaz/açık renk (255'e kadar).
3. Rakam, en-boy oranı korunarak 20x20 piksellik bir kutuya sığdırılır.
4. Rakamın kütle merkezi veya sınır kutusu 28x28 alanın tam ortasına yerleştirilir.
5. Normalizasyon: Ortalama 0.1307, Standart Sapma 0.3081.

Bu modül, cep telefonuyla çekilen kağıt üzerindeki el yazısı fotoğraflarındaki
gölgeleri, defter çizgilerini ve ışık dengesizliklerini temizleyip
MNIST standartlarına dönüştürür.
"""

import os
import glob
import re
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch


def get_natural_sort_key(filepath):
    """
    Dosya adındaki indeks numarasını sayısal olarak çıkarır (0, 1, 2, ..., 10).
    Böylece 'digit_10' 'digit_1'den önce gelmez.
    """
    filename = os.path.basename(filepath)
    m = re.search(r'digit_(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    digits = re.findall(r'\d+', filename)
    if digits:
        return int(digits[0])
    return 0


def preprocess_single_digit(image_input, target_size=(28, 28), digit_box_size=20):
    """
    Tek bir el yazısı rakam görselini MNIST formatına dönüştürür.
    
    Adımlar:
      1. Gri seviyeye çevirme (Grayscale).
      2. Arka plan aydınlatmasını çıkarma (BoxBlur ile gölge ve kağıt gradyanını yok etme).
      3. Mürekkep çizgilerini izole etme ve eşikleme (Thresholding).
      4. Rakam sınır kutusunu (bounding box) bulma.
      5. En-boy oranını koruyarak 20x20 kutuya sığdırma.
      6. 28x28 siyah tuvalin tam ortasına yerleştirme.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('L')
    else:
        img = image_input.convert('L')

    w, h = img.size

    # 1. Arka Plan Aydınlatmasını Çıkarma (Gölge ve Kağıt Gradyanını Yok Etme):
    # Geniş çaplı bir bulanıklaştırma ile kağıdın gölgeli arka planını tahmin et
    radius = max(w, h) // 25
    if radius < 3:
        radius = 3
    bg = img.filter(ImageFilter.BoxBlur(radius))

    # Kağıt açık renk, mürekkep koyu renktir: (bg - img) doğrudan mürekkep izini verir.
    # Bu işlem hem renkleri tersine çevirir (MNIST gibi siyah arka plan / beyaz çizgi yapar)
    # hem de ortamdaki gölgeleri ve çizgili defterin yumuşak tonlarını sıfırlar.
    diff = ImageChops.subtract(bg, img)
    arr = np.array(diff, dtype=np.float32)

    max_val = arr.max()
    if max_val > 0:
        # En koyu mürekkep darbelerini korumak için dinamik eşikleme (%45)
        thresh = max_val * 0.40
        clean_arr = np.where(arr > thresh, 255, 0).astype(np.uint8)
    else:
        clean_arr = arr.astype(np.uint8)

    clean_img = Image.fromarray(clean_arr)

    # 2. Sınır Kutusunu (Bounding Box) Bulma
    bbox = clean_img.getbbox()
    if bbox is None:
        return Image.new('L', target_size, 0)

    cropped = clean_img.crop(bbox)
    cw, ch = cropped.size

    # 3. En-boy oranını koruyarak 20x20 kutuya sığdırma
    if cw > ch:
        nw = digit_box_size
        nh = max(1, int(round((ch * digit_box_size) / cw)))
    else:
        nh = digit_box_size
        nw = max(1, int(round((cw * digit_box_size) / ch)))

    resized = cropped.resize((nw, nh), Image.Resampling.LANCZOS)

    # 4. 28x28 siyah tuval oluştur ve rakamı ortala
    final_canvas = Image.new('L', target_size, 0)
    paste_x = (target_size[0] - nw) // 2
    paste_y = (target_size[1] - nh) // 2
    final_canvas.paste(resized, (paste_x, paste_y))

    return final_canvas


def preprocess_all_custom_images(raw_dir="data/custom_digits/raw", 
                                 processed_dir="data/custom_digits/processed"):
    """
    raw_dir içindeki tüm fotoğrafları doğal sırayla işleyip processed_dir içine kaydeder.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Eski işlenmiş dosyaları temizle
    for old_file in glob.glob(os.path.join(processed_dir, "*.*")):
        try:
            os.remove(old_file)
        except Exception:
            pass

    extensions = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP")
    files = set()
    for ext in extensions:
        files.update(glob.glob(os.path.join(raw_dir, ext)))
    
    # Sayısal doğal sıralama (0, 1, 2, ..., 10)
    files = sorted(list(files), key=get_natural_sort_key)

    if not files:
        print(f"[Uyarı] '{raw_dir}' dizininde işlenecek görüntü bulunamadı.")
        return []

    processed_paths = []
    print(f"Toplam {len(files)} adet ham el yazısı görseli doğal sıralama ile işleniyor...")
    
    for fpath in files:
        fname = os.path.basename(fpath)
        processed_img = preprocess_single_digit(fpath)
        out_path = os.path.join(processed_dir, fname)
        out_path = os.path.splitext(out_path)[0] + '.png'
        processed_img.save(out_path)
        processed_paths.append(out_path)
        print(f" -> İşlendi ve kaydedildi: {out_path}")

    print("[Tamamlandı] Tüm el yazısı görüntüleri 28x28 MNIST formatına dönüştürüldü.")
    return processed_paths


def create_sample_student_digits(student_id="210102045", output_dir="data/custom_digits/raw"):
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    for idx, char in enumerate(str(student_id)):
        digit_val = int(char)
        img = Image.new('L', (120, 120), color=250)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 75)
        except Exception:
            font = ImageFont.load_default()

        draw.text((35, 15), str(digit_val), fill=10, font=font)
        filename = f"digit_{idx}_val{digit_val}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        created_files.append(filepath)

    return created_files


if __name__ == "__main__":
    preprocess_all_custom_images()
