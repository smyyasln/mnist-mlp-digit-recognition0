"""
dataset.py - MNIST veri seti yükleme, Train/Validation/Test ayrımı ve Veri Yükleyicileri.

Kurallar:
- Train/Validation/Test ayrımı açıkça gösterilmiştir.
- Test seti eğitim veya model geliştirme sürecinde kesinlikle kullanılmaz.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from PIL import Image


def get_mnist_transforms():
    """
    MNIST veri seti için standart dönüşümleri döner.
    Görüntüler [0, 1] aralığına çekilir ve MNIST ortalama/standart sapması ile normalize edilir.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform


def get_data_loaders(data_dir="./data", batch_size=64, val_size=10000, seed=42):
    """
    MNIST veri kümesini indirir ve Train / Validation / Test veri yükleyicilerini oluşturur.
    
    Toplam MNIST:
      - Standart Train: 60,000 örnek -> 50,000 Train + 10,000 Validation olarak ayrılır.
      - Standart Test: 10,000 örnek (Yalnızca nihai test için saklanır).
      
    Args:
        data_dir (str): Verinin indirileceği dizin.
        batch_size (int): Mini-batch boyutu.
        val_size (int): Validation kümesine ayrılacak örnek sayısı.
        seed (int): Tekrarlanabilirlik için rastgelelik tohumu.
        
    Returns:
        train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)
    """
    transform = get_mnist_transforms()

    # 1. 60,000 örnekten oluşan ham eğitim verisi indirilir
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # 2. 10,000 örnekten oluşan test verisi indirilir (Geliştirmede kullanılmaz!)
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # 3. Train ve Validation ayrımı (50,000 Train, 10,000 Validation)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=generator
    )

    # 4. DataLoader nesneleri
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[Veri Seti Bilgisi]")
    print(f" - Toplam Eğitim Kümesi (Train):      {len(train_dataset)} örnek")
    print(f" - Doğrulama Kümesi (Validation):     {len(val_dataset)} örnek")
    print(f" - Test Kümesi (Test - Dokunulmaz):    {len(test_dataset)} örnek")
    print(f" - Sınıf Sayısı:                       10 (0 - 9)")
    print(f" - Görüntü Boyutu:                     28x28 gri seviye (1 kanal)")

    return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)


class CustomDigitDataset(Dataset):
    """
    Öğrencinin kendi el yazısıyla yazdığı rakam görsellerini yükleyen veri sınıfı.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else get_mnist_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Gri seviye
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path
