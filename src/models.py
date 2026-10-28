"""
models.py - Multi-Layer Perceptron (MLP) Mimarileri

Önemli Kural:
- Bu ödevde KESİNLİKLE CNN kullanılmamıştır.
- Yalnızca Fully Connected (Linear) katmanlardan ve ReLU aktivasyonlarından oluşan MLP mimarileri kullanılmıştır.
- Modeller sıfırdan oluşturulur ve eğitilir (Pretrained model kullanılmaz).
"""

import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    """
    Deney 1 için Temel MLP Modeli:
      - Girdi Boyutu: 784 (28x28 piksel düzleştirilmiş vektör)
      - Hidden Layer 1: 512 nöron + ReLU
      - Hidden Layer 2: 256 nöron + ReLU
      - Çıktı Katmanı: 10 nöron (0-9 rakam sınıfları için ham logits)
      
    İsteğe bağlı olarak aşırı öğrenmeyi (overfitting) önlemek için Dropout eklenebilir.
    """
    def __init__(self, input_dim=784, hidden_dims=[512, 256], num_classes=10, dropout_rate=0.2):
        super(BaseMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # 2D veya 4D görüntüyü 1D vektöre düzleştir (Flatten): [B, 1, 28, 28] -> [B, 784]
        x = x.view(x.size(0), -1)
        return self.network(x)


class DeeperMLP(nn.Module):
    """
    Karşılaştırma Deneyi (Daha Derin MLP):
      - 3 Gizli Katman: 784 -> 512 -> 256 -> 128 -> 10
      - Aktivasyon: ReLU
    """
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], num_classes=10, dropout_rate=0.2):
        super(DeeperMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class SimpleMLP(nn.Module):
    """
    Karşılaştırma Deneyi (Daha Hafif / Sade MLP):
      - 2 Gizli Katman: 784 -> 128 -> 64 -> 10
      - Aktivasyon: ReLU
    """
    def __init__(self, input_dim=784, hidden_dims=[128, 64], num_classes=10, dropout_rate=0.0):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def get_model(model_name="base", dropout_rate=0.2):
    """Model fabrikası"""
    model_name = model_name.lower()
    if model_name in ["base", "deney1", "basemlp"]:
        return BaseMLP(hidden_dims=[512, 256], dropout_rate=dropout_rate)
    elif model_name in ["deeper", "deepermlp"]:
        return DeeperMLP(hidden_dims=[512, 256, 128], dropout_rate=dropout_rate)
    elif model_name in ["simple", "simplemlp"]:
        return SimpleMLP(hidden_dims=[128, 64], dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_name}")


def count_parameters(model):
    """Modelin toplam eğitilebilir parametre sayısını hesaplar."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
