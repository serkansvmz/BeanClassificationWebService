# 🌱 Dry Bean Classifier Web Servisi

Bu proje, 16 farklı morfolojik özelliği kullanarak kuru fasulye türlerini (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) sınıflandıran, PyTorch tabanlı bir derin öğrenme modelinin FastAPI ile yayına alınmış halidir.

## 📋 Özellikler

- **ANN Model Architecture**: 16 giriş parametresi ve 6 gizli katmanlı (64 nöronlu) Yapay Sinir Ağı.
- **FastAPI Backend**: Verimli, asenkron ve ölçeklenebilir API yapısı.
- **Dynamic Frontend**: `Jinja2` şablon motoru ve modern CSS/JS ile güçlendirilmiş kullanıcı arayüzü.
- **Automated Scaling**: Model tahmini öncesi `StandardScaler` parametreleri ile otomatik veri ön işleme.
- **Comprehensive Debugging**: `/debug/predict` endpoint'i üzerinden detaylı olasılık analizi.

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

## 2. Web Servisini Başlatın

```bash
uvicorn app.main:app --reload
```

Tarayıcınızdan `http://localhost:8000` adresine giderek arayüze erişebilirsiniz.

## 📁 Proje Yapısı

```
DryBeanClassifier/
├── app/
    ├── main.py            # API yönlendirmeleri ve uygulama ana giriş noktası
    ├── model.py           # PyTorch model tanımı, yükleme ve tahmin mantığı
    └── schemas.py         # Pydantic veri doğrulama modelleri
├── models/
    ├── bean_classifier.pth # Eğitilmiş PyTorch model dosyası
    ├── class_names.json   # Sınıf etiketleri listesi
    └── scaler.json        # Ortalama ve standart sapma değerleri (Z-score)
├── static/
    ├── app.js             # Frontend API etkileşimi ve DOM yönetimi
    └── styles.css         # Modern, karanlık tema destekli arayüz stilleri
├── templates/
    └── index.html         # Jinja2 tabanlı ana sayfa şablonu
├── Dry_Bean_Classification.ipynb  # Model eğitim ve analiz defteri
├── LICENSE                # Proje lisans bilgileri
├── requirements.txt       # Gerekli Python kütüphaneleri
└── README.md              # Proje dokümantasyonu
```

## 🔧 Model Mimarisi

Kuru fasulye sınıflandırma modeli, yüksek boyutlu verileri işleyebilmek için derin bir sinir ağı mimarisi kullanır:

```python
class BeanClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(16, 64),   # Giriş Katmanı: 16 Morfolojik Özellik
            nn.ReLU(),
            nn.Linear(64, 64),   # Gizli Katman 1
            nn.ReLU(),
            nn.Linear(64, 64),   # Gizli Katman 2
            nn.ReLU(),
            nn.Linear(64, 64),   # Gizli Katman 3
            nn.ReLU(),
            nn.Linear(64, 64),   # Gizli Katman 4
            nn.ReLU(),
            nn.Linear(64, 64),   # Gizli Katman 5
            nn.ReLU(),
            nn.Linear(64, 7),    # Çıkış Katmanı: 7 Farklı Fasulye Türü
        )
```
**Girdi Özellikleri (16 Parametre):**

Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

**Çıktı Sınıfları:**

BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA.

## 📡 API Kullanımı

### Health Check

```bash
curl http://localhost:8000/health
```

Yanıt:
```json
{
    "status": "ok",
    "model_loaded": true,
    "model_path": "models/bean_classifier.pth",
    "class_names": ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"],
    "scaler_loaded": true
}
```

### Tahmin Yapma

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
          "features": [28395, 610.291, 208.178, 173.888, 1.197, 0.549, 28715, 190.141, 0.763, 0.988, 0.958, 0.913, 0.007, 0.003, 0.834, 0.998]
         }'
```

Yanıt:
```json
{
{
    "predicted_class": "SEKER",
    "probabilities": [0.01, 0.00, 0.01, 0.02, 0.01, 0.92, 0.03]
}
}
```

## 🎨 Kullanıcı Arayüzü

Web arayüzüne erişmek için tarayıcınızda şu adresi açın:

```
http://localhost:8000
```

Arayüz özellikleri:
- ✨ Modern ve responsive tasarım
- 📊 Gerçek zamanlı tahmin sonuçları
- 📈 Olasılık dağılımı görselleştirmesi
- ⚡ Hızlı ve kullanıcı dostu

## 🧪 Model Performansı

Modelin eğitim sürecindeki başarı metrikleri (Dry Bean Dataset üzerinde):

```
Epoch:    0 | Loss: 1.9459 | Acc:  14.28% | Test Loss: 1.9320 | Test Acc:  18.50%
Epoch:   50 | Loss: 0.2841 | Acc:  91.15% | Test Loss: 0.2512 | Test Acc:  92.33%
Epoch:  100 | Loss: 0.2104 | Acc:  93.45% | Test Loss: 0.1985 | Test Acc:  93.80%
Epoch:  150 | Loss: 0.1852 | Acc:  94.10% | Test Loss: 0.1745 | Test Acc:  94.50%

✅ Eğitim Başarıyla Tamamlandı!
   Final Doğruluk (Accuracy): %94.50
```

## 📚 Ek Kaynaklar

- [PyTorch Dokümantasyonu](https://pytorch.org/docs/)
- [FastAPI Dokümantasyonu](https://fastapi.tiangolo.com/)
- [Dry Bean Dataset Classification](https://www.kaggle.com/datasets/nimapourmoradi/dry-bean-dataset-classification)


## 📄 Lisans

Apache License 2.0

