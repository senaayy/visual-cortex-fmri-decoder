# 🧠 Visual Cortex fMRI Decoder

Görsel korteks fMRI (BOLD) sinyallerinden, bir kişinin o anda hangi görsel kategoriye baktığını tahmin eden bir beyin-sinyali dekoder projesi. Klasik SVM tabanlı bir taban çizgisiyle başlayıp, PyTorch ile yazılmış bir sinir ağına kadar uzanan iki farklı yaklaşım içerir.

Veri seti: [Haxby (2001)](https://nilearn.github.io/stable/modules/description/haxby.html) — ventral temporal (görsel) kortekste 8 farklı nesne kategorisine (yüz, kedi, ev, ayakkabı, sandalye vb.) verilen tepkileri içeren klasik fMRI benchmark'ı.

## 📌 Proje Ne Yapıyor?

1. Nilearn üzerinden Haxby fMRI veri setini indirir.
2. 4 boyutlu (X, Y, Z, Zaman) beyin taramasını, hemodinamik yanıt gecikmesini (2 TR kaydırma) hesaba katarak `rest` durumları çıkarılmış temiz bir zaman serisine dönüştürür.
3. `NiftiMasker` ile 4D hacmi, sadece görsel korteks vokselini içeren 2D bir özellik matrisine (zaman × voksel) indirger.
4. Bu matris üzerinde:
   - **Baseline:** Doğrusal SVM + 5 katlamalı çapraz doğrulama
   - **İleri model:** PyTorch ile eğitilen sinir ağı (train/test ayrımıyla)
5. Modelin öğrendiği ağırlıkları tekrar 3D beyin uzayına gömüp bir ısı haritası (`brain_weights_map.png`) olarak kaydeder.

## 📊 Sonuçlar

| Model | Doğruluk | Şans Seviyesi |
|---|---|---|
| Doğrusal SVM (5-fold CV) | değişken, `train_model.py` çıktısında raporlanır | %12.5 (1/8 sınıf) |
| PyTorch Sinir Ağı (gerçek test seti) | **~%66.5** | %12.5 (1/8 sınıf) |

8 sınıflı bir çoklu sınıflandırma problemi için şans seviyesinin oldukça üzerinde bir performans.

## 🗂️ Proje Yapısı

```
visual-cortex-fmri-decoder/
├── fetch_data.py           # Haxby veri setini indirir
├── inspect_brain.py        # İndirilen taramanın boyutlarını/özelliklerini inceler
├── train_model.py          # SVM baseline: eğitim, çapraz doğrulama, ısı haritası
├── Untitled33.ipynb        # PyTorch sinir ağı ile eğitim (Google Colab, GPU destekli)
├── src/
│   ├── data_loader.py      # NIfTI dosyalarını yükleyen fMRILoader sınıfı
│   └── spatial_analysis.py # 4D veriyi 2D voksel matrisine indirgeyen yardımcı fonksiyon
├── Dockerfile               # Bağımlılıkları izole bir ortamda kurmak için
└── requirements.txt
```

## 🛠️ Teknolojiler

- **Analiz / ML:** `nilearn`, `nibabel`, `scikit-learn`, `PyTorch`
- **Veri işleme:** `numpy`, `pandas`
- **Görselleştirme:** `matplotlib`, `nilearn.plotting`
- **Ortam:** `Docker`, Google Colab (CUDA/GPU)

## 🚀 Kurulum ve Çalıştırma

### Yerelde

```bash
git clone https://github.com/senaayy/visual-cortex-fmri-decoder.git
cd visual-cortex-fmri-decoder
pip install -r requirements.txt

# 1) Veri setini indir (~300 MB, tek denek)
python fetch_data.py

# 2) İndirilen taramayı incele (opsiyonel)
python inspect_brain.py

# 3) SVM baseline'ı eğit ve ısı haritasını üret
python train_model.py
```

### Docker ile

```bash
docker build -t fmri-decoder .
docker run -it fmri-decoder
```

### PyTorch sinir ağı (Google Colab, GPU önerilir)

`Untitled33.ipynb` dosyasını Colab'da açıp hücreleri sırayla çalıştırın. Not defteri CUDA'yı otomatik algılar; GPU yoksa CPU'ya düşer.

## 🔬 Yöntem Detayları

- **Hemodinamik gecikme düzeltmesi:** Kan oksijen seviyesi tepkisi uyarandan birkaç saniye sonra geldiği için etiketler 2 TR kaydırılır (`shift(2)`).
- **`rest` filtreleme:** Sadece aktif görsel uyaran anları modele veriliyor, dinlenme anları çıkarılıyor.
- **Uzaysal maskeleme:** `NiftiMasker`, ventral temporal bölgeyi izole edip sinyalleri standartlaştırıyor, böylece ~1500 civarı voksel özelliğe indirgeniyor.

## 📈 Yol Haritası

- [ ] Denekler arası genelleme (transfer learning) — farklı beyin anatomileri arasında ortak örüntülerin test edilmesi
- [ ] Öğrenilen ağırlıkların tam 3D NIfTI beyin haritasına dönüştürülüp interaktif görselleştirilmesi
- [ ] BIDS formatına tam uyumlu, çok denekli bir veri işleme hattı

## 📚 Referans

Haxby, J. V. et al. (2001). *Distributed and Overlapping Representations of Faces and Objects in Ventral Temporal Cortex.* Science.

---

Beyin-bilgisayar arayüzleri (BCI) ve nörobilim odaklı makine öğrenmesi üzerine araştırma amaçlı geliştirilmiştir.
