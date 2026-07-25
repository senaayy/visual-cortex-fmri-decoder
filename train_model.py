import pandas as pd
import numpy as np
from nilearn import datasets
from nilearn.image import index_img
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

print("🧠 fMRI verisi ve deney etiketleri yükleniyor...")
# Veri yollarını otomatik al
haxby_dataset = datasets.fetch_haxby(data_dir='./data', subjects=[1])
fmri_filename = haxby_dataset.func[0]
mask_filename = haxby_dataset.mask_vt[0] # Yalnızca görsel korteksi (Ventral Temporal) izole eden anatomik maske

behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")

# 5 saniyelik biyolojik kan akışı gecikmesini koda öğretiyoruz
conditions = behavioral['labels'].shift(2).bfill()

# Artık sadece kedi/yüz yok. Hasta "dinleniyorken (rest)" hariç TÜM nesneleri yakalıyoruz!
condition_mask = conditions != 'rest'

y = conditions[condition_mask].values
X_raw = index_img(fmri_filename, condition_mask)

print(f"🎭 Uzaysal maskeleme başlatılıyor... (4D Hacim -> 2D Matris dönüşümü)")
# Masker objesi: Hem görsel korteksi filtreler hem de sinyalleri standartlaştırır
masker = NiftiMasker(mask_img=mask_filename, standardize=True)

# 4D veriyi 2D matrise çevir (rest hariç tüm nesne zaman dilimleri)
X = masker.fit_transform(X_raw)

print(f"📊 Veri mühendisliği tamamlandı! Makine öğrenmesine hazır yeni matris boyutu: {X.shape}")
print(f"(Bu, {X.shape[0]} farklı zaman anında, {X.shape[1]} adet vokselli görsel korteks alanından alınan veridir.)")

print("\n🤖 Destek Vektör Makinesi (Linear SVM) eğitiliyor...")
svc = SVC(kernel='linear')

# Modeli 5 Katlamalı Çapraz Doğrulama (Cross-Validation) ile test et
cv_scores = cross_val_score(svc, X, y, cv=5)
mean_score = np.mean(cv_scores) * 100

print("-" * 50)
print(f"🎯 Model Doğruluğu (8 Sınıflı Çoklu Tahmin): %{mean_score:.2f}")
print("-" * 50)
from nilearn.plotting import plot_stat_map

print("\n🎨 Modelin nöron ağırlıkları (öğrendiği harita) 3D formata geri çevriliyor...")
# SVM'in katsayılarını (karar verirken hangi piksellerin daha önemli olduğunu) al
svc.fit(X, y)
coef_ = svc.coef_

# 1D ağırlık dizesini, masker kullanarak tekrar 3 boyutlu beyin hacmine yerleştir
weight_img = masker.inverse_transform(coef_[0])

print("📸 Isı haritası (Heatmap) çiziliyor...")
# Arka plana hastanın kendi anatomik beyin taramasını (anat) koyarak ağırlıkları üzerine renklendir
plot_stat_map(
    weight_img, 
    bg_img=haxby_dataset.anat[0], 
    title="Yuz vs Kedi Tahmin Haritasi",
    display_mode='z', # Yukarıdan aşağıya (Z ekseni) dilimler halinde göster
    cut_coords=5,     # 5 farklı derinlik kesiti al
    output_file="brain_weights_map.png"
)

print("✅ 'brain_weights_map.png' dosyası ana dizine kaydedildi!")