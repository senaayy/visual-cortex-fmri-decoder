from nilearn import datasets
import os

print("🧠 Görsel Korteks (Haxby) fMRI veri seti indiriliyor...")
# Sistemi yormamak için şimdilik sadece 1 numaralı deneğin verisini indiriyoruz (Yaklaşık 300 MB)
haxby_dataset = datasets.fetch_haxby(data_dir='./data', subjects=[1])

print("\n✅ İndirme tamamlandı!")
print(f"📁 4D fMRI Dosyası: {haxby_dataset.func[0]}")
print(f"📁 Deney Etiketleri (Görülen Nesneler): {haxby_dataset.session_target[0]}")