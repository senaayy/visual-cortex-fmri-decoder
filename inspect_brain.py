from nilearn import datasets
from src.data_loader import fMRILoader

print("🔍 İndirilen beyin taraması analiz ediliyor...")

# Verinin yerini bul
haxby_dataset = datasets.fetch_haxby(data_dir='./data', subjects=[1])
fmri_file_path = haxby_dataset.func[0]

# Hazırladığımız Data Loader motorunu çalıştır
# (Nilearn bize dosyanın tam yolunu verdiği için data_dir parametresini boş bırakıyoruz)
loader = fMRILoader(data_dir="")

# 4 Boyutlu fMRI verisini RAM'e yükle
img = loader.load_nifti(fmri_file_path)

# Verinin anatomik ve zamansal boyutlarını ekrana yazdır
x, y, z, time_points = img.shape

print("\n✅ Veri başarıyla sisteme yüklendi!")
print("-" * 50)
print(f"🧠 Matris Boyutu (X, Y, Z, Zaman): {img.shape}")
print(f"📏 X Ekseni (Sağ-Sol): {x} voksel")
print(f"📏 Y Ekseni (Ön-Arka): {y} voksel")
print(f"📏 Z Ekseni (Alt-Üst): {z} voksel")
print(f"⏱️ Zaman Çizelgesi: Görsel korteksin {time_points} farklı andaki tepkisi kaydedilmiş.")
print(f"🧊 Tek bir andaki toplam 3D Piksel (Voksel) sayısı: {x * y * z}")
print("-" * 50)