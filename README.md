# Potato Grading System

Sistem otomatis untuk mendeteksi, mengukur, dan mengklasifikasikan kentang berdasarkan ukuran menggunakan computer vision (YOLOv8 instance segmentation).

---

## Fitur

- Deteksi kentang secara otomatis dari foto menggunakan model YOLO26 segmentation
- Pengukuran dimensi (panjang × lebar) dalam satuan milimeter menggunakan kalibrasi kamera
- Klasifikasi grade A–D berdasarkan panjang terpanjang kentang
- Antarmuka web interaktif (Streamlit) untuk upload foto dan melihat hasil
- Export hasil ke gambar anotasi (PNG) dan tabel pengukuran (CSV)
- Logging otomatis hasil per sesi ke file CSV harian

---

## Standar Grading

| Grade | Panjang Terpanjang |
| ----- | ------------------ |
| A     |                    |
| B     |                    |
| C     |                    |
| D     |                    |

Threshold dapat diubah di `src/grader.py` → `GRADE_THRESHOLDS`.

---

## Persyaratan Sistem

- Python 3.9+
- GPU (opsional, tapi direkomendasikan untuk inferensi lebih cepat)
- CUDA 11.8+ jika menggunakan GPU NVIDIA

---

## Instalasi

```bash
git clone https://github.com/syifahasan/potato-grading.git
cd potato-grading
pip install -r requirements.txt
```

---

## Setup Awal

### 1. Download Model

Download `best.pt` dari [Releases](https://github.com/syifahasan/potato-grading/releases/tag/model), lalu letakkan di:

```
models/weights/best.pt
```

### 2. Kalibrasi Kamera

Kalibrasi wajib dilakukan sekali untuk mengkonversi pixel ke milimeter. Hasilnya disimpan di `src/calibration.json` dan dipakai oleh semua modul pengukuran.

1. Siapkan foto yang berisi penggaris atau objek dengan panjang yang diketahui
2. Edit dua konstanta di bagian atas `src/calibration.py`:
   ```python
   IMAGE_PATH = "path/ke/foto/kalibrasi.jpg"
   KNOWN_DISTANCE_MM = 100.0  # panjang referensi dalam mm
   ```
3. Jalankan:
   ```bash
   python src/calibration.py
   ```
4. Klik dua titik pada penggaris di jendela yang muncul — program otomatis menghitung dan menyimpan `src/calibration.json`

---

## Cara Penggunaan

### Aplikasi Web (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Buka browser di `http://localhost:8501`, upload foto kentang, dan hasil deteksi + grading akan langsung muncul.

Fitur di aplikasi:

- Slider confidence threshold di sidebar
- Gambar anotasi dengan bounding box berwarna per grade
- Ringkasan jumlah kentang per grade
- Tabel detail pengukuran per kentang
- Tombol download gambar hasil dan CSV
- Tombol simpan ke log harian

### Script Pengukuran (CLI)

Untuk testing cepat tanpa antarmuka web:

```bash
python src/measurement.py
```

Edit `IMAGE_PATH` di bagian atas file sebelum menjalankan. Hasil disimpan ke `outputs/measurement_result.jpg`.

---

## Pelatihan Model (Opsional)

Dataset menggunakan format YOLO segmentation dari Roboflow. Untuk melatih ulang:

```bash
yolo segment train data=datasets/data.yaml model=yolo26n-seg.pt epochs=100 imgsz=640 batch=8
```

Validasi model hasil training:

```bash
yolo segment val data=datasets/data.yaml model=runs/segment/train-5/weights/best.pt
```

> `datasets/data.yaml` menggunakan path relatif, sehingga perintah YOLO harus dijalankan dari root repositori.

---

## Dataset

- **Sumber:** Roboflow (workspace: `potatoyolo`, project: `potato-lqffs-0ka4b`, CC BY 4.0)
- **Jumlah gambar:** ~1.400 asli, 3× augmentasi → 3.585 training images
- **Split:** ~81% train / 11% valid / 8% test
- **Kelas:** 1 kelas (`Potato`, index 0)
- **Format label:** YOLO segmentation polygon (koordinat dinormalisasi [0, 1])

Untuk mendapatkan dataset:

1. Buat akun di [Roboflow](https://roboflow.com)
2. Minta akses ke workspace `potatoyolo` (hubungi pemilik repo)
3. Export dalam format **YOLOv8 Segmentation** dan letakkan di folder `datasets/`

---

## Struktur Proyek

```
potato-grading/
├── app/
│   └── streamlit_app.py      # Antarmuka web Streamlit
├── datasets/
│   ├── data.yaml             # Konfigurasi dataset YOLO
│   ├── train/
│   ├── valid/
│   └── test/
├── models/
│   └── weights/
│       └── best.pt           # Model terlatih (download dari Releases)
├── outputs/                  # Gambar hasil & log CSV (auto-generated)
├── src/
│   ├── calibration.py        # Tool kalibrasi pixel-per-mm (interaktif)
│   ├── detector.py           # PotatoDetector — orkestrasi pipeline utama
│   ├── grader.py             # Logika grading & threshold ukuran
│   ├── logger.py             # Logging hasil sesi ke CSV harian
│   ├── measurer.py           # Pengukuran dimensi dari mask segmentasi
│   ├── measurement.py        # Script standalone untuk testing
│   └── utils.py              # Helper drawing & konversi gambar
├── config.py                 # Konfigurasi path terpusat
├── requirements.txt
└── CLAUDE.md
```
