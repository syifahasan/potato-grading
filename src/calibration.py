import cv2
import json
import os
import math

# ======================================================
# INSTRUKSI PENGGUNAAN:
# 1. Jalankan script ini: python calibrate.py
# 2. Akan muncul jendela gambar
# 3. Klik titik AWAL ruler (misal: tanda 0 cm)
# 4. Klik titik AKHIR ruler (misal: tanda 10 cm)
# 5. Program otomatis hitung dan simpan calibration.json
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- KONFIGURASI : sesuaikan dua baris ini ---
IMAGE_PATH = os.path.join(BASE_DIR, "..", "datasets", "test", "images", "label-meteran_JPG.rf.7d585e9df1fbf9d2bca77e8f459956de.jpg")  # Ganti dengan path gambar kalibrasi Anda
KNOWN_DISTANCE_MM = 100.0  # Ganti dengan jarak yang diketahui pada gambar (misal: 10 cm = 100 mm)


clicks = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(img_display, (x, y), 6, (0, 255, 0), -1)
        cv2.imshow("Calibration", img_display)

        if len(clicks) == 1:
            print(f"Titik pertama: {clicks[0]} - sekarang klik titik kedua")

        if len(clicks) == 2:
            dx = clicks[1][0] - clicks[0][0]
            dy = clicks[1][1] - clicks[0][1]
            pixel_distance = math.sqrt(dx**2 + dy**2)

            pixel_per_mm = pixel_distance / KNOWN_DISTANCE_MM

            print(f"\nJarak pixel: {pixel_distance:.2f} px")
            print(f"Jarak nyata: {KNOWN_DISTANCE_MM:.2f} mm")
            print(f"Pixel per mm: {pixel_per_mm:.4f} px/mm")

            calib_path = os.path.join(BASE_DIR, "calibration.json")

            calib_data = {
                "pixel_per_mm": pixel_per_mm,
                "known_distance_mm": KNOWN_DISTANCE_MM,
                "pixel_distance": pixel_distance,
                "point1": clicks[0],
                "point2": clicks[1],
                "image_used": IMAGE_PATH
            }

            with open(calib_path, "w") as f:
                json.dump(calib_data, f, indent=4)
            
            print(f"\nKalibrasi berhasil disimpan di {calib_path}")
            print("Tekan tombol apapun untuk keluar.")

img_original = cv2.imread(IMAGE_PATH)
img_display = img_original.copy()

cv2.imshow("Calibration", img_display)
cv2.setMouseCallback("Calibration", on_click)

print("Jendela kalibrasi terbuka. Klik dua titik pada gambar untuk kalibrasi.")
print(f"Klik titik pertama di ruler ({KNOWN_DISTANCE_MM} mm = jarak yang akan kamu klik.)")

cv2.waitKey(0)
cv2.destroyAllWindows()