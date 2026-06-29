import cv2
import json
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(BASE_DIR, "..", "datasets", "sizing_calib", "SET 2.jpeg")
KNOWN_DISTANCE_MM = 20.0

# --- Ukuran maksimal jendela tampilan ---
# Sesuaikan dengan resolusi monitor kamu
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720

clicks_display = []   # koordinat di gambar yang ditampilkan (sudah di-resize)
clicks_original = []  # koordinat di gambar asli (yang akan disimpan)

def resize_for_display(img):
    """
    Resize gambar supaya muat di layar, tapi pertahankan aspect ratio.
    Kembalikan gambar yang sudah diresize + scale factor yang dipakai.
    """
    h, w = img.shape[:2]
    
    # hitung scale factor — ambil yang paling kecil supaya pasti muat
    scale_w = DISPLAY_MAX_WIDTH / w
    scale_h = DISPLAY_MAX_HEIGHT / h
    scale = min(scale_w, scale_h)
    
    # kalau gambar sudah lebih kecil dari batas, tidak perlu resize
    if scale >= 1.0:
        return img.copy(), 1.0
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized, scale

def on_click(event, x, y, flags, param):
    """
    x, y di sini adalah koordinat di gambar yang SUDAH diresize (display).
    Kita perlu konversi balik ke koordinat gambar asli sebelum disimpan.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # simpan koordinat display untuk gambar lingkaran
        clicks_display.append((x, y))
        
        # konversi ke koordinat asli dengan membagi scale factor
        x_original = int(x / display_scale)
        y_original = int(y / display_scale)
        clicks_original.append((x_original, y_original))
        
        # gambar lingkaran di posisi klik (di gambar display)
        cv2.circle(img_display, (x, y), 6, (0, 255, 0), -1)
        cv2.imshow("Calibration", img_display)

        if len(clicks_display) == 1:
            print(f"Titik pertama (display): {clicks_display[0]}")
            print(f"Titik pertama (asli)   : {clicks_original[0]}")
            print("Sekarang klik titik kedua...")

        if len(clicks_display) == 2:
            # hitung jarak pixel di gambar ASLI (bukan display)
            dx = clicks_original[1][0] - clicks_original[0][0]
            dy = clicks_original[1][1] - clicks_original[0][1]
            pixel_distance = math.sqrt(dx**2 + dy**2)

            pixel_per_mm = pixel_distance / KNOWN_DISTANCE_MM

            print(f"\nJarak pixel (asli) : {pixel_distance:.2f} px")
            print(f"Jarak nyata        : {KNOWN_DISTANCE_MM:.2f} mm")
            print(f"Pixel per mm       : {pixel_per_mm:.4f} px/mm")

            # ambil dimensi gambar asli untuk disimpan
            h_orig, w_orig = img_original.shape[:2]

            calib_path = os.path.join(BASE_DIR, "calibration.json")
            calib_data = {
                "pixel_per_mm": pixel_per_mm,
                "known_distance_mm": KNOWN_DISTANCE_MM,
                "pixel_distance": pixel_distance,
                "point1": clicks_original[0],
                "point2": clicks_original[1],
                "image_used": IMAGE_PATH,
                # simpan resolusi gambar kalibrasi — penting untuk scaling nanti
                "calibration_image_width": w_orig,
                "calibration_image_height": h_orig,
            }

            with open(calib_path, "w") as f:
                json.dump(calib_data, f, indent=4)
            
            print(f"\nKalibrasi berhasil disimpan di {calib_path}")
            print("Tekan tombol apapun untuk keluar.")

# --- Load gambar asli ---
img_original = cv2.imread(IMAGE_PATH)

if img_original is None:
    print(f"ERROR: Gambar tidak ditemukan di {IMAGE_PATH}")
    exit()

# --- Resize untuk display ---
img_display, display_scale = resize_for_display(img_original)

print(f"Resolusi asli   : {img_original.shape[1]}x{img_original.shape[0]}px")
print(f"Resolusi display: {img_display.shape[1]}x{img_display.shape[0]}px")
print(f"Scale factor    : {display_scale:.4f}")
print()
print("Klik titik pertama pada ruler...")

cv2.imshow("Calibration", img_display)
cv2.setMouseCallback("Calibration", on_click)
cv2.waitKey(0)
cv2.destroyAllWindows()