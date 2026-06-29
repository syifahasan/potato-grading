import os
import cv2
import json
import numpy as np

def load_calibration(calib_path: str) -> float:
    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"File kalibrasi tidak ditemukan di {calib_path}\n"
            f"Jalankan calibration.py terlebih dahulu."
        )
    
    with open(calib_path, "r") as f:
        calib = json.load(f)

    # validasi key yang dibutuhkan 
    if "pixel_per_mm" not in calib:
        raise KeyError(
            f"Key 'pixel_per_mm' tidak ditemukan di {calib_path}. "
            f"File kalibrasi mungkin korup atau formatnya salah."
        )
    
    pixel_per_mm = calib["pixel_per_mm"]
    
    # validasi nilai masuk akal (tidak nol atau negatif)
    if pixel_per_mm <= 0:
        raise ValueError(
            f"Nilai pixel_per_mm tidak valid: {pixel_per_mm}. "
            f"Lakukan kalibrasi ulang."
        )

    return pixel_per_mm

def measure_from_mask(mask_xy: np.ndarray, pixel_per_mm: float) -> dict:
    mask_array = np.array(mask_xy, dtype=np.float32)

    rect = cv2.minAreaRect(mask_array)

    width_px = rect[1][0]
    height_px = rect[1][1]

    # konversi ke mm
    length_mm = max(width_px, height_px) / pixel_per_mm
    width_mm = min(width_px, height_px) / pixel_per_mm

    # hitung luas perkiraan
    area_mm2 = length_mm * width_mm

    return {
        "length_mm" : round(length_mm, 1),
        "width_mm" : round(width_mm, 1),
        "area_mm2" : round(area_mm2, 1),
        "rect" : rect,
    }

def is_mask_valid(mask_xy: np.ndarray, min_points: int = 5, min_area_px: float = 500.0) -> tuple[bool, str]:
    # cek jumlah titik
    if len(mask_xy) < min_points:
        return False, f"titik mask terlalu sedikit ({len(mask_xy)} < {min_points})"

    # hitung area polygon dengan Shoelace formula via openCv
    area = cv2.contourArea(np.array(mask_xy, dtype=np.float32))
    if area < min_area_px:
        return False, f"area mask terlalu kecil ({area:.0f} < {min_area_px} px²)"

    return True, ""