import cv2
import json
import os
import sys
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CALIB_PATH, MODEL_DIR, BEST_MODEL, TEST_DIR, OUTPUT_DIR

# ============================================================
# KONFIGURASI — sesuaikan bagian ini
# ============================================================
CONF        = 0.35
IMAGE_PATH = os.path.join(TEST_DIR, "POR-254001_JPG.rf.94d387ff4edce3d1e522ba972059fec3.jpg")  # Ganti dengan path gambar kalibrasi Anda
# ============================================================

def load_calibration(calib_path):
    with open(calib_path, "r") as f:
        calib = json.load(f)
    return calib['pixel_per_mm']

def get_grade(length_mm, width_mm):
    # Grading berdasarkan dimensi terpanjang dan terpendek
    # Sesuaikan bucket ini dengan standar grading kentang brother kamu
    major = max(length_mm, width_mm)
    minor = min(length_mm, width_mm)

    if major >= 80:
        return "Grade A"
    elif major >= 60:
        return "Grade B"
    elif major >= 40:
        return "Grade C"
    else:
        return "Grade D"
    
def measure_from_mask(mask_xy, pixel_per_mm):
    # mask_xy array koordinat polygon dari segmentation mask
    # cv2.minAreaRect mencari rectangle terkecil yang bisa membungkus polygon tersebut
    # hasilnya adalah rectangle yang bisa miring (bukan harus lurus)

    rect = cv2.minAreaRect(mask_xy)

    #rect berisi: (center_x, center_y), (width, height), angle
    # width dan height di sini dalam satuan pixel
    width_px = rect[1][0]
    height_px = rect[1][1]

    #konversi pixel ke mm menggunakan calibration factor
    length_mm = max(width_px, height_px) / pixel_per_mm
    width_mm = min(width_px, height_px) / pixel_per_mm

    return length_mm, width_mm, rect

def draw_results(img, box_points, length_mm, width_mm, grade, idx):
    #box points: 4 titik sudut dari minAreaRect, perlu dikonversi ke Integer
    points = cv2.boxPoints(box_points) if hasattr(cv2, 'boxPoints') else cv2.boxPoints(box_points)
    points = np.int32(points)

    #Gambar kotak hijau di sekitar kentang
    cv2.drawContours(img, [points], 0, (0, 255, 0), 2)

    #tentukan posisi teks (di atas kotak)
    text_x = int(points[:, 0].min())
    text_y = int(points[:, 1].min()) - 10

    label = f'#{idx} {length_mm:.1f}mm x {width_mm:.1f}mm - {grade}'
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == "__main__":
    #load calibration
    pixel_per_mm = load_calibration(CALIB_PATH)
    print(f'Calibration loaded: {pixel_per_mm:.4f} pixels/mm')

    # load model
    model = YOLO(BEST_MODEL)

    #jalankan inference
    img = cv2.imread(IMAGE_PATH)
    results = model.predict(source=IMAGE_PATH, conf=CONF, verbose=False)

    result = results[0]  #ambil hasil untuk satu gambar
    img_output = img.copy()

    potato_count = 0

    if result.masks is not None:
        for idx, mask_xy in enumerate(result.masks.xy):
            mask_array = np.array(mask_xy, dtype=np.float32)

            if len(mask_array) < 5:
                print(f"  Kentang #{idx+1}: mask terlalu kecil, dilewati")
                continue

            length_mm, width_mm, rect = measure_from_mask(mask_array, pixel_per_mm)
            grade = get_grade(length_mm, width_mm)

            potato_count += 1
            print(f"  Kentang #{potato_count}: {length_mm:.1f}mm x {width_mm:.1f}mm - {grade}")
            draw_results(img_output, rect, length_mm, width_mm, grade, potato_count)
    else:
        print("Tidak ada kentang terdeteksi. Coba turunkan conf")

    print(f"\n Total kentang terdeteksi: {potato_count}")

    #Simpan gambar hasil
    output_path = os.path.join(OUTPUT_DIR, "measurement_result.jpg")
    cv2.imwrite(output_path, img_output)
    print(f"Gambar hasil disimpan di: {output_path}")

    #tampilkan hasil
    cv2.imshow("Measurement Result", img_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
