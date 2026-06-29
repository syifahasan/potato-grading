import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CALIB_PATH, BEST_MODEL, TEST_DIR, OUTPUT_DIR
from src.measurer import load_calibration, measure_from_mask
from src.grader import get_grade

# ============================================================
# KONFIGURASI — sesuaikan bagian ini
# ============================================================
CONF       = 0.35
IMAGE_PATH = os.path.join(TEST_DIR, "POR-254001_JPG.rf.94d387ff4edce3d1e522ba972059fec3.jpg")
# ============================================================


def draw_results(
    img: np.ndarray,
    box_points: tuple,
    length_mm: float,
    width_mm: float,
    grade: str,
    idx: int,
) -> None:
    points = cv2.boxPoints(box_points)
    points = np.int32(points)

    cv2.drawContours(img, [points], 0, (0, 255, 0), 2)

    text_x = int(points[:, 0].min())
    text_y = int(points[:, 1].min()) - 10

    label = f"#{idx} {length_mm:.1f}mm x {width_mm:.1f}mm - {grade}"
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == "__main__":
    pixel_per_mm = load_calibration(CALIB_PATH)
    print(f"Calibration loaded: {pixel_per_mm:.4f} pixels/mm")

    model = YOLO(BEST_MODEL)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Tidak bisa membaca gambar: {IMAGE_PATH}")

    results = model.predict(source=IMAGE_PATH, conf=CONF, verbose=False)

    result = results[0]
    img_output = img.copy()
    potato_count = 0

    if result.masks is not None:
        for idx, mask_xy in enumerate(result.masks.xy):
            mask_array = np.array(mask_xy, dtype=np.float32)

            if len(mask_array) < 5:
                print(f"  Kentang #{idx+1}: mask terlalu kecil, dilewati")
                continue

            measurement = measure_from_mask(mask_array, pixel_per_mm)
            length_mm = measurement["length_mm"]
            width_mm  = measurement["width_mm"]
            rect      = measurement["rect"]
            grade     = get_grade(length_mm, width_mm)

            potato_count += 1
            print(f"  Kentang #{potato_count}: {length_mm:.1f}mm x {width_mm:.1f}mm - {grade}")
            draw_results(img_output, rect, length_mm, width_mm, grade, potato_count)
    else:
        print("Tidak ada kentang terdeteksi. Coba turunkan conf")

    print(f"\nTotal kentang terdeteksi: {potato_count}")

    output_path = os.path.join(OUTPUT_DIR, "measurement_result.jpg")
    cv2.imwrite(output_path, img_output)
    print(f"Gambar hasil disimpan di: {output_path}")

    cv2.imshow("Measurement Result", img_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
