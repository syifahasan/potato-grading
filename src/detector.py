import os
import sys
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import BEST_MODEL, CALIB_PATH
from src.measurer import load_calibration, measure_from_mask, is_mask_valid
from src.grader import get_grade

class PotatoDetector:
    def __init__(
        self,
        model_path: str = BEST_MODEL,
        calib_path: str = CALIB_PATH,
        conf: float = 0.35
    ):
        # Validasi file model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}\n"
                f"Pastikan best.pt ada di folder models/weights/"
            )
        
        print(f"[Detector] Loading model dari: {model_path}")
        self.model = YOLO(model_path)

        print(f"[Detector] Loading kalibrasi dari: {calib_path}")
        self.pixel_per_mm = load_calibration(calib_path)
        print(f"[Detector] pixel_per_mm = {self.pixel_per_mm:.4f}")

        self.conf = conf
    
    def run(self, image_path:str) -> dict:
        # validasi file gambar
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
        
        # jalankan inference - verbose=False supaya tidak banjir output dari terminal
        raw_results = self.model.predict(
            source=image_path,
            conf=self.conf,
            verbose=False,
        )

        result = raw_results[0]
        detections = []
        skipped = 0

        if result.masks is None:
            # model tidak menemukan kentang sama sekali
            print("[Detector] Tidak ada kentang terdeteksi.")
            return {"detections": [], "total": 0, "image_path": image_path}
        
        # ambil conf scores dari boxes
        conf_scores =  result.boxes.conf.cpu().numpy()

        for idx, mask_xy in enumerate(result.masks.xy):
            mask_array = np.array(mask_xy, dtype=np.float32)

            # validasi mask sebelum diukur
            valid, reason = is_mask_valid(mask_array)
            if not valid:
                print(f" [skip] mask #{idx+1} dilewati: {reason}")
                skipped += 1
                continue
        
            measurement = measure_from_mask(mask_array, self.pixel_per_mm)

            grade = get_grade(measurement["length_mm"], measurement["width_mm"])

            # collect semua info ke satu dict
            detection = {
                "id" : len(detections) + 1,
                "length_mm" : measurement["length_mm"],
                "width_mm" : measurement["width_mm"],
                "area_mm2" : measurement["area_mm2"],
                "grade" : grade,
                "rect" : measurement["rect"],
                "conf" : round(float(conf_scores[idx]), 3),
            }

            detections.append(detection)
        
        print(f"[Detector] Selesai: {len(detections)} kentang, {skipped} dilewati")

        return {
            "detections" : detections,
            "total" : len(detections),
            "image_path" : image_path,
        }