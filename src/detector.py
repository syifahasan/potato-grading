import logging
import os
import sys
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import BEST_MODEL, CALIB_PATH
from src.measurer import load_calibration, measure_from_mask, is_mask_valid
from src.grader import get_grade

logger = logging.getLogger(__name__)


class PotatoDetector:
    def __init__(
        self,
        model_path: str = BEST_MODEL,
        calib_path: str = CALIB_PATH,
        conf: float = 0.35
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}\n"
                f"Pastikan best.pt ada di folder models/weights/"
            )

        logger.info("Loading model dari: %s", model_path)
        self.model = YOLO(model_path)

        logger.info("Loading kalibrasi dari: %s", calib_path)
        self.pixel_per_mm = load_calibration(calib_path)
        logger.info("pixel_per_mm = %.4f", self.pixel_per_mm)

        self.conf = conf

    def run(self, image_path: str, conf: float | None = None) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

        effective_conf = conf if conf is not None else self.conf

        raw_results = self.model.predict(
            source=image_path,
            conf=effective_conf,
            verbose=False,
        )

        result = raw_results[0]
        detections = []
        skipped = 0

        if result.masks is None:
            logger.info("Tidak ada kentang terdeteksi.")
            return {"detections": [], "total": 0, "image_path": image_path}

        conf_scores = result.boxes.conf.cpu().numpy()

        for idx, mask_xy in enumerate(result.masks.xy):
            mask_array = np.array(mask_xy, dtype=np.float32)

            valid, reason = is_mask_valid(mask_array)
            if not valid:
                logger.debug("mask #%d dilewati: %s", idx + 1, reason)
                skipped += 1
                continue

            measurement = measure_from_mask(mask_array, self.pixel_per_mm)
            grade = get_grade(measurement["length_mm"], measurement["width_mm"])

            detection = {
                "id"        : len(detections) + 1,
                "length_mm" : measurement["length_mm"],
                "width_mm"  : measurement["width_mm"],
                "area_mm2"  : measurement["area_mm2"],
                "grade"     : grade,
                "rect"      : measurement["rect"],
                "conf"      : round(float(conf_scores[idx]), 3),
            }

            detections.append(detection)

        logger.info("Selesai: %d kentang, %d dilewati", len(detections), skipped)

        return {
            "detections" : detections,
            "total"      : len(detections),
            "image_path" : image_path,
        }
