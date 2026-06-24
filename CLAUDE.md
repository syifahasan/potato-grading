# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer vision project for **potato instance segmentation and grading** using YOLOv8 (Ultralytics). The pipeline detects potatoes in an image, measures their dimensions in mm using a calibrated pixel-per-mm ratio, and assigns grades A–D based on the longest dimension. A trained model (`best.pt`) is available from the [GitHub Releases page](https://github.com/syifahasan/potato-grading/releases/tag/model) and must be placed at `models/weights/best.pt`.

## Setup

```bash
pip install -r requirements.txt
```

The model weight (`best.pt`) must exist at `models/weights/best.pt` before running the app or detector. Calibration (`src/calibration.json`) must also exist — run the calibration tool first if it's missing.

## Running the Application

**Streamlit web UI (primary interface):**
```bash
streamlit run app/streamlit_app.py
```

**Calibration tool** (run once to generate `src/calibration.json`):
1. Edit `IMAGE_PATH` and `KNOWN_DISTANCE_MM` at the top of `src/calibration.py`
2. `python src/calibration.py` — click two points on the ruler in the window that opens

**Standalone measurement script** (prototype, useful for quick testing):
```bash
python src/measurement.py
```
Edit `IMAGE_PATH` at the top of the file before running.

## YOLO Commands

Train (from repo root, hyperparams used: batch=8, device=0):
```bash
yolo segment train data=datasets/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640 batch=8
```

Validate:
```bash
yolo segment val data=datasets/data.yaml model=runs/segment/train-5/weights/best.pt
```

Raw inference:
```bash
yolo segment predict model=runs/segment/train-5/weights/best.pt source=datasets/test/images/
```

`datasets/data.yaml` uses relative paths (`../train/images` etc.), so YOLO CLI commands must be run from the repo root with an absolute or relative path to `data.yaml` as shown.

## Architecture

All paths are centralized in `config.py` (repo root). Every `src/` module and the app insert `ROOT` into `sys.path` to import from `config`.

**Inference pipeline (layered):**

1. `app/streamlit_app.py` — web UI; uploads image, calls `PotatoDetector`, renders annotated image + grade summary table, triggers CSV logging.
2. `src/detector.py` — `PotatoDetector` class; loads YOLO model and calibration once, runs inference, calls `measurer` and `grader` per mask, returns a list of detection dicts.
3. `src/measurer.py` — pure measurement logic: `load_calibration()`, `measure_from_mask()` (uses `cv2.minAreaRect` to get oriented bounding box → px → mm), `is_mask_valid()` (filters degenerate masks).
4. `src/grader.py` — pure grading logic: `get_grade()` classifies by the longest dimension (mm) against `GRADE_TRESHOLDS` (note: typo in variable name — "TRESHOLDS"). Thresholds: A ≥ 80mm, B ≥ 60mm, C ≥ 40mm, D < 40mm. Also exports `GRADE_COLORS` (BGR) and `summarize_grades()`.
5. `src/utils.py` — drawing helpers: `draw_detection()`, `draw_all_detections()`, `save_annotated_image()`, `image_to_bytes()` (OpenCV → PNG bytes for Streamlit).
6. `src/logger.py` — appends per-session results to a daily CSV at `outputs/logs/log_YYYYMMDD.csv`.

**Detection dict schema** (output of `PotatoDetector.run()`):
```python
{
  "detections": [
    {"id": int, "length_mm": float, "width_mm": float,
     "area_mm2": float, "grade": str, "rect": cv2.minAreaRect, "conf": float}
  ],
  "total": int,
  "image_path": str,
}
```

**Key files to know when changing grading logic:** `src/grader.py:GRADE_TRESHOLDS` (thresholds) and `src/grader.py:GRADE_COLORS` (visualization colors). The Streamlit sidebar legend at `app/streamlit_app.py:57–61` must be kept in sync manually.

## Dataset

Single class: `Potato` (class index 0). ~1,400 original images, 3× augmented to 3,585 training images. Sourced from Roboflow (workspace: `potatoyolo`, project: `potato-lqffs-0ka4b`, CC BY 4.0).

Label format — YOLO instance segmentation polygon (normalized [0,1] coords). Empty `.txt` files are negative images (no potatoes).

## Known Issues in Existing Code

- `src/measurer.py`: `import os` is placed at the bottom of the file (line 68) instead of the top — works but is non-standard.
- `test.py`: hardcodes `D:\PotatoProject` — not portable, use `src/measurement.py` or the Streamlit app instead.
- `train.py`: references `yolo26n-seg.pt` (typo) instead of `yolov8n-seg.pt`.
- `src/measurement.py` duplicates grading and measurement logic that now lives in `src/grader.py` and `src/measurer.py` — it is a prototype and not used by the main pipeline.
