# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer vision project for **potato instance segmentation** using YOLO (Ultralytics). The dataset (~1,400 original images, 3× augmented to 3,585 training images) was sourced from Roboflow (workspace: `potatoyolo`, project: `potato-lqffs-0ka4b`, version 1, CC BY 4.0). A trained model (`best.pt`) is available from the GitHub releases page and also at `runs/segment/train-5/weights/best.pt` locally.

## Repository Layout

```
datasets/              # YOLO dataset (data.yaml + train/valid/test splits)
src/calibration.py     # Interactive pixel-to-mm calibration tool (OpenCV)
runs/segment/          # YOLO training outputs (gitignored)
```

## Dataset

`datasets/data.yaml` declares the paths and class list for Ultralytics. It uses relative paths (`../train/images` etc.), so YOLO commands must be run from the **`datasets/` directory** or use an absolute path to `data.yaml`.

**Single class:** `Potato` (class index 0)

**Label format** — YOLO instance segmentation polygon (not bounding box):
```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```
All coordinates are normalized [0, 1]. Empty `.txt` files are negative images.

**Image naming:**
- `P14XXXX_*` / `PO-XXXXXX_*` / `POR-XXXXXX_*` — three potato image sources
- `IMG_XXXX_*` — raw camera captures
- `grading-A/B/C/D_*` — quality-graded images
- `kosongan_*`, `label-meteran_*`, `label-spidol_*` — negative samples
- `.rf.<hash>` suffix — Roboflow augmentation variant of the same source image

## YOLO Commands

Install:
```bash
pip install ultralytics
```

Train (run from repo root, actual hyperparams used: batch=8, device=0):
```bash
yolo segment train data=datasets/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640 batch=8
```

Validate against the trained model:
```bash
yolo segment val data=datasets/data.yaml model=runs/segment/train-5/weights/best.pt
```

Inference on test images:
```bash
yolo segment predict model=runs/segment/train-5/weights/best.pt source=datasets/test/images/
```

## Calibration Tool (`src/calibration.py`)

Interactive OpenCV tool for computing the pixel-per-mm ratio from a reference image containing a ruler or known-length object. Run it before any size-estimation step that converts pixel measurements to real-world units.

Dependencies:
```bash
pip install opencv-python
```

Usage:
1. Edit the two constants at the top of the file: `IMAGE_PATH` (path to a calibration image) and `KNOWN_DISTANCE_MM` (the real-world length in mm that you will click).
2. Run: `python src/calibration.py`
3. Click two points on the ruler/reference object in the window that opens.
4. The script computes `pixel_per_mm` and saves it to `src/calibration.json`.
