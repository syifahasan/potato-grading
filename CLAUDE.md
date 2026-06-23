# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computer vision dataset for **potato instance segmentation** using YOLO. The dataset was sourced from Roboflow (workspace: `potatoyolo`, project: `potato-lqffs-0ka4b`, version 1, license: CC BY 4.0). There is currently no training code — the repo is a dataset ready to be consumed by a YOLO training pipeline.

## Dataset Structure

```
dataset/
  data.yaml          # YOLO dataset config (paths, class list)
  train/images/      # 3,585 training images (.jpg)
  train/labels/      # 3,585 YOLO segmentation label files (.txt)
  valid/images/      # 161 validation images
  valid/labels/      # 161 label files
  test/images/       # 113 test images
  test/labels/       # 113 label files
```

**Single class:** `Potato` (class index 0)

## Label Format

Labels use **YOLO instance segmentation format** (polygon, not bounding box):

```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

All coordinates are normalized to [0, 1] relative to image dimensions. Each line is one object instance. Empty `.txt` files represent negative images (no potatoes).

## Image Naming Conventions

- `P14XXXX_*` — main potato images
- `PO-XXXXXX_*` — potato variant source A
- `POR-XXXXXX_*` — potato variant source B
- `IMG_XXXX_*` — raw camera captures
- `grading-A/B/C/D_*` — images with quality grade annotations
- `kosongan_*`, `label-meteran_*`, `label-spidol_*` — negative samples (empty labels)

Roboflow augmentation hash suffixes (`.rf.<hash>`) allow the same source image to appear multiple times with different augmentations in the training set.

## Training with YOLO (Ultralytics)

Install and train:

```bash
pip install ultralytics
yolo segment train data=dataset/data.yaml model=yolo11n-seg.pt epochs=100 imgsz=640
```

Validate:

```bash
yolo segment val data=dataset/data.yaml model=runs/segment/train/weights/best.pt
```

Run inference on test images:

```bash
yolo segment predict model=runs/segment/train/weights/best.pt source=dataset/test/images/
```

The `data.yaml` uses relative paths (`../train/images` etc.), so commands must be run from the `dataset/` directory, or pass the absolute path to `data.yaml`.
