import os
import sys
from ultralytics import YOLO

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)
    from config import DATA_YAML

    model = YOLO('yolov8n-seg.pt')
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,
        amp=True,
        device=0,
        workers=2   # turun dari default 8
    )