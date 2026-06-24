import os
from ultralytics import YOLO

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = os.path.join(BASE_DIR, 'dataset', 'data.yaml')

    model = YOLO('yolo26n-seg.pt')
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,
        amp=True,
        device=0,
        workers=2   # turun dari default 8
    )