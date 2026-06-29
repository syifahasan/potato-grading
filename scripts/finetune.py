# scripts/finetune.py

from ultralytics import YOLO
import os

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model = YOLO(os.path.join(ROOT, "models", "weights", "best.pt"))

    model.train(
        data=os.path.join(ROOT, "datasets", "data.yaml"),
        epochs=30,
        imgsz=640,
        batch=4,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        freeze=10,
        workers=2,
        device=0,
        amp=False,         # ← ubah ke False, karena GTX 1650 tidak support AMP
                           #   (kamu bisa lihat di log: "AMP will be disabled")
        project=os.path.join(ROOT, "runs", "finetune"),
        name="potato_finetune_v1",
        exist_ok=True,
    )

# ↓ ini yang wajib ada di Windows
if __name__ == '__main__':
    main()