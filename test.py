import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ultralytics import YOLO
from config import BEST_MODEL, TEST_DIR

model = YOLO(BEST_MODEL)
results = model.predict(source=TEST_DIR, save=True, conf=0.25)