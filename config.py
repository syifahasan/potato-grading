import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Path Utama ---
DATASET_DIR = os.path.join(ROOT, "datasets")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

TRAIN_DIR = os.path.join(ROOT, "datasets","train", "images")
VALID_DIR = os.path.join(ROOT, "datasets","valid", "images")
TEST_DIR = os.path.join(ROOT, "datasets","test", "images")

MODEL_DIR = os.path.join(ROOT, "models", "weights")
BEST_MODEL = os.path.join(MODEL_DIR, "best.pt")

CALIB_PATH = os.path.join(ROOT, "src", "calibration.json")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)



