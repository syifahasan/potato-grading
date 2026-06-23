## Download Model

Download `best.pt` dari [Releases](https://github.com/syifahasan/potato-grading/releases/tag/model)

# Dataset

Dataset terdiri dari ~1.400 gambar kentang yang dianotasi untuk segmentasi instance.

- **Source:** Roboflow (workspace: `potatoyolo`, project: `potato-lqffs-0ka4b`)
- **Split:** ~81% train / 11% valid / 8% test
- **Augmentation:** 3× pada training images
- **Format:** YOLO segmentation polygon, `nc: 1`, `names: ['Potato']`

## Cara mendapatkan dataset

1. Buat akun di [Roboflow](https://roboflow.com)
2. Minta akses ke workspace `potatoyolo` (hubungi pemilik repo)
3. Export dalam format **YOLOv8 Segmentation**
4. Letakkan hasil export di folder `datasets/`

## Struktur folder setelah download

datasets/
├── train/
│ ├── images/
│ └── labels/
├── valid/
│ ├── images/
│ └── labels/
└── test/
├── images/
└── labels/
