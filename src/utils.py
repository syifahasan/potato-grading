# src/utils.py
# ============================================================
# Fungsi-fungsi helper untuk:
#   1. Menggambar anotasi di atas gambar (kotak + label)
#   2. Menyimpan gambar hasil ke disk
#   3. Konversi gambar OpenCV → bytes (untuk Streamlit)
# ============================================================

import logging
import cv2
import os
import sys
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.grader import get_grade_color


def draw_detection(img: np.ndarray, detection: dict[str, object]) -> np.ndarray:
    """
    Gambar satu bounding box + label untuk satu kentang di atas gambar.

    Args:
        img       : gambar OpenCV (numpy array BGR)
        detection : satu item dari list detections (output detector.py)

    Returns:
        img yang sudah digambar (in-place, tapi juga di-return)
    """
    rect       = detection["rect"]        # tuple minAreaRect
    length_mm  = detection["length_mm"]
    width_mm   = detection["width_mm"]
    grade      = detection["grade"]
    idx        = detection["id"]

    # Ambil warna sesuai grade (BGR)
    color = get_grade_color(grade)

    # cv2.boxPoints: konversi minAreaRect → 4 titik sudut kotak
    points = cv2.boxPoints(rect)
    points = np.int32(points)  # harus integer untuk digambar

    # Gambar kotak di sekeliling kentang
    cv2.drawContours(img, [points], 0, color, 2)

    # Tentukan posisi label teks — di atas sudut kiri atas kotak
    text_x = int(points[:, 0].min())
    text_y = int(points[:, 1].min()) - 10

    # Kalau teks terlalu ke atas (keluar frame), geser ke bawah
    if text_y < 15:
        text_y = int(points[:, 1].max()) + 20

    label = f"#{idx} {length_mm:.1f}x{width_mm:.1f}mm [{grade}]"

    # Background hitam tipis di belakang teks supaya mudah dibaca
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (text_x - 2, text_y - th - 4), (text_x + tw + 2, text_y + 2), (0, 0, 0), -1)

    # Tulis label teks
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img


def draw_all_detections(img_path: str, detections: list) -> np.ndarray:
    """
    Baca gambar dari path, gambar semua deteksi, kembalikan gambar hasil.

    Args:
        img_path   : path ke gambar asli
        detections : list of dict dari detector.run()

    Returns:
        numpy array gambar yang sudah dianotasi

    Raises:
        FileNotFoundError : kalau gambar tidak bisa dibaca
    """
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Tidak bisa membaca gambar: {img_path}")

    # Gambar setiap deteksi
    for detection in detections:
        draw_detection(img, detection)

    # Tambahkan info ringkasan di sudut kiri atas gambar
    total   = len(detections)
    summary = f"Total: {total} kentang"
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def save_annotated_image(img: np.ndarray, output_dir: str, prefix: str = "result") -> str:
    """
    Simpan gambar hasil anotasi ke disk dengan nama file berdasarkan timestamp.

    Args:
        img        : gambar numpy array (output dari draw_all_detections)
        output_dir : folder tujuan
        prefix     : awalan nama file (default "result")

    Returns:
        path lengkap file yang disimpan
    """
    os.makedirs(output_dir, exist_ok=True)  # buat folder kalau belum ada

    # Nama file pakai timestamp supaya tidak tertimpa
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{prefix}_{timestamp}.jpg"
    filepath  = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, img)
    logger.info("Gambar disimpan: %s", filepath)

    return filepath


def image_to_bytes(img: np.ndarray) -> bytes:
    """
    Konversi gambar OpenCV (numpy array BGR) ke bytes PNG.
    Dipakai oleh Streamlit untuk menampilkan gambar tanpa simpan ke disk dulu.

    Args:
        img : numpy array BGR dari OpenCV

    Returns:
        bytes dalam format PNG
    """
    # cv2.imencode: encode gambar ke buffer memory (bukan file)
    success, buffer = cv2.imencode(".png", img)

    if not success:
        raise RuntimeError("Gagal mengkonversi gambar ke bytes.")

    return buffer.tobytes()