# tests/test_measurer.py
# ============================================================
# Unit test untuk src/measurer.py
# Jalankan dengan: pytest tests/
# ============================================================

import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.measurer import is_mask_valid, measure_from_mask


# --- TEST is_mask_valid() ---

def test_mask_valid_normal():
    # Mask persegi panjang sederhana dengan 4 titik dan area cukup besar
    # Area kotak 100x100 = 10000 px² → jauh di atas minimum 500 px²
    mask = np.array([
        [0, 0], [100, 0], [100, 100], [0, 100],
        [0, 0]   # titik ke-5 supaya memenuhi min_points=5
    ], dtype=np.float32)

    valid, reason = is_mask_valid(mask)
    assert valid == True
    assert reason == ""   # tidak ada pesan error kalau valid

def test_mask_invalid_too_few_points():
    # Mask dengan hanya 3 titik → harus ditolak (min_points=5)
    mask = np.array([
        [0, 0], [50, 0], [25, 50]
    ], dtype=np.float32)

    valid, reason = is_mask_valid(mask)
    assert valid == False
    assert "titik" in reason   # pesan error harus menyebut "titik"

def test_mask_invalid_too_small_area():
    # Mask dengan titik cukup tapi area sangat kecil (kotak 5x5 = 25 px²)
    mask = np.array([
        [0, 0], [5, 0], [5, 5], [0, 5], [0, 0]
    ], dtype=np.float32)

    valid, reason = is_mask_valid(mask)
    assert valid == False
    assert "area" in reason   # pesan error harus menyebut "area"


# --- TEST measure_from_mask() ---

def test_measure_output_keys():
    # Pastikan output measure_from_mask() punya semua key yang dibutuhkan
    mask = np.array([
        [0, 0], [200, 0], [200, 100], [0, 100], [0, 0]
    ], dtype=np.float32)

    pixel_per_mm = 2.0   # 2 pixel = 1 mm (nilai dummy untuk test)
    result = measure_from_mask(mask, pixel_per_mm)

    # Semua key ini harus ada di hasil
    assert "length_mm" in result
    assert "width_mm"  in result
    assert "area_mm2"  in result
    assert "rect"      in result

def test_measure_length_larger_than_width():
    # Untuk kotak 200x100 px dengan pixel_per_mm=2.0:
    # → length = 200/2 = 100mm, width = 100/2 = 50mm
    # length harus selalu >= width (ini kontrak measure_from_mask)
    mask = np.array([
        [0, 0], [200, 0], [200, 100], [0, 100], [0, 0]
    ], dtype=np.float32)

    pixel_per_mm = 2.0
    result = measure_from_mask(mask, pixel_per_mm)

    assert result["length_mm"] >= result["width_mm"]

def test_measure_values_positive():
    # Semua output dimensi harus bilangan positif
    mask = np.array([
        [10, 10], [110, 10], [110, 60], [10, 60], [10, 10]
    ], dtype=np.float32)

    result = measure_from_mask(mask, pixel_per_mm=1.0)

    assert result["length_mm"] > 0
    assert result["width_mm"]  > 0
    assert result["area_mm2"]  > 0