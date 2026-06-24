# src/logger.py
# ============================================================
# Simpan hasil pengukuran ke file CSV per sesi.
# Kenapa penting:
#   - Operator bisa lihat riwayat pengukuran sebelumnya
#   - Data bisa dibuka di Excel untuk laporan
#   - Berguna untuk debugging kalau ada hasil yang aneh
# ============================================================

import csv
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import OUTPUT_DIR

# Subfolder khusus untuk log CSV
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Nama kolom CSV — urutan ini menentukan urutan kolom di file
CSV_COLUMNS = [
    "timestamp",      # kapan pengukuran dilakukan
    "session_id",     # ID sesi (satu sesi = satu foto)
    "potato_id",      # nomor kentang dalam sesi ini
    "length_mm",      # panjang dalam mm
    "width_mm",       # lebar dalam mm
    "area_mm2",       # luas perkiraan dalam mm²
    "grade",          # Grade A / B / C / D
    "confidence",     # confidence score YOLO
    "image_file",     # nama file gambar yang diproses
]


def generate_session_id() -> str:
    """
    Buat ID unik untuk satu sesi pengukuran berdasarkan timestamp.
    Contoh output: "SESSION_20260624_143022"
    """
    return datetime.now().strftime("SESSION_%Y%m%d_%H%M%S")


def log_session(detections: list, image_path: str, session_id: str = None) -> str:
    """
    Simpan seluruh hasil satu sesi pengukuran ke CSV.

    Setiap sesi akan menambahkan baris ke file CSV harian.
    Format nama file: log_YYYYMMDD.csv
    Kalau file sudah ada, baris baru di-append (tidak ditimpa).

    Args:
        detections : list of dict dari detector.run()
        image_path : path gambar yang diproses
        session_id : ID sesi (opsional, auto-generate kalau tidak diisi)

    Returns:
        path ke file CSV yang dipakai
    """
    os.makedirs(LOG_DIR, exist_ok=True)  # buat folder logs kalau belum ada

    # Generate session ID kalau tidak diberikan
    if session_id is None:
        session_id = generate_session_id()

    # Nama file CSV berdasarkan tanggal hari ini
    today    = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(LOG_DIR, f"log_{today}.csv")

    # Cek apakah file sudah ada — kalau belum, tulis header dulu
    file_exists = os.path.exists(csv_path)

    # Buka file dengan mode 'a' (append) supaya data lama tidak tertimpa
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        # Tulis header hanya kalau file baru
        if not file_exists:
            writer.writeheader()

        # Timestamp untuk semua baris dalam sesi ini sama
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_name = os.path.basename(image_path)  # hanya nama file, bukan path penuh

        # Tulis satu baris per kentang
        for det in detections:
            writer.writerow({
                "timestamp"  : timestamp,
                "session_id" : session_id,
                "potato_id"  : det["id"],
                "length_mm"  : det["length_mm"],
                "width_mm"   : det["width_mm"],
                "area_mm2"   : det["area_mm2"],
                "grade"      : det["grade"],
                "confidence" : det["conf"],
                "image_file" : image_name,
            })

    print(f"[Logger] {len(detections)} baris disimpan ke: {csv_path}")
    return csv_path


def get_log_summary(csv_path: str) -> dict:
    """
    Baca file CSV dan kembalikan ringkasan statistik.
    Berguna untuk ditampilkan di dashboard Streamlit.

    Args:
        csv_path : path ke file CSV log

    Returns:
        dict berisi:
            total_potatoes  : total kentang sepanjang hari
            total_sessions  : jumlah sesi foto
            grade_counts    : dict jumlah per grade
            avg_length_mm   : rata-rata panjang
            avg_width_mm    : rata-rata lebar
    """
    if not os.path.exists(csv_path):
        return {}

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {}

    # Hitung statistik dasar
    total_potatoes = len(rows)
    total_sessions = len(set(r["session_id"] for r in rows))  # session unik

    # Jumlah per grade
    grade_counts = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Grade D": 0}
    for row in rows:
        grade = row["grade"]
        if grade in grade_counts:
            grade_counts[grade] += 1

    # Rata-rata dimensi — convert string ke float dulu
    lengths = [float(r["length_mm"]) for r in rows]
    widths  = [float(r["width_mm"])  for r in rows]

    avg_length = sum(lengths) / len(lengths)
    avg_width  = sum(widths)  / len(widths)

    return {
        "total_potatoes" : total_potatoes,
        "total_sessions" : total_sessions,
        "grade_counts"   : grade_counts,
        "avg_length_mm"  : round(avg_length, 1),
        "avg_width_mm"   : round(avg_width, 1),
    }