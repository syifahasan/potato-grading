# ========================================
# Berisi HANYA logic grading kentang.
# ========================================

# definisi batas ukuran per grade - ubah disini kalau standar berubah
# satuan : mm, berdasarkan dimensi TERPANJANG kentang
GRADE_THRESHOLDS = {
    "A": 80,
    "B": 60,
    "C": 40,
    # dibawah 40mm -> grade D
}

# warna per-grade untuk visualisasi (format BGR OpenCV)
GRADE_COLORS = {
    "Grade A": (0, 200, 0),    # hijau
    "Grade B": (0, 165, 255),  # oranye
    "Grade C": (0, 0, 255),    # merah
    "Grade D": (128, 0, 128),  # ungu
}

def get_grade(length_mm: float, width_mm: float) -> str:
    major = max(length_mm, width_mm)

    if major >= GRADE_THRESHOLDS["A"]:
        return "Grade A"
    elif major >= GRADE_THRESHOLDS["B"]:
        return "Grade B"
    elif major >= GRADE_THRESHOLDS["C"]:
        return "Grade C"
    else:
        return "Grade D"

def get_grade_color(grade: str) -> tuple[int, int, int]:
    return GRADE_COLORS.get(grade, (255, 255, 255))

def summarize_grades(detections: list[dict]) -> dict[str, int]:
    """
    Hitung jumlah kentang per grade dari satu sesi pengukuran.

    Args:
        detections : list of dicts, masing masing berisi key 'grade'
        contoh: [{""grade": "Grade A"}, {"grade": "Grade B"}, {"grade": "Grade A"}]

    Returns:
        dict jumlah per grade, contoh: {"Grade A": 2, "Grade B": 1, "Grade C": 0, "Grade D": 0}     
    """

    # inisialisasi semua grade dengan 0
    summary = {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Grade D": 0}

    for det in detections:
        grade = det.get("grade", "")
        if grade in summary:
            summary[grade] += 1
    
    return summary