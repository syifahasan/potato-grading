import os
import sys
import pandas as pd
import streamlit as st
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.detector import PotatoDetector
from src.utils import draw_all_detections, image_to_bytes, save_annotated_image
from src.grader import summarize_grades
from src.logger import log_session
from config import OUTPUT_DIR

# =========================================
# KONFIGURASI HALAMAN
# =========================================
st.set_page_config(
    page_title="Potato Grading System",
    page_icon="🥔",
    layout="wide",
)

# =========================================
# INISIALISASI DETECTOR
# model dan kalibrasi di load sekali
# =========================================

@st.cache_resource
def load_detector():
    return PotatoDetector()

# =========================================
# HEADER
# =========================================
st.title("Potato Grading System")
st.markdown("Upload foto kentang → sistem otomatis mendeteksi, mengukur, dan menentukan grade.")
st.divider()

# =========================================
# SIDEBAR - pengaturan
# =========================================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    conf_treshold = st.slider(
        label = "Confidence Treshold",
        min_value = 0.1,
        max_value = 0.9,
        value = 0.35,
        step = 0.05,
        help = "Semakin tinggi = hanya deteks yang yakin saja. Turunkan kalau ada kentang yang tidak terdeteksi."
    )

    st.markdown("---")
    st.markdown("**Keterangan Grade:**")
    st.markdown("🟢 **Grade A** — panjang ≥ 80mm")
    st.markdown("🟠 **Grade B** — panjang 60–79mm")
    st.markdown("🔴 **Grade C** — panjang 40–59mm")
    st.markdown("🟣 **Grade D** — panjang < 40mm")

# =======================================
# AREA UPLOAD GAMBAR
# =======================================
uploaded_file = st.file_uploader(
    label = "Upload gambar kentang (JPG/PNG)",
    type = ["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload gambar kentang di atas terpal biru untuk mulai.")
    st.stop()

# =======================================
# PROSES GAMBAR
# =======================================

# simpan file upload ke folder temp supaya bisa dibaca openCV
temp_dir = os.path.join(OUTPUT_DIR, "temp")
os.makedirs(temp_dir, exist_ok=True)

temp_path = os.path.join(temp_dir, uploaded_file.name)
with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer()) # tulis bytes file ke disk

# load detector (dari cache kalau sudah pernah di-load)
detector = load_detector()

# update confidence treshold dari sidebar
detector.conf = conf_treshold

# run deteksi
with st.spinner("Mendeteksi dan mengukur kentang ..."):
    result = detector.run(temp_path)

detections = result["detections"]
total = result["total"]

# =====================================
# TAMPILKAN HASIL
# =====================================
if total == 0:
    st.warning("Tidak ada kentang terdeteksi. Coba turunkan confidence treshold di sidebar.")
    st.stop()

# Buat gambar dengan anotasi
annotated_img = draw_all_detections(temp_path, detections)
img_bytes = image_to_bytes(annotated_img)

# Layout dua kolom: gambar | ringkasan
col_img, col_summary = st.columns([3, 2])

with col_img:
    st.subheader("Hasil Deteksi")
    st.image(img_bytes, caption=f"Total: {total} kentang terdeteksi", width="stretch")

with col_summary:
    st.subheader("Ringkasan Grade")
    grade_summary = summarize_grades(detections)
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.metric("🟢 Grade A", grade_summary["Grade A"])
        st.metric("🔴 Grade C", grade_summary["Grade C"])
    with gcol2:
        st.metric("🟠 Grade B", grade_summary["Grade B"])
        st.metric("🟣 Grade D", grade_summary["Grade D"])
    
    st.divider()
    st.metric("Total kentang", total)

st.divider()

# --- Tabel detail per-kentang ---
st.subheader("Detail pengukuran")

# Buat dataframe dari list detections
df = pd.DataFrame([
    {
        "No." : det["id"],
        "Panjang (mm)" : det["length_mm"],
        "Lebar (mm)" : det["width_mm"],
        "Luas (mm²)" : det["area_mm2"],
        "Grade" : det["grade"],
        "Confidence" : f'{det["conf"]:.2f}',
    }
    for det in detections
])

st.dataframe(
    df,
    width='stretch',
    hide_index = True
)

# ======================================
# TOMBOL SIMPAN & EXPORT
# ======================================
st.divider()
st.subheader("Simpan Hasil")

btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    # Download gambar anotasi
    st.download_button(
        label    = "⬇️ Download Gambar Hasil",
        data     = img_bytes,
        file_name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime     = "image/png",
    )
 
with btn_col2:
    # Download tabel sebagai CSV
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️ Download CSV",
        data      = csv_data,
        file_name = f"pengukuran_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime      = "text/csv",
    )
 
with btn_col3:
    # Tombol simpan ke log harian (bukan download — simpan di server)
    if st.button("📁 Simpan ke Log Harian"):
        log_path = log_session(detections, temp_path)
        st.success(f"✅ Tersimpan di: {log_path}")

# =====================================
# FOOTER
# =====================================
st.divider()
st.caption("Potato Grading System — Computer Vision Pipeline | PotatoProject")