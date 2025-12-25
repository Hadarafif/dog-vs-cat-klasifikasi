from pathlib import Path
import streamlit as st
from src.infer_citra import run_citra_ui
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
CITRA_DIR = MODELS_DIR / "citra dataset contoh"

st.set_page_config(
    page_title="Cat vs Dog • Image Classifier",
    page_icon="🐾",
    layout="wide",
)

# ---------- Custom style ----------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
      .hero {
        padding: 1.1rem 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,.12);
        background: linear-gradient(135deg, rgba(59,130,246,.15), rgba(16,185,129,.10));
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
      }
      .hero h1 { margin: 0; font-size: 1.8rem; }
      .hero p { margin: .35rem 0 0; opacity: .9; }
      .chip {
        display:inline-block; padding:0.25rem 0.6rem; border-radius:999px;
        background: rgba(56,189,248,.12); border: 1px solid rgba(56,189,248,.25);
        font-size: 0.8rem;
      }
      .card {
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,.03);
      }
      .muted { opacity: .85; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.markdown("## 🐾 Cat vs Dog")
st.sidebar.caption("Klasifikasi gambar menggunakan model CNN / Transfer Learning yang kamu latih.")
st.sidebar.markdown("---")

st.sidebar.markdown("**📁 Folder model**")
st.sidebar.code(str(CITRA_DIR.as_posix()), language="text")

st.sidebar.markdown("**✅ Checklist cepat**")
st.sidebar.markdown(
    """
- File model ada di `models/citra/*.h5` atau `*.keras`  
- (Opsional) `models/citra/labels.json` ada & benar  
- Folder dataset contoh `models/citra/Cat` dan `models/citra/Dog` (opsional)
    """
)

st.sidebar.markdown("---")
st.sidebar.info("Tip: kalau label ketuker, cek `labels.json` harus sesuai `class_indices` saat training.")

# ---------- Main ----------
st.markdown(
    """
    <div class="hero">
      <span class="chip">Image Classification</span>
      <h1>🐶🐱 Cat vs Dog Classifier</h1>
      <p class="muted">Upload gambar atau pilih dari dataset → prediksi Top-K + probabilitas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Info cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="card">
          <div class="chip">Input</div>
          <h3 style="margin:.5rem 0 0;">Upload / Dataset</h3>
          <p class="muted" style="margin:.35rem 0 0;">
            Terima PNG/JPG/JPEG/WEBP. Bisa juga pilih contoh dari folder dataset.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="card">
          <div class="chip">Model</div>
          <h3 style="margin:.5rem 0 0;">Auto Spec</h3>
          <p class="muted" style="margin:.35rem 0 0;">
            IMG_SIZE & channels dibaca otomatis dari input shape model.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="card">
          <div class="chip">Output</div>
          <h3 style="margin:.5rem 0 0;">Top-K Prob</h3>
          <p class="muted" style="margin:.35rem 0 0;">
            Menampilkan prediksi terbaik + tabel probabilitas.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("---")

# Run only image page
run_citra_ui(models_dir=CITRA_DIR)
