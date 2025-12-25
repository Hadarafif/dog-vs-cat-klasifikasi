from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import streamlit as st
from PIL import Image


@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: Path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path.as_posix())


def _infer_model_spec(model) -> tuple[int, int]:
    """Return (img_size, channels) from model.input_shape; fallback (224,3)."""
    shp = model.input_shape
    if isinstance(shp, list):
        shp = shp[0]

    img_size, channels = 224, 3
    try:
        # common TF: (None, H, W, C)
        if (
            len(shp) == 4
            and isinstance(shp[1], int)
            and isinstance(shp[2], int)
            and shp[1] == shp[2]
        ):
            img_size = int(shp[1])
            if shp[3] in (1, 3):
                channels = int(shp[3])
    except Exception:
        pass

    return img_size, channels


def _load_label_map(models_dir: Path):
    path = models_dir / "labels.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _label_from_map(label_map, idx: int) -> str:
    if label_map is None:
        return str(idx)

    if isinstance(label_map, list) and 0 <= idx < len(label_map):
        return str(label_map[idx])

    if isinstance(label_map, dict):
        return str(label_map.get(str(idx), label_map.get(idx, idx)))

    return str(idx)


def _preprocess(img: Image.Image, img_size: int, channels: int) -> np.ndarray:
    if channels == 1:
        img = img.convert("L").resize((img_size, img_size))
        arr = (np.array(img, dtype=np.float32) / 255.0)[..., None]  # (H,W,1)
    else:
        img = img.convert("RGB").resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32) / 255.0  # (H,W,3)

    return np.expand_dims(arr, 0)  # (1,H,W,C)


def run_citra_ui(models_dir: Path):
    st.title("🖼️ Klasifikasi Citra")

    st.info(
        "Model citra di: models/citra/\n"
        "Dataset contoh (folder kelas) di: models/citra/{Dog,Cat}/\n"
        "Label map: models/citra/labels.json"
    )

    # cari model
    model_files = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.keras"))
    if not model_files:
        st.warning("Tidak ada model citra (.h5/.keras) di models/citra/")
        return

    model_name = st.selectbox("Pilih model", [p.name for p in model_files])
    model_path = models_dir / model_name

    with st.spinner("Load model citra..."):
        model = load_keras_model(model_path)

    img_size_default, channels = _infer_model_spec(model)
    label_map = _load_label_map(models_dir)

    # dataset folder classes (opsional)
    CLASSES = ["Dog", "Cat"]
    dataset_ok = all((models_dir / c).exists() for c in CLASSES)

    st.subheader("Input")
    mode_options = ["Upload gambar"]
    if dataset_ok:
        mode_options.append("Pilih dari dataset (Dog/Cat)")

    mode = st.radio("Sumber gambar", mode_options, horizontal=True)

    img = None
    caption = ""

    if mode.startswith("Pilih") and dataset_ok:
        cls = st.selectbox("Kelas", CLASSES)

        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            files += list((models_dir / cls).glob(ext))

        if not files:
            st.warning(f"Tidak ada gambar di {models_dir/cls}")
            return

        pick = st.radio("Pilih file", ["Acak", "Manual"], horizontal=True)
        chosen = random.choice(files) if pick == "Acak" else st.selectbox(
            "File", files, format_func=lambda p: p.name
        )

        img = Image.open(chosen)
        caption = f"{cls}/{chosen.name}"

    else:
        up = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg", "webp"])
        if up is None:
            st.caption("Upload gambar dulu.")
            return
        img = Image.open(up)
        caption = "Uploaded"

    st.image(img, caption=caption, use_container_width=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        img_size = st.number_input("IMG_SIZE", value=int(img_size_default), step=1)
    with c2:
        st.write("channels (sesuai model)")
        st.selectbox("channels", [channels], index=0, disabled=True)
    with c3:
        topk = st.slider("Top-K", 1, 5, 3)

    if st.button("🔍 Prediksi", type="primary"):
        x = _preprocess(img, int(img_size), int(channels))

        y = model.predict(x, verbose=0)
        y = np.squeeze(y)

        # binary sigmoid vs multiclass
        if y.ndim == 0:
            p = float(y)
            probs = np.array([1 - p, p], dtype=np.float32)
        elif y.ndim == 1 and y.shape[0] == 1:
            p = float(y[0])
            probs = np.array([1 - p, p], dtype=np.float32)
        else:
            probs = y.astype(np.float32)

        s = float(np.sum(probs))

        # normalize / softmax if looks like logits
        if not (0.98 <= s <= 1.02) or np.any(probs < 0):
            probs = np.exp(probs - np.max(probs))
            probs = probs / (np.sum(probs) + 1e-12)
        else:
            probs = probs / (s + 1e-12)

        order = np.argsort(-probs)[: int(topk)]
        best = int(order[0])

        st.subheader("Hasil")
        st.success(
            f"Prediksi: **{_label_from_map(label_map, best)}** "
            f"(prob: **{float(probs[best]):.4f}**)"
        )

        import pandas as pd

        rows = [
            {
                "class_id": int(i),
                "label": _label_from_map(label_map, int(i)),
                "prob": float(probs[i]),
            }
            for i in order
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
