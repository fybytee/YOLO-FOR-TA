import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("Streamlit/yolo26-seg.onnx")

model = load_model()

# =========================
# CLASS NAMES
# =========================
class_names = ['buah', 'karbo', 'nasi', 'protein', 'sayur', 'susu']

# =========================
# UI
# =========================
st.title("🍱 Segmentasi Makanan + Hitung Pixel Nasi")

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    H, W = image_np.shape[:2]

    st.image(image, caption="Gambar Asli")

    # =========================
    # PREDIKSI YOLO
    # =========================
    results = model(image_np)
    result = results[0]

    # =========================
    # VISUALISASI
    # =========================
    annotated = result.plot()
    st.image(annotated, caption="Hasil Segmentasi")

    nasi_pixel_total = 0

    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, mask in enumerate(masks):

            class_id = int(classes[i])
            class_name = class_names[class_id]

            if class_name == "nasi":

                # =========================
                # RESIZE MASK (AMAN)
                # =========================
                mask_img = Image.fromarray(
                    (mask * 255).astype(np.uint8)
                )

                mask_resized = mask_img.resize(
                    (W, H),
                    resample=Image.NEAREST
                )

                mask_resized = (
                    np.array(mask_resized) / 255.0
                )

                # =========================
                # THRESHOLD
                # =========================
                binary_mask = mask_resized > 0.5

                nasi_pixel = np.sum(binary_mask)

                nasi_pixel_total += nasi_pixel

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Hasil Perhitungan")

    if nasi_pixel_total > 0:
        st.success(
            f"Jumlah pixel nasi: {int(nasi_pixel_total)}"
        )
    else:
        st.warning("Tidak ada nasi terdeteksi")