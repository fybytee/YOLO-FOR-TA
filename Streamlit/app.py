import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolo26-seg.pt")  # ganti sesuai model kamu

model = load_model()

# =========================
# CLASS NAMES
# =========================
class_names = ['buah', 'karbo', 'nasi', 'protein', 'sayur', 'susu']

# =========================
# STREAMLIT UI
# =========================
st.title("🍱 Segmentasi Makanan + Hitung Pixel Nasi (FIX SCALE)")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ukuran asli
    H, W = image_np.shape[:2]

    st.image(image, caption="Gambar Asli", use_column_width=True)

    # =========================
    # PREDIKSI
    # =========================
    results = model(image_np)
    result = results[0]

    # =========================
    # VISUALISASI
    # =========================
    annotated = result.plot()
    st.image(annotated, caption="Hasil Segmentasi", use_column_width=True)

    # =========================
    # DEBUG UKURAN (biar jelas)
    # =========================
    st.write("Ukuran gambar asli:", (H, W))

    nasi_pixel_total = 0

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()   # (n, h, w)
        classes = result.boxes.cls.cpu().numpy()

        st.write("Ukuran mask YOLO:", masks.shape)

        for i, mask in enumerate(masks):
            class_id = int(classes[i])
            class_name = class_names[class_id]

            if class_name == "nasi":

                # =========================
                # 🔥 RESIZE MASK KE ORIGINAL
                # =========================
                mask_resized = cv2.resize(mask, (W, H))

                # =========================
                # THRESHOLD
                # =========================
                binary_mask = mask_resized > 0.5

                # =========================
                # HITUNG PIXEL
                # =========================
                nasi_pixel = np.sum(binary_mask)
                nasi_pixel_total += nasi_pixel

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Hasil Perhitungan")

    if nasi_pixel_total > 0:
        st.success(f"Jumlah pixel nasi: {int(nasi_pixel_total)}")
    else:
        st.warning("Tidak ada nasi terdeteksi")


    ###############