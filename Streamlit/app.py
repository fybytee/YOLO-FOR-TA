import streamlit as st
import numpy as np
import cv2
from engine import process_image

st.set_page_config(page_title="Deteksi Nasi", layout="centered")

st.title("🍚 Deteksi & Estimasi Berat Nasi")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Input", use_column_width=True)

    if st.button("🔍 Proses"):
        with st.spinner("Processing..."):
            result_img, weight, found, objects = process_image(image)

        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

        st.subheader("📦 Objek Terdeteksi")
        for obj in objects:
            st.write(f"- {obj['label']} ({obj['confidence']})")

        if not found:
            st.warning("❌ Tidak ditemukan nasi")
        else:
            st.success(f"🍚 Estimasi Berat: {weight:.2f} gram")