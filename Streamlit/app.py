import streamlit as st
import numpy as np
import cv2
from engine import process_image

st.title("🍚 Deteksi & Estimasi Berat Nasi")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # BGR untuk model
    image_bgr = cv2.imdecode(file_bytes, 1)

    # RGB untuk display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Input", use_column_width=True)

    if st.button("🔍 Proses"):
        with st.spinner("Processing..."):
            result_img, weight, found, objects, crops = process_image(image_bgr)

        # convert hasil ke RGB
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

        # ============================
        # OBJEK TERDETEKSI
        # ============================
        st.subheader("📦 Objek Terdeteksi")
        for obj in objects:
            st.write(f"- {obj['label']} ({obj['confidence']})")

        # ============================
        # CROP NASI
        # ============================
        st.subheader("🧩 Crop Nasi")
        if len(crops) == 0:
            st.write("Tidak ada crop nasi")
        else:
            for i, crop in enumerate(crops):
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                st.image(crop_rgb, caption=f"Crop Nasi {i+1}", use_column_width=True)

        # ============================
        # HASIL BERAT
        # ============================
        if not found:
            st.warning("❌ Tidak ditemukan nasi")
        else:
            st.success(f"🍚 Estimasi Berat: {weight:.2f} gram")