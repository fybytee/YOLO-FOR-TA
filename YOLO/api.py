from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import io
import os

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = YOLO("yolo26-seg.pt")

# nama class
class_names = model.names

# =========================
# KONSTANTA
# =========================

# hasil kalibrasi pixel -> gram
KALIBRASI = 0.00011159

# 100 gram nasi = 129 kalori
# sumber:
# https://www.fatsecret.co.id/kalori-gizi/umum/nasi-putih?portionid=53181&portionamount=100,000
KALORI_PER_100_GRAM = 129

# =========================
# ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =========================
        # VALIDASI FILE
        # =========================
        if "file" not in request.files:

            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        # =========================
        # READ IMAGE
        # =========================
        image_bytes = file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # =========================
        # SIMPAN TEMP FILE
        # supaya inferensi sama
        # seperti notebook
        # =========================
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        )

        temp_path = temp_file.name

        # tutup file agar tidak terkunci di Windows
        temp_file.close()

        # save image
        image.save(temp_path)

        # =========================
        # PREDICT YOLO
        # =========================
        results = model(
            temp_path,
            conf=0.2,
            imgsz=640,
            retina_masks=True,
            verbose=False
        )

        result = results[0]

        # =========================
        # VARIABLE
        # =========================
        detected_classes = set()

        detections = []

        nasi_pixel_total = 0

        # =========================
        # DEBUG SHAPE
        # =========================
        print("\n===== DEBUG =====")

        if result.masks is not None:

            print(
                "Mask shape:",
                result.masks.data.shape
            )

        # =========================
        # PROCESS BOXES
        # =========================
        if result.boxes is not None:

            classes = result.boxes.cls.cpu().numpy()

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            for i in range(len(classes)):

                class_id = int(classes[i])

                class_name = class_names[class_id]

                conf = float(confidences[i])

                # simpan class
                detected_classes.add(class_name)

                # simpan detail detection
                detections.append({

                    "class": class_name,

                    "confidence": round(
                        conf,
                        4
                    )

                })

                print(
                    f"{class_name} -> {conf:.4f}"
                )

        # =========================
        # PROCESS MASK
        # =========================
        if result.masks is not None:

            masks = (
                result.masks.data
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            for i, mask in enumerate(masks):

                class_id = int(classes[i])

                class_name = class_names[class_id]

                # =========================
                # HITUNG PIXEL NASI
                # =========================
                if class_name == "nasi":

                    # binary mask
                    binary_mask = mask > 0.5

                    nasi_pixel = np.sum(
                        binary_mask
                    )

                    nasi_pixel_total += nasi_pixel

                    print(
                        "Nasi pixel:",
                        nasi_pixel
                    )

        # =========================
        # HITUNG BERAT NASI
        # =========================
        gram_nasi = (
            nasi_pixel_total
            * KALIBRASI
        )

        # =========================
        # HITUNG KALORI
        # =========================
        kalori_nasi = (
            gram_nasi / 100
        ) * KALORI_PER_100_GRAM

        # =========================
        # HAPUS TEMP FILE
        # =========================
        if os.path.exists(temp_path):

            os.remove(temp_path)

        print("=================\n")

        # =========================
        # RESPONSE
        # =========================
        return jsonify({

            "classes_detected": list(
                detected_classes
            ),

            # "detections": detections,

            "nasi_pixel": int(
                nasi_pixel_total
            ),

            "gram_nasi": round(
                gram_nasi,
                4
            ),

            "kalori_nasi": round(
                kalori_nasi,
                2
            )

        })

    # =========================
    # ERROR HANDLER
    # =========================
    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )