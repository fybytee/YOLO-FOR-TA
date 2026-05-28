from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import io
import os
import time

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
KALORI_PER_100_GRAM = 129

# =========================
# ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    temp_path = None

    try:

        # =========================
        # VALIDASI FILE
        # =========================
        if "file" not in request.files:

            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "error": "Empty filename"
            }), 400

        # =========================
        # READ IMAGE
        # =========================
        image_bytes = file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # =========================
        # AMBIL EXTENSION ASLI
        # =========================
        ext = os.path.splitext(
            file.filename
        )[1].lower()

        # fallback
        if ext == "":
            ext = ".jpg"

        # =========================
        # SIMPAN TEMP FILE
        # =========================
        temp_file = tempfile.NamedTemporaryFile(
            suffix=ext,
            delete=False
        )

        temp_path = temp_file.name

        # penting di Windows
        temp_file.close()

        # save image
        image.save(temp_path)

        # =========================
        # PREDICT YOLO
        # =========================
        results = model(
            temp_path,
            conf=0.01,
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
        # DEBUG
        # =========================
        print("\n===== DETECTION =====")

        # =========================
        # PROCESS BOXES
        # =========================
        if result.boxes is not None:

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            for i in range(len(classes)):

                class_id = int(classes[i])

                class_name = class_names[class_id]

                conf = float(confidences[i])

                detected_classes.add(
                    class_name
                )

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

            print(
                "Mask shape:",
                masks.shape
            )

            for i, mask in enumerate(masks):

                class_id = int(classes[i])

                class_name = class_names[class_id]

                # =========================
                # HITUNG PIXEL NASI
                # =========================
                if class_name == "nasi":

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

        print("=====================\n")

        # =========================
        # RESPONSE
        # =========================
        return jsonify({

            "classes_detected": list(
                detected_classes
            ),

            #"detections": detections,

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
    # ERROR
    # =========================
    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500

    # =========================
    # FINALLY
    # selalu hapus temp file
    # =========================
    finally:

        if temp_path is not None:

            try:

                if os.path.exists(temp_path):

                    # delay kecil supaya
                    # file tidak masih dipakai
                    time.sleep(0.1)

                    os.remove(temp_path)

            except Exception as delete_error:

                print(
                    "Gagal hapus temp file:",
                    delete_error
                )


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )