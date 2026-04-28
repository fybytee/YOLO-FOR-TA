from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# =========================
# LOAD MODEL (sekali saja)
# =========================
model = YOLO("yolo26-seg.pt")  # sesuaikan path
class_names = model.names

# =========================
# ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # =========================
    # READ IMAGE
    # =========================
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    H, W = image_np.shape[:2]

    # =========================
    # PREDICT YOLO
    # =========================
    results = model.predict(
        source=image_np,
        conf=0.25,
        iou=0.45,
        imgsz=640
    )

    result = results[0]

    detected_classes = set()
    nasi_pixel_total = 0

    # =========================
    # PROCESS MASK
    # =========================
    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, mask in enumerate(masks):

            class_id = int(classes[i])
            class_name = class_names[class_id]

            detected_classes.add(class_name)

            if class_name == "nasi":

                # Resize mask ke ukuran asli
                mask_img = Image.fromarray(
                    (mask * 255).astype(np.uint8)
                )

                mask_resized = mask_img.resize(
                    (W, H),
                    resample=Image.NEAREST
                )

                mask_resized = np.array(mask_resized) / 255.0

                binary_mask = mask_resized > 0.25

                nasi_pixel = np.sum(binary_mask)
                nasi_pixel_total += nasi_pixel

    # =========================
    # HITUNG GRAM
    # =========================
    gram_nasi = nasi_pixel_total * 0.0001

    # =========================
    # RESPONSE
    # =========================
    return jsonify({
        "classes_detected": list(detected_classes),
        "nasi_pixel": int(nasi_pixel_total),
        "gram_nasi": round(gram_nasi, 4)
    })


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)