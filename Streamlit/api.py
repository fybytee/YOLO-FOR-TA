from flask import Flask, request, jsonify
import numpy as np
import cv2
from engine import process_image

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result_img, weight, found, objects = process_image(image)

    return jsonify({
        "found_rice": bool(found),
        "estimated_weight": float(weight),
        "detected_objects": objects
    })

if __name__ == "__main__":
    app.run(debug=True)