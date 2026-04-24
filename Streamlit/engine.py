import numpy as np
import cv2
from ultralytics import YOLO

# ============================
# LOAD MODEL
# ============================
model_det = YOLO("best.pt")   # model detection
model_seg = YOLO("bestseg.pt")      # model segmentation

SCALING_FACTOR = 0.00010024


# ============================
# TAMBAH MARGIN
# ============================
def add_margin(box, img_shape, margin=0.05):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = box

    dw = int((x2 - x1) * margin)
    dh = int((y2 - y1) * margin)

    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(w, x2 + dw)
    y2 = min(h, y2 + dh)

    return x1, y1, x2, y2


# ============================
# MAIN FUNCTION
# ============================
def process_image(image):
    results = model_det(image)

    found_rice = False
    total_weight = 0
    output_image = image.copy()
    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model_det.names[cls]

            detected_objects.append({
                "label": label,
                "confidence": round(conf, 2)
            })

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # gambar semua objek
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output_image, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

            # ============================
            # KHUSUS NASI
            # ============================
            if label.lower() == "nasi":
                found_rice = True

                x1, y1, x2, y2 = add_margin((x1, y1, x2, y2), image.shape)

                crop = image[y1:y2, x1:x2]

                seg_result = model_seg(crop)

                if seg_result[0].masks is None:
                    continue

                masks = seg_result[0].masks.data.cpu().numpy()

                for mask in masks:
                    mask = (mask > 0).astype("uint8")

                    pixel = np.sum(mask)
                    weight = pixel * SCALING_FACTOR
                    total_weight += weight

                    mask_resized = cv2.resize(mask, (crop.shape[1], crop.shape[0]))

                    colored_mask = np.zeros_like(crop)
                    colored_mask[:, :, 1] = mask_resized * 255

                    crop_overlay = cv2.addWeighted(crop, 1, colored_mask, 0.5, 0)

                    output_image[y1:y2, x1:x2] = crop_overlay

                # box nasi warna hijau
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_image, "Nasi",
                            (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    return output_image, total_weight, found_rice, detected_objects