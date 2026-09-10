import os
import cv2
import re
import base64
import numpy as np
from ultralytics import YOLO
import easyocr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
SAMPLE_DIR = os.path.join(BASE_DIR, "dataset", "test", "images")

_plate_model = None
_vehicle_model = None
_ocr_reader = None

def get_models():
    global _plate_model, _vehicle_model, _ocr_reader
    if _plate_model is None:
        if os.path.exists(PLATE_MODEL_PATH):
            _plate_model = YOLO(PLATE_MODEL_PATH)
        else:
            print(f"[ALPR Warning] Plate model not found at {PLATE_MODEL_PATH}, using yolov8n.pt")
            _plate_model = YOLO(VEHICLE_MODEL_PATH)

    if _vehicle_model is None:
        _vehicle_model = YOLO(VEHICLE_MODEL_PATH)

    if _ocr_reader is None:
        try:
            _ocr_reader = easyocr.Reader(['en'], gpu=True)
        except Exception:
            _ocr_reader = easyocr.Reader(['en'], gpu=False)

    return _plate_model, _vehicle_model, _ocr_reader

def is_valid_indian_plate(text):
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$'
    return bool(re.match(pattern, text))

def clean_plate_text(text):
    return "".join(c for c in text if c.isalnum()).upper()

def preprocess_plate_crop(crop):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 60:
        factor = max(2, int(100 / max(h, 1)))
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bfilter)
    return enhanced

def frame_to_base64(frame):
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def recognize_plate(image_input):
    """
    image_input can be:
    - np.ndarray (OpenCV image)
    - str (file path)
    - bytes (raw image bytes from upload/camera)
    """
    plate_model, vehicle_model, reader = get_models()

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return {"found": False, "error": f"File not found: {image_input}"}
        frame = cv2.imread(image_input)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        frame = image_input.copy()
    else:
        return {"found": False, "error": "Invalid image input type."}

    if frame is None or frame.size == 0:
        return {"found": False, "error": "Could not decode image."}

    annotated = frame.copy()
    detected_vehicle = "Car"

    # 1. Vehicle detection
    v_results = vehicle_model.predict(frame, conf=0.35, verbose=False)
    for res in v_results:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
            if cls_id in [2, 3, 5, 7]:
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                v_name = res.names[cls_id].title()
                detected_vehicle = v_name
                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (245, 130, 32), 2)
                cv2.putText(annotated, f"Vehicle: {v_name}", (vx1, max(20, vy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 130, 32), 2)

    # 2. License Plate Detection
    p_results = plate_model.predict(frame, conf=0.25, verbose=False)

    best_plate = None
    best_conf = 0.0

    for res in p_results:
        for box in res.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 165, 255), 2)

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            processed_crop = preprocess_plate_crop(crop)
            if processed_crop is None:
                continue

            # Run EasyOCR
            ocr_results = reader.readtext(
                processed_crop,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            )

            for (_, text, prob) in ocr_results:
                clean = clean_plate_text(text)
                if len(clean) >= 4 and prob > best_conf:
                    # Prefer regex matched plate, else take high-confidence plate
                    if is_valid_indian_plate(clean) or (best_plate is None or prob > 0.4):
                        best_plate = clean
                        best_conf = float(prob)
                        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 220, 0), 3)

    # If YOLO didn't lock a plate box, try reading full frame or vehicle region
    if not best_plate:
        # Full frame OCR attempt
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray_full, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        for (_, text, prob) in ocr_results:
            clean = clean_plate_text(text)
            if is_valid_indian_plate(clean) or (len(clean) >= 7 and prob > 0.35):
                best_plate = clean
                best_conf = float(prob)
                break

    # If plate found, draw nice banner
    if best_plate:
        banner = f"DETECTED: {best_plate} ({int(best_conf * 100)}%)"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(annotated, (15, 15), (25 + bw, 45 + bh), (0, 200, 0), -1)
        cv2.putText(annotated, banner, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    b64_image = frame_to_base64(annotated)

    return {
        "found": best_plate is not None,
        "plate_number": best_plate if best_plate else "",
        "confidence": round(best_conf, 2),
        "vehicle_type": detected_vehicle,
        "annotated_image": b64_image,
        "message": f"Plate '{best_plate}' detected successfully!" if best_plate else "No plate detected with high confidence."
    }

def get_sample_images():
    if not os.path.exists(SAMPLE_DIR):
        return []
    valid_exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in os.listdir(SAMPLE_DIR) if os.path.splitext(f)[1].lower() in valid_exts]
    return files[:15]

def get_sample_image_path(filename):
    safe_name = os.path.basename(filename)
    path = os.path.join(SAMPLE_DIR, safe_name)
    if os.path.exists(path):
        return path
    return None
