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

INDIAN_STATES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 'HP', 'HR',
    'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR',
    'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB', 'BH'
}

CHAR_TO_NUM = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6',
    'T': '7',
    'A': '4'
}

NUM_TO_CHAR = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    '6': 'G',
    '7': 'T',
    '4': 'A'
}

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

def clean_alphanumeric(text):
    return "".join(c for c in text if c.isalnum()).upper()

def is_valid_indian_plate(text):
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$'
    return bool(re.match(pattern, text))

def correct_and_score_plate(raw_text, base_conf=0.5):
    """
    Applies domain-specific Indian license plate formatting and error correction.
    Returns (cleaned_plate, score, is_valid_format).
    """
    clean = clean_alphanumeric(raw_text)
    if len(clean) < 6:
        return clean, int(base_conf * 20), False

    chars = list(clean)
    n = len(chars)

    # 1. First 2 characters must be State Code (letters)
    for i in [0, 1]:
        if i < n and chars[i] in NUM_TO_CHAR:
            chars[i] = NUM_TO_CHAR[chars[i]]

    # 2. Next 1 or 2 characters must be District Code (digits)
    if n > 2 and chars[2] in CHAR_TO_NUM:
        chars[2] = CHAR_TO_NUM[chars[2]]
    if n > 3 and chars[3] in CHAR_TO_NUM and not (n >= 9 and chars[3].isalpha() and chars[4].isalpha()):
        chars[3] = CHAR_TO_NUM[chars[3]]

    # 3. Last 4 characters must be registration number (digits)
    start_num_idx = max(4, n - 4)
    for i in range(start_num_idx, n):
        if chars[i] in CHAR_TO_NUM:
            chars[i] = CHAR_TO_NUM[chars[i]]

    candidate = "".join(chars)

    # Calculate match quality score
    score = int(base_conf * 40)
    is_valid = False

    if is_valid_indian_plate(candidate):
        score += 50
        is_valid = True
        state_code = candidate[:2]
        if state_code in INDIAN_STATES:
            score += 40
    elif 8 <= len(candidate) <= 10:
        score += 20
        if candidate[:2] in INDIAN_STATES:
            score += 25

    return candidate, score, is_valid

def sort_ocr_tokens(ocr_items, crop_w, crop_h):
    """
    Sorts OCR text tokens based on spatial layout.
    Horizontal plates: left-to-right by center_x.
    Tall/square plates: top-to-bottom rows, then left-to-right.
    """
    if not ocr_items:
        return []

    aspect_ratio = crop_w / max(1, crop_h)
    parsed = []

    for bbox, text, prob in ocr_items:
        clean = clean_alphanumeric(text)
        if not clean:
            continue
        cx = sum(p[0] for p in bbox) / 4.0
        cy = sum(p[1] for p in bbox) / 4.0
        parsed.append({
            'text': clean,
            'prob': float(prob),
            'cx': cx,
            'cy': cy
        })

    if aspect_ratio > 2.0:
        # Standard single-row horizontal license plate
        parsed.sort(key=lambda item: item['cx'])
    else:
        # 2-line plate (e.g. motorcycles or square plates)
        row_height = max(1, crop_h * 0.4)
        parsed.sort(key=lambda item: (item['cy'] // row_height, item['cx']))

    return parsed

def get_enhanced_crops(crop):
    """
    Produces multi-scale and preprocessed variants of the plate crop
    to maximize OCR detection across different lighting and contrast conditions.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return []

    # Target minimum height of 100-120px for clear OCR recognition
    scale = max(2.0, 120.0 / float(h))
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 1. High-resolution RGB (Cubic interpolation)
    rgb_large = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Contrast enhanced CLAHE
    gray = cv2.cvtColor(rgb_large, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(bfilter)

    # 3. Otsu Adaptive Binarization
    _, binarized = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [
        ("rgb", rgb_large, new_w, new_h),
        ("clahe", clahe_enhanced, new_w, new_h),
        ("binarized", binarized, new_w, new_h)
    ]

def frame_to_base64(frame):
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def recognize_plate(image_input):
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

    H, W = frame.shape[:2]
    annotated = frame.copy()
    detected_vehicle = "Car"

    # 1. Detect Vehicle
    v_results = vehicle_model.predict(frame, conf=0.35, verbose=False)
    for res in v_results:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 3, 5, 7]:
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                v_name = res.names[cls_id].title()
                detected_vehicle = v_name
                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (245, 130, 32), 2)
                cv2.putText(annotated, f"Vehicle: {v_name}", (vx1, max(20, vy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 130, 32), 2)

    # 2. Detect Plate Bounding Boxes
    p_results = plate_model.predict(frame, conf=0.18, verbose=False)

    best_candidate = ""
    best_score = -1
    best_conf = 0.0
    best_plate_box = None

    for res in p_results:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)

            # Add context margin to prevent cropping edge characters
            pad_x = int((x2 - x1) * 0.08)
            pad_y = int((y2 - y1) * 0.12)
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(W, x2 + pad_x)
            py2 = min(H, y2 + pad_y)

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            # Run multi-pass OCR on crop variants
            enhanced_variants = get_enhanced_crops(crop)
            for variant_name, img_variant, var_w, var_h in enhanced_variants:
                try:
                    ocr_results = reader.readtext(
                        img_variant,
                        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        paragraph=False
                    )
                except Exception:
                    continue

                if not ocr_results:
                    continue

                sorted_tokens = sort_ocr_tokens(ocr_results, var_w, var_h)
                if not sorted_tokens:
                    continue

                # Candidate 1: Concatenated full plate from sorted tokens
                full_raw = "".join(t['text'] for t in sorted_tokens)
                avg_conf = sum(t['prob'] for t in sorted_tokens) / len(sorted_tokens)

                corrected, score, is_valid = correct_and_score_plate(full_raw, avg_conf)
                if score > best_score and len(corrected) >= 4:
                    best_candidate = corrected
                    best_score = score
                    best_conf = avg_conf
                    best_plate_box = (x1, y1, x2, y2)

                # Candidate 2: Individual tokens if full was too noisy
                for t in sorted_tokens:
                    t_corr, t_score, t_valid = correct_and_score_plate(t['text'], t['prob'])
                    if t_score > best_score and len(t_corr) >= 6:
                        best_candidate = t_corr
                        best_score = t_score
                        best_conf = t['prob']
                        best_plate_box = (x1, y1, x2, y2)

    # 3. Fallback: Full frame OCR if YOLO missed plate
    if not best_candidate or best_score < 40:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_ocr = reader.readtext(gray_full, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        for item in full_ocr:
            text = clean_alphanumeric(item[1])
            corr, score, is_valid = correct_and_score_plate(text, item[2])
            if score > best_score and len(corr) >= 6:
                best_candidate = corr
                best_score = score
                best_conf = float(item[2])

    # 4. Highlight Plate on frame
    if best_plate_box:
        bx1, by1, bx2, by2 = best_plate_box
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 220, 0), 3)

    if best_candidate:
        conf_pct = int(best_conf * 100) if best_conf > 0 else 80
        banner = f"PLATE: {best_candidate} ({conf_pct}%)"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(annotated, (15, 15), (25 + bw, 45 + bh), (0, 200, 0), -1)
        cv2.putText(annotated, banner, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    b64_image = frame_to_base64(annotated)

    return {
        "found": bool(best_candidate),
        "plate_number": best_candidate,
        "confidence": round(best_conf, 2),
        "vehicle_type": detected_vehicle,
        "annotated_image": b64_image,
        "score": best_score,
        "message": f"Detected License Plate: {best_candidate}" if best_candidate else "No license plate recognized."
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
