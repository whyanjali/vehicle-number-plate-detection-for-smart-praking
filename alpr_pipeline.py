import cv2
import numpy as np
import re
from collections import Counter
from ultralytics import YOLO
import easyocr

def is_valid_indian_plate(text):
    """
    Validates if the text matches standard Indian license plate formats.
    Generic Format: [State Code][District Code][Series][Number]
    Example: MH12AB1234, DL3CAB1234
    """
    # Regex for standard Indian Private vehicle plates
    # State(2) + District(2) + Series(1-2) + Num(4)
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$'
    return bool(re.match(pattern, text))

def preprocess_plate(plate_crop):
    """
    Prepares the license plate image for OCR by applying grayscale,
    noise reduction, and contrast enhancement.
    """
    if plate_crop is None or plate_crop.size == 0:
        return None
        
    # 1. Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise reduction (Bilateral filter maintains sharp edges)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # 3. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bfilter)
    
    return enhanced

# Global dictionary to track plate readings across frames
plate_history = {}

def main():
    # Initialize EasyOCR reader (Downloads the model on first run)
    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=True) # Set gpu=False if you see CUDA errors

    # 1. Load the models
    model_path = r"runs\detect\train\weights\best.pt"
    try:
        # custom plate model
        plate_model = YOLO(model_path)
        # general vehicle model (COCO)
        vehicle_model = YOLO("yolov8n.pt") 
        print(f"Successfully loaded models.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 2. Open Video Capture (0 for your default laptop webcam)
    # To test on an existing video, change 0 to the video path e.g., "test_video.mp4"
    video_source = 0 
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    print("Starting Real-Time License Plate Detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
            
        # 3. Run Detection and Tracking
        # Detect vehicles (cars, trucks, etc.)
        v_results = vehicle_model.predict(frame, conf=0.4, verbose=False)
        # Detect/Track license plates
        # Decreased confidence to 0.25 to catch plates that might not be very high confidence yet
        p_results = plate_model.track(frame, persist=True, conf=0.25, verbose=False)
        
        # 4. Preparation for display
        annotated_frame = frame.copy()

        # --- VEHICLE DETECTION (Draw first so plates are on top) ---
        for res in v_results:
            v_boxes = res.boxes
            for v_box in v_boxes:
                cls = int(v_box.cls[0])
                # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                if cls in [2, 3, 5, 7]:
                    vx1, vy1, vx2, vy2 = map(int, v_box.xyxy[0])
                    v_name = res.names[cls].upper()
                    cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, v_name, (vx1, vy1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- LICENSE PLATE DETECTION ---
        # We loop through all detected boxes to crop out the license plate
        for result in p_results:
            boxes = result.boxes
            for box in boxes:
                # Get the pixel coordinates of the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw a default rectangle to show YOLO detected a license plate
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                
                # Crop the license plate region using Numpy indexing
                plate_crop = frame[y1:y2, x1:x2]
                
                # Show the cropped plate in a separate smaller window
                if plate_crop.size > 0:
                    # --- IMPROVEMENT: Upscale and Preprocess ---
                    # Upscale if the crop is too small (helps OCR)
                    height, width = plate_crop.shape[:2]
                    if height < 60:
                        scaling_factor = 2
                        plate_crop = cv2.resize(plate_crop, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
                    
                    processed_plate = preprocess_plate(plate_crop)
                    
                    if processed_plate is not None:
                        cv2.imshow("Processed Plate (OCR Input)", processed_plate)
                    
                    # --- NEW: Extract Text using EasyOCR with Tracking ---
                    # We use an allowlist to only look for standard alphanumeric characters
                    ocr_results = reader.readtext(processed_plate, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    
                    for (bbox, text, prob) in ocr_results:
                        # Clean the text: remove spaces and special characters
                        clean_text = "".join(e for e in text if e.isalnum()).upper()
                        
                        # DEBUG: Print everything the AI sees so we know why it's not "locking"
                        if len(clean_text) > 2:
                            print(f"[DEBUG] Raw OCR: {clean_text} (Conf: {prob:.2f})")
                            # Show raw text in orange while attempting to verify
                            cv2.putText(annotated_frame, f"OCR: {clean_text}", (x1, y2 + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                        # Add to tracking history if it looks like a valid plate
                        if box.id is not None:
                            track_id = int(box.id[0])
                            
                            # Slightly relaxed pattern for validation
                            if prob > 0.25 and (is_valid_indian_plate(clean_text) or len(clean_text) >= 8):
                                if track_id not in plate_history:
                                    plate_history[track_id] = []
                                plate_history[track_id].append(clean_text)
                                if len(plate_history[track_id]) > 15:
                                    plate_history[track_id].pop(0)

                            # --- CONSENSUS LOGIC ---
                            if track_id in plate_history and plate_history[track_id]:
                                consensus_text, count = Counter(plate_history[track_id]).most_common(1)[0]
                                
                                # RELAXED: Lock after 3 successful readings instead of 5
                                if count >= 3:
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    
                                    # Draw the "Verified" text with a nice background
                                    label = f"ID:{track_id} | {consensus_text}"
                                    print(f"LOCKED PLATE (ID {track_id}): {consensus_text}")
                                    
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.8
                                    thickness = 2
                                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                                    
                                    cv2.rectangle(annotated_frame, (x1, y1 - text_h - 20), (x1 + text_w, y1), (0, 255, 0), -1)
                                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # 5. Display the main frame with bounding boxes and text
        cv2.imshow("ALPR Real-Time Pipeline", annotated_frame)
        
        # 6. Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
