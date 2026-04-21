import cv2
from ultralytics import YOLO
import easyocr
import os

def main():
    # Initialize EasyOCR reader
    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=True)

    # 1. Load the custom trained model
    model_path = r"runs\detect\train\weights\best.pt"
    if not os.path.exists(model_path):
        print(f"Cannot find model at {model_path}")
        return
        
    model = YOLO(model_path)
    print(f"Successfully loaded model from {model_path}")

    # 2. Open an image
    # We will pick one image from the test set
    image_path = r"dataset\test\images\video2_4290_jpg.rf.7c3e622473595124c1e4ad6729c3213b.jpg"
    if not os.path.exists(image_path):
        print(f"Cannot find test image at {image_path}")
        return
        
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error reading image {image_path}")
        return

    print(f"Successfully loaded image {image_path}")

    # 3. Run Inference on the current frame
    results = model(frame, conf=0.25, verbose=False)
    
    # 4. Get the annotated frame directly from YOLO
    annotated_frame = results[0].plot()

    # 5. Crop and OCR
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size > 0:
                ocr_results = reader.readtext(plate_crop)
                for (bbox, text, prob) in ocr_results:
                    if prob > 0.1:
                        print(f"Detected Plate Text: {text} (Confidence: {prob:.2f})")
                        cv2.putText(annotated_frame, text, (x1, y1 - 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    # Save output
    output_path = "output_test_image.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    main()
