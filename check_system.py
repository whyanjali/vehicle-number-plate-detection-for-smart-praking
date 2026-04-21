import cv2
import os
import sys

def check():
    print("--- ALPR Pre-flight Check ---")
    
    # 1. Check Virtual Env
    if 'yolov8_env' not in sys.executable:
        print("[!] WARNING: You are NOT running this inside 'yolov8_env'. Stop and use the correct python path.")
    else:
        print("[OK] Virtual Environment is active.")

    # 2. Check Models
    model_path = r"runs\detect\train\weights\best.pt"
    if os.path.exists(model_path):
        print(f"[OK] License Plate model found at: {model_path}")
    else:
        print(f"[ERROR] License Plate model NOT found at {model_path}. Check your 'runs' folder.")

    # 3. Check Webcam
    print("Checking Webcam access...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        print("[OK] Webcam is available and accessible.")
        cap.release()
    else:
        print("[ERROR] Webcam is NOT accessible. Ensure NO other app (Zoom, Meet, Camera) is using it.")

    # 4. Check Dependencies
    try:
        from ultralytics import YOLO
        import easyocr
        import lapx
        print("[OK] All AI libraries (YOLO, EasyOCR, Lapx) are correctly installed.")
    except ImportError as e:
        print(f"[ERROR] Missing library: {e}. Run 'pip install {str(e).split()[-1]}'")

    print("\nIf all [OK], run: yolov8_env\\Scripts\\python.exe alpr_pipeline.py")

if __name__ == "__main__":
    check()
