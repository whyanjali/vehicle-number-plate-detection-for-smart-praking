# Vehicle Number Plate Detection for Smart Parking 🚗🅿️

A robust Real-Time Automatic License Plate Recognition (ALPR) system designed to detect vehicles and their license plates, and accurately extract alphanumeric characters using a multi-stage Deep Learning pipeline. 

This system is built specifically for identifying standard Indian license plates to facilitate seamless entry, exit, and tracking applications in smart parking environments.

---

## 🚀 How the System Works

The ALPR pipeline (`alpr_pipeline.py`) processes video feeds sequentially in real-time, relying on the following steps:

### 1. Vehicle and Plate Detection (YOLOv8)
The frame is first processed by dual YOLOv8 models:
- **Vehicle Model**: Uses the pre-trained `yolov8n.pt` model to detect bounding boxes for cars, buses, and motorcycles.
- **Custom Plate Model**: Uses a specifically trained custom model (`best.pt` located in `runs/detect/train/weights/`) dedicated to tracking and locating bounding boxes strictly for the license plates within those vehicles.

### 2. Plate Isolation and Image Enhancement
Once a license plate bounding box is identified, that region is cropped out from the main frame. Before sending it to the OCR engine, the image goes through an enhancement pipeline to maximize character readability:
- **Rescaling**: If the cropped image is low resolution (height < 60px), it is enlarged using cubic interpolation.
- **Grayscaling**: Removes colors to rely strictly on light thresholds.
- **Noise Reduction**: A bilateral filter smooths out noise while strictly preserving the sharp edges of alphanumeric characters.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Further enhances the image contrast, crucial for plates suffering from poor lighting or glare.

### 3. OCR Extraction (EasyOCR)
The pre-processed plate image is passed to `EasyOCR`. Our configuration sets a strictly enforced alphanumeric allowlist (`0-9`, `A-Z`) to ignore irrelevant symbols or dirt that might throw off the prediction.

### 4. Format Validation
The system includes validation to ensure extracted text resembles an actual Indian license plate:
- We test the extracted string against a Regular Expression: `^[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{4}$`.
- Example of matched patterns: `MH12AB1234`, `DL3CAB1234`.

### 5. Tracking & Consensus Logic (Anti-Flicker Mechanism)
Because OCR on real-time video can be sporadic and inconsistent between moving frames, the system implements **Consensus Tracking**:
- A dictionary (`plate_history`) keeps a running log of readings for every tracked object ID.
- The pipeline waits until it sees the exact same alphanumeric string **3 times** for the same tracked ID before officially locking it in.
- This effectively solves flickering, drastically reducing false positives and garbled output text. Once a threshold is reached, a solid green confirmation bounding box is overlaid on the video frame.

---

## 🛠️ Tech Stack Used

- **YOLOv8 (Ultralytics)**: Object detection and tracking.
- **OpenCV**: Computer Vision algorithms and Image preprocessing. 
- **EasyOCR**: Fast Optical character reading.
- **Python / NumPy**: Core processing logic.

## 💻 Running the System

To start the pipeline on your default webcam:
```bash
python alpr_pipeline.py
```
> *To change to a pre-recorded video stream, update the `video_source` variable in `alpr_pipeline.py`.*

*The `Q` key can be pressed entirely to terminate the real-time webcam session.*
