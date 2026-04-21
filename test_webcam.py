
from ultralytics import YOLO

# 1. Load your newly trained License Plate model
model = YOLO(r"runs\detect\train\weights\best.pt")

# 2. Start the live webcam! (source="0" means your primary camera)
# show=True will automatically open a video window with the bounding boxes
print("Starting Camera... Press 'q' on your keyboard to exit the camera window when done!")
model.predict(source="0", show=True, conf=0.5)
