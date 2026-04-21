from ultralytics import YOLO
import os

def main():
    # Get the absolute path to the dataset
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(project_dir, 'dataset', 'data.yaml')

    # Load a pretrained YOLO model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        device=0,  # Use GPU device 0 (set to 'cpu' if no GPU available)
        patience=20,
        save=True,
        project=os.path.join(project_dir, 'runs'),
        name='detect/train',
        exist_ok=True,
        verbose=True,
        workers=0  # Use 0 workers for Windows compatibility
    )

    # Print training results
    print("Training completed!")
    print(f"Results saved to: {results}")

if __name__ == '__main__':
    main()
