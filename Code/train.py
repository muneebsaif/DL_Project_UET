from ultralytics import settings

# Update a setting
settings.update({"mlflow": True})

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(data="./PPE/dataset.yaml", epochs=500, imgsz=640, device=[0,1])
