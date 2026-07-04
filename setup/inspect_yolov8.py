from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO("yolov8m.pt")

print("=== YOLOv8m Layer Structure ===")

for i, layer in enumerate(model.model.model):
    from_layer = getattr(layer, "f", "-")
    print(f"[{i:2d}] {type(layer).__name__:20s} f={from_layer}")