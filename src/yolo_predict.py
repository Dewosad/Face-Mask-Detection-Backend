from ultralytics import YOLO

def detect_faces(image_path, model_path="models/yolov8_mask.pt"):
    model = YOLO(model_path)
    results = model(image_path)

    detections = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]  # Bounding box coordinates
        cls = int(result.cls)  # Class ID
        conf = float(result.conf)  # Confidence
        detections.append({
            "box": [x1.item(), y1.item(), x2.item(), y2.item()],
            "class": cls,
            "confidence": conf
        })

    return detections

if __name__ == "__main__":
    image = "./images/test/test_image.png"
    output = detect_faces(image)
    print(output)
