import cv2
from yolo_predict import detect_faces
from cnn_predict import classify_face

def combined_predict(image_path, yolo_model="./models/yolov8_mask.pt", cnn_model="./models/cnn_mask_classifier.h5"):
    image = cv2.imread(image_path)
    detections = detect_faces(image_path, yolo_model)

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        cropped_face = image[y1:y2, x1:x2]

        # Classify the cropped face
        label = classify_face(cropped_face, cnn_model)
        detection["classification"] = label

    return detections

if __name__ == "__main__":
    result = combined_predict("./images/maksssksksss321.png")
    print(result)
