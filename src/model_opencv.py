import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import cv2

class MaskDetectionCoordinator:
    def __init__(self, yolo_path, cnn_path):
        self.yolo_model = YOLO(yolo_path)
        self.cnn_model = load_model(cnn_path)
        
        # Define consistent class mappings
        self.yolo_classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']
        self.cnn_classes = ['with_mask', 'without_mask', 'mask_weared_incorrectly']
        
        # Fixed mapping between YOLO and CNN classes
        self.yolo_to_cnn_map = {
            0: 2,  # mask_weared_incorrect -> mask_weared_incorrectly
            1: 0,  # with_mask -> with_mask
            2: 1   # without_mask -> without_mask
        }
        
        # Adjusted thresholds
        self.yolo_threshold = 0.45
        self.cnn_threshold = 0.75
        
    def process_frame(self, frame):
        """Process a single frame with both models"""
        try:
            # Get YOLO predictions
            yolo_results = self.yolo_model(frame, verbose=False)
            
            # Process each detection
            processed_frame = frame.copy()
            detections = []
            
            if len(yolo_results) > 0:
                for result in yolo_results:
                    boxes = result.boxes
                    for box in boxes:
                        # Extract face region
                        x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                        face = frame[y1:y2, x1:x2]
                        
                        # Get ensemble prediction
                        class_idx, confidence, class_name = self.ensemble_prediction(face, box)
                        
                        if class_name and confidence > self.cnn_threshold:
                            detection = {
                                'box': (x1, y1, x2, y2),
                                'class': class_name,
                                'confidence': float(confidence)
                            }
                            detections.append(detection)
                            
                            # Draw detection with explicit color mapping
                            color = self.get_color(class_name)
                            self.draw_detection(processed_frame, detection, color)
            
            return processed_frame, detections
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return frame, []

    def get_color(self, class_name):
        """Get color for class"""
        colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255),    # Red
            'mask_weared_incorrectly': (0, 255, 255)  # Yellow
        }
        return colors.get(class_name, (255, 255, 255))  # White as default

    def draw_detection(self, frame, detection, color):
        """Draw detection with improved visibility"""
        try:
            x1, y1, x2, y2 = detection['box']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Draw semi-transparent box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Draw solid border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence*100:.1f}%"
            
            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            
            # Draw label background
            cv2.rectangle(frame, 
                         (x1, y1 - label_h - 10),
                         (x1 + label_w + 10, y1),
                         color, -1)
            
            # Draw label text with black outline for better visibility
            text_pos = (x1 + 5, y1 - 5)
            cv2.putText(frame, label, text_pos,
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2,
                       cv2.LINE_AA)
            cv2.putText(frame, label, text_pos,
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1,
                       cv2.LINE_AA)
            
        except Exception as e:
            print(f"Error in draw_detection: {str(e)}")

    def preprocess_face(self, face, target_size=(128, 128)):
        """Preprocess face image for CNN classification"""
        try:
            if face is None or face.size == 0:
                return None
            
            # Ensure proper color format and resize
            if len(face.shape) == 2:
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            elif face.shape[2] == 4:
                face = cv2.cvtColor(face, cv2.COLOR_BGRA2RGB)
            elif face.shape[2] == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # Resize and normalize
            face_resized = cv2.resize(face, target_size)
            face_array = face_resized.astype(np.float32) / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            
            return face_array
            
        except Exception as e:
            print(f"Error in face preprocessing: {str(e)}")
            return None

    def ensemble_prediction(self, face, yolo_pred):
        """Enhanced ensemble prediction with better handling of 'no mask' cases"""
        try:
            # Get YOLO prediction
            yolo_class = int(yolo_pred.cls[0])
            yolo_conf = float(yolo_pred.conf[0])
            
            if yolo_conf < self.yolo_threshold:
                return -1, 0.0, None
            
            # Get CNN prediction
            face_processed = self.preprocess_face(face)
            if face_processed is None:
                return -1, 0.0, None
            
            cnn_predictions = self.cnn_model.predict(face_processed, verbose=0)
            cnn_class = np.argmax(cnn_predictions, axis=1)[0]
            cnn_conf = float(np.max(cnn_predictions))
            
            # Map YOLO class to CNN class space
            mapped_yolo_class = self.yolo_to_cnn_map[yolo_class]
            
            # Enhanced decision logic
            if mapped_yolo_class == 1 or cnn_class == 1:  # If either model predicts 'without_mask'
                if cnn_predictions[0][1] > 0.8 or yolo_conf > 0.8:
                    return 1, max(cnn_predictions[0][1], yolo_conf), 'without_mask'
            
            # Use weighted ensemble
            weighted_conf = (yolo_conf + cnn_conf) / 2
            if mapped_yolo_class == cnn_class and weighted_conf > self.cnn_threshold:
                return cnn_class, weighted_conf, self.cnn_classes[cnn_class]
            
            # If models disagree, prefer the more confident prediction
            if yolo_conf > cnn_conf + 0.1:
                return mapped_yolo_class, yolo_conf, self.cnn_classes[mapped_yolo_class]
            elif cnn_conf > yolo_conf + 0.1:
                return cnn_class, cnn_conf, self.cnn_classes[cnn_class]
            
            # Default to more conservative prediction
            return 1, max(yolo_conf, cnn_conf), 'without_mask'
            
        except Exception as e:
            print(f"Error in ensemble prediction: {str(e)}")
            return -1, 0.0, None