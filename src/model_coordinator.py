# model_coordinator.py
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import cv2

class MaskDetectionCoordinator:
    def __init__(self, yolo_path, cnn_path):
        self.yolo_model = YOLO(yolo_path)
        self.cnn_model = load_model(cnn_path)
        
        # class mappings to match YOLO yaml file
        self.yolo_classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']
        self.cnn_classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']


        
        #mapping between YOLO and CNN classes
        self.yolo_to_cnn_map = {
            0: 2,  # mask_weared_incorrect -> mask_weared_incorrectly
            1: 0,  # with_mask -> with_mask
            2: 1   # without_mask -> without_mask
        }
        
        # Confidence thresholds
        self.yolo_threshold = 0.35  # increased for better precision
        self.cnn_threshold = 0.65
        
    def preprocess_face(self, face, target_size=(128, 128)):
        """Preprocess face image for CNN classification with improved handling"""
        try:
            if face is None or face.size == 0:
                return None
                
            # add padding around face for better context
            h, w = face.shape[:2]
            pad = int(min(h, w) * 0.1)
            
            # To Create padded image
            padded = cv2.copyMakeBorder(
                face,
                pad, pad, pad, pad,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            
            # To Ensure proper color format
            if len(padded.shape) == 2:
                padded = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
            elif padded.shape[2] == 4:
                padded = cv2.cvtColor(padded, cv2.COLOR_BGRA2RGB)
            elif padded.shape[2] == 3:
                padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            
            # To Normalize and resize
            face_resized = cv2.resize(padded, target_size)
            face_array = face_resized.astype(np.float32) / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            
            return face_array
            
        except Exception as e:
            print(f"Error in face preprocessing: {str(e)}")
            return None
    
    def map_yolo_to_cnn_class(self, yolo_class):
        """Map YOLO class index to CNN class index based on dataset"""
        return self.yolo_to_cnn_map.get(yolo_class, -1)
    
    def ensemble_prediction(self, face, yolo_pred):
        """Combine YOLO and CNN predictions with weighted confidence"""
        try:
            # To Get YOLO prediction
            yolo_class = int(yolo_pred.cls[0])
            yolo_conf = float(yolo_pred.conf[0])
            
            # Only proceed if YOLO confidence is above threshold
            if yolo_conf < self.yolo_threshold:
                return -1, 0.0, None
            
            # To Get CNN prediction
            face_processed = self.preprocess_face(face)
            if face_processed is None:
                return -1, 0.0, None
            
            cnn_predictions = self.cnn_model.predict(face_processed, verbose=0)
            cnn_class = np.argmax(cnn_predictions, axis=1)[0]
            cnn_conf = float(np.max(cnn_predictions))
            
            # Map YOLO class to CNN class space
            mapped_yolo_class = self.map_yolo_to_cnn_class(yolo_class)
            
            # Decision logic for combining predictions
            if mapped_yolo_class == cnn_class:
                # If both models agree, use higher confidence
                final_class = cnn_class
                final_conf = max(yolo_conf, cnn_conf)
            else:
                # If models disagree, use prediction with higher confidence
                if yolo_conf > cnn_conf:
                    final_class = mapped_yolo_class
                    final_conf = yolo_conf
                else:
                    final_class = cnn_class
                    final_conf = cnn_conf
            
            # Apply final confidence threshold
            if final_conf > self.cnn_threshold:
                return final_class, final_conf, self.cnn_classes[final_class]
            
            return -1, 0.0, None
            
        except Exception as e:
            print(f"Error in ensemble prediction: {str(e)}")
            return -1, 0.0, None
    
    def process_frame(self, frame):
        """Process a single frame with both models"""
        try:
            # Get YOLO predictions
            yolo_results = self.yolo_model(frame)
            
            # Process each detection
            processed_frame = frame.copy()
            detections = []
            
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    # Extract face region with padding
                    x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                    
                    # Add padding to face region
                    pad = int(min(x2-x1, y2-y1) * 0.1)
                    y1_pad = max(0, y1 - pad)
                    y2_pad = min(frame.shape[0], y2 + pad)
                    x1_pad = max(0, x1 - pad)
                    x2_pad = min(frame.shape[1], x2 + pad)
                    
                    face = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    # Get ensemble prediction
                    class_idx, confidence, class_name = self.ensemble_prediction(face, box)
                    
                    if class_name:
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'class': class_name,
                            'confidence': confidence
                        })
                        
                        self.draw_detection(processed_frame, 
                                         (x1, y1, x2, y2),
                                         class_name,
                                         confidence)
            
            return processed_frame, detections
            
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            return frame, []
    
    def draw_detection(self, frame, box, class_name, confidence):
        """Draw detection with improved visualization"""
        colors = {
            'with_mask': (0, 255, 0),      # Green
            'without_mask': (0, 0, 255),    # Red
            'mask_weared_incorrect': (0, 0, 0)  # Yellow
        }
        
        x1, y1, x2, y2 = box
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw semi-transparent box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        
        # Draw label with background
        label = f"{class_name}: {confidence*100:.1f}%"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Draw label background
        cv2.rectangle(frame, 
                     (x1, y1 - label_h - 10),
                     (x1 + label_w + 10, y1),
                     color, -1)
        
        # Draw label text
        cv2.putText(frame, label,
                   (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255),
                   1, cv2.LINE_AA)