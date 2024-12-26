import cv2
import time
from model_opencv import MaskDetectionCoordinator

def main():
    # Initialize the mask detection coordinator
    yolo_path = "models/yolov8_mask.pt"
    cnn_path = "models/cnn_mask_classifier.keras"
    
    try:
        coordinator = MaskDetectionCoordinator(yolo_path, cnn_path)
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Add debug mode flag
    debug_mode = True
    
    print("Starting real-time detection... Press 'q' to quit, 'd' to toggle debug mode")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from webcam")
            break
            
        # Process frame with mask detection
        try:
            processed_frame, detections = coordinator.process_frame(frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 30:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Debug information
            if debug_mode:
                for i, detection in enumerate(detections):
                    # Extract confidence scores and predictions
                    mask_conf = detection.get('mask_confidence', 0)
                    no_mask_conf = detection.get('no_mask_confidence', 0)
                    pred_class = detection.get('class', 'unknown')
                    
                    # Draw detailed debug info for each detection
                    y_pos = 110 + (i * 60)
                    cv2.putText(processed_frame, 
                              f"Det{i}: {pred_class} (Mask: {mask_conf:.2f}, NoMask: {no_mask_conf:.2f})",
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                              0.6, (255, 0, 0), 2)
                    
                    # Draw bounding box with confidence
                    if 'bbox' in detection:
                        x1, y1, x2, y2 = detection['bbox']
                        cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                    (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Add confidence values inside box
                        cv2.putText(processed_frame,
                                  f"M:{mask_conf:.2f} NM:{no_mask_conf:.2f}",
                                  (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 255, 0), 2)
            
            # Always display basic info
            cv2.putText(processed_frame, f"FPS: {fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Detections: {len(detections)}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
            
            # Display the processed frame
            cv2.imshow('Face Mask Detection', processed_frame)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            cv2.imshow('Face Mask Detection', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")