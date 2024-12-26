from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model_coordinator import MaskDetectionCoordinator
from model_opencv import MaskDetectionCoordinator as OpenCVCoordinator
import cv2
import numpy as np
import base64
from flask_cors import CORS
import threading
import time

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class MaskDetectionService:
    def __init__(self, yolo_path, cnn_path, opencv_yolo_path, opencv_cnn_path):
        self.cap = None
        self.is_running = False
        self.debug_mode = True
        self.stream_thread = None
        
        # Initialize both coordinators
        self.main_coordinator = MaskDetectionCoordinator(yolo_path, cnn_path)
        self.opencv_coordinator = OpenCVCoordinator(opencv_yolo_path, opencv_cnn_path)
        
        # Flag to switch between coordinators
        self.use_opencv = False
        print("Both model coordinators loaded successfully!")

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.is_running = True
        
        # Start streaming thread if not already running
        if self.stream_thread is None:
            self.stream_thread = threading.Thread(target=self.stream_frames)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
        return {"status": "Camera started"}

    def stop_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.stream_thread = None
        return {"status": "Camera stopped"}

    def toggle_debug(self):
        self.debug_mode = not self.debug_mode
        return {"debug_mode": self.debug_mode}

    def toggle_coordinator(self):
        self.use_opencv = not self.use_opencv
        return {"coordinator": "OpenCV" if self.use_opencv else "Main"}

    def stream_frames(self):
        """Continuous frame streaming function for WebSocket"""
        while self.is_running:
            if not self.cap:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                continue

            try:
                # Use appropriate coordinator based on flag
                coordinator = self.opencv_coordinator if self.use_opencv else self.main_coordinator
                processed_frame, detections = coordinator.process_frame(frame)

                if self.debug_mode:
                    for i, detection in enumerate(detections):
                        if self.use_opencv:
                            # OpenCV coordinator specific detection handling
                            confidence = detection.get('confidence', 0)
                            pred_class = detection.get('class', 'unknown')
                            
                            y_pos = 110 + (i * 60)
                            cv2.putText(processed_frame, 
                                      f"Det{i}: {pred_class} (Conf: {confidence:.2f})",
                                      (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (255, 0, 0), 2)
                            
                            if 'box' in detection:
                                x1, y1, x2, y2 = detection['box']
                                cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                            (int(x2), int(y2)), (0, 255, 0), 2)
                        else:
                            # Main coordinator specific detection handling
                            mask_conf = detection.get('mask_confidence', 0)
                            no_mask_conf = detection.get('no_mask_confidence', 0)
                            pred_class = detection.get('class', 'unknown')
                            
                            y_pos = 110 + (i * 60)
                            cv2.putText(processed_frame, 
                                      f"Det{i}: {pred_class} (Mask: {mask_conf:.2f}, NoMask: {no_mask_conf:.2f})",
                                      (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (255, 0, 0), 2)
                            
                            if 'bbox' in detection:
                                x1, y1, x2, y2 = detection['bbox']
                                cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                            (int(x2), int(y2)), (0, 255, 0), 2)

                # Add coordinator type indicator
                cv2.putText(processed_frame, 
                           f"Using: {'OpenCV' if self.use_opencv else 'Main'} Coordinator",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Detections: {len(detections)}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Emit frame and detections through WebSocket
                socketio.emit('frame_update', {
                    "frame": frame_base64,
                    "detections": detections
                })

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                socketio.emit('error', {"error": str(e)})

            # Control frame rate
            time.sleep(0.033)  # ~30 FPS

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

# Initialize the coordinators and service
model_dir = "src/models"
yolo_path = os.path.join(model_dir, "best.pt")
cnn_path = os.path.join(model_dir, "cnn_mask_classifier.keras")
opencv_yolo_path = os.path.join(model_dir, "best.pt")
opencv_cnn_path = os.path.join(model_dir, "cnn_mask_classifier.keras")

# Initialize service with both coordinators
mask_service = MaskDetectionService(yolo_path, cnn_path, opencv_yolo_path, opencv_cnn_path)

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_stream')
def handle_start_stream():
    return mask_service.start_camera()

@socketio.on('stop_stream')
def handle_stop_stream():
    return mask_service.stop_camera()

@socketio.on('toggle_debug')
def handle_toggle_debug():
    return mask_service.toggle_debug()

@socketio.on('toggle_coordinator')
def handle_toggle_coordinator():
    return mask_service.toggle_coordinator()

# Keep the REST endpoints for compatibility
@app.route("/webcam", methods=["POST"])
def process_webcam_frame():
    try:
        image_data = request.json.get("image", "")
        use_opencv = request.json.get("use_opencv", False)
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
            
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        coordinator = mask_service.opencv_coordinator if use_opencv else mask_service.main_coordinator
        processed_frame, detections = coordinator.process_frame(frame)

        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'detections': detections
        })

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)