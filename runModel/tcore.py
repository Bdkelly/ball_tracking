import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import torch
import serial
import serial.tools.list_ports
import numpy as np
from models import get_fasterrcnn_model_single_class as fmodel 

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']

class BallTrackerAgent:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.ser = None
        self.W = None
        self.H = None
        self.cap = None
        self.confidence_threshold = 0.98
        self.command_interval = 1.0 
        self.last_command_time = time.time()
        self.is_initialized = False

    def _move_left(self):
        try:
            if self.ser:
                self.ser.write(b"Left")
                print("Sent command: Left")
        except serial.SerialException as e:
            print(f"Serial communication error: {e}")

    def _move_right(self):
        try:
            if self.ser:
                self.ser.write(b"Right")
                print("Sent command: Right")
        except serial.SerialException as e:
            print(f"Serial communication error: {e}")

    def _find_esp32(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "USB" in p.description.upper() or "SERIAL" in p.description.upper() or "ESP" in p.description.upper():
                print(f"Found ESP32: {p.device}")
                return p.device
        print("No ESP32 found. Ensure it's connected and drivers are installed.")
        return None

    def initialize(self):
        if self.is_initialized:
            return True

        model_path = 'fin_comb.pth'
        num_classes = len(GLOBAL_CLASS_NAMES)
        try:
            print(f"Using device: {self.device}")
            self.model = fmodel(num_classes).to(self.device)
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}. Running without model check.")
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
            return False

        self.transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        esp32_port = self._find_esp32()
        if esp32_port:
            try:
                self.ser = serial.Serial(esp32_port, 115200, timeout=1)
                print("Successfully connected to ESP32.")
            except serial.SerialException as e:
                print(f"Failed to connect to ESP32: {e}")
                self.ser = None
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open video stream at index {self.camera_index}.")
            self.cap = None
            return False

        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.is_initialized = True
        return True

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.ser:
            self.ser.close()
        print("BallTrackerAgent cleanup complete.")

    def get_frame(self, inference_enabled):
        if not self.cap:
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        frame_with_detections = frame.copy()
        detected_boxes = []

        if inference_enabled and self.model and self.transform:
            detected_boxes, frame_with_detections = self._get_ball_detection(
                self.model, frame_with_detections, self.transform, self.device
            )

            if detected_boxes and self.ser:
                current_time = time.time()
                if current_time - self.last_command_time >= self.command_interval:
                    ball_box = detected_boxes[0]['box']
                    ball_center_x = (ball_box[0] + ball_box[2]) / 2
                    screen_center_x = self.W / 2
                    
                    if ball_center_x < screen_center_x - 50:
                        self._move_left()
                    elif ball_center_x > screen_center_x + 50:
                        self._move_right()
                    else:
                        if self.ser:
                            self.ser.write(b"Stop")
                            print("Sent command: Stop")
                    
                    self.last_command_time = current_time

        return True, frame_with_detections

    def _get_ball_detection(self, model, frame, transform, device, confidence_threshold=0.98):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = transform(image=frame_rgb)
        image_tensor = transformed['image'].to(device).unsqueeze(0)

        with torch.no_grad():
            predictions = model(image_tensor)

        boxes_raw = predictions[0]['boxes']
        labels_raw = predictions[0]['labels']
        scores_raw = predictions[0]['scores']

        keep_indices = torch.where(scores_raw >= confidence_threshold)[0]
        filtered_boxes = boxes_raw[keep_indices].cpu().numpy()
        filtered_labels = labels_raw[keep_indices].cpu().numpy()
        filtered_scores = scores_raw[keep_indices].cpu().numpy()

        original_size = (frame.shape[1], frame.shape[0])
        x_scale = original_size[0] / 640
        y_scale = original_size[1] / 640

        detected_boxes = []
        for i in range(len(filtered_boxes)):
            box = filtered_boxes[i]
            label_idx = filtered_labels[i]
            score = filtered_scores[i]

            x_min_orig = int(box[0] * x_scale)
            y_min_orig = int(box[1] * y_scale)
            x_max_orig = int(box[2] * x_scale)
            y_max_orig = int(box[3] * y_scale)
            
            detected_boxes.append({
                'box': (x_min_orig, y_min_orig, x_max_orig, y_max_orig),
                'label': GLOBAL_CLASS_NAMES[label_idx],
                'score': score
            })

            cv2.rectangle(frame, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (0, 0, 255), 2)
            text = f"{GLOBAL_CLASS_NAMES[label_idx]} {score:.2f}"
            cv2.putText(frame, text, (x_min_orig, y_min_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            break

        return detected_boxes, frame