import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QMutex
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import torch
import serial
import serial.tools.list_ports
from models import get_fasterrcnn_model_single_class as fmodel

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']

COMMAND_SIGNAL = None 

def move_left(ser):
    """Sends command to ESP32 to move camera left."""
    if ser:
        if COMMAND_SIGNAL: COMMAND_SIGNAL.emit("Left")
        ser.write(b"Left\n") 
        print("Sent command: Left")
    else:
        print("Serial not connected: Left")

def move_right(ser):
    """Sends command to ESP32 to move camera right."""
    if ser:
        if COMMAND_SIGNAL: COMMAND_SIGNAL.emit("Right")
        ser.write(b"Right\n") 
        print("Sent command: Right")
    else:
        print("Serial not connected: Right")

def find_esp32():
    "Edit"
    return "/dev/cu.usbserial-10"

class VideoThread(QThread):
    """Defines the worker thread for video capture and inference."""
    change_pixmap_signal = pyqtSignal(QImage)
    command_log_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initializes thread state, model parameters, and serial context."""
        super().__init__(parent)
        self._run_flag = True
        self.inference_active = False
        self.mutex = QMutex()

        self.model_path = 'fin_comb.pth'
        self.num_classes = len(GLOBAL_CLASS_NAMES)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        
        self.ser = None
        self.command_interval = 1.0 
        self.last_command_time = time.time()
        
    def initialize_components(self):
        """Loads the PyTorch model and establishes the serial connection."""
        global COMMAND_SIGNAL
        COMMAND_SIGNAL = self.command_log_signal

        print(f"Using device: {self.device}")
        
        try:
            self.model = fmodel(self.num_classes).to(self.device)
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            else:
                print(f"Warning: Model file not found at {self.model_path}. Using base model.")
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        self.transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        esp32_port = find_esp32()
        if esp32_port:
            try:
                self.ser = serial.Serial(esp32_port, 115200, timeout=1)
                self.command_log_signal.emit(f" Serial connected to {esp32_port}")
            except Exception as e:
                self.command_log_signal.emit(f" Failed to connect: {e}")
                self.ser = None
        
        return True

    def run(self):
        """Contains the main video capture and frame processing loop."""
        if not self.initialize_components():
            self._run_flag = False
            self.command_log_signal.emit("Initialization failed.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            self._run_flag = False
            return
            
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                frame_to_display = frame.copy()
                
                self.mutex.lock()
                is_inference_active = self.inference_active
                self.mutex.unlock()
                
                if is_inference_active:
                    detected_boxes, frame_with_detections = self._get_ball_detection(
                        self.model, frame_to_display, self.transform, self.device
                    )

                    if detected_boxes and self.ser: 
                        current_time = time.time()
                        if current_time - self.last_command_time >= self.command_interval:
                            ball_box = detected_boxes[0]['box']
                            ball_center_x = (ball_box[0] + ball_box[2]) / 2
                            screen_center_x = W / 2
                            
                            if ball_center_x < screen_center_x - 50:
                                move_left(self.ser)
                            elif ball_center_x > screen_center_x + 50:
                                move_right(self.ser)
                            else:
                                if self.ser: 
                                    self.ser.write(b"Stop\n") 
                                    if COMMAND_SIGNAL: COMMAND_SIGNAL.emit("Stop")
                                    print("Sent command: Stop")

                            self.last_command_time = current_time
                    
                    frame_to_display = frame_with_detections
                
                qt_image = self._convert_cv_qt(frame_to_display)
                self.change_pixmap_signal.emit(qt_image)
            
            time.sleep(0.03)

        cap.release()
        if self.ser:
            self.ser.close() 
            print("Serial connection closed.")
        print("VideoThread finished.")

    def stop(self):
        """Sets the internal flag to stop the run loop and waits for the thread to exit."""
        self._run_flag = False
        self.wait()
    
    @pyqtSlot(bool)
    def toggle_inference(self, state):
        """Slot to enable or disable inference and tracking logic."""
        self.mutex.lock()
        self.inference_active = state
        self.mutex.unlock()
        self.command_log_signal.emit(f"--- Inference {'STARTED' if state else 'STOPPED'} ---")

    def _convert_cv_qt(self, cv_img):
        """Converts an OpenCV BGR NumPy array into a QPixmap for display in the GUI."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return p

    def _get_ball_detection(self, model, frame, transform, device, confidence_threshold=0.98):
        """Performs ball detection using the PyTorch model and draws the bounding box on the frame."""
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

class MainWindow(QMainWindow):
    """Defines the main window and centralizes all UI elements."""
    inference_toggle_signal = pyqtSignal(bool)

    def __init__(self):
        """Initializes the UI components, layout, and connects signals to the worker thread."""
        super().__init__()
        self.setWindowTitle("PyTorch Ball Tracker GUI")
        self.setGeometry(100, 100, 950, 650) 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_h_layout = QHBoxLayout(central_widget)
        
        video_control_v_layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setText("Video Feed will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480)
        video_control_v_layout.addWidget(self.image_label)
        
        self.control_button = QPushButton("Start Inference")
        self.control_button.setCheckable(True)
        self.control_button.clicked.connect(self.toggle_inference)
        video_control_v_layout.addWidget(self.control_button)
        
        main_h_layout.addLayout(video_control_v_layout)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumWidth(250)
        self.log_display.setMaximumHeight(550)
        self.log_display.setFont(QFont('Courier', 10))
        self.log_display.setText("--- ESP32 Command Log ---\n")
        main_h_layout.addWidget(self.log_display)


        self.thread = VideoThread()
        
        self.thread.change_pixmap_signal.connect(self.update_image)
        
        self.thread.command_log_signal.connect(self.update_log_display)
        
        self.inference_toggle_signal.connect(self.thread.toggle_inference)
        
        self.thread.start()

    @pyqtSlot(str)
    def update_log_display(self, command_string):
        """Slot that appends the received command string to the text display."""
        self.log_display.append(command_string)

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        """Slot that receives the processed frame from the worker thread and displays it."""
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def toggle_inference(self, checked):
        """Toggles the text of the button and emits a signal to start or stop inference in the thread."""
        if checked:
            self.control_button.setText("Stop Inference")
            self.inference_toggle_signal.emit(True)
        else:
            self.control_button.setText("Start Inference")
            self.inference_toggle_signal.emit(False)

    def closeEvent(self, event):
        """Handles graceful application shutdown by stopping the worker thread."""
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())