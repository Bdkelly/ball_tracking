import sys
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, 
    QLabel, QHBoxLayout, QTextEdit, QGroupBox, QSpinBox, QSlider
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QMutex
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import torch
import serial
import serial.tools.list_ports
from utils.models import get_fasterrcnn_model_single_class as fmodel

# Import the main processing and control functions from external files
from utils.vidpro import videorun, init_video_comp 
from utils.ser_con import move_left, move_right

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']

class VideoThread(QThread):
    # This signal emits a processed QImage to the main window for display
    change_pixmap_signal = pyqtSignal(QImage)
    # This signal emits command strings to the main window log display
    command_log_signal = pyqtSignal(str)
    
    # This method initializes the thread's state variables and mutex
    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True
        self.inference_active = False
        self.mutex = QMutex()

        self.ser = None 
        self.command_interval = 1.0 
        self.last_command_time = time.time()
        
    # This method sets up video capture, loads components externally, and delegates the main loop to video_run_loop
    def run(self):
        success, model, transform, ser_connection = init_video_comp(self)
        
        if not success:
            self._run_flag = False
            self.command_log_signal.emit("Initialization failed.")
            return
            
        self.ser = ser_connection 

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.command_log_signal.emit("Error: Could not open video stream.")
            self._run_flag = False
            return
            
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        videorun(self, cap, W, H, model, transform, device, self.ser)

    def stop(self):
        self._run_flag = False
        self.wait()
    
    # This slot receives the toggle state from the GUI and updates the inference flag safely
    @pyqtSlot(bool)
    def toggle_inference(self, state):
        self.mutex.lock()
        self.inference_active = state
        self.mutex.unlock()
        self.command_log_signal.emit(f"--- Inference {'STARTED' if state else 'STOPPED'} ---")

    # This slot receives the new command interval from the GUI and updates the interval variable safely
    @pyqtSlot(float)
    def set_command_interval(self, interval):
        self.mutex.lock()
        self.command_interval = interval
        self.mutex.unlock()
        self.command_log_signal.emit(f"Command Interval set to: {interval:.2f}s")
        
    # This method converts an OpenCV image (numpy array) into a scaled QImage for display
    def _convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return p


#Main Window 
class MainWindow(QMainWindow):
    #This signal is emitted to toggle the inference status in the VideoThread
    inference_toggle_signal = pyqtSignal(bool)
    # This signal is emitted to send the new command interval to the VideoThread
    command_interval_update_signal = pyqtSignal(float) 

    # This method initializes the thread and calls helper methods to set up the GUI
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTorch Ball Tracker GUI")
        self.setGeometry(100, 100, 950, 650) 
        
        self.thread = VideoThread() 
        
        self._create_widgets()
        self._setup_layout()
        self._connect_signals()
        
        self.thread.start()

    #This method creates all individual UI components (buttons, labels, logs, etc.)
    def _create_widgets(self):
        self.image_label = QLabel("Video Feed will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480)
        
        self.control_button = QPushButton("Start Inference")
        self.control_button.setCheckable(True)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumWidth(250)
        self.log_display.setMaximumHeight(550)
        self.log_display.setFont(QFont('Courier', 10))
        self.log_display.setText("--- ESP32 Command Log ---\n")
        
        self.settings_group = QGroupBox("Tracking Settings")
        
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(1, 20)
        self.interval_slider.setValue(10)
        
        self.interval_label = QLabel(f"Command Interval: {self.interval_slider.value() / 10.0}s")
        
        self.manual_left_button = QPushButton("Manual Left")
        self.manual_right_button = QPushButton("Manual Right")


    #This method organizes the UI components into layouts and sets the central widget
    def _setup_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_h_layout = QHBoxLayout(central_widget)
        
        video_control_v_layout = QVBoxLayout()
        video_control_v_layout.addWidget(self.image_label)
        video_control_v_layout.addWidget(self.control_button)
        main_h_layout.addLayout(video_control_v_layout)
        
        right_v_layout = QVBoxLayout()
        
        settings_layout = QVBoxLayout(self.settings_group)
        settings_layout.addWidget(self.interval_label)
        settings_layout.addWidget(self.interval_slider)
        
        manual_h_layout = QHBoxLayout()
        manual_h_layout.addWidget(self.manual_left_button)
        manual_h_layout.addWidget(self.manual_right_button)
        settings_layout.addLayout(manual_h_layout)

        right_v_layout.addWidget(self.settings_group)
        right_v_layout.addWidget(QLabel("Command Log:"))
        right_v_layout.addWidget(self.log_display)
        
        main_h_layout.addLayout(right_v_layout)

    #This method connects all widget signals to their respective slots and cross-thread signals
    def _connect_signals(self):
        self.control_button.clicked.connect(self.toggle_inference)
        self.interval_slider.valueChanged.connect(self.update_interval_ui)
        self.interval_slider.valueChanged.connect(self.update_interval_thread)

        self.manual_left_button.clicked.connect(lambda: move_left(self.thread.ser))
        self.manual_right_button.clicked.connect(lambda: move_right(self.thread.ser))
        
        self.inference_toggle_signal.connect(self.thread.toggle_inference)
        self.command_interval_update_signal.connect(self.thread.set_command_interval)
        
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.command_log_signal.connect(self.update_log_display)
    
    #This slot receives a comand string from the thread and appends it to the log display
    @pyqtSlot(str)
    def update_log_display(self, command_string):
        self.log_display.append(command_string)

    # This slot receives the processed QImage from the thread and displays it in the QLabel
    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    # This method is called when the control button is clicked and emits the inference toggle signal
    def toggle_inference(self, checked):
        if checked:
            self.control_button.setText("Stop Inference")
            self.inference_toggle_signal.emit(True)
        else:
            self.control_button.setText("Start Inference")
            self.inference_toggle_signal.emit(False)

    # This method updates the text label to reflect the new slider value
    def update_interval_ui(self, value):
        self.interval_label.setText(f"Command Interval: {value / 10.0:.1f}s")
        
    # This method emits the new value to be processed by the VideoThread
    def update_interval_thread(self, value):
        self.command_interval_update_signal.emit(value / 10.0)

    # This method handles the appliation close event by gracefully stopping the worker thread
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

# This block creates the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())