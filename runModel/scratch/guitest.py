import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2
import numpy as np

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Camera Feed")
        self.setGeometry(100, 100, 700, 600)
        
        # --- 1. Video Capture Setup ---
        self.camera_index = 0  # ⚠️ Change this index if your camera is not at 0 (e.g., 1 or 2)
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}.")
            # Provide feedback to the user and exit gracefully
            self.show_error_message()
            return
        
        # --- 2. UI Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Video Label: This QLabel will display the frames
        self.image_label = QLabel(self)
        self.image_label.setText("Streaming Live Video...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480) 
        self.layout.addWidget(self.image_label)
        
        # --- 3. Timer Setup (Drives the Frame Rate) ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update approximately every 30 milliseconds (~33 FPS)

    def show_error_message(self):
        """Displays a message if camera initialization fails."""
        error_label = QLabel(f"ERROR: Failed to open camera index {self.camera_index}. Check connections.")
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("color: red; font-size: 16pt;")
        self.setCentralWidget(error_label)

    def _convert_cv_qt(self, cv_img):
        """Converts a BGR OpenCV NumPy image array to a QPixmap for display."""
        if cv_img is None or cv_img.size == 0:
            return QImage()
            
        # OpenCV reads BGR, QImage expects RGB, so we convert.
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create QImage from the buffer
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit the QLabel size
        p = QPixmap.fromImage(convert_to_Qt_format.scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
        return p

    def update_frame(self):
        """
        Slot called by the QTimer: reads a frame, converts it, and displays it.
        """
        # Ensure the capture object is still valid
        if not self.cap or not self.cap.isOpened():
            self.timer.stop()
            return

        ret, frame = self.cap.read()
        
        if ret:
            # Display the successful frame
            qt_pixmap = self._convert_cv_qt(frame)
            self.image_label.setPixmap(qt_pixmap)
        else:
            # If read fails, stop the timer
            print("Camera failed to read frame or stream ended.")
            self.timer.stop()

    def closeEvent(self, event):
        """Releases the camera resource and stops the timer when the application closes."""
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    if window.cap and window.cap.isOpened():
        window.show()
        sys.exit(app.exec_())
    else:
        # Exit if camera failed to open during initialization
        sys.exit(1)