import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QMutex
import cv2
import time

try:
    from tcore import BallTrackerAgent
except ImportError:
    print("Error: Could not import BallTrackerAgent")
    sys.exit(1)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    
    def __init__(self, camera_index=1, parent=None):
        super().__init__(parent)
        self._run_flag = True
        self.inference_active = False
        self.mutex = QMutex() 
        self.agent = BallTrackerAgent(camera_index)

    def run(self):
        if not self.agent.initialize():
            print("Agent initialization failed. Thread stopping.")
            self._run_flag = False
            
        while self._run_flag:
            self.mutex.lock()
            is_inference_active = self.inference_active
            self.mutex.unlock()
            
            ret, frame = self.agent.get_frame(is_inference_active)
            
            if ret and frame is not None:
                qt_image = self._convert_cv_qt(frame)
                self.change_pixmap_signal.emit(qt_image)
            
            time.sleep(0.033) 
        
        self.agent.cleanup()
        print("VideoThread finished.")

    def stop(self):
        self._run_flag = False
        self.wait()
    
    @pyqtSlot(bool)
    def toggle_inference(self, state):
        self.mutex.lock()
        self.inference_active = state
        self.mutex.unlock()
        print(f"Inference {'STARTED' if state else 'STOPPED'}")

    def _convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return p

class MainWindow(QMainWindow):
    inference_toggle_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ball Tracker GUI")
        
        camera_index = 1 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        self.image_label = QLabel(self)
        self.image_label.setText("Video Feed (Camera Index 1)")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480) 
        self.layout.addWidget(self.image_label)
        
        self.control_button = QPushButton("Start Inference")
        self.control_button.setCheckable(True)
        self.control_button.clicked.connect(self.toggle_inference)
        self.layout.addWidget(self.control_button)

        self.thread = VideoThread(camera_index=camera_index)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.inference_toggle_signal.connect(self.thread.toggle_inference)
        self.thread.start()

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def toggle_inference(self, checked):
        if checked:
            self.control_button.setText("Stop Inference")
            self.inference_toggle_signal.emit(True)
        else:
            self.control_button.setText("Start Inference")
            self.inference_toggle_signal.emit(False)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())