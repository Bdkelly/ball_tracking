import pytest
import sys
import numpy as np
from unittest.mock import MagicMock, patch, call
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, Qt

from ..guicore import MainWindow, VideoThread 

@pytest.fixture(scope="session")
def qapp():
    if QApplication.instance() is None:
        return QApplication(sys.argv)
    return QApplication.instance()

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.initialize.return_value = True
    # Mock get_frame to return a successful read and an empty NumPy image
    agent.get_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return agent

@pytest.fixture
@patch('gui_app.BallTrackerAgent')
def video_thread(MockAgent, mock_agent):
    MockAgent.return_value = mock_agent 
    thread = VideoThread(camera_index=5)
    return thread