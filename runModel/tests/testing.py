import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import serial
import serial.tools.list_ports
from trackTest import move_left, move_right, get_ball_detection, main

# Serial Communication 

def test_move_left_sends_command():
    mock_ser = MagicMock()
    move_left(mock_ser)
    mock_ser.write.assert_called_once_with(b"Left")

def test_move_right_sends_command():
    mock_ser = MagicMock()
    move_right(mock_ser)
    mock_ser.write.assert_called_once_with(b"Right")

@patch('builtins.print')
def test_move_left_handles_serial_error(mock_print):
    mock_ser = MagicMock()
    mock_ser.write.side_effect = serial.SerialException("Error")
    move_left(mock_ser)
    mock_print.assert_called()

#  Detection Logic
def get_mock_predictions(boxes, labels, scores):
    return [{'boxes': torch.tensor(boxes), 
             'labels': torch.tensor(labels), 
             'scores': torch.tensor(scores)}]

@patch('torch.no_grad')
@patch('cv2.rectangle')
@patch('cv2.putText')
@patch('cv2.cvtColor', return_value=np.zeros((720, 1280, 3), dtype=np.uint8))
def test_get_ball_detection_successful(mock_cvt, mock_puttext, mock_rect, mock_nograd):
    mock_model = MagicMock()
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_transform = MagicMock()
    mock_transform.return_value = {'image': torch.zeros((3, 640, 640))}
    device = torch.device("cpu")

    mock_model.return_value = get_mock_predictions(
        boxes=[[0., 0., 100., 100.]],
        labels=[1],
        scores=[0.99]
    )
    
    detected_boxes, _ = get_ball_detection(
        mock_model, mock_frame, mock_transform, device, confidence_threshold=0.98
    )
    
    assert len(detected_boxes) == 1
    assert detected_boxes[0]['label'] == 'Ball'
    mock_rect.assert_called_once()


@patch('torch.no_grad')
@patch('cv2.rectangle')
@patch('cv2.putText')
@patch('cv2.cvtColor', return_value=np.zeros((720, 1280, 3), dtype=np.uint8))
def test_get_ball_detection_no_high_confidence(mock_cvt, mock_puttext, mock_rect, mock_nograd):
    mock_model = MagicMock()
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_transform = MagicMock()
    mock_transform.return_value = {'image': torch.zeros((3, 640, 640))}
    device = torch.device("cpu")

    mock_model.return_value = get_mock_predictions(
        boxes=[[0., 0., 100., 100.]],
        labels=[1],
        scores=[0.50]
    )
    
    detected_boxes, _ = get_ball_detection(
        mock_model, mock_frame, mock_transform, device, confidence_threshold=0.98
    )
    
    assert len(detected_boxes) == 0
    mock_rect.assert_not_called()

# Main Loop Logic

@patch('tracker_agent.get_ball_detection')
@patch('tracker_agent.find_esp32', return_value='COM_MOCK')
@patch('serial.Serial')
@patch('cv2.VideoCapture')
@patch('cv2.imshow')
@patch('cv2.waitKey', side_effect=[ord('a'), ord('q')])
@patch('tracker_agent.move_left')
@patch('tracker_agent.move_right')
@patch('torch.cuda.is_available', return_value=False)
@patch('torch.load')
@patch('torch.nn.Module.load_state_dict')
@patch('os.path.exists', return_value=True)
@patch('time.time', side_effect=[0.0, 1.1])
def test_main_logic_moves_left(mock_time, *args):
    # Setup mocks
    FRAME_WIDTH = 640
    mock_move_left = args[2]
    mock_serial = args[5].return_value
    mock_cap = args[6].return_value
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [FRAME_WIDTH, 480]
    mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)]
    
    mock_get_detection = args[7]
    mock_get_detection.return_value = ([{'box': (50, 50, 150, 150), 'label': 'Ball', 'score': 0.99}], MagicMock())
    
    main()

    mock_move_left.assert_called_once()
    mock_serial.close.assert_called_once()

@patch('tracker_agent.get_ball_detection')
@patch('tracker_agent.find_esp32', return_value='COM_MOCK')
@patch('serial.Serial')
@patch('cv2.VideoCapture')
@patch('cv2.imshow')
@patch('cv2.waitKey', side_effect=[ord('a'), ord('q')])
@patch('tracker_agent.move_left')
@patch('tracker_agent.move_right')
@patch('torch.cuda.is_available', return_value=False)
@patch('torch.load')
@patch('torch.nn.Module.load_state_dict')
@patch('os.path.exists', return_value=True)
@patch('time.time', side_effect=[0.0, 1.1])
def test_main_logic_stays_in_deadzone(mock_time, *args):
    # Setup mocks
    FRAME_WIDTH = 640
    mock_move_left = args[2]
    mock_move_right = args[3]
    mock_serial = args[5].return_value
    mock_cap = args[6].return_value
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [FRAME_WIDTH, 480]
    mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)]
    
    # Box in center (center X = 320, within [270, 370])
    mock_get_detection = args[7]
    mock_get_detection.return_value = ([{'box': (300, 300, 340, 340), 'label': 'Ball', 'score': 0.99}], MagicMock())
    
    main()

    mock_move_left.assert_not_called()
    mock_move_right.assert_not_called()
    mock_serial.write.assert_called_once_with(b"Stop")