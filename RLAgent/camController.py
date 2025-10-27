import cv2
import numpy as np
import torch
import serial
from ballfind import get_ball_detection

class CameraControlEnv:
    def __init__(self, cap, detection_model, transform, device, 
                 frame_center_x, frame_center_y, max_action, 
                 reward_weights):
        
        self.cap = cap
        self.detection_model = detection_model
        self.transform = transform
        self.device = device
        
        # Frame and center properties
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FRAME_CENTER = (self.W // 2, self.H // 2)
        
        # Control parameters
        self.max_action = max_action 
        self.reward_weights = reward_weights 

        
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.current_frame = None

        #Training window size
        self.window_width = int(0.25 * self.W)
        self.window_height = self.H
        self.x_position = (self.W - self.window_width) // 2 #
        self.y_position = 0 # 

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, self.current_frame = self.cap.read()
        self.prev_action = np.zeros(2, dtype=np.float32)

        # Get the initial ball error
        ball_x, ball_y, frame_with_detections = self._detect_ball()
        
        # Calculate initial state S0
        dx = (ball_x - self.FRAME_CENTER[0]) / self.W  # Normalized error
        dy = (ball_y - self.FRAME_CENTER[1]) / self.H  # Normalized error
        
        initial_state = np.array([dx, dy, self.prev_action[0], self.prev_action[1]], dtype=np.float32)
        
        return initial_state, frame_with_detections

    def _detect_ball(self):
        if self.current_frame is None:
            return 0.0, 0.0, None

        detected_boxes, frame_with_detections = get_ball_detection(
            self.detection_model, self.current_frame.copy(), self.transform, self.device
        )
        
        if detected_boxes:
            box = detected_boxes[0]['box']
            x_min, y_min, x_max, y_max = box
            
            ball_x = (x_min + x_max) // 2
            ball_y = (y_min + y_max) // 2
        else:
            ball_x = self.FRAME_CENTER[0] 
            ball_y = self.FRAME_CENTER[1]
            
        return ball_x, ball_y, frame_with_detections


    def step(self, action):
        action = np.array(action, dtype=np.float32).flatten()
        
        
        if self.ser:
            # Format and send the command (e.g., "PAN:4.5,TILT:-1.2\n")
            command = f"P:{action[0]:.2f},T:{action[1]:.2f}\n"
            self.ser.write(command.encode('utf-8'))
        
        # Read next frame
        ret, next_frame = self.cap.read()
        if not ret:
            # End of video file
            done = True
            next_state = np.zeros(4, dtype=np.float32) # Zero state on termination
            reward = 0.0
            frame_with_detections = self.current_frame
            return next_state, reward, done, frame_with_detections

        self.current_frame = next_frame
        
        ball_x, ball_y, frame_with_detections = self._detect_ball()
        
        # Calculate the normalized error
        dx = (ball_x - self.FRAME_CENTER[0]) / self.W
        dy = (ball_y - self.FRAME_CENTER[1]) / self.H
        
        next_state = np.array([dx, dy, action[0], action[1]], dtype=np.float32)
        
        c1 = self.reward_weights.get('centering', 10.0)
        c2 = self.reward_weights.get('effort', 0.1)
        c3 = self.reward_weights.get('stability', 20.0)

        # Centering Penalty (primary goal: minimize distance from center)
        R_centering = -c1 * (dx**2 + dy**2)

        R_effort = -c2 * np.sum(np.abs(action))

        acceleration = action - self.prev_action
        R_stability = -c3 * np.sum(np.abs(acceleration))

        reward = R_centering + R_effort + R_stability
    
        self.prev_action = action
        done = False
        return next_state, reward, done, frame_with_detections