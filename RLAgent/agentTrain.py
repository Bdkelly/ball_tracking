from RLAgent import RLAgent
from camController import CameraControlEnv
from albumentations.pytorch import ToTensorV2
import albumentations as A
from utils.models import get_fasterrcnn_model_single_class
import os
import cv2
import numpy
import torch

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']

def vidget(videopth):
    cap = cv2.VideoCapture(videopth)
    if not cap.isOpened():
        print(f"Error: Could not open video file {videopth}")
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap,W,H,


def train_agent(videopth,model_path):
    num_classes = len(GLOBAL_CLASS_NAMES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap,W,H = vidget(videopth)

    state_size = 4
    action_size = 2
    max_action = 5.0

    RWD_Weights = {
        "centering" : 100.0,
        "effort" : 0.1,
        "stability" : 50.0
    }

    agent = RLAgent(state_size,action_size,max_action,device)
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    detection_model = get_fasterrcnn_model_single_class(num_classes).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Detection model file not found at {model_path}")
        return
    detection_model.load_state_dict(torch.load(model_path, map_location=device))
    detection_model.eval()
    #Enviroment
    #To Cam Controller
    env = CameraControlEnv(
        cap=cap, 
        detection_model=detection_model, 
        transform=transform, 
        device=device,
        frame_center_x=W // 2,
        frame_center_y=H // 2,
        max_action=max_action,
        reward_weights=RWD_Weights,
    )