from RLAgent import RLAgent
from camController import CameraControlEnv
from albumentations.pytorch import ToTensorV2
import albumentations as A
from utils.models import get_fasterrcnn_model_single_class
import os
import cv2
import numpy as np
import torch
from collections import deque 

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']

class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        np.random.seed(seed)
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def vidget(videopth):
    cap = cv2.VideoCapture(videopth)
    if not cap.isOpened():
        print(f"Error: Could not open video file {videopth}")
        return None, 0, 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, W, H


def train_agent(videopth, model_path, num_episodes=100, max_t=500):
    num_classes = len(GLOBAL_CLASS_NAMES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap, W, H = vidget(videopth)

    if cap is None:
        return

    state_size = 3
    action_size = 1
    max_action = 5.0

    RWD_Weights = {
        "centering": 100.0,
        "effort": 0.1,
        "stability": 50.0
    }

    
    agent = RLAgent(state_size, action_size, max_action, device)
    noise = OUNoise(action_size, seed=0) 
    
    
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

    
    print(f"Starting DDPG Training for {num_episodes} episodes...")
    scores_deque = deque(maxlen=100)
    
    for i_episode in range(1, num_episodes + 1):
        state, frame_with_detections = env.reset() 
        noise.reset() 
        score = 0
        
        
        for t in range(max_t):
            
            action = agent.choose_action(state)
            action_with_noise = action + noise.sample()
            action_with_noise = np.clip(action_with_noise, -max_action, max_action)

            
            next_state, reward, done, action_taken, frame_with_detections = env.step(action_with_noise)

            
            agent.add_experience(state, action_with_noise, reward, next_state, done)
            
            
            agent.learn()
            
            
            state = next_state
            score += reward

            
            if len(action_taken) == 1:
                text = f"Pan: {action_taken[0]:.2f}"
            else:
                text = f"Pan: {action_taken[0]:.2f}, Tilt: {action_taken[1]:.2f}"
            cv2.putText(frame_with_detections, text, 
                        (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            
            cv2.imshow('DDPG Tracking (Green Box is Agent Control Window)', frame_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if done:
                break
        
        scores_deque.append(score)
        avg_score = np.mean(scores_deque)

        
        print(f'\rEpisode {i_episode}\tAverage Score (100 eps): {avg_score:.2f}\tScore: {score:.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score (100 eps): {avg_score:.2f}')
            

    cap.release()
    cv2.destroyAllWindows()
    print("\nTraining complete.")
    return agent.actor_local.state_dict()


if __name__ == '__main__':
    
    VIDEO_PATH = r'/Users/Ben/Documents/dever/python/ptorch/data/SmallVideo.mp4' 
    MODEL_PATH = r'/Users/Ben/Documents/dever/python/ptorch/models/fin_comb.pth' 
    
    
    trained_actor_weights = train_agent(VIDEO_PATH, MODEL_PATH)
    print("Actor weights trained.")
    pass
