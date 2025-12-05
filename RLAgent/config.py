import torch
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
# Use relative paths based on the location of this config file
# This assumes the directory structure is project_root/RLAgent/config.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'SmallVideo.mp4')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fin_comb.pth')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Visualization
# Check for HEADLESS environment variable or default to False (GUI enabled)
# Users can set HEADLESS=1 to disable GUI
HEADLESS = os.environ.get('HEADLESS', '0') == '1'

# Model parameters
STATE_SIZE = 4
ACTION_SIZE = 1
MAX_ACTION = 5.0
NUM_CLASSES = 2 

# Training hyperparameters
NUM_EPISODES = 2000
MAX_T = 5000
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
SOFT_UPDATE = 1e-3
BATCH_SIZE = 128
MEMORY_SIZE = 100000

# Noise parameters
NOISE_SIGMA = 0.2
NOISE_THETA = 0.15
NOISE_DECAY = 0.999
NOISE_SIGMA_MIN = 0.01

# Reward weights
RWD_WEIGHTS = {
    "centering": 100.0,
    "effort": 0.1,
    "stability": 50.0,
    "lost_ball_penalty": 100.0,
    "window_bonus": 10.0
}
