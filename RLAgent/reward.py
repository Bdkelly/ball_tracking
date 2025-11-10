import numpy as np

class RewardSystem:
    def __init__(self, reward_weights):
        self.reward_weights = reward_weights
        self.prev_action = np.zeros(1, dtype=np.float32)

    def calculate_reward(self, dx, dy, pan_action):
        c1 = self.reward_weights.get('centering', 100.0)
        c2 = self.reward_weights.get('effort', 1)
        c3 = self.reward_weights.get('stability', 20.0)

        R_centering = -c1 * (dx**2 + dy**2)
        R_effort = -c2 * np.abs(pan_action)
        
        acceleration = pan_action - self.prev_action[0]
        R_stability = -c3 * np.abs(acceleration)

        reward = R_centering + R_effort + R_stability
        return reward

    def update_prev_action(self, pan_action):
        self.prev_action[0] = pan_action

    def reset(self):
        self.prev_action = np.zeros(1, dtype=np.float32)
