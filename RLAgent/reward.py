import numpy as np

class RewardSystem:
    def __init__(self, reward_weights):
        self.reward_weights = reward_weights
        self.prev_action = np.zeros(1, dtype=np.float32)

    def calculate_reward(self, dx, dy, pan_action, is_detected=True):
        if not is_detected:
             return -self.reward_weights.get('lost_ball_penalty', 100.0)

        c1 = self.reward_weights.get('centering', 100.0)
        c2 = self.reward_weights.get('effort', 1)
        c3 = self.reward_weights.get('stability', 20.0)
        window_bonus = self.reward_weights.get('window_bonus', 10.0)

        R_centering = -c1 * (dx**2 + dy**2)
        R_effort = -c2 * np.abs(pan_action)
        
        acceleration = pan_action - self.prev_action[0]
        R_stability = -c3 * np.abs(acceleration)

        reward = R_centering + R_effort + R_stability

        # Add bonus if ball is within roughly 12.5% of center (window is 25%)
        # dx is normalized by width, so window width 0.25 -> +/- 0.125
        if abs(dx) < 0.125:
            reward += window_bonus

        return reward

    def update_prev_action(self, pan_action):
        self.prev_action[0] = pan_action

    def reset(self):
        self.prev_action = np.zeros(1, dtype=np.float32)
