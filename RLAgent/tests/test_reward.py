import pytest
import numpy as np
from RLAgent.reward import RewardSystem

def test_reward_calculation():
    reward_system = RewardSystem(reward_weights={
        'centering': 100.0, 'effort': 1.0, 'stability': 20.0,
        'window_bonus': 10.0, 'lost_ball_penalty': 500.0
    })
    
    # Test case 1: Ball is perfectly centered, no effort, no acceleration
    # Should get window_bonus because dx=0 < 0.125
    reward = reward_system.calculate_reward(dx=0, dy=0, pan_action=0, is_detected=True)
    assert reward == 10.0

    # Test case 2: Ball is off-center but inside window (dx=0.1 < 0.125)
    reward = reward_system.calculate_reward(dx=0.1, dy=0.1, pan_action=0.5, is_detected=True)
    expected_reward = -100.0 * (0.1**2 + 0.1**2) - 1.0 * np.abs(0.5) - 20.0 * np.abs(0.5 - 0) + 10.0
    assert np.isclose(reward, expected_reward)

    # Test case 3: Ball is off-center outside window (dx=0.2 > 0.125)
    reward = reward_system.calculate_reward(dx=0.2, dy=0.1, pan_action=0.5, is_detected=True)
    expected_reward = -100.0 * (0.2**2 + 0.1**2) - 1.0 * np.abs(0.5) - 20.0 * np.abs(0.5 - 0)
    # No window bonus
    assert np.isclose(reward, expected_reward)

    # Test case 4: Test stability component
    reward_system.update_prev_action(0.5)
    reward = reward_system.calculate_reward(dx=0.2, dy=0, pan_action=0.2, is_detected=True) # Outside window
    expected_reward = -100.0 * (0.2**2) - 1.0 * np.abs(0.2) - 20.0 * np.abs(0.2 - 0.5)
    assert np.isclose(reward, expected_reward)

    # Test case 5: Lost ball penalty
    reward = reward_system.calculate_reward(dx=0, dy=0, pan_action=0, is_detected=False)
    assert reward == -500.0

def test_reward_reset():
    reward_system = RewardSystem(reward_weights={})
    reward_system.update_prev_action(0.5)
    reward_system.reset()
    assert reward_system.prev_action[0] == 0.0
