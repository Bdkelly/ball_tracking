import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256 + action_size, 128) 
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat([xs, action], 1)
        
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value