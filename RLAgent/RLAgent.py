import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

from ActorNet import Actor
from CriticNet import Critic

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class RLAgent:
    def __init__(self, state_size, action_size, max_action, device):
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.device = device
        self.gamma = 0.99
        self.softup = 0.005
        #Actor
        self.actor_local = Actor(state_size, action_size, max_action).to(device)
        self.actor_target = Actor(state_size, action_size, max_action).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        #Critic
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        #Replay
        self.memory = deque(maxlen=20000)
        self.batch_size = 5000

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy().squeeze()
        self.actor_local.train()
            
        return np.clip(action, -self.max_action, self.max_action)
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.softup * local_param.data + (1.0 - self.softup) * target_param.data)
    
    def add_experience(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float() 
        reward = torch.tensor([reward], dtype=torch.float)
        next_state = torch.from_numpy(next_state).float()
        done = torch.tensor([done], dtype=torch.float)

        self.memory.append(Transition(state, action, reward, next_state, done))
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.stack(batch.state).to(self.device)
        actions = torch.stack(batch.action).to(self.device)
        rewards = torch.stack(batch.reward).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.stack(batch.done).to(self.device).float()
        #For Critic
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #For Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        #For Target
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)