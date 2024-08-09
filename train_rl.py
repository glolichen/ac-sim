import math
import random
from collections import namedtuple, deque
from itertools import count
import numpy as np
import signal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import gym_environment
import const
import argparse
import sys
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse arguments
parser = argparse.ArgumentParser(prog="TrainRL")
parser.add_argument("episodes", type=int, help="Number of episodes to train")
parser.add_argument("-o", "--output", default="model.pt", help="Output file for the model")
parser.add_argument("-m", "--model", help="Input file for pre-trained model")
parser.add_argument("-s", "--save_interval", type=int, default=100, help="Interval for saving the model")
args = parser.parse_args()

# Initialize environment
env = gym_environment.Environment("2r_simple.json")

# Hyperparameters
BATCH_SIZE = 188
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.001
EPS_DECAY = 800
TAU = 0.001
LR = 1e-4

# Define Transition namedtuple
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
    def __init__(self, size):
        self.memory = deque([], maxlen=size)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(observation_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Initialize networks and optimizer
action_size = env.action_space.n
state, _ = env.reset()
observation_size = len(state)

policy_net = DQN(observation_size, action_size).to(const.DEVICE)
target_net = DQN(observation_size, action_size).to(const.DEVICE)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

# Load pre-trained model if specified
if args.model:
    policy_net.load_state_dict(torch.load(args.model))
    logging.info(f"Loaded pre-trained model from {args.model}")

steps_done = 0
episode_rewards = []
epsilon_values = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    epsilon_values.append(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=const.DEVICE, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=const.DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=const.DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def save_model_and_stats(episode):
    model_filename = f"{args.output}_episode_{episode}.pt"
    torch.save(policy_net.state_dict(), model_filename)
    
    stats = {
        "episode": episode,
        "total_steps": steps_done,
        "epsilon": epsilon_values[-1],
        "episode_rewards": episode_rewards,
        "epsilon_values": epsilon_values
    }
    
    stats_filename = f"stats_episode_{episode}.json"
    with open(stats_filename, 'w') as f:
        json.dump(stats, f)
    
    logging.info(f"Saved model and stats at episode {episode}")

def signal_handler(signal, frame):
    logging.info("Ctrl+C pressed. Saving model and stats before exiting.")
    save_model_and_stats(i_episode)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Training loop
# Training loop
# Training loop
for i_episode in range(args.episodes):
    state, info = env.reset(num_setpoints=random.randint(2, 7))
    state = state.unsqueeze(0)
    episode_reward = 0

    for t in count():
        action = select_action(state)
        ac_status, dampers = env.get_action(action.item())
        next_state, reward, terminated = env.step(ac_status, dampers)

        reward = torch.tensor([reward], device=const.DEVICE)
        episode_reward += reward.item()

        if terminated:
            next_state = None
        else:
            next_state = next_state.clone().detach().to(const.DEVICE).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if terminated:
            episode_rewards.append(episode_reward)
            break

    # Log progress
    if (i_episode + 1) % 10 == 0:
        avg_reward = sum(episode_rewards[-10:]) / 10
        logging.info(f"Episode {i_episode+1}/{args.episodes} - Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon_values[-1]:.4f}")

    # Save model and stats periodically
    if (i_episode + 1) % args.save_interval == 0:
        save_model_and_stats(i_episode + 1)

# Save final model and stats
save_model_and_stats(args.episodes)
logging.info("Training complete")