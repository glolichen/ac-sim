import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import gym_environment
import const
import logging
import json
import argparse
import random
from collections import namedtuple, deque
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="Optimize hyperparameters using Optuna")
parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials to run")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per trial")
args = parser.parse_args()

env = gym_environment.Environment("2r_simple.json")

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
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

def select_action(state, EPS_START, EPS_END, EPS_DECAY):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=const.DEVICE, dtype=torch.long)

def optimize_model(memory, BATCH_SIZE, GAMMA, TAU, optimizer, policy_net, target_net):
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

def objective(trial):
    global steps_done
    steps_done = 0
    BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 32, 256)
    GAMMA = trial.suggest_float('GAMMA', 0.8, 0.999)
    EPS_START = trial.suggest_float('EPS_START', 0.8, 1.0)
    EPS_END = trial.suggest_float('EPS_END', 0.0001, 0.1)
    EPS_DECAY = trial.suggest_int('EPS_DECAY', 100, 1000)
    TAU = trial.suggest_float('TAU', 0.001, 0.1)
    LR = trial.suggest_float('LR', 1e-5, 1e-3, log=True)

    action_size = env.action_space.n
    state, _ = env.reset()
    observation_size = len(state)

    global policy_net, target_net
    policy_net = DQN(observation_size, action_size).to(const.DEVICE)
    target_net = DQN(observation_size, action_size).to(const.DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    total_reward = 0
    for i_episode in range(args.episodes):
        state, _ = env.reset()
        state = state.unsqueeze(0)
        episode_reward = 0

        for t in range(200):
            action = select_action(state, EPS_START, EPS_END, EPS_DECAY)
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

            optimize_model(memory, BATCH_SIZE, GAMMA, TAU, optimizer, policy_net, target_net)

            if terminated:
                break

        total_reward += episode_reward

    return total_reward

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.trials)

best_params = study.best_params
with open("best_hyperparameters.txt", "w") as f:
    f.write(json.dumps(best_params, indent=4))

logging.info("Optimization complete. Best hyperparameters saved to best_hyperparameters.txt")