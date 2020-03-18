import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from trading_session_gym.envs.trading_session_gym import TradingSession

MEAN_REWARD_BOUND = 70
GAMMA = 0
BATCH_SIZE = 100
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_STEPS = 1000
REPLAY_START_SIZE = 10000
EPSILON_DECAY = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones), np.array(next_states)

class DQN(nn.Module):
    """Deep Q-network with target network"""

    def __init__(self, n_inputs, n_outputs):
        super(DQN, self).__init__()
        # network
        self.fc = nn.Sequential(
                    nn.Linear(n_inputs, n_inputs),
                    nn.ReLU(),
                    nn.Linear(n_inputs, n_outputs)
        )


    def forward(self, x):
        x = x.float()
        return self.fc(x)

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu", cuda_async=False, gamma=0):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    if device=="cuda":
        states_v = states_v.cuda(non_blocking=cuda_async)
        next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        actions_v = actions_v.cuda(non_blocking=cuda_async)
        rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        done_mask = done_mask.cuda(non_blocking=cuda_async)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1).long()).squeeze(-1)

    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def train():
    writer = SummaryWriter(comment="-trading_session")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda available")
    else:
        device = torch.device("cpu")
        print("cuda not available")

    env = TradingSession(action_space_config = 'discrete')
    net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    step_idx = 0
    ts_step = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        step_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - step_idx / EPSILON_DECAY)

        reward = agent.play_step(net, epsilon, device=device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (step_idx - ts_step) / (time.time() - ts)
            ts_step = step_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d episodes, mean reward %.3f, eps %.2f, speed %.2f steps/s" % (step_idx, len(total_rewards), mean_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, step_idx)
            writer.add_scalar("speed", speed, step_idx)
            writer.add_scalar("reward_100", mean_reward, step_idx)
            writer.add_scalar("reward", reward, step_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), "model.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d steps!" % step_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if step_idx % SYNC_TARGET_STEPS == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device, cuda_async = True, gamma = GAMMA)
        loss_t.backward()
        optimizer.step()
    writer.close()

if __name__ == '__main__':
    train()
