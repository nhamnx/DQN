from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
from baselines_wrappers.dummy_vec_env import DummyVecEnv

from pytorch_wrappers import make_atari_deepmind

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ = 1000
NUM_ENVS = 4

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))
        self.num_actions = env.action_space.n

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        q_values = self(obses_t)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()
        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions
    

    def compute_loss(self, transitions):
        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute Targets
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        return loss
# env = gym.make('Breakout-v0')

# env = gym.make('CartPole-v0')
make_env = lambda: make_atari_deepmind('Breakout-v0')
env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
# env = Subpro

replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)

episode_reward = 0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize replay buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    actions = [env.action_space.sample() for _ in range(NUM_ENVS)]
    new_obses, rews, dones, _ = env.step(actions)

    for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):    
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
    
    obses = new_obses



# Main Training Loop
obs = env.reset()
for step in itertools.count():
    epsilon = np.interp(step*NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    actions = online_net.act(obses, epsilon)

    new_obses, rews, dones, _ = env.step(actions)

    for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):    
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)
    
    obses = new_obses


    transitions = random.sample(replay_buffer, BATCH_SIZE)

    loss = online_net.compute_loss(transition)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Net
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print()
        print('Step:', step)
        print('Avg Rew:', np.mean(rew_buffer))










