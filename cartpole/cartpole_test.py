import torch
from torch import nn
from network import Network
import gym


env = gym.make('CartPole-v0')
net = torch.load('../models/cartpole_v0.pt')

obs = env.reset()
done = False

while not done:
    action = net.act(obs)

    obs, _, done, _ = env.step(action)
    env.render()
