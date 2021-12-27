import gym
import numpy as np

env = gym.make('BreakoutNoFrameskip-v4')
print(env.action_space.n)