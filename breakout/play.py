import gym
from gym.utils.play import play
from pytorch_wrappers import *
from baselines_wrappers import *

play(make_atari_deepmind('SeaquestNoFrameskip-v4', scale_values=True), zoom=3)