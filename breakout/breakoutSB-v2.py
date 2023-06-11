# pip install "stable-baselines3[extra]>=2.0.0a4"

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

import sys

import matplotlib.pyplot as plt
import numpy as np

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=1e4)

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

model.learn(total_timesteps=2e4)

sys.stdout = orig_stdout
f.close()

model.save("a2c_breakout")

f = open('out.txt', 'r')
lines = f.readlines()
rewards = []
for l in lines:
    if "ep_rew_mean" in l:
        rewards.append(float(l.split('|')[2]))


plt.plot(np.arange(2000, 2e4+1, 2000), rewards)
plt.title("Mean episode rewards during training")
plt.ylabel("Mean rewards")
plt.xlabel("Number of timesteps")
plt.savefig("rewards_breakout")

        
        
        
        
        