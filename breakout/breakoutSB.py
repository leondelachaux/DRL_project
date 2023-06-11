# pip install "stable-baselines3[extra]>=2.0.0a4"

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# There already exists an environment generator that will make and wrap atari environments correctly.
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1e7)

model.save("a2c_pong")