import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from math import *

from environment import Env
from robot import Robot

import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import TD3, A2C, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import Logger

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

plt.ion()

def make_env():
    # create environment with a set of obstacles
    env = Env(world_size=[40,10], 
          obstacles=[[5,1],[10,4],[15,7],[20,5],[30,3],[35,7],[25,7]])
    robot = Robot(env=env,initial_loc=[1,5], inital_orient=0)
    return env


if __name__ == "__main__":
    env = make_env()

    # define RL model
    model = DQN("MultiInputPolicy", env, verbose=1, device='cuda', 
                tensorboard_log="./dqn_collision_tensorboard/",
                buffer_size=1_000_000, learning_starts=100_000, 
                exploration_fraction=0.1)    
    
    # Save a checkpoint every n steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./logs/",
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
        )

    # train model
    model.learn(total_timesteps=2_000_000, log_interval=100,
                progress_bar=True, callback=checkpoint_callback, 
                reset_num_timesteps=False)
    # save final model
    model.save(f"models\dqn_collision_model")

    vec_env = model.get_env()
    obs = vec_env.reset()

    # training is done, wait for user input to run simulation with trained RL model
    input("Training Complete...")
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")