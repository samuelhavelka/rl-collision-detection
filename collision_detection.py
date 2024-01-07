import matplotlib.pyplot as plt
from math import *

from stable_baselines3 import TD3, A2C, DQN

from environment import Env
from robot import Robot


plt.ion()

def make_env():
    # create environment with a set of obstacles
    env = Env(world_size=[40,10], 
          obstacles=[[5,1],[10,4],[15,7],[20,5],[30,3],[35,7],[25,7]])
    robot = Robot(env=env,initial_loc=[1,5], inital_orient=0,
                  robot_dims=[0.8,0.8])
    return env


if __name__ == "__main__":
    env = make_env()

    # load trained model
    model = DQN.load(f"models\dqn_collision_model.zip")
    model.set_env(env)

    vec_env = model.get_env()
    obs = vec_env.reset()

    # run simulation with trained RL model
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")