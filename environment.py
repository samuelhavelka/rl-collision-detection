import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import random
from math import *

import gymnasium as gym
from gymnasium import spaces

PI = 3.14159265359

class Env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, world_size=[85,13], 
                 bocces=None, pallino=None, obstacles=None):

        self.world_size = world_size
        self.x = world_size[0]
        self.y = world_size[1]
        self.step_count = 0
        self.k_obstacles = 8

        # define action space
        self.action_space = spaces.Discrete(n=5)
        # define observation space, multiinput Dict observation space
        self.observation_space = spaces.Dict({'position' : spaces.Box(low=np.array([0, 0]), high=np.array([1,1])),
                                              'orientation' : spaces.Box(low=-1, high=1),
                                              'measurements' : spaces.Box(low=0, high=1, shape=(30,))
                                              })

        # define borders of court
        self.borders = []
        self.borders.append(plt.Line2D((0, 0), (0, world_size[1]), c='k', linestyle="-"))
        self.borders.append(plt.Line2D((0, world_size[0]), (0, 0), c='k', linestyle="-"))
        self.borders.append(plt.Line2D((0, world_size[0]), (world_size[1], world_size[1]), c='k', linestyle="-"))
        self.borders.append(plt.Line2D((world_size[0], world_size[0]), (0, world_size[1]), c='k', linestyle="-"))
        
        # define obstacle locations
        self.obstacles_list = []
        if obstacles is not None:
            for i, obstacle in enumerate(obstacles):
                # uses bocces list to simplify measurements
                self.obstacles_list.append(plt.Circle((obstacle[0], obstacle[1]), radius=0.8, fc='r'))

        # draw plot
        self.initialize_plot()


    def initialize_plot(self):
        # using seaborn, set background grid to gray
        sns.set_style("dark")
        plt.rcParams["figure.figsize"] = [x/2 for x in self.world_size]

        # Set minor axes in between the labels
        self.fig = plt.figure()
        self.ax = plt.gca()
        cols = self.world_size[0]+1
        rows = self.world_size[1]+1

        self.ax.set_xticks([x for x in range(1,cols)],minor=True )
        self.ax.set_yticks([y for y in range(1,rows)],minor=True)

        plt.xlim([0,self.x])
        plt.ylim([0,self.y])
        
        # Plot grid on minor axes in gray (width = 1)
        plt.grid(which='minor',ls='-',lw=1, color='white')
        
        # Plot grid on major axes in larger width
        plt.grid(which='major',ls='-',lw=2, color='white')

        self.fig.canvas.draw()

        # draw boarders on plot
        for border in self.borders:
            plt.gca().add_line(border)
        
        # draw goal line
        plt.gca().add_line(plt.Line2D((self.x-2, self.x-2), 
                                      (0, self.y), 
                                      c='g', linestyle="-", linewidth=2))
        
        # draw obstacles on plot
        if self.obstacles_list is not None:
            for obstacle in self.obstacles_list:
                plt.gca().add_patch(obstacle)
            

    def reset(self, randomize_obstacles=True, seed=False):
        """Reset robot location and obstacles."""
        super().reset(seed=seed)
        self.step_count = 0

        # reset robot location
        self.robot.set_robot_position(self.robot.origin[0][0], 
                                      self.robot.origin[0][1], 
                                      self.robot.origin[1])
        
        # remove all obstacles
        for b in self.obstacles_list:
            b.remove()

        # randomize obstacles
        if randomize_obstacles: self.randomize_obstacles(k=self.k_obstacles, seed=seed)

        observation = self._get_obs()

        return observation, {}

    def step(self, action):

        action = (action-2) * 15
        self.robot.vector_move(0.25,action)

        observation = self._get_obs()
        reward, terminated, truncated = self.get_reward()
        info = self._get_info()

        self.step_count += 1

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        measurements = self.robot.observe_environment()

        if self.robot.orient < 0:
            abs_angle = 360 + self.robot.orient
        else:
            abs_angle = self.robot.orient
    
        if abs_angle > 180:
            abs_angle = abs_angle - 360
        angle = abs_angle / 180

        position = np.array([self.robot.x/self.x, self.robot.y/self.y], dtype=np.float32)
        orientation = np.array([angle], dtype=np.float32)
        measurements = np.array(measurements, dtype=np.float32)

        if orientation.shape == (1, 1):
            orientation = orientation[0]
        
        observation = {'position' : position,
                       'orientation' : orientation,
                       'measurements' : measurements}

        return observation

    def _get_info(self):
        return { "distance": (self.robot.x) }

    def get_reward(self):
        """Returns: reward, terminated, truncated"""

        collision = self.robot.check_collision()
        if collision:
            return float(-100), True, False

        if self.robot.x >= self.x-2:
            return float(1_000), True, False
        
        if self.step_count > 1000:
            self.step_count = 0
            return float(0), True, True

        reward = float(self.robot.x / self.x)

        # reward gates:
        if self.robot.x >= 30:
            return reward * 10, False, False
        if self.robot.x >= 15:
            return reward * 5, False, False

        return reward, False, False
    

    def randomize_obstacles(self, k=5, seed=False):
        possible_locations = [[5,1],[5,7],
                              [10,3],[10,6],[10,9],
                              [15,2],[15,5],[15,8],
                              [20,1],[20,4],[20,7],
                              [25,2],[25,5],[25,8],
                              [30,3],[30,6],[30,9],
                              [35,2],[35,5],[35,8]]
                            #   ,[10,1],[20,9]]
        if seed: random.seed(seed)
        
        obstacles_list = random.sample(possible_locations,k=k)
        self.obstacles_list = []

        for i, obstacle in enumerate(obstacles_list):
            # uses bocces list to simplify measurements
            self.obstacles_list.append(plt.Circle((obstacle[0], obstacle[1]), radius=0.8, fc='r'))
            plt.gca().add_patch(self.obstacles_list[i])


    def randomize_pallino(self, x_max=40, y_max=13):
        """Move pallino to random location within specified range."""
        if self.pallino is not None: self.pallino.remove()

        x = random.random() * x_max
        y = random.random() * y_max

        self.pallino = plt.Circle((x, y), radius=0.0655, fc='m')
        self.pallino.xy = (x, y)
        plt.gca().add_patch(self.pallino)
    
    def randomize_bocces(self, num_bocce=1, x_max=40, y_max=13):
        """Randomize boccee ball locations within specified range."""
        for b in self.obstacles_list:
            b.remove()

        for _ in range(num_bocce):
            x = random.random() * x_max
            y = random.random() * y_max

            bocce_patch = plt.Circle((x, y), radius=0.175, fc='r')

            self.obstacles_list.append(bocce_patch)
            plt.gca().add_patch(bocce_patch)
    
    def register_robot(self, robot):
        self.robot = robot


if __name__ == "__main__":
    plt.ion()

    obs = [[1,1]]
    env = Env(world_size=[40,10],
              obstacles=obs)
    
    
    input("Press Enter")
