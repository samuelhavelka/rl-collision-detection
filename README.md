# Reinforcement learning for collision detection. 
Uses StableBaselines3 Deep Q Learning RL algorithm to control agent.

Goal: navigate through an unknown environment without colliding with any walls or obstacles.

Completed Features:
- Custom environment with randomly placed obstacles for the robot to avoid.
- Lidar-type sensor to scan environment.
- Discrete action space.
- Dict observation space.

Work in progress features:
- SLAM: simultaneous localization and modeling.
  - Testing genetic algorithm using DEAP package to estimate both the robot's location and map of environment based on the sensor measurements.
- Alternative RL algorithm with continuous action space capabilities (possibly TD3).
- Alternative neuroevolutionary RL algorithms such as Enforced Sub-Populations (ESP).

## Robot Sensor Demo:
Each green dot represents a measurement taken by the agent.

The only information the agent has about its environment is these measurements, its current x and y position, and its current orientation.

<img src="https://github.com/samuelhavelka/rl_collision_detection/blob/main/gifs/sensor_animation.gif" width="1000" height="250"/>

## Robot RL-controlled collision avoidance demo:
The environment is randomized every episode and the robot receives its maximum reward for crossing the green finish line on the right.

<img src="https://github.com/samuelhavelka/rl_collision_detection/blob/main/gifs/rl_animation.gif" width="1000" height="250"/>
