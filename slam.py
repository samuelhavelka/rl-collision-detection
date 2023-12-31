import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import random
from math import *
import time
from deap import creator, base, tools, algorithms

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

from helper import euclidian_distance, find_outliers, calculate_linear_regression_error, partition_on_index, distance_to_rectangle
from environment import Env
from robot import Robot

VISUALIZE = True
ANIMATE = True
 
# enable interactive plot
if VISUALIZE: plt.ion()

PI = 3.14159265359
anim_points = []

class SLAM:

    def __init__(self, robot, env, method='deap'):
        self.robot = robot
        self.env = env

        self.method = method
        
        if self.method == 'deap':
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            self.toolbox = base.Toolbox()
            self.toolbox.register("attr_float", random.random)

            # define individual attributes [float, float]
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_float, n=2)
        
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

            # update toolbox parameters
            self.toolbox.register("evaluate", self.calculate_sum_of_distance)
            self.toolbox.register("mate", tools.cxBlend, alpha=0.1)
            self.toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.1, indpb=0.1)
            self.toolbox.register("select", tools.selTournament, tournsize=3)

            self.stats = tools.Statistics(lambda ind: ind.fitness.values)
            self.stats.register("avg", np.mean)
            self.stats.register("min", np.min)
            self.stats.register("max", np.max)


    def calculate_sum_of_distance(self, args):
        """
        Calculate the total distance from reach measurement and the known court shape.

        :param args: (x,y) tuple of floats representing location as percent of x/y

        returns: (total_error,)

        """
        DEBUG = False
        # unpack arguments
        x_offset, y_offset = args
        # scale inputs by environment dimensions
        x_offset *= self.env.x
        y_offset *= self.env.y

        
        if (x_offset < 0) or (x_offset > self.env.x) or (y_offset < 0) or (y_offset > self.env.y):
            if DEBUG: print("calculate_sum_of_distance() - X OR Y OUT OF RANGE")
            return (float(1_000_000),)

        num_points = len(self.robot.x_list)
        total_err = 0
        
        for i in range(num_points):
            x = self.robot.x_list[i]
            y = self.robot.y_list[i]

            total_err += distance_to_rectangle((x+x_offset, y+y_offset), 
                                               (0,self.env.x,self.env.y,0))
        
        return (total_err,)
    

    def deap_location_estimate(self, initialize=False):
        """
        DEAP Genetic Algorithm optimization to estimate current position based 
        on last sensor reading. Fitness = total distance to environment model.

        :param initialize: [x,y], otherwise initialize with the average [x,y] 
                           of the environment scan 

        returns: [x,y], error as floats
        
        """
        DEBUG = False
        # starttime = time.time()

        x_list, y_list, labels = self.robot.observe_environment_xy()

        pop = self.toolbox.population(n=50)
        # replace first individual with means coord of measurements
        if initialize:
            pop[0] = creator.Individual(initialize)
        else:
            pop[0] = creator.Individual([np.abs(np.max(x_list)), 
                                         np.abs(np.mean(y_list))])

        hof = tools.HallOfFame(1)
        error = 10
        i = 0
        while error > 0.8:
            pop = self.toolbox.population(n=50)
            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.1, ngen=15, 
                                               stats=self.stats, halloffame=hof, verbose=False)

            error = self.calculate_sum_of_distance(hof[0])[0] / len(x_list)
            
            i += 1
            if i > 3: break
        
        # print('\033[35m' + f'--- DEAP in ' + str(round(time.time() - starttime, 5)) + 's secs ---' + '\033[0m')
        
        coords = [hof[0][0] * self.env.x, hof[0][1] * self.env.y]
        real_error = euclidian_distance(np.array([self.robot.x,self.robot.y]), np.array(coords))
        
        if DEBUG and real_error > 3:
            foo = self.calculate_sum_of_distance((self.robot.x/self.env.x, self.robot.y/self.env.y))[0] / len(x_list)
            print("DEAP ERROR > 3:",real_error, error, foo)

        return coords, error

    
    def naive_location_estimate(self, precision=0.5):
        """Check every point to minimize total distance to model for location estimate"""

        x_list, y_list, labels = self.robot.observe_environment_xy()

        best_loc = [None, None]
        best_err = 1_000_000

        for i, dx in enumerate(np.linspace(0, 1, int(self.env.x/precision))):
            for j, dy in enumerate(np.linspace(0, 1, int(self.env.y/precision))):

                err = self.calculate_sum_of_distance((dx,dy))[0]

                if err < best_err:
                    best_err = err
                    best_loc = (dx,dy)

        best_err /= len(self.robot.x_list)

        return best_loc, best_err
    

    def mapping(self):
        """Estimate state using environemnt scan and current position"""
        DEBUG = False

        # scan environment
        x_list = self.robot.x_list
        y_list = self.robot.y_list
        
        # estimate current position
        deap_loc, deap_err = self.deap_location_estimate()
        deap_x = deap_loc[0][0] * self.env.x
        deap_y = deap_loc[0][1] * self.env.y

        if DEBUG: 
            print("---------- MAPPING ----------")
            print("DEAP ERROR FROM ACTUAL XY:", deap_x-self.robot.x, deap_y-self.robot.y)
            print("X RANGE:", min(x_list), max(x_list))
            print("Y RANGE:", min(y_list), max(y_list))
            print("ESTIMATED LOCATION:", deap_x, deap_y)
            print("LEN OF MEASUREMENTS:",len(x_list), len(y_list))
            print("-----------------------------")

        dist_list = np.empty_like(x_list)

        # offset all x and y mneasurements by estimated location
        x_list = np.add(x_list, deap_x).tolist()
        y_list = np.add(y_list, deap_y).tolist()

        # calculate distance to court border
        for i in range(len(x_list)):
            dist = distance_to_rectangle((x_list[i], y_list[i]), (0,self.env.x,self.env.y,0))
            dist_list[i] = dist

        # extract outliers from distance calculations
        # OUTLIER EXTRACTION SENSITIVE TO Q3 VALUES
        outliers = find_outliers(dist_list, q1=25, q3=80)
        outliers_i = [x[0] for x in outliers]

        neighbors = []
        for point in outliers:
            count = 0

            for target_point in outliers:
                dist = euclidian_distance(np.array(point), np.array(target_point))
                if dist < 5:
                    count += 1
            
            neighbors.append(count)
        
        if DEBUG:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

            ax1.scatter(x_list, y_list, color='blue')
            ax1.scatter([x_list[i] for i in outliers_i], 
                        [y_list[i] for i in outliers_i], 
                        color='red', label='Red Points')
            ax1.scatter(deap_x, deap_y, color='k')

            ax2.scatter(range(len(dist_list)), dist_list)
            ax2.scatter(outliers_i, 
                        [dist_list[i] for i in outliers_i], 
                        color='red', label='Red Points')
            plt.show()
            input()


def init():
    """Animation initialization"""
    x,y,t = robot.history[0]
    robot.robot_rec.set_xy((x, y))
    robot.robot_rec.set_angle(t)
    lines = [robot.robot_rec]

    robot.points = []

    for x,y in zip(robot.m_history[0][0], robot.m_history[0][1]):
        pt_circle = plt.Circle((x, y), radius=0.1, fc='g')
        robot.points.append(pt_circle)
        plt.gca().add_patch(pt_circle)

    return lines,

def animate(i):
    """Animation function steps through logged positions"""

    x,y,t = robot.history[i]
    p = robot.m_history[i]

    robot.robot_rec.set_xy((x, y))
    robot.robot_rec.set_angle(t)

    xdim = robot.robot_rec.get_width()
    ydim  = robot.robot_rec.get_height()

    robot.sensor_line.set_xdata((x+xdim/2, x + 100*cos(radians(t))+xdim/2))
    robot.sensor_line.set_ydata((y+ydim/2, y + 100*sin(radians(t))+ydim/2))

    for circle in robot.points:
        circle.set_radius(0)

    for x,y in zip(robot.m_history[i][0], robot.m_history[i][1]):
        pt_circle = plt.Circle((x, y), radius=0.1, fc='g')
        robot.points.append(pt_circle)
        plt.gca().add_patch(pt_circle)

    lines = [robot.robot_rec, robot.sensor_line, robot.points]

    return lines

if __name__ == "__main__":

    env = Env(world_size=[40,10], 
              obstacles=[[1,1],[10,4],[8,8], [5,8], [30,6], [20,5]])
    
    # orientation in degrees
    robot = Robot(env=env,initial_loc=[35,3], inital_orient=135, max_speed=0.25)
    slam = SLAM(robot, env, method='deap')

    max_frames = 360
    fps = 30

    input("Press Enter to run sim...")
    fps_time = time.time()

    counter = 0
    move_vector = [0,0]
    status1 = False

    for frame in range(max_frames):

        if not status1:
            robot.observe_environment_xy(num_points=75)
            status1, move_vector = robot.navigation_step(10,1,135)

        if status1: 
            robot.observe_environment_xy(num_points=75)
            status2, move_vector = robot.navigation_step(5,5,135)
            if status2: break


    # naive localization method, checks every possible location
    naive_loc, naive_err = slam.naive_location_estimate()
    naive_loc = [naive_loc[0] * env.x, 
                 naive_loc[1] * env.y]
    naive_loc = np.round(naive_loc, 1)
    naive_err = np.round(naive_err, 3)
    print("BEST NAIVE LOC EST: Coords:", naive_loc, "Error:", naive_err)
    
    # DEAP search alogirthm for localization
    # much faster but occaisionally fails
    deap_loc, deap_err = slam.deap_location_estimate()
    deap_loc = np.round(deap_loc, 1)
    deap_err = np.round(deap_err, 3)
    print("BEST DEAP  LOC EST: Coords:", deap_loc, "Error:", deap_err)


    if ANIMATE:
        anim = animation.FuncAnimation(env.fig, animate, init_func=init, 
                                       frames=frame, interval=5000, blit=False)

    if ANIMATE:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save("animation.mp4", writer=writer)