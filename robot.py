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

from helper import circle_line_segment_intersection, collision_rec_circle, euclidian_distance, find_outliers, calculate_linear_regression_error, partition_on_index, distance_to_rectangle
from environment import Env

# enable interactive plot
plt.ion()

PI = 3.14159265359


class Robot:
    
    def __init__(self, env, 
                 initial_loc=[85/2,13/2], inital_orient=0, 
                 robot_dims=[1,1], 
                 range=100, measure_noise=0.5, 
                 motion_noise=0.1, turn_noise=1.0,
                 max_speed=0.5, max_turn_angle=30):
        
        self.env = env
        self.range = range

        # robot orign
        self.origin = (initial_loc, inital_orient)

        # robot dimensions
        self.dimx = robot_dims[0]
        self.dimy = robot_dims[1]

        self.max_speed = max_speed
        self.max_turn_angle = max_turn_angle

        # starting coords
        self.pos = initial_loc
        self.x = initial_loc[0]
        self.y = initial_loc[1]
        self.orient = inital_orient

        self.x_est = self.x
        self.y_est = self.y

        # noise objects
        self.motion_noise = motion_noise
        self.measurement_noise = measure_noise
        self.turn_noise = turn_noise

        # define robot
        self.robot_rec = plt.Rectangle((self.x, self.y), 
                                       self.dimx, self.dimy, 
                                       angle=self.orient, 
                                       fc='b', rotation_point='center')
        # define sensor line
        self.sensor_line = plt.Line2D((self.x+self.dimx/2, self.x + self.range*cos(radians(self.orient))+self.dimx/2), 
                                      (self.y+self.dimy/2, self.y + self.range*sin(radians(self.orient))+self.dimy/2), 
                                      linestyle="--")
        
        # draw robot and sensor on plot
        plt.gca().add_patch(self.robot_rec)
        plt.gca().add_line(self.sensor_line)

        # add robot to env
        self.env.register_robot(self)

        # coord tracker for animations
        self.x_list = []
        self.y_list = []
        self.history = [[self.x, self.y, self.orient]]
        self.observe_environment_xy()
        self.m_history = [[self.x_list, self.y_list]]
        

    # returns a positive, random float
    def rand(self):
        return random.random() * 2.0 - 1.0

    def update_visual(self):
        self.env.fig.canvas.draw_idle()
        self.env.fig.canvas.flush_events()

    def observe_environment(self, num_points=30):
        """ 
        Take sensor measurements for localization. 
        Returns as list of distances for measurements within 120 deg from current orientation.

        :param: num_points is the number of measurements to take within 120deg window.
        """

        measure_list = []
        for angle in np.linspace(self.orient-120,self.orient+120, num_points):
            m, _ = self.measure_distance(B_angle=angle)
            measure_list.append(m)

        # normalize measurements
        arrayint=np.array(ceil(sqrt(self.env.x**2 + self.env.y**2)))
        measure_list = measure_list / arrayint
        measure_list = [min(x, 1) for x in measure_list]

        return measure_list


    def observe_environment_xy(self, num_points=30):
        """ 
        Take sensor measurements for localization. 
        Returns as list of xy coordinates and cooresponding labels 

        :param: num_points is the number of measurements to take within 120deg window.

        returns: x_list, y_list, labels = (list, list, list)
        """

        x_list = []
        y_list = []
        labels = []

        for angle in np.linspace(self.orient-120,self.orient+120, num_points):
            
            m, label = self.measure_distance(B_angle=angle)

            mx = m*cos(radians(angle)) + self.x+self.dimx/2
            my = m*sin(radians(angle)) + self.y+self.dimy/2

            x_list.append(mx)
            y_list.append(my)
            labels.append(label)

        self.x_list = x_list
        self.y_list = y_list

        return x_list, y_list, labels
    

    def navigation_step(self, x_dest, y_dest, t_dest):
        """Take a single step in the direction of a destination point."""
        DEBUG = False

        dx = x_dest - self.x
        dy = y_dest - self.y
        move = [0,0]

        travel_angle = degrees(atan2(dy,dx))
        if travel_angle < 0: travel_angle += 360
        if DEBUG: print(dx, dy, travel_angle, self.orient, travel_angle-self.orient)
        
        # check if at locaiton
        if abs(self.orient-t_dest) < 0.5 and sqrt(dx**2 + dy**2) < 0.5: return True, [0,0]

        # turn to angle
        if abs(self.orient-travel_angle) > 0.5 and sqrt(dx**2 + dy**2) > 0.5:
            angle_diff = travel_angle-self.orient
            if DEBUG: print("TURN",angle_diff)
            self.vector_move(0, angle_diff)
        
        # move to target coords
        elif sqrt(dx**2 + dy**2) > 0.5:
            self.vector_move(sqrt(dx**2 + dy**2), 0)
            speed = min(sqrt(dx**2 + dy**2), self.max_speed)
            move = [speed*cos(radians(self.orient)), speed*sin(radians(self.orient))]

            if DEBUG: print("MOVE", min(self.max_speed, sqrt(dx**2 + dy**2)))

        # turn to angle
        elif abs(self.orient-t_dest) > 0.5:
            angle_diff = t_dest - self.orient
            self.vector_move(0, angle_diff)
            if DEBUG: print("TURN_FIN", angle_diff)
        
        return False, move


    def vector_move(self, speed, turn_angle):
        """ Move robot with given speed and turning angle in degrees. """
        speed = min(speed, self.max_speed)

        turn_sign = np.sign(turn_angle)
        turn_angle = min(abs(turn_angle), self.max_turn_angle) * turn_sign

        t = self.orient + turn_angle
        x = speed * cos(radians(t))
        y = speed * sin(radians(t))

        status = self.move(x,y,turn_angle)
        self.update_visual()

        return status
    
    def direct_move(self, speed, angle):
        """ Move robot with given speed and turning angle in degrees. """
        speed = min(speed, self.max_speed)

        t = angle
        x = speed * cos(radians(t))
        y = speed * sin(radians(t))

        status = self.move(x,y,0)
        self.update_visual()

        return status


    def move(self, dx, dy, dt):
        """ Move robot relative to current position. dt in degrees """

        # absolute coordinates after move
        x = self.x + dx
        y = self.y + dy
        t = self.orient + dt

        self.set_robot_position(x,y,t)
        return True
    

    def set_robot_position(self, x, y, t):
        """Update robot position using absolute values."""
        
        # update class variables
        self.pos = [x,y]
        self.x = x
        self.y = y
        self.orient = t

        # log position history for playback animation
        self.history.append([x,y,t])
        self.m_history.append([self.x_list, self.y_list])

        # set position and angle of robot
        self.robot_rec.set_xy((x, y))
        self.robot_rec.set_angle(t)

        # call update to sensor line
        self.update_sensor_position()

        return (x,y), t

    def update_sensor_position(self):
        """Update sensor line based on current position and orientation"""
        x, y = np.array(self.robot_rec.get_xy())
        orient = self.robot_rec.get_angle()

        xdim = self.robot_rec.get_width()
        ydim  = self.robot_rec.get_height()

        self.sensor_line.set_xdata((x+xdim/2, x + 100*cos(radians(orient))+xdim/2))
        self.sensor_line.set_ydata((y+ydim/2, y + 100*sin(radians(orient))+ydim/2))


    def measure_distance(self, B_angle=False):
        """Return distance measurement of sensor"""
        best_dist = 1_000_000
        label = 0

        # check distance to all balls
        if len(self.env.obstacles_list) > 0:
            for circle in self.env.obstacles_list:
                x = self.lineCircleIntersection(self.sensor_line, circle, B_angle)
                if (x != False) and (x != None):
                    if (x < best_dist):
                        best_dist = x
                        label = 1

        # check distance to all court borders
        for line in self.env.borders:
            x = self.lineLineIntersection(self.sensor_line, line, B_angle)
            if (x != False) and (x != None):
                if (x < best_dist):
                    best_dist = x
                    label = 0
        
        if best_dist == 1_000_000: 
            return 1_000_000, False

        # random measurement noise range of +measurement_noise% to -measurement_noise%
        # max_noise = self.measurement_noise/100
        # best_dist *= np.random.uniform(1-max_noise, 1+max_noise)

        return best_dist, label


    def lineLineIntersection(self, sensor_line, line2, B_angle=False):
        """Check if a line intersects with another line. Returns distance as float"""
        DEBUG = False

        line1_data = sensor_line.get_data()
        line2_data = line2.get_data()

        # sensor line 
        A = line1_data[0][0], line1_data[1][0]
        if B_angle:
            B = A[0] + self.range*cos(radians(B_angle)), A[1] + self.range*sin(radians(B_angle))
        else:
            B_angle = self.orient
            B = line1_data[0][1], line1_data[1][1]
        
        # target line (court borders)
        C = line2_data[0][0], line2_data[1][0]
        D = line2_data[0][1], line2_data[1][1]

        # Line AB represented as a1x + b1y = c1
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1*(A[0]) + b1*(A[1])
    
        # Line CD represented as a2x + b2y = c2
        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2*(C[0]) + b2*(C[1])
    
        determinant = a1*b2 - a2*b1
    
        if (determinant == 0):
            # The lines are parallel
            return False
        else:
            if DEBUG: print("A", A, "B",B, "C",C, "D", D)
            x = (b2*c1 - b1*c2)/determinant
            y = (a1*c2 - a2*c1)/determinant

            rel_x = min(x - A[0], 1000)
            rel_y = min(y - A[1], 1000)

            # get angle of measurement vector to filter out negative measurements
            if rel_x == 0:
                if rel_y == 0: print("ERROR ZERO-ZERO MEASUREMENT")
                if np.sign(rel_y) > 0: measurement_angle = 90
                if np.sign(rel_y) < 0: measurement_angle = 270
                if np.sign(rel_y) == 0: measurement_angle = 0
            else:
                measurement_angle = degrees(atan2(rel_y, rel_x))
                if measurement_angle < 0: measurement_angle += 360
            
            # print("rels", rel_x,rel_y)
            quadrant = (B_angle % 360) // 90
            measure_quad = (measurement_angle % 360) // 90
            if DEBUG: print("angles:", measurement_angle, B_angle)
            if DEBUG: print("quadrants:",measure_quad,quadrant)

            # only allow measurements in direction of robot's orientation
            if quadrant != measure_quad: return False

            distance = sqrt(rel_x**2 + rel_y**2)
            if distance > 1000: return False
            
            return distance
    
    def lineCircleIntersection(self, line, circle, B_angle=False):
        """Calculates distance to circle if intersecting. Returns distance as float."""
        
        line = self.sensor_line.get_data()
        A = np.array([line[0][0], line[1][0]])

        if B_angle:
            B = A[0] + self.range*cos(radians(B_angle)), A[1] + self.range*sin(radians(B_angle))
        else:
            B = line[0][1], line[1][1]

        circle_xy = circle.get_xy()
        radius = circle.get_radius()

        projection = self.orthogonal_projection(A, B, circle_xy)
        proj_dist = euclidian_distance(circle_xy, projection)

        if proj_dist <= radius:

            pts = circle_line_segment_intersection(circle_xy, radius, A, B)
            if len(pts) > 0:
                closest_dist=1_000
                for pt in pts:
                    dist = euclidian_distance(A,pt)
                    if dist < closest_dist:
                        closest_pt = pt
                        closest_dist = dist
                return euclidian_distance(closest_pt, A)
        else:
            return False
        

    def projection(self, b, a):
        """orthogonal projection of a onto a straight line parallel to b"""
        return ((b @ a) / (b @ b)) * b
    
    def orthogonal_projection(self, A, B, circle):
        """Calculates the orthogonal projection from circle onto the line AB"""
        AB = B - A
        AC = circle - A
        # projection = AC - AB * ((AB @ AC) / (AB @ AB))
        proj = AC - self.projection(AB, AC)

        orthogonal_circle_line = circle - proj
        return orthogonal_circle_line

    def check_collision(self):
        """ Check if robot is intesecting any obstacle on court. """

        rectangle = list(self.robot_rec.get_xy())
        rectangle.append(self.robot_rec.get_width())
        rectangle.append(self.robot_rec.get_height())

        # check if robot is out of bounds
        if (self.x < 0.0) or (self.x+self.dimx > self.env.x) or (self.y < 0.0) or (self.y+self.dimy > self.env.y):
            return True

        # check if robot is colliding with obstacle
        for obstacle in self.env.obstacles_list:
            circle = list(obstacle.get_xy())
            circle.append(obstacle.get_radius())

            collision = collision_rec_circle(rectangle, circle)
            if collision:
                return True
        
        return False


if __name__ == "__main__":
    env = Env(world_size=[40,10], 
              obstacles=[[8,8],[10,5],[15,2],[4,1]])

    robot = Robot(env=env,initial_loc=[8,5.5], inital_orient=0)

    input("Press Enter to run sim...")

    for _ in range(50):
        
        # move robot
        robot.vector_move(0.1,0)

        # scan environment and plot measurement points
        x_list, y_list, labels = robot.observe_environment_xy()
        pts = plt.scatter(x_list,y_list,color='g')

        # check for collision
        collision_bool = robot.check_collision()
        if collision_bool: 
            print("Collision!")
            robot.robot_rec.set_color("k")
        else:
            robot.robot_rec.set_color("b")
        
        # update plot
        env.fig.canvas.draw_idle()
        env.fig.canvas.flush_events()

        time.sleep(0.2)
        pts.remove()

