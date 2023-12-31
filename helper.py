from math import *
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def euclidian_distance(A, B):
        """Calculates the euclidian distance between two points"""
        return np.linalg.norm(A - B)


def find_outliers(numbers, q1=25, q3=75):
    # Calculate the first and third quartiles (Q1 and Q3)
    q1 = np.percentile(numbers, q1)
    q3 = np.percentile(numbers, q3)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define a lower bound and upper bound for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Extract outliers and their indices
    outliers = [(index, value) for index, value in enumerate(numbers) if value < lower_bound or value > upper_bound]

    return outliers


def calculate_linear_regression_error(x, y):
    """
    Calculate the Mean Squared Error for a linear regression model.

    Parameters:
    - x: List or array of x-coordinates.
    - y: List or array of y-coordinates.

    Returns:
    - mse: Mean Squared Error.
    """

    # Reshape the data to meet scikit-learn's requirements
    X = np.array(x).reshape(-1, 1)

    # Fit a linear regression model
    model = LinearRegression().fit(X, y)

    # Predict the y values based on the model
    y_pred = model.predict(X)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y, y_pred)

    return model, mse


def partition_on_index(it, indices):
    indices = set(indices)   # convert to set for fast lookups
    l1, l2 = [], []
    l_append = (l1.append, l2.append)
    for idx, element in enumerate(it):
        l_append[idx in indices](element)
    return l1, l2


def distance_to_rectangle(point, rectangle):

    point_x, point_y = point
    rect_left, rect_right, rect_top, rect_bottom = rectangle
    
    distance = min(
        abs(point_x - rect_left),
        abs(point_x - rect_right),
        abs(point_y - rect_top),
        abs(point_y - rect_bottom)
    )
    return distance

def collision_rec_circle(rectangle,   # rectangle definition
              circle):  # circle definition
    """ Detect collision between a rectangle and circle. """

    # input objects
    rleft, rbottom, width, height = rectangle
    center_x, center_y, radius = circle

    # complete boundbox of the rectangle
    rright = rleft + width
    rtop = rbottom + height

    # bounding box of the circle
    cleft, ctop     = center_x-radius, center_y+radius
    cright, cbottom = center_x+radius, center_y-radius

    # trivial reject if bounding boxes do not intersect
    if rright < cleft or rleft > cright or rbottom > ctop or rtop < cbottom:
        return False  # no collision possible

    # check whether any point of rectangle is inside circle's radius
    for x in (rleft, rright):
        for y in (rbottom, rtop):
            # compare distance between circle's center point and each point of
            # the rectangle with the circle's radius
            if hypot(x-center_x, y-center_y) <= radius:
                return True  # collision detected

    # check if center of circle is inside rectangle
    if (rleft <= center_x <= rright) and (rbottom <= center_y <= rtop):
        return True  # overlaid


    return False  # no collision detected

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


if __name__ == "__main__":
    PI = 3.14159
    DEBUG = False
    # plt.ion()
    from environment import Robot, Env

    env = Env(bocces=[[5,5], [5,10]],
              pallino=[8,13/2])
    
    rob = Robot(world=env, inital_orient=radians(8.6955028774247695983812), robot_dims=[1,1])