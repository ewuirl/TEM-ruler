import argparse
import pandas as pd
import numpy as np
from scipy import signal
# from scipy import special

# Data Preparation Methods
def exclude_NaN(i, length_df):
    x_data = length_df[i][np.logical_not(np.isnan(length_df[i]))]
    y_data = length_df[i+1][np.logical_not(np.isnan(length_df[i+1]))]
    return(x_data, y_data)

# # Smoothing Methods

# Derivative Methods
def get_diff(x_vals, y_vals):
    """
    get_diff(x_vals, y_vals)

    This function calculates an approximate first derivative using the 

    Arguments:
        
    Returns:
        x_diffs (numpy array):
        diffs (numpy array):
    """
    num_diffs = len(y_vals)-1
    diffs = np.zeros(num_diffs)
    x_diffs = np.zeros(num_diffs)
    for i in range(num_diffs):
        diffs[i] = (y_vals[i+1]-y_vals[i])/(x_vals[i+1]-x_vals[i])
        x_diffs[i] = (x_vals[i+1]-x_vals[i])/2+x_vals[i]
    return(x_diffs, diffs)


# Finding Edge Boundaries
# # Min Max Method
# # # Zero Crossing Methods
def find_zero_crossing(derivative, mu, step_size, direction, threshold=2, max_steps=20):
    """
    find_zero_crossing(derivative, mu, step_size, direction, threshold=2, max_steps=20)

    This argument takes the derivative of a function, a peak location, a 
    direction (-1,1), and looks for the point at which the derivative crosses zero in
    that direction from the peak.

    Arguments:

    Returns: 

    """
    # ADD ASSERT FOR DIRECTION

    crossing_point = -1
    step = int(step_size*direction)
    check_point = mu + step
    num_steps = 0
    if derivative[check_point] > 0:
        sign = 1
    else:
        sign = -1
    while crossing_point < 0 and num_steps < max_steps:
#         print("checking step")
#         print(sign*derivative[check_point+step])
        num_steps += 1
        if sign*derivative[check_point+step] > 0:
#             print("take step")
            check_point = check_point+step
        else:
#             print("check next step")
#             print(sign*derivative[check_point+threshold*step])
            if sign*derivative[check_point+threshold*step] < 0 and abs(step) > abs(direction):
#                 print("reset checkpoint")
                check_point = check_point-step
#                 print("take smaller steps now")
                step = direction
            elif sign*derivative[check_point+threshold*step] < 0 and abs(step) == abs(direction):
#                 print("found crossing point")
                crossing_point = check_point
            else:
#                 print("checked next step, take step")
                check_point = check_point+step
    return crossing_point

def find_min_max_bounds(derivative_1, rising_peak_loc, falling_peak_loc, \
    step_size, threshold=2, max_steps=20):
    rising_peak_min = find_zero_crossing(derivative_1, rising_peak_loc, \
        step_size, -1, threshold=threshold,max_steps=max_steps)
    rising_peak_max = find_zero_crossing(derivative_1, rising_peak_loc, \
        step_size, 1, threshold=threshold,max_steps=max_steps)
    falling_peak_max = find_zero_crossing(derivative_1, falling_peak_loc, \
        step_size, -1, threshold=threshold,max_steps=max_steps)
    falling_peak_min = find_zero_crossing(derivative_1, falling_peak_loc, \
        step_size, 1, threshold=threshold,max_steps=max_steps)
    return(rising_peak_min, rising_peak_max, falling_peak_min, falling_peak_max)

# Half Max Calculation Methods
def calculate_half_max(edge_max, edge_min):
    """
    calculate_half_max(edge_max, edge_min)

    Calculates the half max of the edge given the max and min values of the edge

    Arguments:

    Returns:
    """
    return (edge_max-edge_min)/2.0 + edge_min

def find_half_max_neighbors(half_max, y_data, direction, left_bound, right_bound):
    """
    find_half_max_neighbors(half_max, y_data, direction, left_bound, right_bound)

    Finds the neighboring points of the half max value.

    Arguments:

    Returns:
    """
    i = 0
    j = -1
    if direction > 0:
        y_range = y_data[left_bound:right_bound+1]
    else:
        y_range = y_data[right_bound:left_bound-1:-1]
    while j < 0 and i < len(y_range):
        if y_range[i] < half_max:
            i += 1
        else:
            j = i+1
    if direction > 0:
        return(left_bound+i,left_bound+j)
    else:
        return(right_bound-j+1,right_bound-i+1)

def find_half_max_pos(half_max, half_max_bounds, x_data, y_data):
    """
    find_half_max_pos(half_max, half_max_bounds, x_data, y_data)

    Calculates a linear fit using the neighboring points of the half max value,
    and uses this fit to extrapolate the x position of the half max value.

    Arguments:

    Returns:
    """
    slope = y_data[half_max_bounds[0]]-y_data[half_max_bounds[1]]/(x_data[half_max_bounds[0]]-x_data[half_max_bounds[1]])
    intercept = y_data[half_max_bounds[0]] - slope*x_data[half_max_bounds[0]]
    half_max_pos = (half_max-intercept)/slope
    return half_max_pos

# Width calculation methods
def calculate_width_min_max(x_data, y_data, smooth_func, smooth_params)

if __name__ == "__main__":
    # add some arg parse stuff here

    file_path = "/Users/emilywu/OneDrive - Massachusetts Institute of Technology/TEM width/42hb polyplex no stain xy values length.xlsx"
    
    # Read the file in 
    len_data = pd.read_excel(file_path, header=None,names=None)

    # Get the shape
    rows, cols = len_data.shape
    print(len_data[1])
    mean = len_data[1].mean()
    print(mean)
    # Iterate through all of the grayscale columns
    # for i in range(cols/2):
    #     pass
