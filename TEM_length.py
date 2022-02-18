import argparse
import pandas as pd
import numpy as np
from scipy import signal
# from scipy import special

# Data Preparation Methods
def exclude_NaN(x_index, length_df):
    x_data = length_df[x_index][np.logical_not(np.isnan(length_df[x_index]))]
    y_data = length_df[x_index+1][np.logical_not(np.isnan(length_df[x_index+1]))]
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
    # print(f"peak: {mu}")
    crossing_point = -1
    found_crossing_point = False
    step = int(step_size*direction)
    check_point = mu + step
    num_steps = 0
    if derivative[check_point] > 0:
        sign = 1
    else:
        sign = -1
    while not found_crossing_point and num_steps < max_steps:
        # print(f"current check point: {check_point}")
        # print("checking step")
        # print(sign*derivative[check_point+step])
        test_step = check_point+step
        # Check if the test step is in bounds
        # print(f"check if test step in bounds: {test_step}")
        if 0 <= test_step < len(derivative):
            if sign*derivative[test_step] > 0:
                # print(f"take step to {test_step}")
                check_point = test_step
            else:
                # print("found crossing point")
                crossing_point = check_point
                found_crossing_point = True
            num_steps += 1
        else: 
            # If the test step is out of bounds and the step size is greater than 1
            # Try using a smaller step size
            if abs(step) > 1:
                # print(f"test step out of bounds, smaller steps")
                step = direction
            # If the test step is out of bounds, set the crossing point to the test step
            else:
                # print(f"test set out of bounds, set crossing point to check point")
                found_crossing_point = True
                crossing_point = check_point

    # print(f"found crossing point: {found_crossing_point}")
    # print(f"crossing point: {crossing_point}")
    # print(f"num steps: {num_steps}")
    # print(f"checkpoint after while loop: {check_point}")
    return crossing_point

T


# def find_min_max_bounds(derivative_1, rising_peak_loc, falling_peak_loc, \
#     step_size, threshold=2, max_steps=20):
#     rising_peak_min = find_zero_crossing(derivative_1, rising_peak_loc, \
#         step_size, -1, threshold=threshold,max_steps=max_steps)
#     rising_peak_max = find_zero_crossing(derivative_1, rising_peak_loc, \
#         step_size, 1, threshold=threshold,max_steps=max_steps)
#     falling_peak_max = find_zero_crossing(derivative_1, falling_peak_loc, \
#         step_size, -1, threshold=threshold,max_steps=max_steps)
#     falling_peak_min = find_zero_crossing(derivative_1, falling_peak_loc, \
#         step_size, 1, threshold=threshold,max_steps=max_steps)
#     return(rising_peak_min, rising_peak_max, falling_peak_min, falling_peak_max)

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
    slope = (y_data[half_max_bounds[0]]-y_data[half_max_bounds[1]])/(x_data[half_max_bounds[0]]-x_data[half_max_bounds[1]])
    intercept = y_data[half_max_bounds[0]] - slope*x_data[half_max_bounds[0]]
    half_max_pos = (half_max-intercept)/slope
    return half_max_pos

# Width calculation methods
def calculate_width_min_max(x_data, y_data, smooth_func, smooth_params, \
    zero_cross_params):
    # Smooth the function
    y_smooth = smooth_func(y_data, *smooth_params)

    # Calculate the first derivative
    x_d1, y_smooth_d1 = get_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find  the peaks (hill and valley) of the first derivative
    rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[4:-5]))[0][0]
    falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[4:-5]))[0][0]
    # print(f"rising_peak_loc: {rising_peak_loc}")
    # print(f"falling_peak_loc: {falling_peak_loc}")

    # Calculate the zero crossing points of the first derivative
    step_size, threshold, max_steps = zero_cross_params
    # Rising peak
    # print("######### rising_peak_zero_left")
    rising_peak_zero_left = find_zero_crossing(y_smooth_d1, rising_peak_loc, \
        step_size, -1, threshold=threshold, max_steps=max_steps)
    # print("######### rising_peak_zero_right")
    rising_peak_zero_right = find_zero_crossing(y_smooth_d1, rising_peak_loc, \
        step_size, 1, threshold=threshold, max_steps=max_steps)
    # Falling peak
    # print("######### falling_peak_zero_left")
    falling_peak_zero_left = find_zero_crossing(y_smooth_d1, falling_peak_loc, \
        step_size, -1, threshold=threshold, max_steps=max_steps)
    # print("######### falling_peak_zero_right")
    falling_peak_zero_right = find_zero_crossing(y_smooth_d1, falling_peak_loc, \
        step_size, 1, threshold=threshold, max_steps=max_steps)
    # print(f"rising_peak_zero_left: {rising_peak_zero_left}")
    # print(f"rising_peak_zero_right: {rising_peak_zero_right}")
    # print(f"falling_peak_zero_left: {falling_peak_zero_left}")
    # print(f"falling_peak_zero_right: {falling_peak_zero_right}")
    zero_loc_arr = np.array([rising_peak_zero_left, rising_peak_zero_right, \
        falling_peak_zero_left, falling_peak_zero_right])
    # Check if all the zero crossing points were successfully found
    check_string = check_found_zeros(zero_loc_arr)
    if "0" in check_string:
        print(check_string)
        print(f"rising_peak_loc: {rising_peak_loc}")
        print(f"falling_peak_loc: {falling_peak_loc}")
        print(f"rising_peak_zero_left: {rising_peak_zero_left}")
        print(f"rising_peak_zero_right: {rising_peak_zero_right}")
        print(f"falling_peak_zero_left: {falling_peak_zero_left}")
        print(f"falling_peak_zero_right: {falling_peak_zero_right}")
        width = np.nan
    else:
        # Calculate the half max values
        rising_peak_half_max = calculate_half_max(y_smooth[rising_peak_zero_right], \
            y_smooth[rising_peak_zero_left])
        falling_peak_half_max = calculate_half_max(y_smooth[falling_peak_zero_left], \
            y_smooth[falling_peak_zero_right])
        # print(f"rising_peak_half_max: {rising_peak_half_max}")
        # print(f"falling_peak_half_max: {falling_peak_half_max}")
        # Find the half max neighbors
        rising_half_max_bounds = find_half_max_neighbors(rising_peak_half_max, \
            y_smooth, 1, rising_peak_zero_left, rising_peak_zero_right)
        falling_half_max_bounds = find_half_max_neighbors(falling_peak_half_max, \
            y_smooth, -1, falling_peak_zero_left, falling_peak_zero_right)
        # print(f"rising_half_max_bounds: {rising_half_max_bounds}")
        # print(f"falling_half_max_bounds: {falling_half_max_bounds}")
        # Find the half max positions
        rising_half_max_pos = find_half_max_pos(rising_peak_half_max, \
            rising_half_max_bounds, x_data, y_smooth)
        falling_half_max_pos = find_half_max_pos(falling_peak_half_max, \
            falling_half_max_bounds, x_data, y_smooth)
        # print(f"rising_half_max_pos: {rising_half_max_pos}")
        # print(f"falling_half_max_pos: {falling_half_max_pos}")

        # Calculate the half max full width
        width = falling_half_max_pos - rising_half_max_pos

    return width, check_string

# Writing Output Files
def write_header(custom_name, file_name, smooth_method, smooth_params, \
    zero_method, zero_params, width_method, error_list, summary_stats, note):
    with open(f"{file_name}_header_{custom_name}.txt", 'w') as header_file:
        # Record the calculation methods and settings
        header_file.write(f"Smoothing method: {smooth_method}\n")
        header_file.write(f"Smoothing parameters: {smooth_params}\n")
        header_file.write(f"Zero finding method: {zero_method}\n")
        header_file.write(f"Zero parameters: {zero_params}\n")
        header_file.write(f"Width calculation method: {width_method}\n")
        header_file.write(f"Note: {note}\n")
        # Unpack the summary stats
        num_samples, summary_num_samples, mean, median, stdev = summary_stats
        header_file.write(f"Summary stats:\n")
        header_file.write(f"Number of samples: {num_samples}\n")
        header_file.write(f"Number of summary stat samples: {summary_num_samples}\n")
        header_file.write(f"mean: {mean}\n")
        header_file.write(f"median: {median}\n")
        header_file.write(f"stdev: {stdev}\n")
        header_file.write(f"Number of errors: {len(error_list)}\n")
        # Record any errors
        header_file.write("Errors: \n")
        for error in error_list:
            header_file.write(f"{error}\n")

def write_measurement_data_serial(custom_name, file_name, width_array):
    with open(f"{file_name}_measurements_{custom_name}.txt", 'w') as data_file:
        for width in width_array:
            data_file.write(f"{width}\t")

if __name__ == "__main__":
    # add some arg parse stuff here

    # Hard code some stuff
    smooth_func = signal.savgol_filter
    smooth_params = (21, 3)
    step_size = 2
    threshold = 2
    max_steps = 20
    zero_cross_params = (step_size, threshold, max_steps)
    custom_name = "serial"
    file_path = "/Users/emilywu/OneDrive - Massachusetts Institute of Technology/TEM width/42hb polyplex no stain xy values length.xlsx"
    # file_path = "/Users/emilywu/OneDrive - Massachusetts Institute of Technology/TEM width/42hb polyplex no stain xy values width.xlsx"
    file_name = "42hb polyplex no stain xy values length"
    # file_name = "42hb polyplex no stain xy values width"
    smooth_method = "savgol"
    zero_method = "crossing"
    width_method = "min_max"
    note = ""

    # Read the file in 
    length_df = pd.read_excel(file_path, header=None,names=None)

    # Get the shape
    rows, cols = length_df.shape
    num_samples = int(cols/2)
    
    width_array = np.zeros(num_samples)
    error_list = []

    # i = 4
    # # Pick the x columns
    # x_index = 2*i
    # # Remove the NaN values
    # x_data, y_data = exclude_NaN(x_index, length_df)
    # # Calculate the width
    # width, check_string = calculate_width_min_max(x_data, y_data, \
    #     smooth_func, smooth_params, zero_cross_params)
    # # print(f"width: {width}")
    # # print(f"check_string: {check_string}")
    # # Record the data
    # width_array[i] = width
    # # Record any errors
    # if "0" in check_string:
    #     error_message = f"Sample: {i} Zero_Finding_Error: {check_string}"
    #     error_list.append(error_message)
    # else:
    #     pass

    # Serial analysis
    for i in range(num_samples):
        print(i)
        # Pick the x columns
        x_index = 2*i
        # Remove the NaN values
        x_data, y_data = exclude_NaN(x_index, length_df)
        # Calculate the width
        width, check_string = calculate_width_min_max(x_data, y_data, \
            smooth_func, smooth_params, zero_cross_params)
        # Record the data
        width_array[i] = width
        # Record any errors
        if "0" in check_string:
            error_message = f"Sample: {i} Zero_Finding_Error: {check_string}"
            error_list.append(error_message)
        else:
            pass


    # Calculate the mean, median, and standard deviation
    width_array_clean = width_array[np.logical_not(np.isnan(width_array))]
    mean = np.mean(width_array_clean)
    median = np.median(width_array_clean)
    stdev = np.std(width_array_clean)
    # Pack up the stats
    summary_stats = num_samples, len(width_array_clean), mean, median, stdev 

    # Save the data
    # Write a header file
    write_header(custom_name, file_name, smooth_method, smooth_params, \
    zero_method, zero_cross_params, width_method, error_list, summary_stats, note)
    # Write a data file
    write_measurement_data_serial(custom_name, file_name, width_array)