import argparse
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
# from scipy import special

# Data Preparation Methods
def exclude_NaN(i, length_df):
    x_data = length_df[i][np.logical_not(np.isnan(length_df[i]))]
    y_data = length_df[i+1][np.logical_not(np.isnan(length_df[i+1]))]
    return(x_data, y_data)

# # Smoothing Methods

# Derivative Methods
def central_diff(x_vals, y_vals):
    """
    central_diff(x_vals, y_vals)

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
def check_found_edges(base_loc_arr):
    check_string = ""
    for location in base_loc_arr:
        if location < 0:
            check_string += "0"
        else:
            check_string += "1"
    return check_string

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

<<<<<<< HEAD
def find_base_zero(x_d1, y_smooth_d1, peak_list, directions_list, base_params):
    # Unpack the parameters
    step_size, threshold, max_steps = base_params

    # Create an array to store the base positions in
    base_loc_arr = np.zeros(len(peak_list),dtype=int)

    # Calculate the zero crossing points of the first derivative
    # Rising peak
    for i in range(len(base_loc_arr)):
        base_loc_arr[i] = find_zero_crossing(y_smooth_d1, peak_list[i], \
            step_size, directions_list[i], threshold=threshold, max_steps=max_steps)
    
    # Check if bases were found
    base_string = check_found_edges(base_loc_arr)

    if "0" in base_string:
        error_string = f"Zero_Base_Finding_Error: {base_string}"
    else:
        error_string = ""

    return(base_loc_arr, error_string, base_string)

    

# # Second Derivative Threshold Method
# # #
def find_local_supremum(data, start_point, direction, max_steps=20):
    supremum = -1
    num_steps = 0
    found_local_supremum = False
    check_point = start_point + direction
    if data[check_point]-data[start_point] > 0:
        sign = 1
    else:
        sign = -1
    while not found_local_supremum and num_steps < max_steps:
        test_step = check_point+direction
        # Check if the test step is in bounds
        if 0 <= test_step < len(data):
            if sign*(data[test_step]-data[check_point]) > 0:
                check_point = test_step
            else:
                supremum = check_point
                found_local_supremum = True
            num_steps += 1
        else: 
            # If the test step is out of bounds, set the crossing point to the test step
            found_local_supremum = True
            supremum = check_point
    return supremum

def find_local_value(target_value, data, start_point, direction, max_steps=20):
    """
    """
    position = -1
    num_steps = 0
    found_local_value = False
    check_point = start_point
    if data[start_point] > 0:
        sign = 1
    else:
        sign = -1
    while not found_local_value and num_steps < max_steps:
        test_step = check_point+direction
        # Check if the test step is in bounds
        if 0 <= test_step < len(data):
            if sign*(data[test_step]) > target_value:
                check_point = test_step
            else:
                # Pick the point that's closest to the target value
                if sign*(data[test_step]) - target_value > sign*(data[check_point]) - target_value:
                    position = check_point
                else:
                    position = test_step
                found_local_value = True
            num_steps += 1
        else: 
            # If the test step is out of bounds, set the crossing point to the test step
            found_local_value = True
            position = check_point
    return position

def find_base_d2(x_d1, y_smooth_d1_s, peak_list, directions_list, base_params):
    global smooth_func
    global smooth_params
    # Unpack the parameters
    target_value, step_size, threshold, max_steps = base_params
    # Create an array to store the local supremum in
    supremum_loc_arr = np.zeros(len(peak_list),dtype=int)
    # Create an array to store the base positions in
    base_loc_arr = np.zeros(len(peak_list),dtype=int)

    # Calculate the second derivative
    x_d2, y_smooth_d2 = central_diff(x_d1, y_smooth_d1_s)
    # Smooth the first derivative
    y_smooth_d2_s = smooth_func(y_smooth_d2, *smooth_params)

    # Find local supremum around the rising and falling peak locations of the
    # 1st derivative
    for i in range(len(supremum_loc_arr)):
        supremum_loc_arr[i] = find_local_supremum(y_smooth_d2_s, peak_list[i], \
                directions_list[i], max_steps=max_steps)

    # Make sure local supremum was found  
    suprema_string = check_found_edges(supremum_loc_arr)

    for i in range(len(supremum_loc_arr)):
        # If a local supremum was not found, find the base location using the 
        # first derivativezero calculation method
        if supremum_loc_arr[i] < 0:
            base_loc_arr[i] = find_zero_crossing(y_smooth_d1_s, peak_list[i], \
                step_size, directions_list[i], threshold=threshold, \
                max_steps=max_steps)
        # If a local supremum was found, find the base locatin using the 2nd 
        # derivative threshold method
        else:
            base_loc_arr[i] = find_local_value(target_value, y_smooth_d2_s, \
            supremum_loc_arr[i], directions_list[i], max_steps=max_steps)

    # Check if bases were found
    base_string = check_found_edges(base_loc_arr)

    if "0" in suprema_string and "0" not in base_string:
        error_string = f"Supremum_Finding_Error: {suprema_string}\tFound all bases: {base_string}"
    elif "0" in suprema_string and "0" in base_string:
        error_string = f"Supremum_Finding_Error: {suprema_string}\tD2_Base_Finding_Error: {base_string}"
    elif "0" not in suprema_string and "0" in base_string:
        error_string = f"D2_Base_Finding_Error: {base_string}"
    else:
        error_string = ""
    return(base_loc_arr, error_string, base_string)

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
=======
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
>>>>>>> parent of 17f1639... added file processing to TEM_length



# Baseline Correction 
def fit_mid_baseline(x_data, y_data, base_end_arr):
    slope = (y_data[base_end_arr[1]]-y_data[base_end_arr[0]])/(x_data[base_end_arr[1]]-x_data[base_end_arr[0]])
    intercept = y_data[base_end_arr[0]] - slope*x_data[base_end_arr[0]]
    def mid_baseline_fit(x):
        return slope*x+intercept
    return mid_baseline_fit, slope, intercept

def fit_baseline(x_data, y_smooth, base_end_arr):
    # Fit the baseline to straighten the middle portion
    mid_baseline_fit, bl_slope, bl_intercept = fit_mid_baseline(x_data, y_smooth, base_end_arr)
    x_baseline = x_data[base_end_arr[0]:base_end_arr[1]+1]
    mid_fit_baseline = mid_baseline_fit(x_baseline)

    # Prepare the outer edge data (center them so the base endpoints are at 0,0)
    # Left side
    x_left_centered = x_data[:base_end_arr[0]+1] - x_data[base_end_arr[0]]
    x_left_centered = x_left_centered.values
    y_smooth_left_centered  = y_smooth[:base_end_arr[0]+1]-y_smooth[base_end_arr[0]]
    # Right side
    x_right_centered = x_data[base_end_arr[1]:] - x_data[base_end_arr[1]]
    x_right_centered = x_right_centered.values
    y_smooth_right_centered = y_smooth[base_end_arr[1]:]-y_smooth[base_end_arr[1]]

    # Fit left base
    lm_left = LinearRegression(fit_intercept = False)
    lm_left.fit(x_left_centered.reshape(-1,1), y_smooth_left_centered)
    # Fit the right base
    lm_right = LinearRegression(fit_intercept = False)
    lm_right.fit(x_right_centered.reshape(-1,1), y_smooth_right_centered)

    # Assemble the baseline correction
    baseline_correction = np.zeros(len(x_data))
    baseline_correction[:base_end_arr[0]] = lm_left.predict(x_left_centered[:-1].reshape(-1,1)) +  y_smooth[base_end_arr[0]]
    baseline_correction[base_end_arr[0]:base_end_arr[1]+1] = mid_fit_baseline
    baseline_correction[base_end_arr[1]+1:] = lm_right.predict(x_right_centered[1:].reshape(-1,1)) +  y_smooth[base_end_arr[1]]

    # Add the baseline correction to the y data
    y_smooth_blc = y_smooth - baseline_correction
    return y_smooth_blc, baseline_correction

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
<<<<<<< HEAD
def calculate_width_min_max(x_data, y_data, smooth_func, smooth_params, \
    base_func, base_params, adjust_index):
    # Smooth the function
    y_smooth = smooth_func(y_data, *smooth_params)

    # Calculate the first derivative
    x_d1, y_smooth_d1 = central_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find the peaks (hill and valley) of the first derivative
    rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[4:-5]))[0][0]
    falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[4:-5]))[0][0]
    # print(f"rising_peak_loc: {rising_peak_loc}")
    # print(f"falling_peak_loc: {falling_peak_loc}")
    
    # Store peaks in peak list
    peak_list = [rising_peak_loc, rising_peak_loc, falling_peak_loc, falling_peak_loc]
    directions_list = [-1, 1, -1, 1]

    # Find base points of the peaks of the first derivative
    base_loc_arr, error_string, base_string = base_func(x_d1, y_smooth_d1_s, \
        peak_list, directions_list, base_params)

    if "0" in base_string:
        width = np.nan
    else:
        # Adjust the indices (for the 2nd derivative method)
        base_loc_arr = base_loc_arr + adjust_index

        # Figure out the min and max values of the edges
        rising_peak_min = np.min(y_smooth[base_loc_arr[0]:base_loc_arr[1]+1]) 
        rising_peak_max = np.max(y_smooth[base_loc_arr[0]:base_loc_arr[1]+1])
        falling_peak_max = np.max(y_smooth[base_loc_arr[2]:base_loc_arr[3]+1])
        falling_peak_min = np.min(y_smooth[base_loc_arr[2]:base_loc_arr[3]+1])
        # print(f"rising_peak_min: {rising_peak_min}")
        # print(f"rising_peak_max: {rising_peak_max}")
        # print(f"falling_peak_max : {falling_peak_max}") 
        # print(f"falling_peak_min: {falling_peak_min}")
        # Calculate the half max values
        rising_peak_half_max = calculate_half_max(rising_peak_max, \
            rising_peak_min)
        falling_peak_half_max = calculate_half_max(falling_peak_max, \
            falling_peak_min)
        # print(f"rising_peak_half_max: {rising_peak_half_max}")
        # print(f"falling_peak_half_max: {falling_peak_half_max}")
        # Find the half max neighbors
        rising_half_max_bounds = find_half_max_neighbors(rising_peak_half_max, \
            y_smooth, 1, base_loc_arr[0], base_loc_arr[1])
        falling_half_max_bounds = find_half_max_neighbors(falling_peak_half_max, \
            y_smooth, -1, base_loc_arr[2], base_loc_arr[3])
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

    return width, error_string

def calc_width_baseline_correction(x_data, y_data, smooth_func, smooth_params, \
    base_func, base_params, adjust_index): 
    # Smooth the function
    y_smooth = smooth_func(y_data, *smooth_params)

    # Calculate the first derivative
    x_d1, y_smooth_d1 = central_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find the peaks (hill and valley) of the first derivative
    rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[4:-5]))[0][0]
    falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[4:-5]))[0][0]
    # print(f"rising_peak_loc: {rising_peak_loc}")
    # print(f"falling_peak_loc: {falling_peak_loc}")

    # Store peaks in peak list
    base_peak_list = [rising_peak_loc, falling_peak_loc]
    base_directions_list = [-1, 1]
    # Find base points of the peaks of the first derivative
    base_end_arr, error_string, base_string = base_func(x_d1, y_smooth_d1_s, \
        base_peak_list, base_directions_list, base_params)
    # print(base_peak_list)
    # print(base_end_arr)

    # Adjust the indices (for the 2nd derivative method)
    base_end_arr = base_end_arr + adjust_index
    # print("adjusted base end array")
    # print(base_end_arr)

    if "0" in base_string:
        error_string = "Baseline correction failed: " + error_string
        width = np.nan
    else:
        # Perform baseline correction
        y_smooth_blc, baseline_correction = fit_baseline(x_data, y_smooth, \
            base_end_arr)

        # print("corrected base")
        # print(y_smooth_blc[:10])

        # Calculate the first derivative
        x_d1, y_smooth_d1_blc = central_diff(x_data, y_smooth_blc)
        # Smooth the first derivative
        y_smooth_d1_blc_s = smooth_func(y_smooth_d1_blc, *smooth_params)

        
        # Find the peaks (hill and valley) of the baseline corrected first derivative
        rising_peak_loc_blc = np.where(y_smooth_d1_blc_s==np.max(y_smooth_d1_blc_s[4:-5]))[0][0]
        falling_peak_loc_blc = np.where(y_smooth_d1_blc_s==np.min(y_smooth_d1_blc_s[4:-5]))[0][0]
        # print("found peaks")
        # print(rising_peak_loc_blc)
        # print(falling_peak_loc_blc)
        # Store peaks in peak list
        peak_list = [rising_peak_loc_blc, rising_peak_loc_blc, \
        falling_peak_loc_blc, falling_peak_loc_blc]
        directions_list = [-1, 1, -1, 1]
        
        # Find base points of the peaks of the first derivative
        base_loc_arr, error_string, base_string = base_func(x_d1, y_smooth_d1_blc_s, \
            peak_list, directions_list, base_params)
        # print("found base points")
        # print(base_loc_arr)
        if "0" in base_string:
            # print(error_string)
            width = np.nan
        else:
            # Adjust the indices (for the 2nd derivative method)
            base_loc_arr = base_loc_arr + adjust_index

            # Figure out the min and max values of the edges
            rising_peak_min = np.min(y_smooth_blc[base_loc_arr[0]:base_loc_arr[1]+1]) 
            rising_peak_max = np.max(y_smooth_blc[base_loc_arr[0]:base_loc_arr[1]+1])
            falling_peak_max = np.max(y_smooth_blc[base_loc_arr[2]:base_loc_arr[3]+1])
            falling_peak_min = np.min(y_smooth_blc[base_loc_arr[2]:base_loc_arr[3]+1])
            # print(f"rising_peak_min: {rising_peak_min}")
            # print(f"rising_peak_max: {rising_peak_max}")
            # print(f"falling_peak_max : {falling_peak_max}") 
            # print(f"falling_peak_min: {falling_peak_min}")
            # Calculate the half max values
            rising_peak_half_max = calculate_half_max(rising_peak_max, \
                rising_peak_min)
            falling_peak_half_max = calculate_half_max(falling_peak_max, \
                falling_peak_min)
            # print(f"rising_peak_half_max: {rising_peak_half_max}")
            # print(f"falling_peak_half_max: {falling_peak_half_max}")
            # Find the half max neighbors
            rising_half_max_bounds = find_half_max_neighbors(rising_peak_half_max, \
                y_smooth_blc, 1, base_loc_arr[0], base_loc_arr[1])
            falling_half_max_bounds = find_half_max_neighbors(falling_peak_half_max, \
                y_smooth_blc, -1, base_loc_arr[2], base_loc_arr[3])
            # print(f"rising_half_max_bounds: {rising_half_max_bounds}")
            # print(f"falling_half_max_bounds: {falling_half_max_bounds}")
            # Find the half max positions
            rising_half_max_pos = find_half_max_pos(rising_peak_half_max, \
                rising_half_max_bounds, x_data, y_smooth_blc)
            falling_half_max_pos = find_half_max_pos(falling_peak_half_max, \
                falling_half_max_bounds, x_data, y_smooth_blc)
            # print(f"rising_half_max_pos: {rising_half_max_pos}")
            # print(f"falling_half_max_pos: {falling_half_max_pos}")

            # Calculate the half max full width
            width = falling_half_max_pos - rising_half_max_pos

    return width, error_string

# Writing Output Files
def write_header(custom_name, file_name, smooth_method, smooth_params, \
    base_method, base_params, width_method, error_list, summary_stats, note):
    with open(f"{file_name}_header_{custom_name}.txt", 'w') as header_file:
        # Record the calculation methods and settings
        header_file.write(f"Smoothing method: {smooth_method}\n")
        header_file.write(f"Smoothing parameters: {smooth_params}\n")
        header_file.write(f"Base finding method: {base_method}\n")
        header_file.write(f"Base finding parameters: {base_params}\n")
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
=======
def calculate_width_min_max(x_data, y_data, smooth_func, smooth_params)
>>>>>>> parent of 17f1639... added file processing to TEM_length

if __name__ == "__main__":
    # add some arg parse stuff here

<<<<<<< HEAD
    # Hard code some stuff
    smooth_func = signal.savgol_filter
    smooth_params = (21, 3)
    d2_threshold = 0.5
    step_size = 2
    threshold = 2
    max_steps = 20
    # Base finding method
    # Zero crossing method
    # base_method = "1st derivative zero crossing"
    # adjust_index = 0
    # base_params = (step_size, threshold, max_steps)
    # base_func = find_base_zero
    # custom_name = "serial_d1_zero"
    # 2nd derivative method
    base_method = "2nd derivative threshold"
    adjust_index = 1
    base_params = (d2_threshold, step_size, threshold, max_steps)
    base_func = find_base_d2
    # custom_name = "serial_d2_threshold"
    custom_name = "serial_d2_threshold_baseline_correction"


    # custom_name = "serial"
    # file_path = "/Users/emilywu/OneDrive - Massachusetts Institute of Technology/TEM width/42hb polyplex no stain xy values length.xlsx"
    file_path = "/Users/emilywu/OneDrive - Massachusetts Institute of Technology/TEM width/42hb polyplex no stain xy values width.xlsx"
    # file_name = "42hb polyplex no stain xy values length"
    file_name = "42hb polyplex no stain xy values width"
    smooth_method = "savgol"
    width_method = "min_max"
    note = ""

=======
    file_path = "/Users/emilywu/OneDrive - Massachusetts Institute of Technology/TEM width/42hb polyplex no stain xy values length.xlsx"
    
>>>>>>> parent of 17f1639... added file processing to TEM_length
    # Read the file in 
    len_data = pd.read_excel(file_path, header=None,names=None)

    # Get the shape
<<<<<<< HEAD
    rows, cols = length_df.shape
    num_samples = int(cols/2)
    
    width_array = np.zeros(num_samples)
    error_list = []

    # i = 398
    # # for i in range(13):
    # print(f"sample {i}")
    # # Pick the x columns
    # x_index = 2*i
    # # Remove the NaN values
    # x_data, y_data = exclude_NaN(x_index, length_df)
    # # Calculate the width
    # width, error_string = calculate_width_min_max(x_data, y_data, \
    #     smooth_func, smooth_params, base_func, base_params, adjust_index)
    # print(f"width: {width}")
    # print(f"error_string: {error_string}")
    # # Record the data
    # width_array[i] = width
    # # Record any errors
    # if len(error_string)>0:
    #     error_message = f"Sample: {i} {error_string}"
    #     error_list.append(error_message)
    #     print(f"error_message: {error_message}")
    # else:
    #     pass

    # Serial analysis (no baseline correction)
    # for i in range(num_samples):
    #     print(i)
    #     # Pick the x columns
    #     x_index = 2*i
    #     # Remove the NaN values
    #     x_data, y_data = exclude_NaN(x_index, length_df)
    #     # Calculate the width
    #     width, error_string = calculate_width_min_max(x_data, y_data, \
    #         smooth_func, smooth_params, base_func, base_params, adjust_index)
    #     # Record the data
    #     width_array[i] = width
    #     # Record any errors
    #     if len(error_string)>0:
    #         error_message = f"Sample: {i} {error_string}"
    #         error_list.append(error_message)
    #         print(error_message)
    #     else:
    #         pass

    # i = 0
    # # for i in range(13):
    # print(f"sample {i}")
    # # Pick the x columns
    # x_index = 2*i
    # # Remove the NaN values
    # x_data, y_data = exclude_NaN(x_index, length_df)
    # # Calculate the width
    # width, error_string = calc_width_baseline_correction(x_data, y_data, \
    #     smooth_func, smooth_params, base_func, base_params, adjust_index)
    # print(f"width: {width}")
    # print(f"error_string: {error_string}")
    # # Record the data
    # width_array[i] = width
    # # Record any errors
    # if len(error_string)>0:
    #     error_message = f"Sample: {i} {error_string}"
    #     error_list.append(error_message)
    #     print(f"error_message: {error_message}")
    # else:
    #     pass

    # Serial analysis (baseline correction)
    for i in range(num_samples):
        print(i)
        # Pick the x columns
        x_index = 2*i
        # Remove the NaN values
        x_data, y_data = exclude_NaN(x_index, length_df)
        # Calculate the width
        width, error_string = calc_width_baseline_correction(x_data, y_data, \
            smooth_func, smooth_params, base_func, base_params, adjust_index)
        # Record the data
        width_array[i] = width
        # Record any errors
        if len(error_string)>0:
            error_message = f"Sample: {i} {error_string}"
            error_list.append(error_message)
            print(error_message)
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
    base_method, base_params, width_method, error_list, summary_stats, note)
    # Write a data file
    write_measurement_data_serial(custom_name, file_name, width_array)

=======
    rows, cols = len_data.shape
    print(len_data[1])
    mean = len_data[1].mean()
    print(mean)
    # Iterate through all of the grayscale columns
    # for i in range(cols/2):
    #     pass
>>>>>>> parent of 17f1639... added file processing to TEM_length
