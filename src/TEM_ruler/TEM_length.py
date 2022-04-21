import argparse
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression

# Data Preparation Methods
def exclude_NaN(x_index, length_df):
    x_data = length_df[x_index][np.logical_not(np.isnan(length_df[x_index]))]
    y_data = length_df[x_index+1][np.logical_not(np.isnan(length_df[x_index+1]))]
    return(x_data, y_data)

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

def find_zero_crossing(derivative, mu, step_size, direction, \
    threshold=2, max_steps=20):
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

def find_base_zero(x_d1, y_smooth_d1, y_smooth_d1_s, peak_list, directions_list, base_params): # Check this
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

    return(base_loc_arr, error_string, base_string, "0"*len(base_loc_arr))

    

# # Second Derivative Threshold Method
# # #
def find_local_supremum(data, start_point, direction, max_steps=20):
    supremum = -1
    num_steps = 0
    found_local_supremum = False
    check_point = start_point
    if data[check_point+direction]-data[start_point] > 0:
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
                # if sign*(data[test_step] - target_value) > sign*(data[check_point]) - target_value:
                if abs(sign*(data[test_step]) - target_value) > abs(sign*(data[check_point]) - target_value):
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

def find_base_d2(x_d1, y_smooth_d1, y_smooth_d1_s, peak_list, directions_list, \
    base_params):
    """
    """
    # Unpack the parameters
    target_value, step_size, threshold, max_steps, smooth_func, smooth_params = base_params
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
            base_loc_arr[i] = find_zero_crossing(y_smooth_d1, peak_list[i], \
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
    return(base_loc_arr, error_string, base_string, suprema_string)


# Baseline Correction 
def fit_plateau_baseline(x_data, y_data, base_end_arr):
    slope = (y_data[base_end_arr[1]]-y_data[base_end_arr[0]])/(x_data[base_end_arr[1]]-x_data[base_end_arr[0]])
    intercept = y_data[base_end_arr[0]] - slope*x_data[base_end_arr[0]]
    def plateau_baseline_fit(x):
        return slope*x+intercept
    return plateau_baseline_fit, slope, intercept

def fit_baseline(x_data, y_smooth, base_end_arr):
    # Create an array to store the baseline correction
    baseline_correction = np.zeros(len(x_data))

    # Fit the baseline to straighten the middle portion
    plateau_baseline_fit, bl_slope, bl_intercept = fit_plateau_baseline(x_data, y_smooth, base_end_arr)
    x_plateau = x_data[base_end_arr[0]:base_end_arr[1]+1]
    plateau_baseline = plateau_baseline_fit(x_plateau)
    baseline_correction[base_end_arr[0]:base_end_arr[1]+1] = plateau_baseline   

    
    # Left side
    if base_end_arr[0] > 0:    
        # Prepare the outer edge data (center them so the base endpoints are at 0,0)
        x_left_centered = x_data[:base_end_arr[0]+1] - x_data[base_end_arr[0]]
        x_left_centered = x_left_centered.values
        y_smooth_left_centered  = y_smooth[:base_end_arr[0]+1]-y_smooth[base_end_arr[0]]
        # Fit left base
        lm_left = LinearRegression(fit_intercept = False)
        lm_left.fit(x_left_centered.reshape(-1,1), y_smooth_left_centered)
        baseline_correction[:base_end_arr[0]] = \
        lm_left.predict(x_left_centered[:-1].reshape(-1,1)) +  y_smooth[base_end_arr[0]]
    else:
        pass
    # Right side    
    if base_end_arr[1] < len(x_data)-1:    
        # Prepare the outer edge data (center them so the base endpoints are at 0,0)
        x_right_centered = x_data[base_end_arr[1]:] - x_data[base_end_arr[1]]
        x_right_centered = x_right_centered.values
        y_smooth_right_centered = y_smooth[base_end_arr[1]:]-y_smooth[base_end_arr[1]]
        # Fit the right base
        lm_right = LinearRegression(fit_intercept = False)
        lm_right.fit(x_right_centered.reshape(-1,1), y_smooth_right_centered)
        baseline_correction[base_end_arr[1]+1:] = \
        lm_right.predict(x_right_centered[1:].reshape(-1,1)) +  y_smooth[base_end_arr[1]]
    
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
    slope = (y_data[half_max_bounds[0]]-y_data[half_max_bounds[1]])/(x_data[half_max_bounds[0]]-x_data[half_max_bounds[1]])
    intercept = y_data[half_max_bounds[0]] - slope*x_data[half_max_bounds[0]]
    half_max_pos = (half_max-intercept)/slope
    return half_max_pos

# Width calculation methods
def calculate_half_max_full_width_pos(x_data, y_smooth, base_loc_arr):
    # print(f"base loc arr: {base_loc_arr}")
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

    # Calculate the half max full width positions
    return np.array([rising_half_max_pos, falling_half_max_pos])

def calculate_half_max_full_width(x_data, y_smooth, base_loc_arr):
    half_max_positions = \
    calculate_half_max_full_width_pos(x_data, y_smooth, base_loc_arr)

    # Calculate the half max full width
    width = half_max_positions[1] - half_max_positions[0]

    return width

def calculate_width_min_max(x_data, y_smooth, smooth_func, smooth_params, \
    base_func, base_params, adjust_index):
    # Calculate the first derivative
    x_d1, y_smooth_d1 = central_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find the peaks (hill and valley) of the first derivative
    rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[5:-5]))[0][0]
    falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[5:-5]))[0][0]
    # print(f"rising_peak_loc: {rising_peak_loc}")
    # print(f"falling_peak_loc: {falling_peak_loc}")
    
    # Store peaks in peak list
    peak_list = [rising_peak_loc, rising_peak_loc, falling_peak_loc, falling_peak_loc]
    directions_list = [-1, 1, -1, 1]

    # Find base points of the peaks of the first derivative
    base_loc_arr, error_string, base_string, suprema_string = base_func(x_d1, \
        y_smooth_d1, y_smooth_d1_s, peak_list, directions_list, base_params)

    if "0" in base_string:
        width = np.nan
    else:
        # Adjust the indices (for the 2nd derivative method)
        for i in range(len(suprema_string)):
            if suprema_string[i] == "1":
                base_loc_arr[i] = base_loc_arr[i] + adjust_index
            else:
                pass

        # Calculate the half max full width
        width = calculate_half_max_full_width(x_data, y_smooth, base_loc_arr)

    return width, error_string

def calc_width_baseline_correction(x_data, y_smooth, smooth_func, smooth_params, \
    base_func, base_params, adjust_index): 
    # Calculate the first derivative
    x_d1, y_smooth_d1 = central_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find the peaks (hill and valley) of the first derivative
    rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[5:-5]))[0][0]
    falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[5:-5]))[0][0]
    # print(f"rising_peak_loc: {rising_peak_loc}")
    # print(f"falling_peak_loc: {falling_peak_loc}")

    # Store peaks in peak list
    base_peak_list = [rising_peak_loc, falling_peak_loc]
    base_directions_list = [-1, 1]
    # Find base points of the peaks of the first derivative
    base_end_arr, error_string, base_string, suprema_string = base_func(x_d1, \
        y_smooth_d1, y_smooth_d1_s, base_peak_list, base_directions_list, base_params)
    # print(base_peak_list)
    # print(base_end_arr)

    # Adjust the indices (for the 2nd derivative method)
    for i in range(len(suprema_string)):
        if suprema_string[i] == "1":
            base_end_arr[i] = base_end_arr[i] + adjust_index
        else:
            pass
    # print("adjusted base end array")
    # print(base_end_arr)

    if "0" in base_string:
        error_string = "Baseline correction failed: " + error_string
        width = np.nan
    else:
        # Perform baseline correction
        y_smooth_blc, baseline_correction = fit_baseline(x_data, y_smooth, \
            base_end_arr)
        
        # Calculate the width
        width, error_string = calculate_width_min_max(x_data, y_smooth_blc, \
            smooth_func, smooth_params, base_func, base_params, adjust_index)

    return width, error_string

# Reading settings
def read_TEM_length_settings(settings_path):
    # Default settings
    smooth_method = "savgol"
    smooth_func = signal.savgol_filter
    smooth_params = (21, 3)
    width_method = "min_max"
    base_method = "2nd derivative threshold"
    base_func = find_base_d2
    step_size = 2
    threshold = 2
    max_steps = 20
    d2_threshold = 0.5
    base_params = (d2_threshold, step_size, threshold, max_steps, smooth_func, smooth_params)
    # x position of 2nd derivative is shifted by 1 from the x position of 
    # the TEM profile
    adjust_index = 1 

    # Handle custom settings
    if settings_path == "default":
        pass 
    else:
        with open(settings_path, "r") as settings_file:
            lines = settings_file.readlines()
            try:
                for line in lines:
                    line = line.rstrip("\n")
                    line_list = line.split(" = ")
                    if line_list[0] == "smooth_method" and line_list[1] != "default":
                        smooth_method = line_list[1]
                    elif line_list[0] == "smooth_params" and line_list[1] != "default":
                        smooth_params_list = line_list[1].strip("()").split(", ")
                        smooth_params_list = [int(param) for param in smooth_params_list]
                        smooth_params = (*smooth_params_list, )
                    elif line_list[0] == "step_size" and line_list[1] != "default":
                        step_size = int(line_list[1])
                    elif line_list[0] == "threshold" and line_list[1] != "default":
                        threshold = int(line_list[1])
                    elif line_list[0] == "max_steps" and line_list[1] != "default":
                        max_steps = int(line_list[1])
                    elif line_list[0] == "base_method" and line_list[1] != "default":
                        base_method = line_list[1]
                    elif line_list[0] == "d2_threshold" and line_list[1] != "default":
                        d2_threshold = float(line_list[1])
                    else:
                        pass
                # Adjust baseline settings
                if base_method == "1st derivative zero crossing": 
                    base_func = find_base_zero
                    base_params = (step_size, threshold, max_steps)
                elif base_method == "2nd derivative threshold":
                    base_func = find_base_d2
                    base_params = (d2_threshold, step_size, threshold, max_steps, \
                        smooth_func, smooth_params)
                else:
                    raise ValueError("ValueError: Suppled base finding method " \
                        + "not recognized. Please use '1st derivative zero " \
                        + "crossing' or '2nd derivative threshold'.")
            except (ValueError) as msg:
                print(msg)
                # Stop the script execution
                sys.exit(1)
    return (smooth_method, smooth_func, smooth_params, width_method, base_method, \
        base_func, base_params, adjust_index)

# Writing Output Files
def write_header(custom_name, file_name, smooth_method, smooth_params, \
    base_method, base_params, width_method, error_list, summary_stats, note):
    with open(f"{file_name}_header_{custom_name}.txt", 'w') as header_file:
        # Record the calculation methods and settings
        header_file.write("######## Calculation settings ########\n")
        header_file.write(f"Smoothing method: {smooth_method}\n")
        header_file.write(f"Smoothing parameters: {smooth_params}\n")
        header_file.write(f"Base finding method: {base_method}\n")
        if base_method == "2nd derivative threshold":
            d2_threshold, step_size, threshold, max_steps, smooth_func, \
            smooth_params = base_params
            base_params = (d2_threshold, step_size, threshold, max_steps)
        else:
            pass
        header_file.write(f"Base finding parameters: {base_params}\n")
        header_file.write(f"Width calculation method: {width_method}\n")
        header_file.write(f"Note: {note}\n")
        # Unpack the summary stats
        num_samples, summary_num_samples, mean, median, sample_stdev = summary_stats
        # Record the summary stats
        header_file.write(f"\n######## Summary stats ########\n")
        header_file.write(f"Number of samples: {num_samples}\n")
        header_file.write(f"Number of summary stat samples: {summary_num_samples}\n")
        header_file.write(f"mean: {mean}\n")
        header_file.write(f"median: {median}\n")
        header_file.write(f"sample stdev: {sample_stdev}\n")
        # Record any errors
        header_file.write("\n######## Errors ########\n")
        header_file.write(f"Number of errors: {len(error_list)}\n")
        for error in error_list:
            header_file.write(f"{error}\n")

def write_measurement_data(custom_name, file_name, width_array):
    with open(f"{file_name}_measurements_{custom_name}.txt", 'w') as data_file:
        for width in width_array:
            data_file.write(f"{width}\t")

def TEM_length_main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Takes in an excel file of TEM \
        grayscale profiles and computes the lengths of the objects using a half \
        max full width approach.")
    parser.add_argument("read_file", type=str, \
        help="The path to the excel file to analyze.")
    parser.add_argument("save_name", type=str, \
        help="The name extension to add to save file names that results are saved to.")
    parser.add_argument("--baseline", type=str, 
        help="If True, applies a baseline correction before trying to determine the length. Defaults to False.")
    parser.add_argument("--settings", type=str, \
        help="The path to the file containing analysis settings.")
    parser.add_argument("--note", type=str, \
        help="A note to add to the header file")
    parser.add_argument("--prog", type=str, \
        help="If True, prints sample number to terminal as it is being analyzed. Defaults to False")

    args = parser.parse_args()

    # Parse the arguments
    read_file = args.read_file
    file_name = read_file.split("/")[-1].split(".")[0]
    custom_name = args.save_name

    true_list = ["True", "true"]
    if args.baseline and args.baseline in true_list:
        use_baseline = True
    else:
        use_baseline = False

    # Get settings
    if args.settings:
        settings_path = args.settings
    else:
        settings_path = "default"
    smooth_method, smooth_func, smooth_params, width_method, base_method, \
    base_func, base_params, adjust_index = read_TEM_length_settings(settings_path)

    # Read in the note
    if args.note:
        note = args.note
    else:
        note = ""

    if args.prog and args.prog in true_list:
        progress = True
    else:
        progress = False

    # Read the file in 
    length_df = pd.read_excel(read_file, header=None, names=None)

    # Get the shape
    rows, cols = length_df.shape
    num_samples = int(cols/2)
    
    # Create array/list to store the widths and errors
    width_array = np.zeros(num_samples)
    error_list = []

    # Serial measurements
    print("Making measurements.")
    for i in range(num_samples):
        if progress:
            print(i)
        else:
            pass
        # Pick the x columns
        x_index = 2*i
        # Remove the NaN values
        x_data, y_data = exclude_NaN(x_index, length_df)
        # Smooth the function
        y_smooth = smooth_func(y_data, *smooth_params)
        # Calculate the width
        if use_baseline:
            width, error_string = calc_width_baseline_correction(x_data, y_smooth, \
                smooth_func, smooth_params, base_func, base_params, adjust_index)
        else:
            width, error_string = calculate_width_min_max(x_data, y_smooth, \
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
    sample_stdev = np.std(width_array_clean, ddof=1)
    # Pack up the stats
    summary_stats = num_samples, len(width_array_clean), mean, median, sample_stdev 

    # Save the data
    print("Writing header.")
    # Write a header file
    write_header(custom_name, file_name, smooth_method, smooth_params, \
    base_method, base_params, width_method, error_list, summary_stats, note)
    

    # Write a data file
    print("Writing measurement data file.")
    write_measurement_data(custom_name, file_name, width_array)
    print("Finished.")

if __name__ == "__main__":
    TEM_length_main()
    
    # i = 11
    # # for i in range(13):
    # print(f"sample {i}")
    # # Pick the x columns
    # x_index = 2*i
    # # Remove the NaN values
    # x_data, y_data = exclude_NaN(x_index, length_df)
    # # Smooth the function
    # y_smooth = smooth_func(y_data, *smooth_params)
    # # Calculate the width
    # width, error_string = calculate_width_min_max(x_data, y_smooth, \
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

    # # Serial analysis (no baseline correction)
    # for i in range(num_samples):
    #     print(i)
    #     # Pick the x columns
    #     x_index = 2*i
    #     # Remove the NaN values
    #     x_data, y_data = exclude_NaN(x_index, length_df)
    #     # Smooth the function
    #     y_smooth = smooth_func(y_data, *smooth_params)
    #     # Calculate the width
    #     width, error_string = calculate_width_min_max(x_data, y_smooth, \
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

    # i = 11
    # # for i in range(13):
    # print(f"sample {i}")
    # # Pick the x columns
    # x_index = 2*i
    # # Remove the NaN values
    # x_data, y_data = exclude_NaN(x_index, length_df)
    # # Smooth the function
    # y_smooth = smooth_func(y_data, *smooth_params)
    # # Calculate the width
    # width, error_string = calc_width_baseline_correction(x_data, y_smooth, \
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
    # for i in range(num_samples):
    #     print(i)
    #     # Pick the x columns
    #     x_index = 2*i
    #     # Remove the NaN values
    #     x_data, y_data = exclude_NaN(x_index, length_df)
    #     # Smooth the function
    #     y_smooth = smooth_func(y_data, *smooth_params)
    #     # Calculate the width
    #     width, error_string = calc_width_baseline_correction(x_data, y_smooth, \
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
    # 