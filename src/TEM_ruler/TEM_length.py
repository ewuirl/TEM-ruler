import sys 
import argparse
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression

class FileInputError(Exception):
    """Read file is not an accepted format."""
    pass

def determine_os():
    """Determines the OS type to figure out how paths are formatted.

    Returns:
        file_separation (str): Returns '\\' if Windows, '/' if other.

    """
    # Windows
    if sys.platform == "win32":
        return "\\"
    # Not Windows
    else:
        return "/"

# Data Reading Methods
def read_TEM_data(read_file_path, transpose=False):
    """Reads in a TEM data file (.xlsx or tab separated .txt).

    Args:
        read_file_path (str): The path to the TEM data file. 
        transpose (bool): True if the data is transposed (rows). (Default value 
            = False)

    Returns:
        TEM_df (pd.DataFrame): A pandas dataframe containing the TEM data.

    Raises:
        FileInputError: If the read file is not a supported format (xlsx or tab 
            separated .txt)

    """
    try:
        if read_file_path[-5:] == ".xlsx":
            # Read the file in 
            TEM_df = pd.read_excel(read_file_path, header=None, names=None, \
                dtype=float)

        elif read_file_path[-4:] == ".txt":
            TEM_df = pd.read_table(read_file_path, header=None, \
                    names=None, dtype=float)
        else:
            raise FileInputError("FileInputError: Supplied file is not an xlsx or tab separated txt file.")

        if transpose:
            TEM_df = TEM_df.transpose()
        else:
            pass

        return TEM_df

    except(FileInputError) as msg:
        print(msg)
        # Stop the script execution
        sys.exit(1)

# Data Preparation Methods
def exclude_NaN(x_index, TEM_df):
    """Selects the specified x (position) column and corresponding y (height) column.
    Removes any NaN values.

    Args:
        x_index (int): The index of the desired x column. 
        TEM_df (pd.DataFrame): A pandas dataframe containing the TEM data.

    Returns:
        x_data (pd.Series): The desired x column with NaN removed.
        y_data (pd.Series): The corresponding y column with NaN removed.

    """
    x_data = TEM_df[x_index][np.logical_not(np.isnan(TEM_df[x_index]))]
    y_data = TEM_df[x_index+1][np.logical_not(np.isnan(TEM_df[x_index+1]))]
    return(x_data, y_data)

# Derivative Methods
def central_diff(x_vals, y_vals):
    """This function calculates an approximate first derivative using the central
    difference method.

    Args:
        x_vals (arr-like): The x values.
        y_vals (arr-like): The y values.

    Returns:
        x_diffs (numpy arr): The x positions of the derivative.
        diffs (numpy arr): The derivative calculated with the central difference 
            method.

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
    """Checks if the 1st derivative base locations were successfully found and 
    creates a record string.

    Args:
      base_loc_arr (numpy arr): An array of the indices of the base locations.

    Returns:
        check_string (str): Contains a "1" if the base was found and "0" if not.

    """
    check_string = ""
    for location in base_loc_arr:
        if location < 0:
            check_string += "0"
        else:
            check_string += "1"
    return check_string

def find_zero_crossing(derivative, mu, direction, step_size, \
    threshold=2, max_steps=20):
    """This function takes the first derivative, a peak location, a direction, 
    and looks for the point at which the derivative crosses zero (or the threshold) 
    in that direction from the peak.

    Args:
        derivative (numpy arr): The first derivative.
        mu (int): The x index of the 1st derivative peak location.
        direction (int): -1 if searching locations left of the peak, +1 if 
            searching right. 
        step_size (int): The step size to take in the search. Minimum is 1.
        threshold (float): The threshold value at which 'zero' is found. 
            (Default value = 2)
        max_steps (int): The maximum number of steps before the function gives 
            up. (Default value = 20)

    Returns:
        crossing_point (int): The x index of the 'zero crossing point'. Returns 
            -1 if not found.

    """
    crossing_point = -1
    found_crossing_point = False
    step = int(step_size*direction)
    check_point = mu + step
    num_steps = 0
    step_count = 1
    if derivative[check_point] > 0:
        sign = 1
    else:
        sign = -1
    while not found_crossing_point and num_steps < max_steps:
        test_step = check_point+step
        # Check if the test step is in bounds
        if 0 <= test_step < len(derivative):
            if sign*derivative[test_step] > threshold:
                check_point = test_step
                num_steps += step_count
            else:
                if abs(step) > 1:
                    step = direction
                    step_count = 0.5
                else:
                    # Pick the point closest to the threshold
                    if abs(sign*derivative[test_step]-1) > abs(sign*derivative[check_point]-1):
                        crossing_point = check_point
                    else:
                        crossing_point = test_step
                    found_crossing_point = True
        else: 
            # If the test step is out of bounds and the step size is greater than 1
            # Try using a smaller step size
            if abs(step) > 1:
                step = direction
            # If the test step is out of bounds, set the crossing point to the test step
            else:
                found_crossing_point = True
                crossing_point = check_point

    return crossing_point

def find_base_d1(x_d1, y_smooth_d1, y_smooth_d1_s, peak_list, directions_list, base_params, \
    verbose=False): 
    """ Uses the 1st derivative threshold crossing method to find the base 
    locations of 1st derivative peaks.

    Args:
        x_d1 (numpy arr): 1st derivative x positions.
        y_smooth_d1 (numpy arr): 1st derivative of the smoothed y data.
        y_smooth_d1_s (numpy arr): Smoothed 1st derivative.
        peak_list (list): A list of 1st derivative peak locations (x indices) to
            find base locations for. 
        directions_list (list): A list of the corresponding directions to search 
            in.
        base_params (tuple): Parameters for find_zero_crossing.
            step_size (int): The step size to take in the search. Minimum is 1.
            threshold (float): The threshold value at which 'zero' is found. 
            max_steps (int): The maximum number of steps to use in find_zero_crossing
        verbose (bool): Does not affect functionality. (Default value = False)      

    Returns:
        base_loc_arr (numpy arr): An array of the base locations (indices) of 
            the 1st deriative peaks.
        base_dict (dict): Contains auxiliary information.
            error_string (str): Describes any errors that have occurred.
            base_string (str): Describes which bases were found.
            suprema_string (str): Describes which suprema were found (not \
                relevant for the 1st derivative method.)

    """
    # Unpack the parameters
    step_size, threshold, max_steps = base_params
    # Create an array to store the base positions in
    base_loc_arr = np.zeros(len(peak_list),dtype=int)
    # Calculate the zero crossing points of the first derivative
    # Rising peak
    for i in range(len(base_loc_arr)):
        base_loc_arr[i] = find_zero_crossing(y_smooth_d1, peak_list[i], \
            directions_list[i], step_size, threshold=threshold, max_steps=max_steps)    
    # Check if bases were found
    base_string = check_found_edges(base_loc_arr)
    # Handle errors
    if "0" in base_string:
        error_string = f"Zero_Base_Finding_Error: {base_string}"
    else:
        error_string = ""
    # Add strings to base_dict
    base_dict = {"error_string": error_string, "base_string": base_string, \
    "suprema_string": "0"*4}

    return(base_loc_arr, base_dict)

    

# # Second Derivative Threshold Method
def find_local_supremum(data, start_point, direction, max_steps=20):
    """Finds the index of the local suprema in the specified direction from 
    the start point.

    Args:
        data (numpy arr): The data to search for a suprema in.
        start_point (int): The index of the point to start searching from.
        direction (int): -1 if searching locations left of the starting point, 
            +1 if searching right. 
        max_steps (int): The maximum number of steps before the function gives 
            up. (Default value = 20)

    Returns:
        supremum (int): The index of the local supremum. Returns -1 if not found.

    """
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
                next_test_step = test_step+direction
                # Check if next test step is in bounds
                if 0 <= next_test_step < len(data):            
                    if sign*(data[next_test_step]-data[test_step]) > 0:
                        check_point = next_test_step
                        num_steps+=1
                    else:
                        supremum = check_point
                        found_local_supremum = True
                # If the next test step is out of bounds, set the crossing point to the test step
                else:
                    supremum = check_point
                    found_local_supremum = True
            num_steps += 1
        else: 
            # If the test step is out of bounds, set the crossing point to the test step
            supremum = check_point
            found_local_supremum = True
    return supremum

def find_local_value(target_value, data, start_point, direction, max_steps=20):
    """Finds the index of the target value.

    Args:
        target_value (float): The target value to find.
        data (numpy arr): The data to search.
        start_point (int): The index of the point to start searching from.
        direction (int): -1 if searching locations left of the starting point, 
            +1 if searching right. 
        max_steps (int): The maximum number of steps before the function gives 
            up. (Default value = 20)

    Returns:
        position (int): The index of the target value location. Returns -1 if 
            not found.

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
            # if sign*(data[test_step]) > target_value:
            if sign*(data[test_step]) > target_value and sign*(data[test_step]) < sign*(data[check_point]):
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
    base_params, verbose=False):
    """Uses the 2nd derivative threshold method to find the base locations of 
    1st derivative peaks. If unable to find a base location with this method, it
    falls back on the 1st derivative threshold crossing method.

    Args:
        x_d1 (numpy arr): 1st derivative x positions.
        y_smooth_d1 (numpy arr): 1st derivative of the smoothed y data.
        y_smooth_d1_s (numpy arr): Smoothed 1st derivative.
        peak_list (list): A list of 1st derivative peak locations (x indices) to
            find base locations for. 
        directions_list (list): A list of the corresponding directions to search 
            in.
        base_params (tuple): Parameters for find_local_supremum, 
            find_zero_crossing, and find_local_value.
            target_value (float): The target value to find in the 2nd derivative.
            step_size (int): The step size to take in the search. Minimum is 1.
            threshold (float): The threshold value at which 'zero' is found. 
            max_steps (int): The maximum number of steps to use in the search 
                functions.
            smooth_func (func): A smoothing function.
            smooth_params (tuple): Parameters for smooth_func.
        verbose (bool): If True, saves 2nd derivative data in base_dict. 
            (Default value = False)      

    Returns:
        base_loc_arr (numpy arr): An array of the base locations (indices) of 
            the 1st deriative peaks.
        base_dict (dict): Contains auxiliary information.
            error_string (str): Describes any errors that have occurred.
            base_string (str): Describes which bases were found.
            suprema_string (str): Describes which local suprema were found. 
                Contains a "1" if the suprema was found and "0" if not.
            d2_data (tuple): 2nd derivative data. Present if verbose=TRUE
                x_d2 (numpy arr): 2nd derivative x positions.
                y_smooth_d2 (numpy arr): 2nd derivative (using y_smooth_d1_s)
                y_smooth_d2_s (numpy arr): Smoothed 2nd derivative.

    """
    # Unpack the parameters
    target_value, step_size, threshold, max_steps, smooth_func, smooth_params = base_params
    # Create an array to store the local supremum in
    suprema_loc_arr = np.zeros(len(peak_list),dtype=int)
    # Create an array to store the base positions in
    base_loc_arr = np.zeros(len(peak_list),dtype=int)

    # Calculate the second derivative
    x_d2, y_smooth_d2 = central_diff(x_d1, y_smooth_d1_s)
    # Smooth the first derivative
    y_smooth_d2_s = smooth_func(y_smooth_d2, *smooth_params)

    # Find local supremum around the rising and falling peak locations of the
    # 1st derivative
    for i in range(len(suprema_loc_arr)):
        suprema_loc_arr[i] = find_local_supremum(y_smooth_d2_s, peak_list[i], \
                directions_list[i], max_steps=max_steps)

    # Make sure local supremum was found  
    suprema_string = check_found_edges(suprema_loc_arr)
    for i in range(len(suprema_loc_arr)):
        # If a local supremum was not found, find the base location using the 
        # first derivative threshold calculation method
        if suprema_loc_arr[i] < 0:
            base_loc_arr[i] = find_zero_crossing(y_smooth_d1, peak_list[i], \
                directions_list[i], step_size, threshold=threshold, \
                max_steps=max_steps)
        # If a local supremum was found, find the base location using the 2nd 
        # derivative threshold method
        else:
            base_loc_arr[i] = find_local_value(target_value, y_smooth_d2_s, \
            suprema_loc_arr[i], directions_list[i], max_steps=max_steps)

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
    # Add additional information to dictionary
    base_dict = {"error_string": error_string, "base_string": base_string, \
    "suprema_loc_arr": suprema_loc_arr, "suprema_string": suprema_string}
    
    if verbose:                
        base_dict["d2_data"] = (x_d2, y_smooth_d2, y_smooth_d2_s)
    else:
        pass
    return(base_loc_arr, base_dict)


# Baseline Correction 
def fit_plateau_baseline(x_data, y_data, base_end_arr):
    """Fits a line between the exterior base points of the plateau feature.

    Args:
        x_data (arr-like): x (position) data.
        y_data (arr-like): y (height) data.
        base_end_arr (numpy arr): An array containing the indices of the exterior
            base points of the feature.

    Returns:
        plateau_baseline_fit (func): Provided an x position, calculates the 
            corresponding y value along the fitted line.
        slope (float): The slope of the fitted line.
        intercept (float): The intercept of the fitted line.

    """
    slope = (y_data[base_end_arr[1]]-y_data[base_end_arr[0]])/(x_data[base_end_arr[1]]-x_data[base_end_arr[0]])
    intercept = y_data[base_end_arr[0]] - slope*x_data[base_end_arr[0]]
    def plateau_baseline_fit(x):
        """Calculates points along the fitted line.

        Args:
            x (float): x position.

        Returns:
            y (float): Corresponding y position along the fitted line.

        """
        return slope*x+intercept
    return plateau_baseline_fit, slope, intercept

def fit_baseline(x_data, y_smooth, base_end_arr):
    """Performs a baseline correction to a plateau feature.

    Args:
        x_data (arr-like): The x (position) data.
        y_smooth (numpy arr): The smoothed y (height) data.
        base_end_arr (numpy arr): An array containing the indices of the exterior
            base points of the feature.

    Returns:
        y_smooth_blc (numpy arr): The baseline corrected smoothed y data.
        baseline_correction (numpy arr): The applied baseline correction.

    """
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
    """Calculates the half max of the edge given the max and min values of the 
    edge.

    Args:
        edge_max (float): The maximum value of the edge (height).
        edge_min (float): The minimum value of the edge (base).

    Returns:
        half_max (float): The midpoint value of the provided max/min.

    """
    return (edge_max-edge_min)/2.0 + edge_min

def find_half_max_neighbors(half_max, y_data, direction, left_bound, right_bound):
    """Finds the neighboring points of the half max value.

    Args:
        half_max (float): The half max value of the edge. 
        y_data (arr-like): The y (height) data.
        direction (int): The slope of the edge (+1 if rising, -1 if falling).
        left_bound (int): The index of the lower bound of the search.
        right_bound (int): The index of the upper bound of the search.

    Returns:
        left_neighbor (int): The index of the x position left-bounding the half
            max.
        right_neighbor (int): The index of the x position right-bounding the half
            max.

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
    """Calculates a linear fit using the neighboring points of the half max 
    value, and uses this fit to extrapolate the x position of the half max value.

    Args:
        half_max (float): The half max value of the edge.
        half_max_bounds (tuple): The indices of the x positions bounding the half
            max value. 
        x_data (arr-like): The x (position) data.
        y_data (arr-like): The y (height) data.

    Returns:
        half_max_pos (float): The extrapolated x position for the half max value.

    """
    slope = (y_data[half_max_bounds[0]]-y_data[half_max_bounds[1]])/(x_data[half_max_bounds[0]]-x_data[half_max_bounds[1]])
    intercept = y_data[half_max_bounds[0]] - slope*x_data[half_max_bounds[0]]
    half_max_pos = (half_max-intercept)/slope
    return half_max_pos

# Width calculation methods
def calculate_half_max_full_width_pos(x_data, y_smooth, base_loc_arr):
    """Calculates the positions of the half maxes of the plateau feature.

    Args:
        x_data (arr-like): The x (position) data.
        y_smooth (numpy arr): The smoothed y (height) data.
        base_loc_arr (numpy arr): An array of the base locations (indices) of 
            the 1st deriative peaks.

    Returns:
        half_max_pos (numpy arr): The extrapolated x positions for the half max 
            values.
        half_max_vals (numpy arr): The half max values of the edges.

    """
    # Figure out the min and max values of the edges
    rising_peak_min = np.min(y_smooth[base_loc_arr[0]:base_loc_arr[1]+1]) 
    rising_peak_max = np.max(y_smooth[base_loc_arr[0]:base_loc_arr[1]+1])
    falling_peak_max = np.max(y_smooth[base_loc_arr[2]:base_loc_arr[3]+1])
    falling_peak_min = np.min(y_smooth[base_loc_arr[2]:base_loc_arr[3]+1])    
    # Calculate the half max values
    rising_peak_half_max = calculate_half_max(rising_peak_max, \
        rising_peak_min)
    falling_peak_half_max = calculate_half_max(falling_peak_max, \
        falling_peak_min)
    # Find the half max neighbors
    rising_half_max_bounds = find_half_max_neighbors(rising_peak_half_max, \
        y_smooth, 1, base_loc_arr[0], base_loc_arr[1])
    falling_half_max_bounds = find_half_max_neighbors(falling_peak_half_max, \
        y_smooth, -1, base_loc_arr[2], base_loc_arr[3])
    # Find the half max positions
    rising_half_max_pos = find_half_max_pos(rising_peak_half_max, \
        rising_half_max_bounds, x_data, y_smooth)
    falling_half_max_pos = find_half_max_pos(falling_peak_half_max, \
        falling_half_max_bounds, x_data, y_smooth)
    

    half_max_pos = np.array([rising_half_max_pos, falling_half_max_pos])
    half_max_vals = np.array([rising_peak_half_max, falling_peak_half_max])
    return (half_max_pos, half_max_vals)        


def calculate_half_max_full_width(x_data, y_smooth, base_loc_arr, verbose=False):
    """Calculates the distance between two half max points (half max full width).

    Args:
        x_data (arr-like): The x (position) data.
        y_smooth (numpy arr): The smoothed y (height) data.
        base_loc_arr (numpy arr): An array of the base locations (indices) of 
            the 1st derivative peaks.
        verbose (bool): If True, saves half max data in base_dict. (Default 
            value = False)

    Returns:
        width (float): The distance between the two half max points.
        half_max_dict (dict): Contains auxiliary information.
            error_string (str): Records an error if the width was negative.
            half_max_positions (numpy arr): The extrapolated x positions for the 
                half max values. Present if verbose=TRUE.
            half_max_vals (numpy arr): The half max values of the edges. Present 
                if verbose=TRUE.

    """
    half_max_pos, half_max_vals = \
    calculate_half_max_full_width_pos(x_data, y_smooth, base_loc_arr)

    # Calculate the half max full width
    width = half_max_pos[1] - half_max_pos[0]
    
    if width >= 0:
        error_string = ""
    else:
        width = np.nan
        error_string = "Negative_Width_Error"
    half_max_dict = {"width_error_string": error_string}
    if verbose:
        half_max_dict["half_max_positions"] = half_max_pos
        half_max_dict["half_max_vals"] = half_max_vals
    else:
        pass

    return width, half_max_dict

def calculate_width_min_max(x_data, y_smooth, smooth_func, smooth_params, \
    base_func, base_params, verbose=False):
    """Calculates the width of a plateau with the specified base function.

    Args:
        x_data (arr-like): The x (position) data.
        y_smooth (numpy arr): The smoothed y (height) data.
        smooth_func (func): A smoothing function.
        smooth_params (tuple): Parameters for smooth_func.
        base_func (func): The function to find the base locations of the 1st 
            derivative. find_base_d1 or find_base_d2.
        base_params (tuple): Parameters for the specified base_func.
        verbose: Records auxiliary information in base_dict if True. (Default 
            value = False)

    Returns:
        width (float): The distance between the two half max points.
        base_dict (dict): Contains error messages and auxiliary information.
            error_string (str): Records any errors that may have occurred.
            d1_data (tuple): A tuple of 1st derivative data. Present if verbose=TRUE.
                x_d1 (numpy arr): 1st derivative x positions.
                y_smooth_d1 (numpy arr): 1st derivative of the smoothed y data.
                y_smooth_d1_s (numpy arr): Smoothed 1st derivative.
            d1_peak_list (list): A list of 1st derivative peak locations 
                (x indices). Present if verbose=TRUE.
            base_loc_arr (numpy arr): An array of the base locations (indices) of 
                the 1st deriative peaks. Present if verbose=TRUE.
            half_max_positions (numpy arr): The extrapolated x positions for the 
                half max values. Present if verbose=TRUE.
            half_max_vals (numpy arr): The half max values of the edges. Present 
                if verbose=TRUE.

    """
    # Calculate the first derivative
    x_d1, y_smooth_d1 = central_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find the peaks (hill and valley) of the first derivative
    rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[5:-5]))[0][0]
    falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[5:-5]))[0][0]
    
    # Store peaks in peak list
    peak_list = [rising_peak_loc, rising_peak_loc, falling_peak_loc, falling_peak_loc]
    directions_list = [-1, 1, -1, 1]

    # Find base points of the peaks of the first derivative
    base_loc_arr, base_dict = base_func(x_d1, \
        y_smooth_d1, y_smooth_d1_s, peak_list, directions_list, base_params, \
        verbose=verbose)

    if "0" in base_dict["base_string"]:
        width = np.nan
    else:
        # Adjust the indices (for the 2nd derivative method)
        for i in range(len(base_dict["suprema_string"])):
            if base_dict["suprema_string"][i] == "1":
                base_loc_arr[i] = base_loc_arr[i] + 1
            else:
                pass

        # Calculate the half max full width
        width, half_max_dict = calculate_half_max_full_width(x_data, y_smooth, \
            base_loc_arr, verbose=verbose)
        # Handle any width calculation errors
        if len(base_dict["error_string"]) > 0:
            base_dict["error_string"] += f"\t{half_max_dict['width_error_string']}"
        else:
            base_dict["error_string"] = half_max_dict["width_error_string"]
    # Handle the verbose
    if verbose:
        base_dict["d1_data"] = (x_d1, y_smooth_d1, y_smooth_d1_s)
        base_dict["d1_peak_list"] = [rising_peak_loc, falling_peak_loc]
        base_dict["base_loc_arr"] = base_loc_arr
        base_dict["half_max_positions"] = half_max_dict["half_max_positions"]
        base_dict["half_max_vals"] = half_max_dict["half_max_vals"]
    else:
        pass
    return width, base_dict

def calc_width_baseline_correction(x_data, y_smooth, smooth_func, smooth_params, \
    base_func, base_params, verbose=False): 
    """Performs a baseline correction and then calculates the width of a plateau 
    with the specified base function.

    Args:
        x_data (arr-like): The x (position) data.
        y_smooth (numpy arr): The smoothed y (height) data.
        smooth_func (func): A smoothing function.
        smooth_params (tuple): Parameters for smooth_func.
        base_func (func): The function to find the base locations of the 1st 
            derivative. find_base_d1 or find_base_d2.
        base_params (tuple): Parameters for the specified base_func.
        verbose: Records auxiliary information in base_dict if True. (Default 
            value = False)

    Returns:
        width (float): The distance between the two half max points.
        base_dict (dict): Contains error messages and auxiliary information. If 
            verbose=TRUE, also contains auxiliary information used for the 
            baseline correction (keys end in '_blc').
            error_string (str): Records any errors that may have occurred.
            d1_data (tuple): A tuple of 1st derivative data. Present if verbose=TRUE.
                x_d1 (numpy arr): 1st derivative x positions.
                y_smooth_d1 (numpy arr): 1st derivative of the smoothed y data.
                y_smooth_d1_s (numpy arr): Smoothed 1st derivative.
            d1_peak_list (list): A list of 1st derivative peak locations 
                (x indices). Present if verbose=TRUE.
            base_loc_arr (numpy arr): An array of the base locations (indices) of 
                the 1st deriative peaks. Present if verbose=TRUE.
            half_max_positions (numpy arr): The extrapolated x positions for the 
                half max values. Present if verbose=TRUE.
            half_max_vals (numpy arr): The half max values of the edges. Present 
                if verbose=TRUE.


    """
    # Calculate the first derivative
    x_d1, y_smooth_d1 = central_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find the peaks (hill and valley) of the first derivative
    rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[5:-5]))[0][0]
    falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[5:-5]))[0][0]

    # Store peaks in peak list
    base_peak_list = [rising_peak_loc, falling_peak_loc]
    base_directions_list = [-1, 1]
    # Find base points of the peaks of the first derivative
    base_loc_arr_blc, base_dict_blc = base_func(x_d1, \
        y_smooth_d1, y_smooth_d1_s, base_peak_list, base_directions_list, base_params,
        verbose=verbose)
    # Adjust the indices (for the 2nd derivative method)
    for i in range(len(base_dict_blc["suprema_string"])):
        if base_dict_blc["suprema_string"][i] == "1":
            base_loc_arr_blc[i] = base_loc_arr_blc[i] + 1
        else:
            pass

    if "0" in base_dict_blc["base_string"]:
        base_dict_blc["base_string"] = "Baseline correction failed: " + base_dict_blc["base_string"]
        width = np.nan
        base_dict = base_dict_blc
    else:
        # Perform baseline correction
        y_smooth_blc, baseline_correction = fit_baseline(x_data, y_smooth, \
            base_loc_arr_blc)
        
        # Calculate the width
        width, base_dict = calculate_width_min_max(x_data, y_smooth_blc, \
            smooth_func, smooth_params, base_func, base_params, verbose=verbose)

        if verbose:
            # Add the blc dict to the base_dict
            for key, item in base_dict_blc.items():
                base_dict[f"{key}_blc"] = item
            base_dict["y_smooth_blc"] = y_smooth_blc
            base_dict["baseline_correction"] = baseline_correction
        else:
            pass
            
    if verbose:
        base_dict["d1_data_blc"] = (x_d1, y_smooth_d1, y_smooth_d1_s)
        base_dict["d1_peak_list_blc"] = base_peak_list
        base_dict["base_loc_arr_blc"] = base_loc_arr_blc
    else:
        pass

    return width, base_dict

# Reading settings
def read_TEM_length_settings(settings_path):
    """Used to read in custom TEM_length settings. If settings are not specified
    in the settings file, default values are used.

    Args:
        settings_path (str): Path to a settings .txt file. 

    Returns:
        smooth_method (str): The smoothing method. Currently only uses
            'savgol'.
        smooth_func (func): A smoothing function.
        smooth_params (tuple): Parameters for smooth_func.
        width_method (str): The width calculation method. Currently only uses
            'min_max'.
        base_method (str): Which base method to use. '1st derivative threshold'
            or '2nd derivative threshold'.
        base_func (func): The function to find the base locations of the 1st 
            derivative. find_base_d1 or find_base_d2.
        base_params (tuple): Parameters for the specified base_func.

    """
    # Default settings
    smooth_method = "savgol"
    smooth_func = signal.savgol_filter
    smooth_params = (21, 3)
    width_method = "min_max"
    base_method = "1st derivative threshold"
    base_func = find_base_d1
    step_size = 2
    threshold = 1
    max_steps = 20
    d2_threshold = 0.5
    base_params = (step_size, threshold, max_steps)

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
                        if step_size < 1:
                            raise ValueError("ValueError: step size must be >= 1.")
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
                if base_method == "1st derivative threshold": 
                    base_func = find_base_d1
                    base_params = (step_size, threshold, max_steps)
                elif base_method == "2nd derivative threshold":
                    base_func = find_base_d2
                    base_params = (d2_threshold, step_size, threshold, max_steps, \
                        smooth_func, smooth_params)
                else:
                    raise ValueError("ValueError: Suppled base finding method " \
                        + "not recognized. Please use '1st derivative " \
                        + "threshold' or '2nd derivative threshold'.")
            except (ValueError) as msg:
                print(msg)
                # Stop the script execution
                sys.exit(1)

    return (smooth_method, smooth_func, smooth_params, width_method, base_method, \
        base_func, base_params)

# Writing Output Files
def write_header(custom_name, file_name, smooth_method, smooth_params, \
    base_method, base_params, use_baseline, width_method, error_list, \
    summary_stats, note):
    """Writes a header file of the methods used and summary statistics of the
    analysis.

    Args:
        custom_name (str): Appends the custom name to the end of the file. 
        file_name (str): The name of the read file.
        smooth_method (str): The smoothing method. Currently only uses
            'savgol'.
        smooth_params (tuple): Parameters for smooth_func.
        base_method (str): Which base method to use. '1st derivative threshold'
            or '2nd derivative threshold'.
        base_params (tuple): Parameters for the specified base_func.
        use_baseline:
        width_method (str): The width calculation method. Currently only uses
            'min_max'.
        error_list (list): A list of all the errors generated.
        summary_stats (tuple): 
            num_samples (int): The total number of measurements.
            summary_num_samples (int): The number of measurements used for the
                mean, median, and sample standard deviation.
            mean (float): The mean of the widths.
            median (float): The median of the widths.
            sample_stdev (float): The sample standard deviation of the widths.
        note (str): A custom note.

    Returns:
        None. Outputs a .txt header file.

    """
    if len(custom_name) > 0:
        add_name = f"_{custom_name}"
    else:
        add_name = ""
    with open(f"{file_name}_header{add_name}.txt", 'w') as header_file:
        # Record the calculation methods and settings
        header_file.write("######## Calculation settings ########\n")
        header_file.write(f"Smoothing method: {smooth_method}\n")
        header_file.write(f"Smoothing parameters: {smooth_params}\n")
        header_file.write(f"Baseline correction: {use_baseline}\n")
        header_file.write(f"Edge end estimation method: {base_method}\n")
        if base_method == "2nd derivative threshold":
            d2_threshold, step_size, threshold, max_steps, smooth_func, \
            smooth_params = base_params
            base_params = (d2_threshold, step_size, threshold, max_steps)
        else:
            pass
        header_file.write(f"Edge end estimation parameters: {base_params}\n")
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
    """Writes the calculated widths to a .txt file.

    Args:
        custom_name (str): Appends the custom name to the end of the file. 
        file_name (str): The name of the read file.
        width_array (numpy arr): Contains all the calculated widths. 

    Returns:
        None. Outputs a .txt data file.

    """
    if len(custom_name) > 0:
        add_name = f"_{custom_name}"
    else:
        add_name = ""
    with open(f"{file_name}_measurements{add_name}.txt", 'w') as data_file:
        for width in width_array:
            data_file.write(f"{width}\t")

def TEM_length_main():
    """Handles commandline input and runs width calculation analyses of a 
    supplied file of plateau data.

    Args: 
        None. Uses commandline input.

    Returns:
        None. Outputs header and data .txt files.

    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Takes in a .xslx or .txt file of TEM \
        grayscale profiles and computes the lengths of the objects using a half \
        max full width approach. Results are saved in the same folder as the input file.")
    parser.add_argument("read_file_path", type=str, \
        help="The path to the .xslx or .txt data file to analyze.")
    parser.add_argument("--save_name", type=str, \
        help="A custom name to append to names of files that results are saved to.")
    parser.add_argument("--save_folder", type=str, \
        help="Path to the folder to save results in.")
    parser.add_argument("--baseline", type=str, 
        help="If True, applies a baseline correction before trying to determine the length. Defaults to True.")
    parser.add_argument("--settings", type=str, \
        help="The path to a file containing analysis settings.")
    parser.add_argument("--transpose", type=str, \
        help="Set as True if the input data is transposed (samples are organized in rows). Defaults to False.")
    parser.add_argument("--note", type=str, \
        help="A note to add to the header file.")
    parser.add_argument("--prog", type=str, \
        help="If True, prints sample number to terminal as it is being analyzed. Defaults to False.")

    args = parser.parse_args()

    # Parse the arguments
    read_file_path = args.read_file_path
    
    # Get custom save name
    if args.save_name:
        custom_name = args.save_name
    else:
        custom_name = ""

    # Get custom save folder
    if args.save_folder:
        print("Save folder provided")
        file_split_string = determine_os()
        save_folder = args.save_folder
        file_name = read_file_path.split(file_split_string)[-1].split(".")[0]
        file_name = f"{save_folder}{file_split_string}{file_name}"

    else:
        file_name = read_file_path.split(".")[0]
    
    # Get baseline setting
    true_list = ["True", "true", "TRUE"]
    if args.baseline and args.baseline not in true_list:
        use_baseline = False
    else:
        use_baseline = True

    # Get settings
    if args.settings:
        settings_path = args.settings
    else:
        settings_path = "default"
    smooth_method, smooth_func, smooth_params, width_method, base_method, \
    base_func, base_params = read_TEM_length_settings(settings_path)

    # Read in the note
    if args.note:
        note = args.note
    else:
        note = ""

    # Determine if data is transposed
    if args.transpose and args.transpose in true_list:
        is_transpose = True
    else:
        is_transpose = False

    if args.prog and args.prog in true_list:
        progress = True
    else:
        progress = False

    # Read the file in 
    TEM_df = read_TEM_data(read_file_path, transpose=is_transpose)

    # Get the shape
    rows, cols = TEM_df.shape
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
        x_data, y_data = exclude_NaN(x_index, TEM_df)
        # Smooth the function
        y_smooth = smooth_func(y_data, *smooth_params)
        # Calculate the width
        if use_baseline:
            width, base_dict = calc_width_baseline_correction(x_data, y_smooth, \
                smooth_func, smooth_params, base_func, base_params)
        else:
            width, base_dict = calculate_width_min_max(x_data, y_smooth, \
                smooth_func, smooth_params, base_func, base_params)
        # Record the data
        width_array[i] = width
        # Record any errors
        if len(base_dict["error_string"])>0:
            error_message = f"Sample: {i} {base_dict['error_string']}"
            error_list.append(error_message)
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
    base_method, base_params, use_baseline, width_method, error_list, \
    summary_stats, note)
    

    # Write a data file
    print("Writing measurement data file.")
    write_measurement_data(custom_name, file_name, width_array)
    print("Finished.")

if __name__ == "__main__":
    TEM_length_main()