import argparse
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
from TEM_ruler.TEM_length import exclude_NaN
from TEM_ruler.TEM_length import central_diff
from TEM_ruler.TEM_length import find_zero_crossing
from TEM_ruler.TEM_length import find_local_value
from TEM_ruler.TEM_length import find_base_d2
from TEM_ruler.TEM_length import fit_plateau_baseline
from TEM_ruler.TEM_length import fit_baseline
from TEM_ruler.TEM_length import calculate_half_max_full_width_pos
# from TEM_ruler.TEM_length import calculate_half_max_full_width
from TEM_ruler.TEM_length import write_header


# Figure out midpoint of ring
def find_ring_center(y_smooth, window):
    mid_point = int(len(y_smooth)/2)
    left_bound = mid_point-int(window/2)
    right_bound = mid_point+int(window/2)
    target_value = min(y_smooth[left_bound], y_smooth[right_bound])
    # Find the closest point on the other side of data midpoint
    if y_smooth[left_bound] == target_value:
        right_bound = find_local_value(target_value, y_smooth, right_bound, -1, max_steps=20)
    else:
        left_bound = find_local_value(target_value, y_smooth, left_bound, 1, max_steps=20)
    
    return (left_bound, right_bound, int((right_bound-left_bound)/2)+left_bound)

# Width calculation methods
def calculate_half_max_full_width_ring(x_data, y_smooth, base_loc_arr):
    widths_array = np.zeros(3)

    left_half_max_positions = \
    calculate_half_max_full_width_pos(x_data, y_smooth, base_loc_arr[:4])

    right_half_max_positions = \
    calculate_half_max_full_width_pos(x_data, y_smooth, base_loc_arr[4:])

    # Calculate the half max full width
    widths_array[0] = left_half_max_positions[1] - left_half_max_positions[0]
    widths_array[1] = right_half_max_positions[1] - right_half_max_positions[0]
    widths_array[2] = right_half_max_positions[0] - left_half_max_positions[1]

    return widths_array

def calculate_ring_min_max(x_data, y_smooth, smooth_func, smooth_params, \
    base_func, base_params, adjust_index, midpoint_window):
    
    # Find the midpoint of the ring
    left_bound, right_bound, midpoint = find_ring_center(y_smooth, midpoint_window)

    # print(f"center point stuff")
    # print(left_bound, right_bound, midpoint)

    # Calculate the first derivative
    x_d1, y_smooth_d1 = central_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find the peaks (hill and valley) of the first derivative
    left_rising_peak_loc_smooth = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[5:midpoint+1]))[0][0]
    left_falling_peak_loc_smooth = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[5:midpoint+1]))[0][0]
    right_rising_peak_loc_smooth = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[midpoint:-5]))[0][0]
    right_falling_peak_loc_smooth = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[midpoint:-5]))[0][0]
    # print("peaks")
    # print(f"left: {left_rising_peak_loc_smooth}, {left_falling_peak_loc_smooth}")
    # print(f"right: {right_rising_peak_loc_smooth}, {right_falling_peak_loc_smooth}")
    
    # Store peaks in peak list
    peak_list = [left_rising_peak_loc_smooth, left_rising_peak_loc_smooth, \
    left_falling_peak_loc_smooth, left_falling_peak_loc_smooth, \
    right_rising_peak_loc_smooth, right_rising_peak_loc_smooth, \
    right_falling_peak_loc_smooth, right_falling_peak_loc_smooth]

    directions_list = [-1, 1, -1, 1, -1, 1, -1, 1]

    # Find base points of the peaks of the first derivative
    base_loc_arr, error_string, base_string, suprema_string = base_func(x_d1, \
        y_smooth_d1_s, peak_list, directions_list, base_params)

    # print(f"bases: {base_loc_arr}")
    # print(f"base string: {base_string}")
    # print(f"suprema_string: {suprema_string}")

    if "0" in base_string:
        widths_array = np.array([np.nan, np.nan, np.nan])
    else:
        # Adjust the indices (for the 2nd derivative method)
        for i in range(len(suprema_string)):
            if suprema_string[i] == "1":
                base_loc_arr[i] = base_loc_arr[i] + adjust_index
            else:
                pass
        # print(f"adjusted bases: {base_loc_arr}")

        # Calculate the half max full width ring and pore widths
        widths_array = calculate_half_max_full_width_ring(x_data, y_smooth, base_loc_arr)

    return widths_array, error_string

# Baseline Fitting Functions
def fit_ring_baseline(x_data, y_smooth, base_end_arr):
    # Fit the baseline to straighten the plateaus of the ring
    left_plateau_baseline_fit, left_bl_slope, left_bl_intercept = fit_plateau_baseline(x_data, y_smooth, base_end_arr[:2])
    left_x_plateau = x_data[base_end_arr[0]:base_end_arr[1]+1]
    left_plateau_baseline = left_plateau_baseline_fit(left_x_plateau)
    
    right_plateau_baseline_fit, right_bl_slope, right_bl_intercept = fit_plateau_baseline(x_data, y_smooth, base_end_arr[2:])
    right_x_plateau = x_data[base_end_arr[2]:base_end_arr[3]+1]
    right_plateau_baseline = right_plateau_baseline_fit(right_x_plateau)

    # Prepare the outer edge data (center them so the base endpoints are at 0,0)
    # Left side
    x_left_centered = x_data[:base_end_arr[0]+1] - x_data[base_end_arr[0]]
    x_left_centered = x_left_centered.values
    y_smooth_left_centered  = y_smooth[:base_end_arr[0]+1]-y_smooth[base_end_arr[0]]        
    # Right side
    x_right_centered = x_data[base_end_arr[3]:] - x_data[base_end_arr[3]]
    x_right_centered = x_right_centered.values
    y_smooth_right_centered = y_smooth[base_end_arr[3]:]-y_smooth[base_end_arr[3]]        
    
    # Fit left base
    lm_left = LinearRegression(fit_intercept = False)
    lm_left.fit(x_left_centered.reshape(-1,1), y_smooth_left_centered)
    # Fit the right base
    lm_right = LinearRegression(fit_intercept = False)
    lm_right.fit(x_right_centered.reshape(-1,1), y_smooth_right_centered)

    # Assemble the baseline correction
    baseline_correction = np.zeros(len(x_data))
    baseline_correction[:base_end_arr[0]] = lm_left.predict(x_left_centered[:-1].reshape(-1,1)) +  y_smooth[base_end_arr[0]]
    baseline_correction[base_end_arr[0]:base_end_arr[1]+1] = left_plateau_baseline
    baseline_correction[base_end_arr[2]:base_end_arr[3]+1] = right_plateau_baseline
    baseline_correction[base_end_arr[3]+1:] = lm_right.predict(x_right_centered[1:].reshape(-1,1)) +  y_smooth[base_end_arr[3]]

    # Handle the middle
    if base_end_arr[1] != base_end_arr[2]:
        mid_baseline_fit, mid_bl_slope, mid_bl_intercept = fit_plateau_baseline(x_data, y_smooth, base_end_arr[1:3])
        mid_x = x_data[base_end_arr[1]:base_end_arr[2]+1]
        mid_baseline = mid_baseline_fit(mid_x)
        baseline_correction[base_end_arr[1]:base_end_arr[2]+1] = mid_baseline
    else:
        pass
    
    # Add the baseline correction to the y data
    y_smooth_blc = y_smooth - baseline_correction
    return y_smooth_blc, baseline_correction

def calculate_ring_baseline_correction(x_data, y_smooth, smooth_func, smooth_params, \
    base_func, base_params, adjust_index, midpoint_window):
    # Find the midpoint of the ring
    left_bound, right_bound, midpoint = find_ring_center(y_smooth, midpoint_window)

    # print(f"center point stuff")
    # print(left_bound, right_bound, midpoint)

    # Calculate the first derivative
    x_d1, y_smooth_d1 = central_diff(x_data, y_smooth)
    # Smooth the first derivative
    y_smooth_d1_s = smooth_func(y_smooth_d1, *smooth_params)

    # Find the peaks (hill and valley) of the first derivative
    left_rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[5:midpoint+1]))[0][0]
    left_falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[5:midpoint+1]))[0][0]
    right_rising_peak_loc = np.where(y_smooth_d1_s==np.max(y_smooth_d1_s[midpoint:-5]))[0][0]
    right_falling_peak_loc = np.where(y_smooth_d1_s==np.min(y_smooth_d1_s[midpoint:-5]))[0][0]
    # print("peaks")
    # print(f"left: {left_rising_peak_loc_smooth}, {left_falling_peak_loc_smooth}")
    # print(f"right: {right_rising_peak_loc_smooth}, {right_falling_peak_loc_smooth}")
    
    # Store peaks in peak list
    base_peak_list = [left_rising_peak_loc, left_falling_peak_loc, \
    right_rising_peak_loc, right_falling_peak_loc]

    base_directions_list = [-1, 1, -1, 1]

    # Find base points of the peaks of the first derivative
    base_end_arr, error_string, base_string, suprema_string = base_func(x_d1, \
        y_smooth_d1_s, base_peak_list, base_directions_list, base_params)
    # print(f"bases: {base_end_arr}")
    # print(f"base string: {base_string}")
    # print(f"suprema_string: {suprema_string}")

    # Adjust the indices (for the 2nd derivative method)
    for i in range(len(suprema_string)):
        if suprema_string[i] == "1":
            base_end_arr[i] = base_end_arr[i] + adjust_index
        else:
            pass

    if "0" in base_string:
        error_string = "Baseline correction failed: " + error_string
        widths_array = np.array([np.nan, np.nan, np.nan])
    else:
        # Perform baseline correction 
        y_smooth_blc, baseline_correction = fit_ring_baseline(x_data, y_smooth, \
            base_end_arr)

        # Calculate the half max full width
        widths_array, error_string = calculate_ring_min_max(x_data, y_smooth_blc, \
            smooth_func, smooth_params, base_func, base_params, adjust_index, \
            midpoint_window)

    return widths_array, error_string


# Writing Output Files
def write_ring_header(custom_name, file_name, midpoint_window, smooth_method, \
    smooth_params, base_method, base_params, width_method, error_list, \
    summary_stats, note):
    with open(f"{file_name}_header_{custom_name}.txt", 'w') as header_file:
        # Record the calculation methods and settings
        header_file.write("######## Calculation settings ########\n")
        header_file.write(f"Center finding window size: {midpoint_window}\n")
        header_file.write(f"Smoothing method: {smooth_method}\n")
        header_file.write(f"Smoothing parameters: {smooth_params}\n")
        header_file.write(f"Base finding method: {base_method}\n")
        header_file.write(f"Base finding parameters: {base_params}\n")
        header_file.write(f"Width calculation method: {width_method}\n")
        header_file.write(f"Note: {note}\n")
        # Unpack the summary stats
        num_samples, num_ring_samples, num_ring_measurements, ring_mean, \
        ring_std_err, ring_median, ring_sample_stdev , num_pore_samples, \
        num_pore_measurements, pore_mean, pore_std_err, pore_median, \
        pore_sample_stdev = summary_stats
        # Record the summary stats
        header_file.write(f"\n######## Summary stats ########\n")
        header_file.write(f"Number of samples: {num_samples}\n")
        header_file.write(f"Number of ring measurements: {num_samples*4}\n")
        header_file.write(f"Number of pore measurements: {num_samples*2}\n")
        # Record ring summary stats
        header_file.write(f"\n#### Ring summary stats ####\n")
        header_file.write(f"Number of summary samples: {num_ring_samples}\n")
        header_file.write(f"Number of summary stat measurements: {num_ring_measurements}\n")
        header_file.write(f"mean: {ring_mean}\n")
        header_file.write(f"standard error of the mean: {ring_std_err}\n")
        header_file.write(f"median: {ring_median}\n")
        header_file.write(f"sample stdev: {ring_sample_stdev}\n")
        # Record pore summary stats
        header_file.write(f"\n#### Pore summary stats ####\n")
        header_file.write(f"Number of summary samples: {num_pore_samples}\n")
        header_file.write(f"Number of summary stat measurements: {num_pore_measurements}\n")
        header_file.write(f"mean: {pore_mean}\n")
        header_file.write(f"standard error of the mean: {pore_std_err}\n")
        header_file.write(f"median: {pore_median}\n")
        header_file.write(f"sample stdev: {pore_sample_stdev}\n")
        # Record any errors
        header_file.write("\n######## Errors ########\n")
        header_file.write(f"Number of errors: {len(error_list)}\n")
        for error in error_list:
            header_file.write(f"{error}\n")

def write_ring_measurement_data(custom_name, file_name, ring_width_array, ring_mean_arr, \
    ring_stdev_arr, pore_width_array, pore_mean_arr, pore_stdev_arr):
    with open(f"{file_name}_measurements_{custom_name}.txt", 'w') as data_file:
        for i in range(4):
            data_file.write(f"Ring Width {i+1}\t")
        data_file.write(f"Ring Mean\tRing Sample Std Dev\t")
        data_file.write("Pore Width 1\t Pore Width 2\tPore Mean\tPore Sample Std Dev\n")
        for i in range(len(ring_width_array)):
            for ring_width in ring_width_array[i,:]:
                data_file.write(f"{ring_width}\t")
            data_file.write(f"{ring_mean_arr[i]}\t")
            data_file.write(f"{ring_stdev_arr[i]}\t")
            for pore_width in pore_width_array[i,:]:
                data_file.write(f"{pore_width}\t")
            data_file.write(f"{pore_mean_arr[i]}\t")
            data_file.write(f"{pore_stdev_arr[i]}\n")

if __name__ == "__main__":
    # Add argparse stuff

    # Hard code some stuff
    midpoint_window = 10
    smooth_func = signal.savgol_filter
    smooth_params = (9, 3)
    d2_threshold = 2
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
    base_params = (d2_threshold, step_size, threshold, max_steps, smooth_func, smooth_params)
    base_func = find_base_d2
    # custom_name = "serial_d2_threshold"
    custom_name = "serial_d2_threshold_baseline_correction"


    # custom_name = "serial"
    # file_path = "/Users/emilywu/OneDrive - Massachusetts Institute of Technology/TEM width/42hb polyplex no stain xy values length.xlsx"
    file_path = "/Users/emilywu/OneDrive - Massachusetts Institute of Technology/TEM width/xy dimensions ring 125x silica.xlsx"
    # file_name = "42hb polyplex no stain xy values length"
    file_name = "xy dimensions ring 125x silica"
    smooth_method = "savgol"
    width_method = "min_max"
    note = ""

    # Read the file in 
    length_df = pd.read_excel(file_path, header=None,names=None)

    # Get the shape
    rows, cols = length_df.shape
    num_samples = int(cols/4)
    # num_samples = 2
    
    ring_width_array = np.zeros((num_samples,4))
    pore_width_array = np.zeros((num_samples,2))
    ring_mean_arr = np.zeros(num_samples)
    pore_mean_arr = np.zeros(num_samples)
    ring_sample_stdev_arr = np.zeros(num_samples)
    pore_sample_stdev_arr = np.zeros(num_samples)
    error_list = []

    # Serial analysis (no baseline correction)
    for i in range(num_samples):
        print(i)
        # Pick the x columns
        x_index = 2*i
        
        # Remove the NaN values
        x_data, y_data = exclude_NaN(x_index, length_df)
        x_data_1, y_data_1 = exclude_NaN(x_index+2, length_df)

        # print(f"Measurement 1 # data: {len(x_data)}")
        # print(f"Measurement 2 # data: {len(x_data_1)}")    

        # Smooth the data
        y_smooth = smooth_func(y_data, *smooth_params)
        y_smooth_1 = smooth_func(y_data_1, *smooth_params)
        
        # # Calculate the widths (no baseline correction)
        # width_array, error_string = calculate_ring_min_max(x_data, y_smooth, \
        #     smooth_func, smooth_params, base_func, base_params, adjust_index, \
        #     midpoint_window)
        # width_array_1, error_string_1 = calculate_ring_min_max(x_data_1, y_smooth_1, \
        #     smooth_func, smooth_params, base_func, base_params, adjust_index, \
        #     midpoint_window)

        # Calculate the widths (baseline correction)
        width_array, error_string = calculate_ring_baseline_correction(x_data, \
            y_smooth, smooth_func, smooth_params, base_func, base_params, \
            adjust_index, midpoint_window)
        width_array_1, error_string_1 = calculate_ring_baseline_correction(x_data_1, \
            y_smooth_1, smooth_func, smooth_params, base_func, base_params, \
            adjust_index, midpoint_window)
        
        # Record the data
        ring_width_array[i,:2] = width_array[:2]
        ring_width_array[i, 2:] = width_array_1[:2]
        pore_width_array[i, 0] = width_array[2]
        pore_width_array[i, 1] = width_array_1[2]

        # Calculate mean and standard deviation
        ring_widths_clean = ring_width_array[i,np.logical_not(np.isnan(ring_width_array[i,:]))]
        pore_widths_clean = pore_width_array[i,np.logical_not(np.isnan(pore_width_array[i,:]))]
        if len(ring_widths_clean) > 1:
            ring_width_mean = np.mean(ring_widths_clean)
            # Finite number of measurements --> compute sample stdev
            ring_width_sample_stdev = np.std(ring_widths_clean, ddof=1)  
        elif len(ring_widths_clean) > 1:
            ring_width_mean = ring_widths_clean[0]
            ring_width_sample_stdev = 0
        else:
            ring_width_mean = np.nan 
            ring_width_sample_stdev = np.nan 
        if len(pore_widths_clean) > 1:
            pore_width_mean = np.mean(pore_widths_clean)
            # Finite number of measurements --> compute sample stdev
            pore_width_sample_stdev = np.std(pore_widths_clean, ddof=1)  
        elif len(pore_widths_clean) > 1:
            pore_width_mean = pore_widths_clean[0]
            pore_width_sample_stdev = 0
        else:
            pore_width_mean = np.nan 
            pore_width_sample_stdev = np.nan 
        # Record the mean and standard deviation
        ring_mean_arr[i] = ring_width_mean
        ring_sample_stdev_arr[i] = ring_width_sample_stdev
        pore_mean_arr[i] = pore_width_mean
        pore_sample_stdev_arr[i] = pore_width_sample_stdev

        # Record any errors
        if len(error_string)>0:
            error_message = f"Sample: {i}, Measurement: 1, {error_string}"
            error_list.append(error_message)
            print(error_message)
        else:
            pass
        if len(error_string_1)>0:
            error_message = f"Sample: {i}, Measurement: 2, {error_string_1}"
            error_list.append(error_string_1)
            print(error_message)
        else:
            pass

    # Calculate the mean, median, and standard deviation
    ring_width_array_clean = ring_width_array[np.logical_not(np.isnan(ring_width_array))]
    ring_mean_arr_clean = ring_mean_arr[np.logical_not(np.isnan(ring_mean_arr))]
    ring_sample_stdev_arr_clean = ring_sample_stdev_arr[np.logical_not(np.isnan(ring_sample_stdev_arr))]
    ring_mean = np.mean(ring_mean_arr_clean)
    ring_median = np.median(ring_mean_arr_clean)
    num_ring_samples_clean = len(ring_mean_arr_clean)
    num_ring_measurements_clean = len(ring_width_array_clean)
    ring_std_err = np.sqrt(np.sum(np.square(ring_sample_stdev_arr_clean)))/num_ring_samples_clean
    ring_sample_stdev = np.std(ring_width_array_clean, ddof=1)

    pore_width_array_clean = pore_width_array[np.logical_not(np.isnan(pore_width_array))]
    pore_mean_arr_clean = pore_mean_arr[np.logical_not(np.isnan(pore_mean_arr))]
    pore_sample_stdev_arr_clean = pore_sample_stdev_arr[np.logical_not(np.isnan(pore_sample_stdev_arr))]
    pore_mean = np.mean(pore_mean_arr_clean)
    pore_median = np.median(pore_mean_arr_clean)
    num_pore_samples_clean = len(pore_mean_arr_clean)
    num_pore_measurements_clean = len(pore_width_array_clean)
    pore_std_err = np.sqrt(np.sum(np.square(pore_sample_stdev_arr_clean)))/num_pore_samples_clean
    pore_sample_stdev = np.std(pore_width_array_clean, ddof=1)
    
    # Pack up the stats
    summary_stats = num_samples, num_ring_samples_clean, num_ring_measurements_clean, \
    ring_mean, ring_std_err, ring_median, ring_sample_stdev , \
    num_pore_samples_clean, num_pore_measurements_clean, pore_mean, pore_std_err, \
    pore_median, pore_sample_stdev 

    # Save the data
    # Write a header file
    if base_method == "2nd derivative threshold":
        base_params = (d2_threshold, step_size, threshold, max_steps)
    else:
        pass
    write_ring_header(custom_name, file_name, midpoint_window, smooth_method, \
    smooth_params, base_method, base_params, width_method, error_list, \
    summary_stats, note)
    # Write a data file
    write_ring_measurement_data(custom_name, file_name, ring_width_array, \
        ring_mean_arr, ring_sample_stdev_arr, pore_width_array, pore_mean_arr, \
        pore_sample_stdev_arr)

    