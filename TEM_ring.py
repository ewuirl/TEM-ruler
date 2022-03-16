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
from TEM_ruler.TEM_length import calculate_half_max_full_width
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
def calculate_ring_min_max(x_data, y_data, smooth_func, smooth_params, \
    base_func, base_params, adjust_index, midpoint_window):
    # Smooth the function
    y_smooth = smooth_func(y_data, *smooth_params)

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
        width = np.array([np.nan,np.nan])
    else:
        # Adjust the indices (for the 2nd derivative method)
        for i in range(len(suprema_string)):
            if suprema_string[i] == "1":
                base_loc_arr[i] = base_loc_arr[i] + adjust_index
            else:
                pass
        # print(f"adjusted bases: {base_loc_arr}")

        # Calculate the half max full width
        widths = np.zeros(2)
        widths[0] = calculate_half_max_full_width(x_data, y_smooth, base_loc_arr[:4])
        widths[1] = calculate_half_max_full_width(x_data, y_smooth, base_loc_arr[4:])

    return widths, error_string

# Writing Output Files
def write_ring_header(custom_name, file_name, midpoint_window, smooth_method, \
    smooth_params, base_method, base_params, width_method, error_list, \
    summary_stats, note):
    with open(f"{file_name}_header_{custom_name}.txt", 'w') as header_file:
        # Record the calculation methods and settings
        header_file.write(f"Center finding window size: {midpoint_window}\n")
        header_file.write(f"Smoothing method: {smooth_method}\n")
        header_file.write(f"Smoothing parameters: {smooth_params}\n")
        header_file.write(f"Base finding method: {base_method}\n")
        header_file.write(f"Base finding parameters: {base_params}\n")
        header_file.write(f"Width calculation method: {width_method}\n")
        header_file.write(f"Note: {note}\n")
        # Unpack the summary stats
        num_samples, summary_num_samples, summary_num_measurements, \
        mean, median, stdev = summary_stats
        header_file.write(f"Summary stats:\n")
        header_file.write(f"Number of measurements: {num_samples*4}\n")
        header_file.write(f"Number of samples: {num_samples}\n")
        header_file.write(f"Number of summary stat measurements: {summary_num_samples}\n")
        header_file.write(f"Number of summary stat samples: {summary_num_measurements}\n")
        header_file.write(f"mean: {mean}\n")
        header_file.write(f"median: {median}\n")
        header_file.write(f"stdev: {stdev}\n")
        header_file.write(f"Number of errors: {len(error_list)}\n")
        # Record any errors
        header_file.write("Errors: \n")
        for error in error_list:
            header_file.write(f"{error}\n")

def write_ring_measurement_data(custom_name, file_name, width_array, mean_arr, \
    stdev_arr):
    with open(f"{file_name}_measurements_{custom_name}.txt", 'w') as data_file:
        data_file.write("Width 1\tWidth 2\tWidth 3\tWidth 4\tMean\tStd Dev\n")
        for i in range(len(width_array)):
            for width in width_array[i,:]:
                data_file.write(f"{width}\t")
            data_file.write(f"{mean_arr[i]}\t")
            data_file.write(f"{stdev_arr[i]}\n")


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
    custom_name = "serial_d2_threshold"
    # custom_name = "serial_d2_threshold_baseline_correction"


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
    
    width_array = np.zeros((num_samples,4))
    mean_arr = np.zeros(num_samples)
    stdev_arr = np.zeros(num_samples)
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
        
        # Calculate the widths
        width, error_string = calculate_ring_min_max(x_data, y_data, \
            smooth_func, smooth_params, base_func, base_params, adjust_index, \
            midpoint_window)
        width_1, error_string_1 = calculate_ring_min_max(x_data_1, y_data_1, \
            smooth_func, smooth_params, base_func, base_params, adjust_index, \
            midpoint_window)
        
        # Record the data
        width_array[i,:2] = width
        width_array[i,2:] = width_1

        # Calculate mean and standard deviation
        widths_clean = width_array[i,np.logical_not(np.isnan(width_array[i,:]))]
        if len(widths_clean) > 1:
            mean_width = np.mean(widths_clean)
            stdev_width = np.std(widths_clean)
        elif len(widths_clean) > 1:
            mean_width = widths_clean[0]
            stdev_width = 0 
        else:
            mean_width = np.nan 
            stdev_width = np.nan 
        # Record the mean and standard deviation
        mean_arr[i] = mean_width
        stdev_arr[i] = stdev_width

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
        # print(width_array[i,:])

        # # Serial analysis (baseline correction)
    # for i in range(num_samples):
    #     print(i)
    #     # Pick the x columns
    #     x_index = 2*i
    #     # Remove the NaN values
    #     x_data, y_data = exclude_NaN(x_index, length_df)
    #     # Calculate the width
    #     width, error_string = calc_width_baseline_correction(x_data, y_data, \
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

    # Calculate the mean, median, and standard deviation
    width_array_clean = width_array[np.logical_not(np.isnan(width_array))]
    mean_arr_clean = mean_arr[np.logical_not(np.isnan(mean_arr))]
    stdev_arr_clean = stdev_arr[np.logical_not(np.isnan(stdev_arr))]
    mean = np.mean(mean_arr_clean)
    median = np.median(mean_arr_clean)
    stdev = np.sum(np.square(stdev_arr_clean))
    # Pack up the stats
    summary_stats = num_samples, len(width_array_clean), len(mean_arr_clean), \
    mean, median, stdev 

    # Save the data
    # Write a header file
    write_ring_header(custom_name, file_name, midpoint_window, smooth_method, \
    smooth_params, base_method, base_params, width_method, error_list, \
    summary_stats, note)
    # Write a data file
    write_ring_measurement_data(custom_name, file_name, width_array, mean_arr, \
    stdev_arr)