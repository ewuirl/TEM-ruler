# TEM-ruler
TEM-ruler is a Python-based library for calculating widths of features in TEM images using grayscale profiles. Analysis of data files containing grayscale profiles can be conducted via commandline. Accepted file formats are .xlsx and tab separated .txt. File analysis produces two files: a header file with the analysis settings and summary statistics, and a data file containing the calculated measurements. Default use applies baseline correction and uses the 1st derivative threshold method (see below).

- [Installation](https://github.com/ewuirl/TEM-ruler#installation)
- [Command-line Usage](https://github.com/ewuirl/TEM-ruler#command-line-usage)
  - [Input](https://github.com/ewuirl/TEM-ruler#input)
  - [Output](https://github.com/ewuirl/TEM-ruler#output)
- [Supported Features](https://github.com/ewuirl/TEM-ruler#supported-features)
  - [TEM_length](https://github.com/ewuirl/TEM-ruler#tem_length)
  - [TEM_ring](https://github.com/ewuirl/TEM-ruler#tem_ring)
- [Width Calculation Methods](https://github.com/ewuirl/TEM-ruler#width-calculation-methods)
  - [Edge End Estimation Methods](https://github.com/ewuirl/TEM-ruler#edge-end-estimation-methods)
    - [1st Derivative Threshold](https://github.com/ewuirl/TEM-ruler#1st-derivative-threshold)
    - [2nd Derivative Threshold](https://github.com/ewuirl/TEM-ruler#2nd-derivative-threshold)
  - [Baseline Correction](https://github.com/ewuirl/TEM-ruler#baseline-correction)
- [General Notes and Advice](https://github.com/ewuirl/TEM-ruler#-general-notes-and-advice)

## Installation
Install with pip:
```shell
$ pip install git+https://github.com/ewuirl/TEM-ruler.git
```
## Command-line Usage
File analysis of plateau profiles (TEM_length) can be conducted as follows. File analysis with TEM_ring can be conducted by swapping TEM_length with TEM_ring.
```
TEM_length [read_file_path] [-h] [--save_name SAVE_NAME]
[--save_folder SAVE_FOLDER] [--baseline BASELINE] [--settings SETTINGS]
[--transpose TRANSPOSE] [--note NOTE] [--prog PROG]

Input:
read_file_path          The path to the .xlsx or .txt data file
                        to analyze.

General options:
  -h, --help            show this help message and exit
  --save_name SAVE_NAME
                        A custom name to append to names of files that results
                        are saved to.
  --save_folder SAVE_FOLDER
                        Path to the folder to save results in.
  --baseline BASELINE   If True, applies a baseline correction before trying
                        to determine the length. Defaults to True.
  --settings SETTINGS   The path to a file containing analysis settings.
  --transpose TRANSPOSE
                        Set as True if the input data is transposed (samples
                        are organized in rows). Defaults to False.
  --note NOTE           A note to add to the header file.
  --prog PROG           If True, prints sample number to terminal as it is
                        being analyzed. Defaults to False.

```
### Input
.xlsx or tab separated .txt files, with data organized in columns or rows.
- In an .xlsx file, cells that are not header cells must either contain data or be empty.
- If the data is organized in columns, a measurement is assumed to consist of two columns. The first column contains position data (x) followed immediately by a column of grayscale data (y).
- Data organized in rows can be analyzed with the transpose option.

### Output
File analysis outputs the following files:
1. Header file:
   - Analysis settings (smoothing method, parameters, baseline correction, edge end estimation method, edge end estimation parameters, etc)
   - Summary statistics (mean, median, sample standard deviation, etc)
   - Errors
2. Data file:
   - Measured widths (TEM_length and TEM_ring)
   - Per sample averages, sample standard deviations

## Supported Features
Data file analysis is currently supported for plateau and ring features. Other features can be analysed via adaptation or tuning of the plateau measurement (TEM_length).

### TEM_length
TEM_length can calculate the half max full width of a plateau feature (eg length or width of a continuous feature).

### TEM_ring
TEM_ring can calculate the average ring thickness and pore width of a ring feature. Default use assumes each ring has been measured twice in a row.

## Width Calculation Methods
To calculate the width of a plateau feature, TEM-ruler uses two variations on the following overall approach.
1. Detect of the vertical edges of the plateau
   - Find peaks in the first derivative of the grayscale value. The maximum corresponds to the rising edge, and the minimum corresponds to the falling edge.
2. Estimate the location of the ends (top and the bottom) of the edges.
   - Estimate the base points of the 1st derivative peaks. *This is where the two calculation methods differ.*
3. Determine the vertical midpoint of each edge.
   - Use the heights of the top and bottom of the edges are to find the midpoint of each edge. Fit a line to the points bounding the midpoint to extrapolate the x position of the midpoint.
4. Calculate the half max, full width.
   - Calculate the distance between the midpoints of the edges.

### Edge End Estimation Methods
#### 1st Derivative Threshold
This method finds the base of the 1st derivative peaks by determining approximately where the 1st derivative crosses zero (within a specified threshold).

#### 2nd Derivative Threshold
This method finds the base of the 1st derivative peaks by incorporating the 2nd derivative as well. It finds local suprema in the 2nd derivative around the 1st derivative peaks, and then finds where the 2nd derivative crosses a specified threshold magnitude near these suprema. If this method is unsuccessful in finding a base location, the 1st derivative threshold method is used as a fallback.

### Baseline Correction
Baseline correction is conducted by fitting a line between the bottom points the rising and falling edges of a plateau feature, and fitting lines through the outer base points. These fits are subtracted from the grayscale profile, and the baseline corrected profile is used for futher computation. The selected edge end estimation method is used for both the baseline correction and width calculation.

## General Notes and Advice
### Smoothing
Since the grayscale data is noisy, TEM-ruler uses a <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html">Savitzky-Golay filter<a> to smooth it, as well as its derivatives. The default smoothing parameters for TEM_length and TEM_ring likely need to be adjusted. Smaller filter windows will perform better for features that are less wide.
### Derivatives
Derivatives are approximated using the central difference method.
