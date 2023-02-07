# Script to mask micrographs

# Import packages
import argparse
from tqdm import tqdm
import os

# Set the working directory as the directory that the script is currently in
# so filepaths can be read relative to this path.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import saxtal_functions.py
import saxtal_functions as sax

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument('filenames_in', type=str, help="The file path to a .txt file containing input filenames.")
parser.add_argument('filenames_out', type=str, help="The file path to a .txt file containing output filenames.")
parser.add_argument('params', type=str, help="The file path to the params file.")
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
parser.add_argument("-p", "--plots", help="Show plots", action="store_true")

# Parse the arguments
args = parser.parse_args()

# Read the params file and set the function arguments
with open(args.params) as params:
    lines = params.readlines()
    for line in lines:
        exec(line)
              
# Read the input filenames into a list
with open(args.filenames_in) as inputs:
    input_filenames = [line.strip() for line in inputs]

# Read the output filenames into a list    
with open(args.filenames_out) as outputs:
    output_filenames = [line.strip() for line in outputs]

# Process images in a loop
for input_filename, output_filename in tqdm(zip(input_filenames, output_filenames)):
    sax.mask_image(input_filename,
                   output_filename,
                   threshold_method = threshold_method,
                   pixel_size = pixel_size,
                   verbose = args.verbose,
                   show_plots = args.plots,
                   threads = threads,
                   sigma = sigma,
                   num_sd = num_sd,
                   num_sd_secondpass = num_sd_secondpass,
                   x_window_percent = x_window_percent,
                   y_window_percent = y_window_percent,
                   miller_index_buffer = miller_index_buffer,
                   box_radius = box_radius,
                   min_lattice_size = min_lattice_size,
                   mask_radius= mask_radius,
                   replace_distance_percent = replace_distance_percent)
    
print("All micrographs processed.")