# Script to mask movies

# Import packages
import argparse
from tqdm import tqdm
import os

# Set the working directory as the directory that the script is currently in
# so filepaths can be read relative to this path.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(script_dir)

# Import saxtal_functions.py
import saxtal_functions as sax

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument('movie_filenames_in', type=str, help="The file path to a .txt file containing input movie filenames.")
parser.add_argument('movie_filenames_out', type=str, help="The file path to a .txt file containing output movie filenames.")
parser.add_argument('micrograph_filenames_in', type=str, help="The file path to a .txt file containing input micrograph filenames.")
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
       
# Read the input micrograph filenames into a list
with open(args.movie_filenames_in) as movie_inputs:
    movie_input_filenames = [line.strip() for line in movie_inputs]
    print(movie_input_filenames)

# Read the output filenames into a list    
with open(args.movie_filenames_out) as movie_outputs:
    movie_output_filenames = [line.strip() for line in movie_outputs]
    
# Read the input micrograph filenames into a list
with open(args.micrograph_filenames_in) as micrograph_inputs:
    micrograph_input_filenames = [line.strip() for line in micrograph_inputs]

# Process movies in a loop
for movie_input_filename, movie_output_filename, micrograph_input_filename, micrograph_output_filename in tqdm(zip(movie_input_filenames, movie_output_filenames, micrograph_input_filenames, micrograph_input_filenames)):
    sax.mask_movie(movie_input_filename,
                   movie_output_filename,
                   micrograph_input_filename,
                   micrograph_output_filename,
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
                   replace_distance_percent = replace_distance_percent,
                   return_spots = True)
    
print("All movies processed.")
