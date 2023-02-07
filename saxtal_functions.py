# Import packages

from funcs_mrcio import iwrhdr_opened, irdhdr_opened, iwrsec_opened, irdsec_opened
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from tqdm import tqdm
from itertools import product

import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt
import time
import multiprocessing
import os

def rebin_helper(array, bin_size=2):
    """Take a two dimensional pixel array and average each bin_size x bin_size square to one pixel."""
    shape = (array.shape[0] // bin_size, bin_size,
             array.shape[1] // bin_size, bin_size)
    return array.reshape(shape).mean(-1).mean(1)

def rebin_mrc(filename, bin_size=2):
    """Take the filename of an .mrc file to be rebinned, perform the rebinning, and write out a new file."""
    # Read the .mrc file in binary
    micrograph = open(filename,'rb')
    
    # Use funcs_mrcio to extract image array
    image = irdsec_opened(micrograph,0)
    
    # Rebin the image
    rebinned_image = rebin_helper(image, bin_size)
    
    # Generate a new filename
    new_filename = "rebin_output/" + filename[0:-4] + "_bin_" + str(bin_size) + ".mrc"
    
    # Generate a new header
    nx, ny = rebinned_image.shape
    nxyz = np.array([nx, ny, 1], dtype=np.float32)
    dmin = np.min(rebinned_image)
    dmax = np.max(rebinned_image)
    dmean = np.sum(rebinned_image)/(nx*ny)
    
    # Open a new file
    rebinned_mrc = open(new_filename, 'wb')
    
    # Write the header to the new file
    iwrhdr_opened(rebinned_mrc, 
                  nxyz, 
                  dmin, 
                  dmax, 
                  dmean, 
                  mode=2)
    
    # Write the rebinned array to the new file
    iwrsec_opened(rebinned_image, rebinned_mrc)

def import_mrc(filename):
    """Use funcs_mrcio to open a specified .mrc file"""
    # Read the .mrc file in binary
    micrograph = open(filename,'rb')
    
    # Use funcs_mrcio to extract image array and rescale values to lie between [-1, 1]
    image = rescale_intensity(irdsec_opened(micrograph,0))
    
    # Use funcs_mrcio to extract header info
    header = irdhdr_opened(micrograph)
    
    # Return the rescaled image and header
    return image, header

    
# def pad_image(image, header):
#     """Take image and header from import_mrc() and pad image to make array square"""
#     # Find the maximum dimension of the image
#     max_dimension = np.maximum(header['nx'], header['ny'])
#     
#     # Make an array of the median pixel value of the image array with maximum dimensions
#     padded_image = np.full(shape=(max_dimension, max_dimension),
#                            fill_value=np.median(image))
#     
#     # Transplant image values into padded array
#     padded_image[0:header['nx'],0:header['ny']] = image
#     
#     # Return padded image and header
#     return padded_image, header

def scipy_fft(image, verbose=False, threads=1):
    """Take  image from import_mrc() and return the FFT"""
    
    # Record the time it takes to do the FFT
    # for later comparison with multi-threaded FFTW
    
    # Start the timer
    start_time = time.time()
    
    # Perform the FFT
    padded_fft = sfft.rfft2(image,
                            s=(np.max(image.shape), np.max(image.shape)),
                            workers=threads)
    
    # Stop the timer
    end_time = time.time()
    
    if verbose:
        # Print the elapsed time to the nearest hundreth of a millisecond
        print("scipy_fft(): FFT performed in", np.round((end_time-start_time)*1000, 2), "milliseconds.")
    
    # Eliminate potential artifacts along edge of FFT
    padded_fft[0,:] = padded_fft[1,:]
    padded_fft[:,0] = padded_fft[:,1]
    
    # Return FFT of padded image
    return padded_fft

def west_fft(image, print_timestamps=False):
    """Take the padded image from pad_image() and return the FFT with the FFTW library. WIP!"""    
    
def generate_diff_spectrum(padded_fft, sigma=1):
    """
    Apply gaussian filter to power spectrum of FFT from scipy_fft() or west_fft() 
    to generate a difference spectrum
    """
    
    # Generate the power spectrum of padded FFT
    padded_spectrum = np.abs(padded_fft)
    
    # Smooth the power spectrum with Gaussian filter
    smoothed_spectrum = gaussian(padded_spectrum, sigma)
    
    # Return the difference spectrum
    log_diff_spectrum = np.log(padded_spectrum/smoothed_spectrum)
    
    return log_diff_spectrum, smoothed_spectrum

def find_diffraction_spots_quantile(log_diff_spectrum, quantile=0.99, x_window_percent = (0, 1), y_window_percent = (0, 1)):
    """
    Take a difference spectrum and find outliers greater than a given quantile,
    taking a user-defined window of the FFT.
    """
    
    # Define minimum and maximum indices of the window
    
    y_min = int(log_diff_spectrum.shape[0]*y_window_percent[0])
    y_max = int(log_diff_spectrum.shape[0]*y_window_percent[1])
    x_min = int(log_diff_spectrum.shape[1]*x_window_percent[0])
    x_max = int(log_diff_spectrum.shape[1]*x_window_percent[1])

    # Calculate the threshold on a subset of the spectrum
    quantile_threshold = np.quantile(log_diff_spectrum[y_min:y_max, x_min:x_max].flatten(), 
                                     quantile)
    
    # Find the indices of points where the value is greater than the threshold
    spot_indices = np.where(log_diff_spectrum >= quantile_threshold)
    
    # Discard the points that fall outside the window
    spots_in_window = ((spot_indices[0] >= y_min) & (spot_indices[0] <= y_max)) & ((spot_indices[1] >= x_min) & (spot_indices[1] <= x_max))
    
    # Subset the rows of diffraction_spots that fall within the window
    diffraction_indices = np.array((spot_indices[0][spots_in_window], spot_indices[1][spots_in_window]))
    
    # Extract the amplitudes of these points
    diffraction_amplitudes = np.empty(len(diffraction_indices[0]))
    for i in range(len(diffraction_indices[0])):
        diffraction_amplitudes[i] = log_diff_spectrum[diffraction_indices[0,i], diffraction_indices[1, i]]
    
    return diffraction_indices, diffraction_amplitudes

def find_diffraction_spots_sd(log_diff_spectrum, num_sd=3.0, x_window_percent=(0, 1), y_window_percent=(0, 1)):
    """
    Take a difference spectrum and find outliers past a certain number of SDs from the mean,
    taking a user-defined window of the FFT.
    """
    
    # Define minimum and maximum indices of the window
    y_min = int(log_diff_spectrum.shape[0]*y_window_percent[0])
    y_max = int(log_diff_spectrum.shape[0]*y_window_percent[1])
    x_min = int(log_diff_spectrum.shape[1]*x_window_percent[0])
    x_max = int(log_diff_spectrum.shape[1]*x_window_percent[1])

    # Calculate the mean on a  subset of the spectrum
    mean = np.mean(log_diff_spectrum[y_min:y_max, x_min:x_max]).flatten()
    # Calculate the standard deviation
    sd = np.std(log_diff_spectrum[y_min:y_max, x_min:x_max]).flatten()
   
    # Find the indices of points where the value is greater than the threshold
    spot_indices = np.where(log_diff_spectrum >= mean+(num_sd*sd))
    
    # Discard the points that fall outside the window
    spots_in_window = ((spot_indices[0] >= y_min) & (spot_indices[0] <= y_max)) & ((spot_indices[1] >= x_min) & (spot_indices[1] <= x_max))
    
    # Subset the rows of diffraction_spots that fall within the window
    diffraction_indices = np.array((spot_indices[0][spots_in_window], spot_indices[1][spots_in_window]))
    
    # Extract the amplitudes of these points
    diffraction_amplitudes = np.empty(len(diffraction_indices[0]))
    for i in range(len(diffraction_indices[0])):
        diffraction_amplitudes[i] = log_diff_spectrum[diffraction_indices[0,i], diffraction_indices[1, i]]
    
    return diffraction_indices, diffraction_amplitudes


# Functions to find lattices. This makes remove_hotpixels redundant -----------------------------------------------------------

def unwrap_indices(wrapped_indices, fft):
    """Helper function to take a set of indices and unwrap them according to shape of an FFT."""
    unwrapped_indices = np.copy(wrapped_indices)
    for i in range(len(wrapped_indices[0])):
        if wrapped_indices[0, i] > (np.max(fft.shape)/2):
            unwrapped_indices[0, i] = wrapped_indices[0, i] - np.max(fft.shape)

    return unwrapped_indices

def wrap_indices(unwrapped_indices, fft):
    """Helper function to take a set of indices and wrap them according to shape of an FFT."""
    wrapped_indices = np.copy(unwrapped_indices)
    for i in range(len(unwrapped_indices[0])):
        if unwrapped_indices[0, i] < 0:
            wrapped_indices[0, i] = unwrapped_indices[0, i] + np.max(fft.shape)
    wrapped_indices = wrapped_indices.astype(int)

    return wrapped_indices

def estimate_basis(indices, amplitudes, min_lattice_size):
    
    # Find the top min_lattice_size brightest points
    top_n = np.flip(np.argsort(amplitudes))[0:min_lattice_size]
    basis_points = indices[:,top_n]

    # Initialize variables
    best_error = 1e8
    best_basis = np.empty((2, 2))
    best_miller = np.empty(indices.shape)

    # Loop over all pairs of putative basis vectors
    for i in range(min_lattice_size):
        for j in range((i+1), min_lattice_size):
            

            # Establish the basis matrix
            basis_matrix = basis_points[:,(i,j)]
        
            # Compute the angle between the vectors
            angle = (360/(2*np.pi))*np.arccos(np.dot(basis_matrix[:,0], basis_matrix[:,1])/
                              (np.linalg.norm(basis_matrix[:,0])*np.linalg.norm(basis_matrix[:,1])))
                        
            # If the basis vectors are linearly dependent, skip this iteration
            # As the basis matrix will be non-invertable
            if np.round(angle)==0 or np.round(angle)==180 or np.isnan(angle): 
                continue
        
            # Take the inverse
            inverse_matrix = np.linalg.inv(basis_matrix)
        
            # Transform all peaks into this basis
            miller_matrix = np.matmul(inverse_matrix, indices)
        
            # Compute the sum of the (amplitude peaks * (x_error^2 + y_error^2))
            basis_error = np.sum(np.exp(amplitudes)*np.sum((miller_matrix - np.round(miller_matrix))**2,
                                                                axis=0))
    
            # Compute the angle between the vectors
            angle = (360/(2*np.pi))*np.arccos(np.dot(basis_matrix[:,0], basis_matrix[:,1])/
                              (np.linalg.norm(basis_matrix[:,0])*np.linalg.norm(basis_matrix[:,1])))
        
            # Impose angle penalty if angle is less than 10 degrees
            if angle <= 10: 
                angle_penalty = np.exp(10-angle)
            else:
                angle_penalty = 1
        
            # Compute the overall error for the basis: angle_penalty*sum(amplitude peaks * (x_error^2 + y_error^2))
            error = angle_penalty*basis_error
        
            # Store the choice of basis vectors
            if error < best_error:
                best_error = error
                best_basis = basis_matrix
                best_miller = miller_matrix

    return best_basis, best_miller

def shorten_basis(best_basis, indices, verbose=False):

    # Create a copy of the best basis set to be shortened
    shortened_basis = best_basis.copy()

    # Print the starting basis
    if verbose: print("Starting basis: " + str(shortened_basis))

    # Print the size of the starting basis
    basis_size = np.sum(shortened_basis**2)
    if verbose: print("Starting basis size: " + str(basis_size))

    # Run the loop at least once
    rerun_loop = True

    while rerun_loop:

        # Print starting message
        if verbose: print("Running basis shortening...")

        # Extract the basis vectors
        basis_0 = shortened_basis[:,0]
        basis_1 = shortened_basis[:,1]

        # If one has negative x-values, flip it
        if basis_0[1] <= 0:
            basis_0 = -basis_0
        if basis_1[1] <= 0:
            basis_1 = -basis_1

        # Compute the angle between the basis vectors before shortening and print
        angle = (360/(2*np.pi))*np.arccos(np.dot(basis_0, basis_1)/
                                          (np.linalg.norm(basis_0)*np.linalg.norm(basis_1)))

        if verbose: print("The angle between the basis vectors is: " + str(angle))

        # Determine which basis vector is larger
        larger_basis = np.array([basis_0, basis_1])[np.argmax(np.array([np.sum(basis_0**2), np.sum(basis_1**2)]))]
        smaller_basis = np.array([basis_0, basis_1])[np.argmin(np.array([np.sum(basis_0**2), np.sum(basis_1**2)]))]

        # Print the result of the computation
        if verbose: print(str(larger_basis) + " is larger than " + str(smaller_basis))

        if angle < 90:
            larger_basis = larger_basis - smaller_basis    
        if angle > 90:
            larger_basis = larger_basis + smaller_basis

        # Print the result
        if verbose: print("The updated basis vectors are " + str(larger_basis) + str(smaller_basis))

        # If this is smaller, update and run the loop again
        if (np.sum(larger_basis**2)+np.sum(smaller_basis**2)) < basis_size:
            # Print the result of the above conditional
            if verbose: print("This run resulted in a basis smaller than the last: " + str((np.sum(larger_basis**2)+np.sum(smaller_basis**2))))

            # Update the basis size
            basis_size = (np.sum(larger_basis**2)+np.sum(smaller_basis**2))

            # Update the basis vectors of loop_basis
            shortened_basis[:,0] = larger_basis
            shortened_basis[:,1] = smaller_basis

            # If one vector has a negative x-coordinate, flip it
            if shortened_basis[:,0][1] <= 0:
                shortened_basis[:,0] = -shortened_basis[:,0]   
            if shortened_basis[:,1][1] <= 0:
                shortened_basis[:,1] = -shortened_basis[:,1]

            # Print the result
            if verbose: print("The updated basis vectors are: " + str(shortened_basis))

            # Queue the loop to run again
            rerun_loop = True

        else:
            # Print the result of the above conditional
            if verbose: print("This run resulted in a basis larger than the last: " + str((np.sum(larger_basis**2)+np.sum(smaller_basis**2))))
            if verbose: print("Terminating loop.")
            rerun_loop = False

        if verbose: print("------------------------------------")
            
    # Transform all peaks into shortened Miller basis to find indices
    shortened_miller = np.matmul(np.linalg.inv(shortened_basis), indices)

    # Return the shortened basis and the shortened peaks
    return shortened_basis, shortened_miller

def filter_noninteger_miller(best_miller, epsilon = 0.0707):
    
    # Keep the miller indices that fall within epsilon of the nearest lattice point
    filtered_points_logical = np.logical_and((np.abs(best_miller - np.round(best_miller)) <= epsilon)[0,:],
                                            (np.abs(best_miller - np.round(best_miller)) <= epsilon)[1,:])
    
    # Generate the filtered list
    integer_miller = best_miller[:,filtered_points_logical]
    
    return integer_miller

def filter_redundant_miller(integer_miller):
    """
    For each unique set of miller indices: 1. Pull a subset of redundant peaks,
    2. See which is closest to the true location, 3. Put that point in a new list.
    Collapse the list into an array, return.
    """

    # Initialize variables
    unique_miller = np.unique(np.round(integer_miller), axis=1)
    best_fit_spots = []

    # For each unique set of miller indices
    for miller in np.transpose(unique_miller):
        # Pull a subset of points with the same indices
        redundant_miller = integer_miller[:,np.where((np.round(integer_miller)[0,:] == miller[0]) & (np.round(integer_miller)[1,:] == miller[1]))[0]]

        # See which is closest to the true location
        best_spot = np.argmin((redundant_miller[0,:] - miller[0])**2 + (redundant_miller[1,:] - miller[1])**2)

        # Keep the closest point
        best_fit_spots.append(redundant_miller[:,best_spot])
        
    nonredundant_miller = np.transpose(np.stack(best_fit_spots))
    
    return nonredundant_miller

def refine_basis(shortened_basis, nonredundant_miller, verbose=False):
    
    if verbose: print("Starting basis: " + str(shortened_basis))
    
    # Define the starting variables for LSQ refinement
    lsq_lattice = np.matmul(shortened_basis, nonredundant_miller)
    lsq_miller = np.round(nonredundant_miller)
    lsq_basis = shortened_basis
    
    # Define some intermediate variables
    NN_t_inv = np.linalg.inv(np.matmul(lsq_miller,np.transpose(lsq_miller)))
    N = lsq_miller
    X_t = np.transpose(lsq_lattice)
    
    # Generate the refined basis with LSQ solving of the basis
    refined_basis = np.transpose(np.matmul(np.matmul(NN_t_inv, N), X_t))
    
    if verbose: print("Refined_basis: " + str(refined_basis))
    
    # Return the refined basis
    return refined_basis
    
def generate_lattice_indices(nonredundant_miller, best_basis, miller_index_buffer=2, box_radius=10):
    """
    Generate the Fourier-space indices of a complete lattice of Miller indices to
    do a more sensitive search for other spots that belong to the reciprocal lattice.
    miller_index_buffer controls how far outside the detected lattice to search.
    """
    
    # Generate a combinatorial list of Miller indices to search
    highest_miller_index = int(np.max(np.round(nonredundant_miller))) + miller_index_buffer
    lowest_miller_index = int(np.min(np.round(nonredundant_miller))) - miller_index_buffer
    #lowest_miller_index = -highest_miller_index
    permute_list = np.linspace(lowest_miller_index, highest_miller_index, (highest_miller_index-lowest_miller_index)+1).tolist()
    test_miller = np.transpose(np.array(list(product(permute_list, repeat=2))))
    
    # Convert to Fourier space indices
    test_indices = np.matmul(best_basis, test_miller)

    # Make sure x-coordinates are in range
    lattice_indices = test_indices[:,np.where(test_indices[1,:] > box_radius)]
    lattice_indices = lattice_indices[:,0,:]
    
    return lattice_indices


def search_lattice_indices(wrapped_lattice_indices, log_diff_spectrum, num_sd=3, box_radius=10):
    
    # inrange_indices is a list of places in the image we need to filter.
    # Write a loop over all index pairs that finds and plots 5x5 areas
    # around each inrange_index
    
    # Initialize an empty list to hold new points found in the below loop
    new_points = []

    # For each suspected lattice point
    for i in range(wrapped_lattice_indices.shape[1]):

        # Extract the indices
        view_indices = wrapped_lattice_indices[:,i].astype(int)

        # Extract the box around the indices
        view_array = log_diff_spectrum[(view_indices[0]-box_radius):(view_indices[0]+(box_radius+1)),
                                       (view_indices[1]-box_radius):(view_indices[1]+(box_radius+1))]

        # Look for points >num_sd SD above the background
        mean = np.mean(view_array)
        sd = np.std(view_array)

        # If a point is found
        if(np.sum(view_array >= mean + num_sd*sd) > 0):

            # Find its indices
            relative_indices = np.where(view_array >= mean + num_sd*sd)

            # Adjust them relative to the starting index
            absolute_indices = np.copy(relative_indices)
            absolute_indices[0] = relative_indices[0] + (view_indices[0]-10)
            absolute_indices[1] = relative_indices[1] + (view_indices[1]-10)
            new_points.append(absolute_indices)
    
    # Collapse the list of arrays into an array
    new_indices = np.concatenate(new_points, axis=1)
            
    return new_indices

def find_lattice(indices, 
                 amplitudes, 
                 log_diff_spectrum, 
                 pixel_size,
                 show_plots=False,
                 verbose=False, 
                 num_sd_secondpass=2, 
                 miller_index_buffer=2, 
                 box_radius=10,
                 min_lattice_size=5):
    """
    """
    # Unwrap indices to help find lattice
    unwrapped_indices = unwrap_indices(indices, log_diff_spectrum)
    
    if show_plots: 
        plt.scatter(y = unwrapped_indices[0,:], x = unwrapped_indices[1,:], s = np.exp(amplitudes))
        plt.show()
    
    if unwrapped_indices.shape[1] <= min_lattice_size:
        if verbose: print("Lattice has less than " + str(min_lattice_size) + " candidate basis vectors during first pass. Terminating function.")
        return None, None, None
    
    # Estmate the basis
    best_basis, best_miller = estimate_basis(unwrapped_indices, amplitudes, min_lattice_size)
    # Shorten the basis vectors
    shortened_basis, shortened_miller = shorten_basis(best_basis, unwrapped_indices, verbose)
    # Filter out points not near integer Miller indices
    integer_miller = filter_noninteger_miller(shortened_miller)
    # Filter out points that have been assigned the same Miller indices
    nonredundant_miller = filter_redundant_miller(integer_miller)

    # Recreate the lattice points in Fourier space
    nonredundant_lattice = np.matmul(shortened_basis, nonredundant_miller)

    if show_plots:        
        fig, ax = plt.subplots()
        ax.scatter(y = unwrapped_indices[0,:], x = unwrapped_indices[1,:], s = 5*amplitudes)
        ax.scatter(y = nonredundant_lattice[0,:], x = nonredundant_lattice[1,:], s = 5)
        plt.show()
    
    # If the nonredundant lattice has less than 5 points, don't return a new lattice
    if nonredundant_miller.shape[1] <= min_lattice_size:
        if verbose: print("Lattice has less than " + str(min_lattice_size) + " points. Terminating function.")
        return None, None, None
        
    # Refine the lattice to estimate unit cell dimensions
    refined_basis = refine_basis(shortened_basis, nonredundant_miller, verbose)
    
    # Calculate the dimensions of the unit cell
    num_pix = np.max(log_diff_spectrum.shape)
    basis_0_length = np.sqrt(np.sum(refined_basis[:,0]**2))
    basis_1_length = np.sqrt(np.sum(refined_basis[:,1]**2))
    unit_cell_dimensions = np.round(np.array([pixel_size*num_pix/basis_0_length, pixel_size*num_pix/basis_1_length]), 2)

     # Sanity check: unit cell
    if np.max(unit_cell_dimensions) > 500:
        if verbose: print("Lattice unit cell max dimension is " + str(np.max(unit_cell_dimensions)) + " Angstrom. Terminating function.")
        return None, None, None
    if np.min(unit_cell_dimensions) < 5:
        if verbose: print("Lattice unit cell min dimension is " + str(np.min(unit_cell_dimensions)) + " Angstrom. Terminating function.")
        return None, None, None
    
    # Calculate the resolution of the farthest lattice spot from first-pass points
    num_pix = np.max(log_diff_spectrum.shape)
    max_rad = np.max(np.sqrt((nonredundant_lattice[0,:]**2 + nonredundant_lattice[1,:]**2)))
    highest_resolution = np.round(pixel_size*num_pix/max_rad, 2)
    
    # Generate the lattice points along which to search for more peaks
    lattice_indices = generate_lattice_indices(nonredundant_miller, shortened_basis, miller_index_buffer, box_radius)
    
    if show_plots:
        fig, ax = plt.subplots()
        ax.scatter(y = nonredundant_lattice[0,:], x = nonredundant_lattice[1,:], s = 10)
        ax.scatter(y = lattice_indices[0,:], x = lattice_indices[1,:], s = 1)
        plt.show()
        
    # Wrap the indices to extract boxes
    wrapped_lattice_indices = wrap_indices(lattice_indices, log_diff_spectrum)
    
    # Search for new indices
    new_indices = search_lattice_indices(wrapped_lattice_indices, log_diff_spectrum, num_sd_secondpass, box_radius)
    
    # Unwrap the indices to combine with original lattice points
    new_indices_unwrapped = unwrap_indices(new_indices, log_diff_spectrum)
    
    # Combine the new candidate points with the old points, filter for unique points
    combined_indices = np.unique(np.hstack((nonredundant_lattice, new_indices_unwrapped)), axis=1)
    
    # Extract the respective amplitudes for combined_indices from log_diff_spectrum
    combined_amplitudes = log_diff_spectrum[wrap_indices(combined_indices, log_diff_spectrum)[0,:], 
                                            wrap_indices(combined_indices, log_diff_spectrum)[1,:]]
    
    if show_plots:
        fig, ax = plt.subplots()
        ax.scatter(y = combined_indices[0,:], x = combined_indices[1,:], s = np.exp(combined_amplitudes))
        ax.scatter(y = lattice_indices[0,:], x = lattice_indices[1,:], s = 1)
        plt.show()
    
    if combined_indices.shape[1] <= 8:
        if verbose: print("Lattice has less than " + str(min_lattice_size) + " candidate basis vectors during combine. Returning non-combined lattice.")
        return nonredundant_lattice, unit_cell_dimensions, highest_resolution
    
    # Re-run the above pipeline on the candidate points generated
    combined_best_basis, combined_best_miller = estimate_basis(combined_indices, combined_amplitudes, min_lattice_size)
    combined_shortened_basis, combined_shortened_miller = shorten_basis(combined_best_basis, combined_indices)
    combined_integer_miller = filter_noninteger_miller(combined_shortened_miller)
    combined_nonredundant_miller = filter_redundant_miller(combined_integer_miller)

    # Recreate the lattice points in Fourier space
    combined_nonredundant_lattice = np.matmul(combined_shortened_basis, combined_nonredundant_miller)
    
    if show_plots:
        fig, ax = plt.subplots()
        ax.scatter(y = combined_nonredundant_lattice[0,:], x = combined_nonredundant_lattice[1,:], s = 10)
        plt.show()
        
    # Calculate the resolution of the farthest lattice spot from second-pass data
    num_pix = np.max(log_diff_spectrum.shape)
    max_rad = np.max(np.sqrt((combined_nonredundant_lattice[0,:]**2 + combined_nonredundant_lattice[1,:]**2)))
    highest_resolution = np.round(pixel_size*num_pix/max_rad, 2)
    
    # Wrap the points in Fourier space before returning
    combined_nonredundant_lattice = wrap_indices(combined_nonredundant_lattice, log_diff_spectrum)
    
    return combined_nonredundant_lattice, unit_cell_dimensions, highest_resolution
    
    
def generate_lattice_mask_indices(combined_nonredundant_lattice, mask_radius=2):
    
    # Generate a linspace mask template
    mask_indices_template = np.transpose(np.array(list(product(np.linspace(-mask_radius, mask_radius, (2*mask_radius)+1), repeat=2))))
    
    # Initialize an empty list
    mask_indices_list = []
    
    # Add points according to all lattice spots in combined_nonredundant_lattice
    for i in range(combined_nonredundant_lattice.shape[1]):
        indices = combined_nonredundant_lattice[:,i]
        mask_indices = np.copy(mask_indices_template)
        mask_indices[0,:] = mask_indices_template[0,:] + indices[0]
        mask_indices[1,:] = mask_indices_template[1,:] + indices[1]
        mask_indices_list.append(mask_indices)

    # Collapse the list into an array
    mask_indices_array = np.transpose(np.concatenate(mask_indices_list, axis=1))
#     print(mask_indices_array.shape)

#  101_19 or 101_20 reproduces the error I think...
#     # Make sure a point doesn't fall at the very edge of array(?)
    mask_indices_array = mask_indices_array[np.where(mask_indices_array[:,0] < 5760),:]
    mask_indices_array = mask_indices_array[0,:,:]
#     print(mask_indices_array.shape)
    
    return mask_indices_array
    
# ------------------------------------------------------------------------------------------------------------

def remove_hotpixels(diffraction_indices, verbose=False):
    """
    Removes isolated "hot" pixels that exceed amplitude threshold but
    have no immediate neighbours, so are likely not part of the
    reciprocal lattice.
    """
    filter_list = []
    for ex_spot in tqdm(diffraction_indices, disable=not verbose):
        y_neighbours = np.sum((diffraction_indices == (ex_spot[0]+1, ex_spot[1])).all(axis=1)) + np.sum((diffraction_indices == (ex_spot[0]-1, ex_spot[1])).all(axis=1))
        x_neighbours = np.sum((diffraction_indices == (ex_spot[0], ex_spot[1]+1)).all(axis=1)) + np.sum((diffraction_indices == (ex_spot[0]-1, ex_spot[1]-1)).all(axis=1))
        if x_neighbours >= 1 or y_neighbours >= 1:
            filter_list.append(True)
        else: 
            filter_list.append(False)
    return filter_list



def replace_diffraction_spots(padded_fft, diffraction_indices, replace_distance_percent=0.05):
    """
    Take FFT from scipy_fft() or west_fft() and replace diffraction spots according to indices from find_diffraction_spots.
    replace_distance_percent: fraction of x-dimension to move along the diagonal when finding new amplitude.
    """
    # Generate a masked fft
    masked_fft = np.copy(padded_fft)
    
    # Generate a vector of random phases with the same length as number of diffraction_spots
    phases = np.random.uniform(low = 0.0,
                               high = 2*np.pi,
                               size = diffraction_indices.shape[0])
    phase_count = 0
    
    # Figure out the movement distances horizontally and vertically
    replace_distance = int((np.min(padded_fft.shape)*replace_distance_percent)/np.sqrt(2))
    
    # Loop through axis-0, axis-1 coordinates
    for indices in diffraction_indices:
        # If we're in the top quadrant
        if indices[0] < int(padded_fft.shape[0]/2):
            # Construct the complex number by moving down and right
            real = np.real(masked_fft[int(indices[0]+replace_distance), int(indices[1]+replace_distance)])
            imaginary = phases[phase_count]
            replacement = real + np.imag(imaginary)
            # Replace
            masked_fft[int(indices[0]), int(indices[1])] = replacement
        # If we're in the bottom quadrant
        else:
            # Construct the complex number by moving up and right
            real = np.real(masked_fft[int(indices[0]-replace_distance), int(indices[1]+replace_distance)])
            imaginary = phases[phase_count]
            replacement = real + np.imag(imaginary)
            # Replace
            masked_fft[int(indices[0]), int(indices[1])] = replacement
        # Increment the phase count
        phase_count =+ 1
    
    return masked_fft

def scipy_inverse_fft(masked_fft, verbose=False, threads=1):
    """Take masked FFT from replace_diffraction_spots and return the padded image array"""
    
    # Record the time it takes to do the FFT
    # for later comparison with multi-threaded FFTW
    
    # Start the timer
    start_time = time.time()
    
    # Perform the FFT
    padded_masked_image = sfft.irfft2(masked_fft,
                                      workers=threads)
    
    # Stop the timer
    end_time = time.time()
    
    if verbose:
        # Print the elapsed time to the nearest hundreth of a millisecond
        print("scipy_ifft(): iFFT performed in", np.round((end_time-start_time)*1000, 2), "milliseconds.")
    
    # Return padded masked image
    return padded_masked_image

def unpad_image(padded_masked_image, original_shape):
    """
    Take padded masked image from scipy_inverse_fft() or west_inverse_fft() 
    and original image shape and return unpadded image
    """
    return padded_masked_image[0:original_shape[0], 0:original_shape[1]]

def export_masked_mrc(masked_image, filename_out, verbose=False):
    """Take unpadded masked image from unpad_image() and write out a new header"""
    
#     # Generate a new filename
#     new_filename = "masked_output/" + filename[0:-4] + "_masked.mrc"
    
    # Generate a new header
    nx, ny = masked_image.shape
    nxyz = np.array([nx, ny, 1], dtype=np.float32)
    dmin = np.min(masked_image)
    dmax = np.max(masked_image)
    dmean = np.sum(masked_image)/(nx*ny)
    
    # Open a new file
    masked_mrc = open(filename_out, 'wb')
    
    # Write the header to the new file
    iwrhdr_opened(masked_mrc, 
                  nxyz, 
                  dmin, 
                  dmax, 
                  dmean, 
                  mode=2)
    
    # Write the masked image array to the new file
    iwrsec_opened(masked_image, masked_mrc)
    
    if verbose: print("Export complete!")
    
def mask_image(filename,
               filename_out,
               threshold_method,
               pixel_size,
               verbose=False,
               show_plots=False,
               threads=1,
               sigma=1,
               quantile=0.99,
               num_sd=3.0,
               num_sd_secondpass=2.0,
               x_window_percent=(0, 1),
               y_window_percent=(0, 1),
               miller_index_buffer=2,
               box_radius=10,
               min_lattice_size=5,
               mask_hotpixels = False,
               mask_radius=3,
               replace_distance_percent=0.05,
               return_spots=False,
               return_stats=False):
    """
    Take the filename of a micrograph in the .mrc format and subtract the streptavidin crystal lattice.
    print_timestamps: Boolean, whether to print how long it takes to perform the FFT and iFFT.
    sigma: Int. The SD of the Gaussian filter applied.
    quantile: Float. The quantile above which points will be designated diffraction spots to be masked.
    num_sd: Float. The number of SDs above the mean at which diffraction spots are detected.
    threshold_method: String. The thresholding method used to detect diffraction spots. Can be "quantile" or "sd".
    x_window_percent: Tuple. The upper and lower bounds of the x-dimension window in which to mask points (axis 1).
    x_window_percent: Tuple. The upper and lower bounds of the y-dimension window in which to mask points (axis 0).
    replace_distance_percent: Float. The fraction of the lesser dimension of the FFT to use as a radial distance to find replacement point values.
    """
    
    # Import the image
    image, header = import_mrc(filename)
    
    # Perform an FFT of the image
    padded_fft = scipy_fft(image, verbose, threads)
    
    # Subtract the FFT from a Gaussian-smoothed FFT
    log_diff_spectrum, smoothed_spectrum = generate_diff_spectrum(padded_fft, sigma)
        
    # Find diffraction spots
    if threshold_method == "quantile":
        diffraction_indices, diffraction_amplitudes = find_diffraction_spots_quantile(log_diff_spectrum, quantile, x_window_percent, y_window_percent)
    if threshold_method == "sd":
        diffraction_indices, diffraction_amplitudes = find_diffraction_spots_sd(log_diff_spectrum, num_sd, x_window_percent, y_window_percent)
    else:
        print("No thresholding method specified. Please specify a method using the threshold_method parameter.")
        return

    # Return some info if function is verbose
    if verbose:
        print("Number of first-pass spots found: " + str(diffraction_indices.shape[1]))
    
    
    # Look for the first lattice
    combined_nonredundant_lattice, unit_cell_dimensions, highest_resolution = find_lattice(diffraction_indices, 
                                                                                           diffraction_amplitudes, 
                                                                                           log_diff_spectrum,
                                                                                           pixel_size,
                                                                                           show_plots=show_plots,
                                                                                           verbose=verbose, 
                                                                                           num_sd_secondpass=num_sd_secondpass, 
                                                                                           miller_index_buffer=miller_index_buffer,
                                                                                           box_radius=box_radius,
                                                                                           min_lattice_size=min_lattice_size)
    
    # Store the results in arrays
    unit_cell_dimensions_array = np.array(unit_cell_dimensions)
    highest_resolution_array = np.array([highest_resolution])
    
    # Initialize combined_nonredundant_lattice_2lat to something that is not None to start the lattice-finding loop
    combined_nonredundant_lattice_2lat = 1
    
    # While find_lattice has returned something:
    while combined_nonredundant_lattice_2lat is not None:
    
        # Filter out the hot pixels - leave off, depreciated
        if mask_hotpixels:
            if verbose: print("Removing hot pixels...")
            combined_nonredundant_lattice = combined_nonredundant_lattice[remove_hotpixels(combined_nonredundant_lattice, verbose)]
            if verbose: print(str(num_spots - combined_nonredundant_lattice.shape[0]) + " hot pixels removed.")
        
        # Generate the mask indices
        mask_indices_array = generate_lattice_mask_indices(combined_nonredundant_lattice, mask_radius)
        
        # Replace the diffraction spots
        masked_fft = replace_diffraction_spots(padded_fft, mask_indices_array, replace_distance_percent)
        
        # Generate the diff spectrum again
        log_diff_spectrum, smoothed_spectrum = generate_diff_spectrum(masked_fft, sigma)
        
        # Threshold again
        diffraction_indices, diffraction_amplitudes = find_diffraction_spots_sd(log_diff_spectrum, num_sd, x_window_percent, y_window_percent)
        
        # Look for another lattice
        combined_nonredundant_lattice_2lat, unit_cell_dimensions, highest_resolution = find_lattice(diffraction_indices, 
                                                                                           diffraction_amplitudes, 
                                                                                           log_diff_spectrum,
                                                                                           pixel_size,
                                                                                           show_plots=show_plots,
                                                                                           verbose=verbose, 
                                                                                           num_sd_secondpass=num_sd_secondpass, 
                                                                                           miller_index_buffer=miller_index_buffer,
                                                                                           box_radius=box_radius,
                                                                                           min_lattice_size=min_lattice_size)
        
        # If another lattice was found
        if combined_nonredundant_lattice_2lat is not None:
            # Combine the suspected lattice points across all lattices
            combined_nonredundant_lattice = np.hstack((combined_nonredundant_lattice, combined_nonredundant_lattice_2lat))
            # Append information about the current lattice
            unit_cell_dimensions_array = np.vstack((unit_cell_dimensions_array, unit_cell_dimensions))
            highest_resolution_array = np.append(highest_resolution_array, highest_resolution)
                        
    if np.sum(highest_resolution_array/highest_resolution_array[0] < 0.5) > 0:
        if verbose:
            print("Warning: maximum resolution of lattices differs significantly. The highest resolution lattice is more than twice the resolution of the first detected lattice.")
            print(highest_resolution_array/highest_resolution_array[0])
    
    # Return the indices of diffraction spots if return_spots is true
    # Used for mask_movies
    if return_spots:
        return mask_indices_array
    
    if return_stats:
        return unit_cell_dimensions_array, highest_resolution_array
    
    # Perform the inverse FFT
    padded_masked_image = scipy_inverse_fft(masked_fft, verbose, threads)
    
    # Extract the original image from the padded inverse FFT
    masked_image = unpad_image(padded_masked_image, image.shape)
    
    # Export the image as an .mrc file
    export_masked_mrc(masked_image, filename_out, verbose)
    
    if verbose:
        print(filename + " masked successfully!")
        

# Functions to handle movies -----------------------------------------------------------------------------------
        
        
def import_movie(filename):
    
    # Open the movie
    raw_movie = open(filename,'rb')
    
    # Read in the header
    header = irdhdr_opened(raw_movie)
    
    # Extract info from the header
    nx = header['nx']
    ny = header['ny']
    nz = header['nz']
    mode = header['datatype']
    
    # Read each frame into a numpy array
    movie = np.empty((nx, ny, nz), dtype=np.float32)
    for z in range(nz):
        movie[:,:,z] = rescale_intensity(irdsec_opened(raw_movie, z))
    
    # Return the array and the header
    return movie, header

def scipy_batch_fft(movie, verbose=False, threads=1):
    """Take movie from import_movie() and return the FFT"""
    
    # Start the timer
    start_time = time.time()

    # Perform an FFT over the 0th and 1st axis of the movie
    padded_movie_fft = sfft.rfftn(movie, 
                                  s=(np.max(movie[:,:,0].shape), np.max(movie[:,:,0].shape)), 
                                  axes=(0,1),
                                  overwrite_x=True,
                                  workers=threads)

    # Stop the timer
    end_time = time.time()
    
    if verbose:
        # Print the elapsed time to the nearest hundreth of a millisecond
        print("scipy_batch_fft(): FFT performed in", np.round((end_time-start_time)*1000, 2), "milliseconds.")
        print("scipy_batch_fft():", np.round((end_time-start_time)*1000/movie.shape[2], 2), "milliseconds per frame.")
    
    # Eliminate potential artifacts along edge of FFT
    # padded_fft[0,:,:] = padded_fft[1,:,:]
    # padded_fft[:,0,:] = padded_fft[:,1,:]
    
    # Return FFT of padded movie
    return padded_movie_fft


def scipy_inverse_batch_fft(masked_movie_fft, verbose=False, threads=1):
    """Take masked FFT of movie and return the padded movie array"""
    
    # Record the time it takes to do the FFT
    # for later comparison with multi-threaded FFTW
    
    # Start the timer
    start_time = time.time()

    # Perform an FFT over the 0th and 1st axis of the movie
    padded_masked_movie = sfft.irfftn(masked_movie_fft, 
                                      axes=(0,1),
                                      overwrite_x=True,
                                      workers=threads)

    # Stop the timer
    end_time = time.time()
    
    if verbose:
        # Print the elapsed time to the nearest hundreth of a millisecond
        print("scipy_inverse_batch_fft(): iFFT performed in", np.round((end_time-start_time)*1000, 2), "milliseconds.")
        print("scipy_inverse_batch_fft():", np.round((end_time-start_time)*1000/masked_movie_fft.shape[2], 2), "milliseconds per frame.")
    
    # Return padded masked movie
    return padded_masked_movie

def unpad_movie(padded_masked_movie, original_movie_shape):
    """
    Take padded masked movie from scipy_inverse_fft() or west_inverse_fft() 
    and original movie shape and return unpadded image
    """
    return padded_masked_movie[0:original_movie_shape[0], 0:original_movie_shape[1], 0:original_movie_shape[2]]

def export_masked_movie(masked_movie, movie_filename_out, verbose=False):
    """
    """

    # Generate a new header
    nx, ny, nz = masked_movie.shape
    nxyz = np.array([nx, ny, nz], dtype=np.float32)
    dmin = np.min(masked_movie)
    dmax = np.max(masked_movie)
    dmean = np.sum(masked_movie)/(nx*ny*nz)
    
    # Open a new file
    masked_movie_mrc = open(movie_filename_out, 'wb')
    
    # Write the header to the new file
    iwrhdr_opened(masked_movie_mrc, 
                  nxyz, 
                  dmin, 
                  dmax, 
                  dmean, 
                  mode=2)

    # Write the masked movie array to the new file
    iwrsec_opened(masked_movie, masked_movie_mrc)
    
    if verbose: print("Export complete!")

def mask_movie(movie_filename,
               movie_filename_out,
               micrograph_filename,
               micrograph_filename_out,
               threshold_method,
               pixel_size,
               verbose=False,
               show_plots=False,
               threads=1,
               sigma=1,
               quantile=0.99,
               num_sd=3.0,
               num_sd_secondpass=2.0,
               x_window_percent=(0, 1),
               y_window_percent=(0, 1),
               miller_index_buffer=2,
               box_radius=10,
               min_lattice_size=5,
               mask_hotpixels = False,
               mask_radius=3,
               replace_distance_percent=0.05,
               return_spots=True,
               return_stats=False):
    
    # Find spots to mask by running mask_image
    diffraction_spots = mask_image(micrograph_filename,
                                   micrograph_filename_out,
                                   threshold_method,
                                   pixel_size,
                                   verbose = verbose,
                                   show_plots = show_plots,
                                   threads = threads,
                                   sigma = sigma,
                                   num_sd = num_sd,
                                   num_sd_secondpass = num_sd_secondpass,
                                   x_window_percent = x_window_percent,
                                   y_window_percent = y_window_percent,
                                   miller_index_buffer = miller_index_buffer,
                                   box_radius = box_radius,
                                   min_lattice_size = min_lattice_size,
                                   mask_hotpixels = mask_hotpixels,
                                   mask_radius= mask_radius,
                                   replace_distance_percent = replace_distance_percent,
                                   return_spots = return_spots,
                                   return_stats = return_stats)
    
    # Import the movie
    movie, header = import_movie(movie_filename)
    
    # Save the shape
    movie_shape = movie.shape
    
    # Perform an FFT of the movie
    padded_movie_fft = scipy_batch_fft(movie, verbose, threads)
    
    # Delete movie to save room
    del movie
    
    # Make a new array to hold masked movie
    masked_movie_fft = np.empty(padded_movie_fft.shape, dtype=np.complex64)
        
    # Replace diffraction spots in each frame
    for z in range(header['nz']):
        masked_movie_fft[:,:,z] = replace_diffraction_spots(padded_movie_fft[:,:,z], 
                                                            diffraction_spots, 
                                                            replace_distance_percent)
  
    # Perform the inverse FFT of the movie
    padded_masked_movie = scipy_inverse_batch_fft(masked_movie_fft, verbose, threads)
    
    # Unpad the movie
    masked_movie = unpad_movie(padded_masked_movie, movie_shape)
    
    # Export masked movie
    export_masked_movie(masked_movie, movie_filename_out, verbose)
    
    if verbose:
        print(movie_filename_out + " masked successfully!")
        