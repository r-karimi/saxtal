# Import packages

from funcs_mrcio import iwrhdr_opened, irdhdr_opened, iwrsec_opened, irdsec_opened
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from tqdm import tqdm

import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt
import time
import pyfftw
import multiprocessing
import os

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
    
def pad_image(image, header):
    """Take image and header from import_mrc() and pad image to make array square"""
    # Find the maximum dimension of the image
    max_dimension = np.maximum(header['nx'], header['ny'])
    
    # Make an array of the median pixel value of the image array with maximum dimensions
    padded_image = np.full(shape=(max_dimension, max_dimension),
                           fill_value=np.median(image))
    
    # Transplant image values into padded array
    padded_image[0:header['nx'],0:header['ny']] = image
    
    # Return padded image and header
    return padded_image, header

def scipy_fft(image, verbose=False, threads=1):
    """Take padded image from pad_image() and return the FFT"""
    
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
    diff_spectrum = padded_spectrum - smoothed_spectrum
    
    return diff_spectrum, smoothed_spectrum

def find_diffraction_spots_quantile(diff_spectrum, quantile=0.99, x_window_percent = (0, 1), y_window_percent = (0, 1)):
    """
    Take a difference spectrum and find outliers greater than a given quantile,
    taking a user-defined window of the FFT.
    """
    
    # Define minimum and maximum indices of the window
    
    y_min = int(diff_spectrum.shape[0]*y_window_percent[0])
    y_max = int(diff_spectrum.shape[0]*y_window_percent[1])
    x_min = int(diff_spectrum.shape[1]*x_window_percent[0])
    x_max = int(diff_spectrum.shape[1]*x_window_percent[1])

    # Calculate the threshold on a subset of the spectrum
    quantile_threshold = np.quantile(diff_spectrum[y_min:y_max, x_min:x_max].flatten(), 
                                     quantile)
    
    # Find the indices of points where the value is greater than the threshold
    diffraction_spots = np.transpose(np.where(diff_spectrum >= quantile_threshold))
    
    # Discard the points that fall outside the window
    diffraction_spots_yfilt = np.logical_and(diffraction_spots[:,0] >= y_min, diffraction_spots[:,0] <= y_max)
    diffraction_spots_xfilt = np.logical_and(diffraction_spots[:,1] >= x_min, diffraction_spots[:,1] <= x_max)
    diffraction_spots_filt = np.array([diffraction_spots_yfilt, diffraction_spots_xfilt]).min(axis=0)
    
    # Subset the rows of diffraction_spots that fall within the window
    diffraction_spots = diffraction_spots[diffraction_spots_filt,:]

    return diffraction_spots

def find_diffraction_spots_sd(diff_spectrum, num_sd=3.0, x_window_percent=(0, 1), y_window_percent=(0, 1)):
    """
    Take a difference spectrum and find outliers past a certain number of SDs from the mean,
    taking a user-defined window of the FFT.
    """
    
    # Define minimum and maximum indices of the window
    y_min = int(diff_spectrum.shape[0]*y_window_percent[0])
    y_max = int(diff_spectrum.shape[0]*y_window_percent[1])
    x_min = int(diff_spectrum.shape[1]*x_window_percent[0])
    x_max = int(diff_spectrum.shape[1]*x_window_percent[1])

    # Calculate the mean on a subset of the spectrum
    mean = np.mean(diff_spectrum[y_min:y_max, x_min:x_max].flatten())
    # Calculate the standard deviation
    sd = np.std(diff_spectrum[y_min:y_max, x_min:x_max].flatten())
    
    # Find the indices of points where the value is greater than the threshold
    diffraction_spots = np.transpose(np.where(diff_spectrum >= mean+(num_sd*sd)))
    
    # Discard the points that fall outside the window
    diffraction_spots_yfilt = np.logical_and(diffraction_spots[:,0] >= y_min, diffraction_spots[:,0] <= y_max)
    diffraction_spots_xfilt = np.logical_and(diffraction_spots[:,1] >= x_min, diffraction_spots[:,1] <= x_max)
    diffraction_spots_filt = np.array([diffraction_spots_yfilt, diffraction_spots_xfilt]).min(axis=0)
    
    # Subset the rows of diffraction_spots that fall within the window
    diffraction_spots = diffraction_spots[diffraction_spots_filt,:]

    return diffraction_spots

def plot_diffraction_spots(diffraction_spots):
    """
    Helper function to plot relative locations of diffraction spots to
    evaluate how well the reciprocal lattice was found.
    """
  
def remove_hotpixels(diffraction_spots):
    """
    Removes isolated "hot" pixels that exceed amplitude threshold but
    have no immediate neighbours, so are likely not part of the
    reciprocal lattice.
    """
    filter_list = []
    for ex_spot in tqdm(diffraction_spots):
        y_neighbours = np.sum((diffraction_spots == (ex_spot[0]+1, ex_spot[1])).all(axis=1)) + np.sum((diffraction_spots == (ex_spot[0]-1, ex_spot[1])).all(axis=1))
        x_neighbours = np.sum((diffraction_spots == (ex_spot[0], ex_spot[1]+1)).all(axis=1)) + np.sum((diffraction_spots == (ex_spot[0]-1, ex_spot[1]-1)).all(axis=1))
        if x_neighbours >= 1 and y_neighbours >= 1:
            filter_list.append(True)
        else: 
            filter_list.append(False)
    return(filter_list)

def replace_diffraction_spots(padded_fft, diffraction_spots, replace_distance_percent = 0.05):
    """
    Take FFT from scipy_fft() or west_fft() and replace diffraction spots according to indices from find_diffraction_spots.
    replace_distance_percent: fraction of x-dimension to move along the diagonal when finding new amplitude.
    """
    # Generate a masked fft
    masked_fft = np.copy(padded_fft)
    
    # Generate a vector of random phases with the same length as number of diffraction_spots
    phases = np.random.uniform(low = 0.0,
                               high = 2*np.pi,
                               size = diffraction_spots.shape[0])
    phase_count = 0
    
    # Figure out the movement distances horizontally and vertically
    replace_distance = int((np.min(padded_fft.shape)*replace_distance_percent)/np.sqrt(2))
    
    # Loop through axis-0, axis-1 coordinates
    for indices in diffraction_spots:
        # If we're in the top quadrant
        if indices[0] < int(padded_fft.shape[0]/2):
            # Construct the complex number by moving down and right
            real = np.real(masked_fft[indices[0]+replace_distance, indices[1]+replace_distance])
            imaginary = phases[phase_count]
            replacement = real + np.imag(imaginary)
            # Replace
            masked_fft[indices[0], indices[1]] = replacement
        # If we're in the bottom quadrant
        else:
            # Construct the complex number by moving up and right
            real = np.real(masked_fft[indices[0]-replace_distance, indices[1]+replace_distance])
            imaginary = phases[phase_count]
            replacement = real + np.imag(imaginary)
            # Replace
            masked_fft[indices[0], indices[1]] = replacement
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
    
    # Return FFT of padded image
    return padded_masked_image

def unpad_image(padded_masked_image, original_shape):
    """
    Take padded masked image from scipy_inverse_fft() or west_inverse_fft() 
    and original image shape and return unpadded image
    """
    return padded_masked_image[0:original_shape[0], 0:original_shape[1]]

def export_masked_mrc(masked_image, filename):
    """Take unpadded masked image from unpad_image() and write out a new header"""
    
    # Generate a new filename
    new_filename = "masked_output/" + filename[0:-4] + "_masked.mrc"
    
    # Generate a new header
    nx, ny = masked_image.shape
    nxyz = np.array([nx, ny, 1], dtype=np.float32)
    dmin = np.min(masked_image)
    dmax = np.max(masked_image)
    dmean = np.sum(masked_image)/(nx*ny)
    
    # Open a new file
    masked_mrc = open(new_filename, 'wb')
    
    # Write the header to the new file
    iwrhdr_opened(masked_mrc, 
                  nxyz, 
                  dmin, 
                  dmax, 
                  dmean, 
                  mode=2)
    
    # Write the rebinned array to the new file
    iwrsec_opened(masked_image, masked_mrc)
    
    print("Export complete!")
    
def mask_image(filename, 
               threshold_method,
               verbose=False,
               threads=1,
               sigma=1,
               quantile=0.99,
               num_sd=3.0,
               x_window_percent=(0, 1),
               y_window_percent=(0, 1),
               mask_hotpixels = True,
               replace_distance_percent=0.05):
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
    diff_spectrum, smoothed_spectrum = generate_diff_spectrum(padded_fft, sigma)
    
    # Find diffraction spots
    if threshold_method == "quantile":
        diffraction_spots = find_diffraction_spots_quantile(diff_spectrum, quantile, x_window_percent, y_window_percent)
    if threshold_method == "sd":
        diffraction_spots = find_diffraction_spots_sd(diff_spectrum, num_sd, x_window_percent, y_window_percent)
    else:
        print("No thresholding method specificed. Please specify a method using the threshold_method parameter.")
        return

    num_spots = diffraction_spots.shape[0]
    
    if verbose:
        print("Number of diffraction spots found: " + str(num_spots))
    
    # Filter out the hot pixels
    if mask_hotpixels:
        print("Removing hot pixels...")
        diffraction_spots = diffraction_spots[remove_hotpixels(diffraction_spots)]
        print(str(num_spots - diffraction_spots.shape[0]) + " hot pixels removed.")
              
    # Replace the diffraction spots
    masked_fft = replace_diffraction_spots(padded_fft, diffraction_spots, replace_distance_percent)
    
    # Perform the inverse FFT
    padded_masked_image = scipy_inverse_fft(masked_fft, verbose, threads)
    
    # Extract the original image from the padded inverse FFT
    masked_image = unpad_image(padded_masked_image, image.shape)
    
    # Export the image as an .mrc file
    export_masked_mrc(masked_image, filename)
    
    if verbose:
        print(filename + " masked successfully!")