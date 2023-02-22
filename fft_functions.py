# Functions to perform FFTs of images and particle stacks

from funcs_mrcio import iwrhdr_opened, irdhdr_opened, iwrsec_opened, irdsec_opened
from skimage.exposure import rescale_intensity
import numpy as np
import scipy.fft as sfft
import time
import multiprocessing

def import_stack(filename):
    
    # Open the movie
    raw_stack = open(filename,'rb')
    
    # Read in the header
    header = irdhdr_opened(raw_stack)
    
    # Extract info from the header
    nx = header['nx']
    ny = header['ny']
    nz = header['nz']
    mode = header['datatype']
    
    # Read each frame into a numpy array
    stack = np.empty((nx, ny, nz), dtype=np.float32)
    for z in range(nz):
        stack[:,:,z] = rescale_intensity(irdsec_opened(raw_stack, z))
    
    # Return the array and the header
    return stack, header

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
    
    # Return FFT of padded image
    return padded_fft

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
    return masked_image

def export_masked_stack(masked_stack, stack_filename_out, verbose=False):
    """
    """

    # Generate a new header
    nx, ny, nz = masked_stack.shape
    nxyz = np.array([nx, ny, nz], dtype=np.float32)
    dmin = np.min(masked_stack)
    dmax = np.max(masked_stack)
    dmean = np.sum(masked_stack)/(nx*ny*nz)
    
    # Open a new file
    masked_stack_mrc = open(stack_filename_out, 'wb')
    
    # Write the header to the new file
    iwrhdr_opened(masked_movie_mrc, 
                  nxyz, 
                  dmin, 
                  dmax, 
                  dmean, 
                  mode=2)

    # Write the masked movie array to the new file
    iwrsec_opened(masked_stack, masked_stack_mrc)
    
    if verbose: print("Export complete!")