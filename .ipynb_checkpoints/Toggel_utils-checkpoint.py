import numpy as np

def parse_header(header_str):
    """
    Parses the header from the .fli file. Detects sections and converts metadata into nested dictionaries of key:value pairs.

    Parameters:
        header_str (string): header from .fli file, ascii format.

    Returns:
        Nested dictionaries: Each [SECTION] is a dictionary of key:value pairs for reading metadata values.
    """
    
    metadata = {}
    current_section = None

    for line in header_str.splitlines():
        line = line.strip()
        if not line:
            continue

        # Detect section headers
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]")
            metadata[current_section] = {}
            continue

        # Detect key = value lines
        if "=" in line:
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()

            # Store in current section if one is active, otherwise top-level
            if current_section:
                metadata[current_section][key] = value
            else:
                metadata[key] = value

    return metadata

# open raw phases
# reader for .fli files, requires known values of height, width and number of phases
# returns header as string, phases as a 3D numpy stack, and the dark image as a numpy array
def read_fli_file(file_path):
    
    """
    Reader for .fli file.

    Parameters:
        file_path (string): full path to .fli file.

    Returns:
        header_metadata (nested dictionaries): Each [SECTION] contains key:value pairs for reading metadata values
        phases_stack (numpy array): raw intensity values for each phase
        dark_img (numpy array): camera dark image for this dataset
    """
    
    with open(file_path, 'rb') as f:
        # Step 1: Read the header until the "{END}" marker
        header = b""
        while not header.endswith(b"{END}"):
            header += f.read(1)
        header_str = header.decode('ascii')

       # Step 2: Parse header and extract metadata
        header_metadata = parse_header(header_str)
        width = header_metadata['LAYOUT']['x']
        height = header_metadata['LAYOUT']['y']
        n_phases = header_metadata['LAYOUT']['phases']
    
        # Step 3: Read the binary data (image stack)
        binary_data = f.read()
    
        # Step 4: Reshape the binary data into a stack of frames
        frame_size = int(width) * int(height)
        total_pixels = (int(n_phases) + 1) * frame_size
    
        image_stack = np.frombuffer(binary_data[:total_pixels*2], dtype=np.uint16)
    
        # Step 5: Reshape and separate phase images from dark image 
        image_stack = image_stack.reshape((int(n_phases) + 1), int(height), int(width))
    
        phases_stack = image_stack[0:int(n_phases)]
        dark_img = image_stack[int(n_phases)]

        return header_metadata, phases_stack, dark_img


def find_page_id(indexmap_array, c=0, z=0, t=0):
    """
    Find the row in the MicroManager IndexMap array
    that matches the given (c, z, t).
    Returns row index and uid
    """
    for i, row in enumerate(indexmap_array):
        chan, zs, ts, pos, uid = row
        if (chan, zs, ts) == (c, z, t):
            return i, pos, uid
    return None  # if no match

import tkinter as tk
from tkinter import filedialog

def get_filepath ():
    """
    Dialog to select file.
    
    """
    root = tk.Tk()
    root.lift()            # Lift it above all windows
    root.attributes("-topmost", True)   # Make sure it stays on top
    root.after_idle(root.attributes, "-topmost", False)  # Reset "always on top"
        
    # select file to open
    path_to_file = filedialog.askopenfilename(title=f"Select experimental info CSV file", parent=root)

    root.withdraw()
    
    return path_to_file

from datetime import datetime as dt
import re

def find_phase_series_by_timestamp(list_of_image_names, received_time, acquisition_type):
    """
    Find the image name from a timestamp.
    Parameters:
        list_of_image_names (list): list of image names as found in omero, image name to contain acquisition timestamp.
        received_time (string): received time read from sample image, in format '%Y-%m-%d %H:%M:%S.nnnn'
        acquisition_type (string): 'reference' or 'sample'

    Returns:
        closest_image (string): image name (from list_of_image_names) that is closest to but before received_time
    
    """
    
    timestamp, _ = received_time.split(".")
    timestamp_datetime = dt.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    pattern = re.compile(rf'(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{2}}-\d{{2}}-\d{{2}})-{acquisition_type}-')

    closest_image = None
    closest_time_diff = None

    for image_name in list_of_image_names:
        match = pattern.match(image_name)
        if match:
            date_str, time_str = match.groups()
            file_datetime_str = f"{date_str} {time_str}"
            file_datetime = dt.strptime(file_datetime_str, "%Y-%m-%d %H-%M-%S")
            
            if file_datetime < timestamp_datetime:
                time_diff = timestamp_datetime - file_datetime
                if closest_time_diff is None or time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    closest_image = image_name

    return closest_image

# calculate expected values based on modulation frequency, reference lifetime and number of phases acquired
def calculate_expected_values (frequency, ref_lifetime, nPhases):

    # scale variables
    frequency_hz = float(frequency)
    lifetime_s = float(ref_lifetime) * 1e-9
    
    # Angular frequency
    omega = 2 * np.pi * frequency_hz
    
    # phases in radians
    phases = np.linspace(0, 2 * np.pi, int(nPhases))  # 12 phases from 0 to 2π

    # Calculate phase delay in radians
    phase_delay_radians = np.arctan(omega * lifetime_s)
    
    # Convert phase delay to degrees
    phase_delay_degrees = np.degrees(phase_delay_radians)
    
    # Calculate the expected modulation depth
    modulation_depth = 1 / np.sqrt(1 + (omega * lifetime_s)**2)
    
    return (omega, phases, phase_delay_radians, modulation_depth)

# Define a sine wave function to fit the data
def sine_wave(x, amplitude, phase_shift, offset):
    phase_shift = (phase_shift + 180) % 360
    return amplitude * np.sin(x + phase_shift) + offset

from scipy.optimize import curve_fit

# System calibration using reference dataset
def calibrate_reference (signal_array, phases, phase_delay_radians, modulation_depth, width, height):
    
    # Initialize arrays to hold the modulation and phase results
    calibration_m = np.zeros((height, width)) # modulation depth calibration values
    calibration_phi = np.zeros((height, width)) # phase shift calibration values
    A_fitted = np.zeros((height, width))
    phi_fitted = np.zeros((height, width))
    offset_fitted = np.zeros((height, width))
    ref_mean = np.zeros((height, width))
       
    # Calculate amplitude modulation (m) and phase (φ) for each pixel
    non_zeroes = np.argwhere(np.min(signal_array, axis=0) > 0)
    
    for x, y in non_zeroes: # all pixels in bkg_ref_array > 0
        # Extract the signal at the current pixel location
        signal = signal_array[:, x, y]

        # Normalise signal to mean
        signal_mean = signal.mean()
        ref_mean[x, y] = signal_mean
        
        normalised_signal = signal / signal_mean 

        # Fit the sine wave model to the data
        popt, _ = curve_fit(sine_wave, phases, normalised_signal, 
                            p0=[modulation_depth, phase_delay_radians, 1.0] # initial values are the expected reference values
                           )

        # Extract fitted parameters (Amplitude, Phase shift, average/offset)
        A_fitted[x, y], phi_fitted[x, y], offset_fitted[x, y] = popt
        
        # Calibrate modulation
        calibration_m[x, y] = modulation_depth / (A_fitted[x, y]/offset_fitted[x, y])

        # Calculate phase
        calibration_phi[x, y] = phase_delay_radians - phi_fitted[x, y] 
    
    return (ref_mean, A_fitted, phi_fitted, offset_fitted, phases, calibration_m, calibration_phi)