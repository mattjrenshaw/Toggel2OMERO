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