import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def flim_colormap():
    colors = [
        (0, 0, 0),  # Black (under-range values)
        (0, 0, 1),  # Blue (Short lifetime)
        (0, 1, 1),  # Cyan
        (1, 1, 1),  # White (Mid-range)
        (1, 1, 0),  # Yellow
        (1, 0, 0),  # Red (Long lifetime)
        (1, 0, 1)   # Magenta (over-range values)
    ]
    cmap = LinearSegmentedColormap.from_list("FLIM_LUT", colors[1:-1], N=256)
    cmap.set_under(colors[0])  # Black for very short lifetimes
    cmap.set_over(colors[-1])  # Magenta for very long lifetimes
    return cmap

# set FLIM visualisation settings
FLIM_cmap = flim_colormap()
FLIM_cmap.set_bad(color='black')

gray32_cmap = cm.gray.copy()
gray32_cmap.set_bad(color = 'black')

# Function to compute g and s for a given lifetime
def phasor_coords(tau, omega):
    lifetime = float(tau) * 1e-9
    g = 1 / (1 + (omega * lifetime) ** 2)
    s = (omega * lifetime) / (1 + (omega * lifetime) ** 2)
    return (g, s)

def hexbin_phasor_plot (conn, image_id, keys = None, gridsize = 250, vmin=1000, vmax = 5000):
    
    """
    Function to generate phasor plot from FLIM data uploaded onto OMERO that already has g and s values calculated.
    Reads image metadata to get omega value and donor/acceptor info
    Plots g and s values with a hex-bin plot and draws a calibrated universal semi-circle.

    Parameters:
        conn: OMERO blitz gateway connection
        image_id (int): OMERO image ID
        keys (list): optional list of relevant metadata keys (strings) to add to the plot title 
        gridsize (int): size of hexbin
        vmin, vmax (int): min and max values for normalising lifetime images

    Returns:
        fig, gs: phasor plot, intensity, modulation and phase lifetime images.
    """
    
    # get OMERO image object and read metadata
    img = conn.getObject("Image", image_id)
    image_name = img.getName()
    channel_labels = img.getChannelLabels()
    pixel_size_um = img.getPixelSizeX()

    # read metadata annotations (k-v pairs)
    metadata_dict = {}
    for ann in img.listAnnotations():
        map_ann = conn.getObject("MapAnnotation", ann.getId())
        for k, v in map_ann.getValue():
            metadata_dict[k] = v
    
    omega = float(metadata_dict["omega"])
    donor = metadata_dict["Donor"]
    acceptor = metadata_dict["Acceptor"]

    # get image planes
    px = img.getPrimaryPixels()
    int_array = np.array(px.getPlane(0, 0, 0))
    mod_lft_array = np.array(px.getPlane(0, 1, 0))
    phase_lft_array = np.array(px.getPlane(0, 2, 0))
    g_values_array = np.array(px.getPlane(0, 3, 0))
    s_values_array = np.array(px.getPlane(0, 4, 0))

    # convert s and g values to pandas dataframe
    output = []
    for x,y in np.argwhere(int_array):
        output.append([g_values_array[x,y], s_values_array[x,y]] )

    df = pd.DataFrame(output, columns = ["g_value", "s_value"])
    
    # Create a figure with a gridspec layout
    fig = plt.figure(figsize=(18, 15)) 
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.5]) 

    # Set a global title for the entire grid
    # define plot title
    plot_title = [
        f"ImageID: {image_id}",
        f"{image_name}"
    ]
    if keys != None:
        plot_title += [f"{key}: {metadata_dict[key]}" for key in keys]
        
    plot_title.append(f"Donor: {donor} | Acceptor: {acceptor}")
    plot_title = "\n".join(plot_title)
    
    fig.suptitle(f"{plot_title}\n", fontsize=24)
    
    int_min = np.nanpercentile(int_array, 2)
    int_max = np.nanpercentile(int_array, 98)
    
    # Intensity Image
    ax0 = fig.add_subplot(gs[0, 0])
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax0.imshow(int_array, cmap=gray32_cmap, norm=Normalize(vmin=int_min, vmax=int_max))
    
    ax0.set_title(channel_labels[0], fontsize=18)
    fig.colorbar(im, cax=cax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('intensity', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    ax0.axis("off")
    
    scalebar_position, length, pixel_size = [(20, 480), 50, pixel_size_um]
    scalebar_length = round(length/pixel_size)
    
    # Plot the scale bar
    ax0.plot([scalebar_position[0], scalebar_position[0] + scalebar_length],
            [scalebar_position[1], scalebar_position[1]], 
            linewidth=5, color="white")
    
    # Lifetime from modulation
    ax1 = fig.add_subplot(gs[0, 1])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax1.imshow(mod_lft_array, cmap=FLIM_cmap, norm=Normalize(vmin, vmax))
    
    ax1.set_title(f"{channel_labels[1]} ({np.nanmean(mod_lft_array):.0f}ps)", fontsize=18)
    fig.colorbar(im, cax=cax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('lifetime (picoseconds)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    ax1.axis("off")
    
    # Plot the scale bar
    ax1.plot([scalebar_position[0], scalebar_position[0] + scalebar_length],
            [scalebar_position[1], scalebar_position[1]], 
            linewidth=5, color="white")
    
    # Phase Lifetime Image
    ax2 = fig.add_subplot(gs[0, 2])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax2.imshow(phase_lft_array, cmap=FLIM_cmap, norm=Normalize(vmin, vmax))
    
    ax2.set_title(f"{channel_labels[2]} ({np.nanmean(phase_lft_array):.0f}ps)", fontsize=18)
    fig.colorbar(im, cax=cax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('lifetime (picoseconds)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    ax2.axis("off")

    # Plot the scale bar
    ax2.plot([scalebar_position[0], scalebar_position[0] + scalebar_length],
            [scalebar_position[1], scalebar_position[1]], 
            linewidth=5, color="white")
    
    # Phasor plot (occupying the full width of the bottom row)
    ax3 = fig.add_subplot(gs[1:, :])  
    ax3.set_aspect('equal')
    
    # Hexbin plot of phasors
    # Plot the phasor plot
    #fig, ax = plt.subplots(figsize=(10, 6))
        
    hb = ax3.hexbin(
        df["g_value"], df["s_value"],
        gridsize=gridsize,
        cmap='inferno',
        extent = (0, 1.1, 0, 0.6), #(xmin, xmax, ymin,ymax)
        vmin=2,
    )
    
    # Plot the unit circle for reference
    circle = plt.Circle((0.5, 0), 0.5, color='white', fill=False, linestyle='--', linewidth=1)
    ax3.add_artist(circle)
    
    # Lifetimes to plot on universal semi-circle (in ns)
    lifetimes = [5, 4, 3, 2, 1]
    
    # Add single lifetime values
    for tau in (lifetimes):
        g, s = phasor_coords(tau, omega)
        ax3.plot(g, s, '.', color='w') 
        ax3.text(g, s+0.025, f'{tau}ns', color='w', fontsize=18, ha='left', va='top', ) 

    # add colorbar
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(hb, cax=cax)#, label = "Density")
    cbar.set_label("Density", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    # Set axis limits and labels
    ax3.set_aspect('equal')
    ax3.tick_params(labelsize=14)
    ax3.set_xlim([0, 1.1])
    ax3.set_ylim([0, 0.6])
    ax3.set_xlabel('g', fontsize=18)
    ax3.set_ylabel('s', fontsize=18)
    ax3.set_facecolor('k')
    ax3.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

    return fig, gs

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Function to compute g and s for a given lifetime
def phasor_coords(tau, omega):
    lifetime = float(tau) * 1e-9
    g = 1 / (1 + (omega * lifetime) ** 2)
    s = (omega * lifetime) / (1 + (omega * lifetime) ** 2)
    return (g, s)

# Function to add unit circle to plot
def add_unit_circle (ax, omega, lifetimes = [5, 4, 3, 2, 1]):
    # Plot the unit circle for reference
    circle = plt.Circle((0.5, 0), 0.5, color='k', fill=False, linestyle='--', linewidth=1)
    ax.add_artist(circle)
    
    # Add single lifetime values
    for tau in (lifetimes):
        g, s = phasor_coords(tau, omega)
        ax.plot(g, s, '.', color='k') 
        ax.text(g, s+0.025, f'{tau}ns', color='k', fontsize=12, ha='left', va='top', )

    # Set axis limits and labels
    ax.set_xlim([0, 1.1])
    ax.set_ylim([0, 0.6])
    ax.set_xlabel('g', fontsize=14)
    ax.set_ylabel('s', fontsize=14)
    ax.set_facecolor('w')
    ax.grid(True)

    return ax

def get_values_image_id (conn, image_id, keys = None, output = []):
    
    """
    Gets image object from OMERO.
    Converts image planes to nested lists.
    Reads metadata to aid with plotting.

    Parameters:
        conn: OMERO blitz gateway connection
        image_id (int): OMERO image ID
        keys (list): optional list of relevant metadata keys (strings) to add to the output array 
        output (list): nested lists for values at x,y coords.

    Returns:
        headings, output: phasor plot, intensity, modulation and phase lifetime images.
    """

    # get OMERO image object and read metadata
    img = conn.getObject("Image", image_id)
    image_name = img.getName()
    
    # read metadata map annotations (k-v pairs)
    metadata_dict = {}
    for ann in img.listAnnotations():
        map_ann = conn.getObject("MapAnnotation", ann.getId())
        for k, v in map_ann.getValue():
            metadata_dict[k] = v

    if type(keys) != list:
        keys = [keys]
        
    list_of_keys = ["Donor", "Acceptor"] + keys

    values = []
    for key in list_of_keys:
        value = metadata_dict[key]
        values.append(value)

    # get image planes
    zct_list = list((0, c, 0) for c in range(5))
    pixels = img.getPrimaryPixels()
    planes = pixels.getPlanes(zct_list)  # This is a generator, loading only when needed
    int_array, mod_lft_array, phase_lft_array, g_values_array, s_values_array = np.array([plane for plane in planes])  # Ensure correct shape

    headings = ["image_id"] + list_of_keys + ["intensity", "modulation_lifetime_ps", "phase_lifetime_ps", "g_value", "s_value"]

    # append values to output
    for x,y in np.argwhere(int_array):
        metadata_values = [image_id] + values
        output.append(metadata_values + [int_array[x,y], mod_lft_array[x,y], phase_lft_array[x,y], g_values_array[x,y], s_values_array[x,y]])

    return headings, output
