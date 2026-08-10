# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 2024

@author: BarellaM
"""

import numpy as np
import pandas as pd

def DNH_tip_contour(amplitude, offset, phase, N):
    array = np.arange(N)
    angle = array*np.pi/N
    # x component
    contour_x = offset + amplitude*np.sin(angle - phase)
    return contour_x

# Function to minimize: S-squared. We will find the best parameters
# that minimize the difference between the acquired data and the function
def s_squared(params, data, N):    
    function = DNH_tip_contour(*params, N)
    s2 = 0
    residuals = data - function
    s2_residuals = residuals**2 
    s2 = np.sum(s2_residuals)
    return s2

def calc_r2(observed, fitted):
    # Calculate coefficient of determination
    avg_y = observed.mean()
    # sum of squares of residuals
    ssres = ((observed - fitted)**2).sum()
    # total sum of squares
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

def func_powerlaw(x, m, c):
    return c * x**m

def calculate_pixel_size(number_of_pixels, folder_name, view_field_filepath):
    # Calculate pixel size in nm
    view_field = get_number_for_folder(view_field_filepath, folder_name)    
    if view_field is not None:
        print(f"view_field for folder '{view_field_filepath}': {view_field}")
    else:
        print(f"Folder '{folder_name}' not found in the CSV file")
    pixel_size = (view_field * 1000) / number_of_pixels  # convert um to nm
    return pixel_size

def get_number_for_folder(csv_file, target_folder, number_col='viewfield_um', folder_col='folder'):
    """
    Retrieve the number associated with a given folder name from a CSV file with headers.
    Args:
        csv_file (str): Path to the CSV file
        target_folder (str): Folder name to search for
        number_col (str): Name of the column containing numbers (default 'viewfield_um')
        folder_col (str): Name of the column containing folder names (default 'folder')
    Returns:
        The associated number if found, None otherwise
    """
    try:
        # Read CSV with headers
        df = pd.read_csv(csv_file)
        # Find matching row(s)
        matches = df[df[folder_col] == target_folder][number_col]
        if len(matches) > 0:
            # Return first match if multiple exist
            print(f"Found {len(matches)} matches for folder '{target_folder}' in '{csv_file}'")
            return matches.iloc[0]
        return None
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        return None
    except KeyError as e:
        print(f"Error: Column {e} not found in CSV file")
        return None