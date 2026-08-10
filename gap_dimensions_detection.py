# -*- coding: utf-8 -*-
"""
Created on Wed Sept 25 2024

@author: BarellaM
"""

import os
import matplotlib.pyplot as plt
import step1_crop_and_thresholding as step1
import step2_find_parameters as step2
import step3_do_stats as step3
from auxiliary_functions_for_gap_dimensions_detection import calculate_pixel_size

##############################################################################

def run_gap_characterization(base_folder, folder_name, view_field_filename):

    # run analysis on all folder inside the base folder
    working_folder = os.path.join(base_folder, folder_name)

    plt.ioff()
    plt.close('all')    

    # create folder to save output figures
    save_folder = os.path.join(working_folder, 'figures')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # WHICH SAMPLE DO YOU WANT TO ANALYZE?
    # sample_text = 'sample_1'
    # sample_text = 'sample_2'
    # sample_text = 'sample_3'
    # sample_text = 'sample_4'
    sample_text = 'all'
    
    # FILES TO IGNORE
    ignore_files = ['sample_2_DNH_D3.tif', 'sample_2_DNH_B6.tif']
       
    # PARAMETERS FOR STEP 1
    crop_size = 240 # in px from the center of mass
    save_images_flag = True
    save_filter_demo_image = True
    # thresh_custom_level DO NOT CHANGE WITHOUT OBSERVING DEMO THRESHOLDING IMAGES
    thresh_custom_level = 2700 # this number was arbitrary selected from observations 
    n_bins = 100 # number of bins for the intensity histogram 
    n_window = 11 # number of bins to smooth the histogram
    
    # PARAMETERS FOR STEP 2
    # get pixel size from the view field file
    number_of_pixels = 2048 # SEM image size in pixels
    view_field_filepath = os.path.join(base_folder, view_field_filename)
    pixel_size = calculate_pixel_size(number_of_pixels, folder_name, view_field_filepath) # in nm for SEM
    print(f'Pixel size: {pixel_size} nm')
    # INPUTS
    search_name = "filtered_smoothed.png" # filename ending to be analyzed
    save_image_with_measurements = True
    plot_fitting = False
    circle_fit_distance_from_center = 100 # in nm
    circle_fit_distance_from_center_px = int(round(circle_fit_distance_from_center/pixel_size, 0))
    gap_length = 80 # in nm for SEM
    gap_px_length_half = int(round(gap_length/pixel_size/2, 0))
    tilted_images_flag = False
    # tilted_images_flag = False
    angle = 35 # in degree
    tilt_analysis_distance_px = 50 # in pixels
    tilt_gap_offset = 3 # in pixels
    simulations_flag = False # in case of simulated DNH images

    # WHICH STEPS DO YOU WANNA RUN?
    # do_step1, do_step2, do_step3 = 1, 1, 1
    # do_step1, do_step2, do_step3 = 1, 1, 0
    # do_step1, do_step2, do_step3 = 0, 0, 1
    # do_step1, do_step2, do_step3 = 1, 0, 0
    do_step1, do_step2, do_step3 = 0, 1, 0

    #####################################################################
    ##################################################################### 
    
    if do_step1:
        print('\nExecuting STEP 1...')
        # run step
        step1.crop_and_threshold(working_folder, sample_text, ignore_files, \
                                 thresh_custom_level, crop_size, n_bins, n_window, \
                                 save_images_flag, save_filter_demo_image)
    else:
        print('\nSTEP 1 was not executed.')

    if do_step2:
        print('\nExecuting STEP 2...')
        # run step
        step2.find_contours_and_measure(working_folder, folder_name, sample_text, 
                                        search_name, ignore_files, \
                                        pixel_size, circle_fit_distance_from_center_px, \
                                        gap_px_length_half, angle, tilt_gap_offset, \
                                        tilted_images_flag, tilt_analysis_distance_px, \
                                        save_images_flag, \
                                        save_image_with_measurements, plot_fitting, \
                                        simulations_flag)
    else:
        print('\nSTEP 2 was not executed.')

    if do_step3:
        print('\nExecuting STEP 3...')
        # run step
        step3.do_stats(working_folder, folder_name, sample_text, \
                       tilted_images_flag)
    else:
        print('\nSTEP 3 was not executed.')
        
#####################################################################
#####################################################################
#####################################################################

if __name__ == '__main__':
    # define working folder
    base_folder = 'C:\\datos_mariano\\posdoc\\ami\\plasmonic_optical_trapping\\SEM\\gap_characterization'
    
    view_field_filename = 'view_field_INPUT.csv'

    folder_name = '20251029_sample_21_gap_widths'

    run_gap_characterization(base_folder, folder_name, view_field_filename)


