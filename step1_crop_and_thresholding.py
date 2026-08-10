# -*- coding: utf-8 -*-
"""
Created on Wed Sept 25 2024

@author: BarellaM
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from tifffile import imread
from PIL import Image as img
from PIL import ImageFilter as imgfil
import scipy.signal as sig
from scipy.ndimage import center_of_mass
import re

##############################################################################

def crop_and_threshold(working_folder, sample_text, ignore_files, \
                       thresh_custom_level, crop_size, n_bins, n_window, \
                       save_images_flag, save_filter_demo_image):

    plt.ioff()
    plt.close('all')  

    print('Cropping and thresholding all files inside:', working_folder)
    print('\n---> For samples:', sample_text)
    
    # create folder to save output figures
    save_folder = os.path.join(working_folder, 'figures')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    list_of_files = os.listdir(working_folder)
    list_of_files = [f for f in list_of_files if f.endswith(".tif")]
    if sample_text != 'all':
        list_of_files = [f for f in list_of_files if re.search(sample_text,f)]
    
    for i in range(len(list_of_files)):
        filename = list_of_files[i]
        # uncomment for debugging or check a particular nanostructure
        # if not re.search('A4', filename): continue
        # ignore the nanostructure if it's in the above ignore list
        if filename in ignore_files: 
            print('\n ... Ignoring file', filename)
            continue
        # if re.search(irrad_25mW[i], filename):
        print(filename)

        # load
        filepath = os.path.join(working_folder, filename)
        image = imread(filepath)
        # get image size and crop the footer of the SEM image
        image_size = image.shape
        image = image[0:image_size[1], 0:image_size[1]]
        
        # make intensity histogram
        histogram, bin_edges = np.histogram(image, 
                                            bins=n_bins, 
                                            density=False)
        
        histogram_smooth = sig.savgol_filter(histogram, 
                                             n_window, 1, axis = 0, 
                                             mode='interp')
    
        # find local minima
        local_minima = sig.argrelmin(histogram_smooth)[0]
        bin_min = [int(x) for x in list(bin_edges[local_minima])]
        
        if bin_min:
            thresh = bin_min[0]
        else:
            thresh = thresh_custom_level
            print('Warning! Histogram shows no min in %s' % filename) 
            print('Threshold set to', thresh)
        
        # thresholding
        binary_image = image > thresh
        binary_image_inverted = np.invert(binary_image)
        
        # find center of mass (in pixels)
        cm = center_of_mass(binary_image_inverted)
        cm_y_px = int(round(cm[0], 0))
        cm_x_px = int(round(cm[1], 0))
        
        # further cropping, crop closer to the nanostructure
        binary_image_inverted_cropped = binary_image_inverted[cm_y_px - crop_size: cm_y_px + crop_size, 
                                                              cm_x_px - crop_size: cm_x_px + crop_size]  
        image_cropped = image[cm_y_px - crop_size: cm_y_px + crop_size, 
                                              cm_x_px - crop_size: cm_x_px + crop_size]  
        # smooth edges of the binary image to get the contour
        # 4 is for 1st neightbours, 8 is for 1st and 2nd neighbours
        aux_image = img.fromarray(binary_image_inverted_cropped)
        binary_image_inverted_cropped_smoothed = aux_image.filter(imgfil.ModeFilter(size=4))
                   
        # SAVE IMAGES
        # create folder to save output figures
        save_folder_cropped = os.path.join(working_folder, 'figures\\step1\\image_cropped')
        if not os.path.exists(save_folder_cropped):
            os.makedirs(save_folder_cropped)

        if save_images_flag:
            # save images
            cropped_image_name = filename[:-4] + '_cropped'
            cropped_image_path = os.path.join(save_folder_cropped, '%s.png' % cropped_image_name)
            image_to_save = img.fromarray(image_cropped) 
            image_to_save.save(cropped_image_path)
            
            thresholded_image_name = filename[:-4] + '_filtered'
            thresholded_image_path = os.path.join(save_folder_cropped, '%s.png' % thresholded_image_name)
            image_to_save = img.fromarray(binary_image_inverted_cropped) 
            image_to_save.save(thresholded_image_path)
            
            thresholded_smooth_image_name = filename[:-4] + '_filtered_smoothed'
            thresholded_smooth_image_path = os.path.join(save_folder_cropped, '%s.png' % thresholded_smooth_image_name)
            binary_image_inverted_cropped_smoothed.save(thresholded_smooth_image_path)
              
        # SAVE PLOTS
        if save_filter_demo_image:
            fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
            ax = axes.ravel()
            ax[0] = plt.subplot(1, 3, 1)
            ax[1] = plt.subplot(1, 3, 2)
            ax[2] = plt.subplot(1, 3, 3)
            
            ax[0].imshow(image, cmap=plt.cm.gray)
            ax[0].set_title('Original')
            ax[0].axis('off')
            
            ax[1].hist(image.ravel(), bins=n_bins, density=False)
            ax[1].plot(bin_edges[:-1], histogram_smooth)     
            ax[1].plot(bin_edges[local_minima], histogram_smooth[local_minima],'bx')
            ax[1].set_yscale('log')
            ax[1].set_title('Histogram @ log scale')
            ax[1].axvline(thresh, color='r')
            
            ax[2].imshow(binary_image, cmap = plt.cm.gray)
            ax[2].set_title('Thresholded')
            ax[2].axis('off')
            
            # create folder to save output figures
            save_folder_demo = os.path.join(working_folder, 'figures\\step1\\filter_demo')
            if not os.path.exists(save_folder_demo):
                os.makedirs(save_folder_demo)
            figure_name = filename[:-4] + '_filter_demo'
            figure_path = os.path.join(save_folder_demo, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
            #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
            plt.close()
        
    print('\nSTEP 1 finished.')

    return
