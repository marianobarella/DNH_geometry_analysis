# -*- coding: utf-8 -*-
"""
Created on Wed Sept 25 2024

@author: BarellaM
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.ndimage import center_of_mass
from PIL import Image as img
import re
from skimage import measure
import scipy.optimize as opt
from circle_fit import taubinSVD, plot_data_circle
from auxiliary_functions_for_gap_dimensions_detection import DNH_tip_contour, s_squared, calc_r2, func_powerlaw
import pandas as pd

##############################################################################

def find_contours_and_measure(initial_folder, folder_name, sample_text, search_name, \
                              ignore_files, \
                              pixel_size, circle_fit_distance_from_center_px, \
                              gap_px_length_half, angle, tilt_gap_offset, \
                              tilted_images_flag, tilt_analysis_distance_px, \
                              save_images_flag, save_image_with_measurements, \
                              plot_fitting, simulations_flag):

    plt.ioff()
    plt.close('all')  
    
    # Check if step 1 was executed on this dataset
    working_folder = os.path.join(initial_folder, 'figures\\step1\\image_cropped')
    if not os.path.exists(working_folder):
        print(working_folder, 'DOES NOT EXIST')
        print('Run step 1 or double check input folder')
        return
    
    print('Finding countours and measuring nanostructures in:', initial_folder)
    print('\n---> For samples:', sample_text)
    print()
    
    list_of_files = os.listdir(working_folder)
    list_of_files = [f for f in list_of_files if f.endswith(search_name)]
    if sample_text != 'all':
        list_of_files = [f for f in list_of_files if re.search(sample_text, f)]
        
    # allocate
    gap_width_array = np.zeros(len(list_of_files))
    dnh_length_array = np.zeros(len(list_of_files))
    radius_circle_top_array = np.zeros(len(list_of_files))
    center_circle_top_array = np.zeros((len(list_of_files), 2))
    residual_err_circle_top_array = np.zeros(len(list_of_files))
    radius_circle_bottom_array = np.zeros(len(list_of_files))
    center_circle_bottom_array = np.zeros((len(list_of_files), 2))
    residual_err_circle_bottom_array = np.zeros(len(list_of_files))
    amplitude_left_array = np.zeros(len(list_of_files))
    tip_curvature_left_array = np.zeros(len(list_of_files))
    offset_left_array = np.zeros(len(list_of_files))
    gap_r2_left_array = np.zeros(len(list_of_files))
    amplitude_right_array = np.zeros(len(list_of_files))
    tip_curvature_right_array = np.zeros(len(list_of_files))
    offset_right_array = np.zeros(len(list_of_files))
    gap_r2_right_array = np.zeros(len(list_of_files))
    tip_curvature_mean_array = np.zeros(len(list_of_files))
    file_number_array = np.zeros(len(list_of_files), dtype='int')
    tilted_flag_array = np.zeros(len(list_of_files), dtype='bool')
    tilt_tip_degree_left_array = np.zeros(len(list_of_files))
    tilt_tip_degree_right_array = np.zeros(len(list_of_files))
    
    for i in range(len(list_of_files)):
        filename = list_of_files[i]
        file_number_array[i] = i
        if re.search("_tilted", filename):
            tilted_flag = 1
        else:
            tilted_flag = 0
        tilted_flag_array[i] = tilted_flag
        
        print(filename)
        # uncomment for debugging or check a particular nanostructure
        # if not re.search('A4', filename): continue
        # ignore the nanostructure if it's in the above ignore list
        if filename in ignore_files: 
            print('\n ... Ignoring file', filename)
            continue
        # if re.search(irrad_25mW[i], filename):
            
        # load
        filepath = os.path.join(working_folder, filename)
        binary_image_inverted_cropped_smoothed = img.open(filepath)
        # Find contours at a constant value of 0.5, of the smooth one
        ######################################################################
        # ONLY FOR SIMULATIONS, COMMENT IF NOT SIMULATIONS
        # ONLY FOR SIMULATIONS, COMMENT IF NOT SIMULATIONS
        # ONLY FOR SIMULATIONS, COMMENT IF NOT SIMULATIONS
        if simulations_flag:
            binary_image_inverted_cropped_smoothed.convert(mode='1')
        ######################################################################
        binary_image_inverted_cropped_smoothed_array = np.array(binary_image_inverted_cropped_smoothed, dtype='bool')
        contours = measure.find_contours(binary_image_inverted_cropped_smoothed_array, 0.5)
        contours_flattened = np.array([[np.nan, np.nan]])
        counter_contours = 0
        # print(filename)
        for k in range(len(contours)):
            array = contours[k]
            if array.shape[0] > 100:
                # print(array.shape)
                counter_contours += 1
                contours_flattened = np.concatenate((contours_flattened, array))
        # remove nan elements
        contours_flattened = np.delete(contours_flattened, 0, axis=0)
        # print when DNH shows a closed gap and therefore two lobes
    
        # MEASURE
        # get new center of mass
        cm = center_of_mass(binary_image_inverted_cropped_smoothed_array)
        cm_y_px = int(round(cm[0], 0))
        cm_x_px = int(round(cm[1], 0))
                       
        # create folder to save output figures
        save_folder_contour = os.path.join(initial_folder, 'figures\\step2\\contours')
        if not os.path.exists(save_folder_contour):
            os.makedirs(save_folder_contour)
        save_folder_contour_circle_top = os.path.join(save_folder_contour, 'circle_top')
        if not os.path.exists(save_folder_contour_circle_top):
            os.makedirs(save_folder_contour_circle_top)
        save_folder_contour_circle_bottom = os.path.join(save_folder_contour, 'circle_bottom')
        if not os.path.exists(save_folder_contour_circle_bottom):
            os.makedirs(save_folder_contour_circle_bottom)
        save_folder_contour_gap = os.path.join(save_folder_contour, 'gap')
        if not os.path.exists(save_folder_contour_gap):
            os.makedirs(save_folder_contour_gap)
            
        if counter_contours > 1:
            print(filename, "has been ignored. It presents two contours.")
        else:
            if not tilted_flag:
                # GET WIDTH AT THE CENTER OF MASS
                # get neck/gap profile of the structure
                gap_profile = binary_image_inverted_cropped_smoothed_array[cm_y_px,:]
                gap_width_px = np.count_nonzero(gap_profile) # in pixels
                gap_width = gap_width_px*pixel_size # in nm
                # print(gap_width)
    
                # GET LENGTH AT THE CENTER OF MASS 
                # get length profile of the structure
                length_profile = binary_image_inverted_cropped_smoothed_array[:,cm_x_px]
                # counting the "True" pixels
                dnh_length_px_counting = np.count_nonzero(length_profile) # in pixels
                # by the difference between the last and the first
                length_profile_difference = np.diff(length_profile) # in pixels
                index_length_difference = np.where(length_profile_difference == True)[0]
                dnh_length_px_difference = index_length_difference[-1] - index_length_difference[0]
                # if dnh_length_px_counting != dnh_length_px_difference:
                    # print('\nLength difference in % s' % filename)
                    # print('Counting', dnh_length_px_counting, '/ Difference', dnh_length_px_difference)
                dnh_length = dnh_length_px_difference*pixel_size # in nm
                # print(dnh_length)
                
                gap_width_array[i] = gap_width
                dnh_length_array[i] = dnh_length
                
                # if gap_width > 40:
                #     print('\nOutlier gap width', gap_width, 'nm at', filename)
                #     print('DNH length', dnh_length, 'nm')   
                
                # get contours and convert to distance in nm
                contour_x = (contours_flattened[:,1] - cm_x_px)*pixel_size
                contour_y = (contours_flattened[:,0] - cm_y_px)*pixel_size
                contour = np.vstack((contour_x, contour_y)).T
                # find center in x dimension
                center_index_y = np.where(np.abs(contour_y) < pixel_size)[0]
                index_left = center_index_y[0]
                index_right = center_index_y[-1]
                
                # get contours of the bottom circle
                # x coordinate
                contour_circle_bottom_x_left = contour_x[0:index_left - circle_fit_distance_from_center_px]
                contour_circle_bottom_x_right = contour_x[index_right + circle_fit_distance_from_center_px:]
                contour_circle_bottom_x = np.concatenate((contour_circle_bottom_x_left, contour_circle_bottom_x_right))
                # y coordinate
                contour_circle_bottom_y_left = contour_y[0:index_left - circle_fit_distance_from_center_px]
                contour_circle_bottom_y_right = contour_y[index_right + circle_fit_distance_from_center_px:]
                contour_circle_bottom_y = np.concatenate((contour_circle_bottom_y_left, contour_circle_bottom_y_right))
                contour_circle_bottom = np.vstack((contour_circle_bottom_x, contour_circle_bottom_y)).T
    
                # get contours of the top circle
                # x coordinate
                contour_circle_top_x = contour_x[index_left + circle_fit_distance_from_center_px:
                                                         index_right - circle_fit_distance_from_center_px]
                # y coordinate
                contour_circle_top_y = contour_y[index_left + circle_fit_distance_from_center_px:
                                                         index_right - circle_fit_distance_from_center_px]
                contour_circle_top = np.vstack((contour_circle_top_x, contour_circle_top_y)).T
    
                # fit the circles
                # top
                xc_top, yc_top, r_top, sigma_top = taubinSVD(contour_circle_top)
                radius_circle_top_array[i] = r_top
                center_circle_top_array[i,:] = np.array([xc_top, yc_top])
                residual_err_circle_top_array[i] = sigma_top
                # plot
                if save_image_with_measurements:
                    plot_data_circle(contour_circle_top, xc_top, yc_top, r_top, plot_fitting)            
                    figure_name = filename[:-4] + '_top_circle_fit'
                    figure_path = os.path.join(save_folder_contour_circle_top, '%s.png' % figure_name)
                    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
                    # figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
                    #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
                    plt.close()
                    
                # bottom
                xc_bottom, yc_bottom, r_bottom, sigma_bottom = taubinSVD(contour_circle_bottom)
                radius_circle_bottom_array[i] = r_bottom
                center_circle_bottom_array[i,:] = np.array([xc_bottom, yc_bottom])
                residual_err_circle_bottom_array[i] = sigma_bottom
                # plot
                if save_image_with_measurements:
                    plot_data_circle(contour_circle_bottom, xc_bottom, yc_bottom, r_bottom, plot_fitting)            
                    figure_name = filename[:-4] + '_bottom_circle_fit'
                    figure_path = os.path.join(save_folder_contour_circle_bottom, '%s.png' % figure_name)
                    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
                    # figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
                    #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
                    plt.close()
            
                ### CALCULATE GAP DIMENSIONS
                # get contours of the gap
                contour_gap_x_left = contour_x[index_left - gap_px_length_half: \
                                             index_left + gap_px_length_half]
                contour_gap_y_left = contour_y[index_left - gap_px_length_half: \
                                             index_left + gap_px_length_half]
    
                contour_gap_x_right = contour_x[index_right - gap_px_length_half: \
                                             index_right + gap_px_length_half]
                contour_gap_y_right = contour_y[index_right - gap_px_length_half: \
                                             index_right + gap_px_length_half]
                # fit gap shapes
                # left side
                # set initial parameters
                amplitude_left = 10
                offset_left_x = contour_x[index_left - gap_px_length_half]
                phase_left = 0
                initial_params = [amplitude_left, offset_left_x, phase_left]
                # get number of points for the fitting procedure
                N = len(contour_gap_x_left)
                initial_contour_x_left = DNH_tip_contour(*initial_params, N)
                # print('\nFinding best fit for the DNH tip contour of %s' % filename)
                # minimize difference between data and function
                # prepare function to store points the method pass through
                road_to_convergence = list()
                road_to_convergence.append(initial_params)
                def callback_fun_trust(X, log_ouput):
                    road_to_convergence.append(list(X))
                    return 
                # define bounds of the minimization problem (any bounded method)
                # [lower bound array], [upper bound array]
                # [amplitude, vertex, phase]
                bnds = opt.Bounds([-1000, -1000, -np.pi], [1000, 1000, np.pi])
                # now minimize
                ################# constrained and bounded methods
                out = opt.minimize(s_squared, initial_params, \
                                   args = (contour_gap_x_left, N),                            
                                    method = 'trust-constr', bounds = bnds,
                                    callback = callback_fun_trust,
                                    options = {'maxiter':5000, 'xtol':1e-16,
                                                'gtol':1e-16, 'disp':False})
                # print(out)
                # grab fitted parameters
                best_params_left = out.x
                amplitude_fitted_left = best_params_left[0]
                offset_fitted_left = best_params_left[1]
                success_flag = out.success
                if not success_flag:
                    print('Success:', success_flag)
                    print('on file:', filename)
                fitted_contour_x_left = DNH_tip_contour(*best_params_left, N)
                # figure of merit of the fit
                R2 = calc_r2(contour_gap_x_left, fitted_contour_x_left)
                gap_r2_left_array[i] = R2
                # print('R-squared %.2f' % R2)
                amplitude_left_array[i] = amplitude_fitted_left
                tip_curvature_left_array[i] = -amplitude_fitted_left*(np.pi/N)**2
                offset_left_array[i] = offset_fitted_left
                
                # right side
                # set initial parameters
                amplitude_right = -10
                offset_right_x = contour_x[index_right - gap_px_length_half]
                phase_right = 0
                initial_params = [amplitude_right, offset_right_x, phase_right]
                # get number of points for the fitting procedure
                N = len(contour_gap_x_right)
                initial_contour_x_right = DNH_tip_contour(*initial_params, N)
                # print('\nFinding best fit for the DNH tip contour of %s' % filename)
                # minimize difference between data and function
                # prepare function to store points the method pass through
                road_to_convergence = list()
                road_to_convergence.append(initial_params)
                def callback_fun_trust(X, log_ouput):
                    road_to_convergence.append(list(X))
                    return 
                # define bounds of the minimization problem (any bounded method)
                # [lower bound array], [upper bound array]
                # [amplitude, vertex, phase]
                bnds = opt.Bounds([-1000, -1000, -np.pi], [1000, 1000, np.pi])
                # now minimize
                ################# constrained and bounded methods
                out = opt.minimize(s_squared, initial_params, \
                                   args = (contour_gap_x_right, N),                            
                                    method = 'trust-constr', bounds = bnds,
                                    callback = callback_fun_trust,
                                    options = {'maxiter':5000, 'xtol':1e-16,
                                                'gtol':1e-16, 'disp':False})
                # print(out)
                # grab fitted parameters
                best_params_right = out.x
                amplitude_fitted_right = best_params_right[0]
                offset_fitted_right = best_params_right[1]
                success_flag = out.success
                if not success_flag:
                    print('Success:', success_flag)
                    print('on file:', filename)
                fitted_contour_x_right = DNH_tip_contour(*best_params_right, N)
                # figure of merit of the fit
                R2 = calc_r2(contour_gap_x_right, fitted_contour_x_right)
                gap_r2_right_array[i] = R2
                # print('R-squared %.2f' % R2)
                amplitude_right_array[i] = amplitude_fitted_right
                tip_curvature_right_array[i] = -amplitude_fitted_right*(np.pi/N)**2
                offset_right_array[i] = offset_fitted_right
                
                # calculate mean tip curvature 
                tip_curvature_mean_array[i] = (np.abs(tip_curvature_left_array[i]) + 
                                            np.abs(tip_curvature_right_array[i]))/2

                # save contours
                data_to_save = contour
                contour_filename = filename[:-4] + '_contour'
                contour_pathfile = os.path.join(save_folder_contour, '%s.txt' % contour_filename)
                np.savetxt(contour_pathfile, data_to_save, fmt='%.6e')
                
                # data_to_save = contour_circle_top
                # contour_filename = filename[:-4] + '_contour_top_circle'
                # contour_pathfile = os.path.join(save_folder_contour_circle_top, '%s.txt' % contour_filename)
                # np.savetxt(contour_pathfile, data_to_save, fmt='%.6e')
                
                # data_to_save = contour_circle_bottom
                # contour_filename = filename[:-4] + '_contour_top_circle'
                # contour_pathfile = os.path.join(save_folder_contour_circle_bottom, '%s.txt' % contour_filename)
                # np.savetxt(contour_pathfile, data_to_save, fmt='%.6e')
                
                # plot contours
                # in xy
                # plt.figure()
                # plt.plot(contour_x, contour_y, color='C0', markersize=2, label='measured')
                # plt.scatter(contour_x[0], contour_y[0], color='k', marker='s', label='initial point')
                # plt.plot(contour_circle_top_x, contour_circle_top_y, color='C3', marker='x', label='top circle')
                # plt.plot(contour_circle_bottom_x, contour_circle_bottom_y, color='C4', marker='x', label='bottom circle')
                # plt.gca().set_aspect('equal')
                # plt.legend()
                # plt.plot(initial_contour_xy[:,0], initial_contour_xy[:,1], color='C1', linestyle='dotted')
                # plt.plot(fitted_contour_xy[:,0], fitted_contour_xy[:,1], color='C2', linestyle='--')
                
                if save_image_with_measurements:
                    # display closest points in the gap
                    closest_x_left = offset_fitted_left + amplitude_fitted_left
                    closest_x_right = offset_fitted_right + amplitude_fitted_right
                    # plot contours
                    plt.figure()
                    plt.plot(contour_gap_x_left, contour_gap_y_left, color='C0', label='x left')
                    plt.plot(initial_contour_x_left, contour_gap_y_left, color='C2', linestyle='dotted', label='x initial')
                    plt.plot(fitted_contour_x_left, contour_gap_y_left, color='C3', linestyle='--', label='x fit')        
                    plt.plot(contour_gap_x_right, contour_gap_y_right, color='C0', label='x right')
                    plt.plot(initial_contour_x_right, contour_gap_y_right, color='C2', linestyle='dotted', label='x initial')
                    plt.plot(fitted_contour_x_right, contour_gap_y_right, color='C3', linestyle='--', label='x fit')
                    plt.axvline(closest_x_left, color='k', linestyle='--')
                    plt.axvline(closest_x_right, color='k', linestyle='--')
                    plt.legend()
                    plt.xlabel('x (nm)')
                    plt.ylabel('y (nm)')
                    # plt.show()
                    figure_name = filename[:-4] + '_gap_fit'
                    figure_path = os.path.join(save_folder_contour_gap, '%s.png' % figure_name)
                    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
                    #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
                    #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
                    plt.close()
                    
            # get contours of the gap when TILTED 
            if tilted_images_flag:
                if re.search("_tilted_", filename):
                    search_lower_limit = cm_y_px - 50
                    search_upper_limit = cm_y_px + 100
                    horizontal_profiles = binary_image_inverted_cropped_smoothed_array[search_lower_limit:search_upper_limit,:]    
                    profiles_length = np.sum(horizontal_profiles, axis = 1)
                    index_of_shortest = np.argmin(profiles_length) + search_lower_limit + tilt_gap_offset
                    # pixel values fopr the image
                    upper_limit_y_tilt_analysis = index_of_shortest
                    lower_limit_y_tilt_analysis = index_of_shortest - tilt_analysis_distance_px
                    # index for the contour
                    upper_limit_y_index_left = np.where(contours_flattened[:,0] == upper_limit_y_tilt_analysis)[0][0]
                    # print(contours_flattened[:,0])
                    lower_limit_y_index_left = np.where(contours_flattened[:,0] == lower_limit_y_tilt_analysis)[0][0]
                    upper_limit_y_index_right = np.where(contours_flattened[:,0] == upper_limit_y_tilt_analysis)[0][-1]
                    lower_limit_y_index_right = np.where(contours_flattened[:,0] == lower_limit_y_tilt_analysis)[0][-1]         
                    # print(upper_limit_y_index_left, lower_limit_y_index_left, \
                    #       upper_limit_y_index_right, lower_limit_y_index_right)
                    # make contours
                    contour_tilted_gap_x_left = contours_flattened[upper_limit_y_index_left: \
                                                                lower_limit_y_index_left, 1]
                    contour_tilted_gap_y_left = contours_flattened[upper_limit_y_index_left: \
                                                                lower_limit_y_index_left, 0]    
                        
                    contour_tilted_gap_x_right = contours_flattened[lower_limit_y_index_right: \
                                                                    upper_limit_y_index_right, 1]
                    contour_tilted_gap_y_right = contours_flattened[lower_limit_y_index_right: \
                                                                    upper_limit_y_index_right, 0]
                        
                    # transform to distances
                    angle_factor = np.sin(angle*(np.pi/180))
                    contour_tilted_gap_y_left_nm = (np.max(contour_tilted_gap_y_left) - contour_tilted_gap_y_left)*pixel_size*angle_factor
                    contour_tilted_gap_y_right_nm = (np.max(contour_tilted_gap_y_right) - contour_tilted_gap_y_right)*pixel_size*angle_factor
                    contour_tilted_gap_x_left_nm = contour_tilted_gap_x_left*pixel_size
                    contour_tilted_gap_x_right_nm = contour_tilted_gap_x_right*pixel_size
                    # scale x for fitting
                    contour_tilted_gap_x_left_scaled = 1 + (-contour_tilted_gap_x_left_nm + np.min(contour_tilted_gap_x_left_nm))/ \
                                                    (np.max(contour_tilted_gap_x_left_nm) - np.min(contour_tilted_gap_x_left_nm))
                    contour_tilted_gap_x_right_scaled = (contour_tilted_gap_x_right_nm - np.min(contour_tilted_gap_x_right_nm))/ \
                                                    (np.max(contour_tilted_gap_x_right_nm) - np.min(contour_tilted_gap_x_right_nm))
                    # fit with powerlaw
                    p_left, cov_left = opt.curve_fit(func_powerlaw, \
                                                contour_tilted_gap_x_left_scaled, \
                                                contour_tilted_gap_y_left_nm, \
                                                p0 = np.asarray([0.5, 35]), \
                                                maxfev = 2000)   
                    y_fit_left = func_powerlaw(contour_tilted_gap_x_left_scaled, *p_left)
                    p_right, cov_right = opt.curve_fit(func_powerlaw, \
                                                contour_tilted_gap_x_right_scaled, \
                                                contour_tilted_gap_y_right_nm, \
                                                p0 = np.asarray([0.5, 35]), \
                                                maxfev = 2000)
                    y_fit_right = func_powerlaw(contour_tilted_gap_x_right_scaled, *p_right)
                    
                    tip_degree_left = p_left[0]
                    tip_degree_right = p_right[0]
                    
                    tilt_tip_degree_left_array[i] = tip_degree_left
                    tilt_tip_degree_right_array[i] = tip_degree_right
                    
                    # plot and save 
                    save_folder_contour_tilted_gap = os.path.join(save_folder_contour, 'tilted_gap')
                    if not os.path.exists(save_folder_contour_tilted_gap):
                        os.makedirs(save_folder_contour_tilted_gap)
                        
                    if save_image_with_measurements:                   
                        # plot contours scaled
                        plt.figure()
                        plt.scatter(contour_tilted_gap_x_left_scaled, contour_tilted_gap_y_left_nm, marker='.', color='C0', label='left')
                        plt.plot(contour_tilted_gap_x_left_scaled, y_fit_left, color='C3', linestyle='--', label='left fit')        
                        plt.scatter(contour_tilted_gap_x_right_scaled, contour_tilted_gap_y_right_nm, marker='.', color='C1', label='right')
                        plt.plot(contour_tilted_gap_x_right_scaled, y_fit_right, color='C4', linestyle='--', label='right fit')        
                        plt.legend()
                        plt.xlabel('x (nm)')
                        plt.ylabel('y (nm)')
                        # plt.xscale('log')
                        # plt.yscale('log')
                        # plt.show()
                        figure_name = filename[:-4] + '_tilted_gap_fit_scaled'
                        figure_path = os.path.join(save_folder_contour_tilted_gap, '%s.png' % figure_name)
                        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
                        #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
                        #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
                        plt.close()
                        
                        # plot contours
                        plt.figure()
                        plt.scatter(contour_tilted_gap_x_left_nm, contour_tilted_gap_y_left_nm, marker='.', color='C0', label='left')
                        plt.plot(contour_tilted_gap_x_left_nm, y_fit_left, color='C3', linestyle='--', label='left fit')        
                        plt.scatter(contour_tilted_gap_x_right_nm, contour_tilted_gap_y_right_nm, marker='.', color='C1', label='right')
                        plt.plot(contour_tilted_gap_x_right_nm, y_fit_right, color='C4', linestyle='--', label='right fit')        
                        plt.legend()
                        plt.xlabel('x (nm)')
                        plt.ylabel('y (nm)')
                        # plt.xscale('log')
                        # plt.yscale('log')
                        # plt.show()
                        figure_name = filename[:-4] + '_tilted_gap_fit'
                        figure_path = os.path.join(save_folder_contour_tilted_gap, '%s.png' % figure_name)
                        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
                        #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
                        #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
                        plt.close()
                
        ########################################################
        
        # SAVING IMAGES
        if save_image_with_measurements:
            # create folder to save output figures
            save_folder_demo = os.path.join(initial_folder, 'figures\\step2\\thresholded_and_measurements')
            if not os.path.exists(save_folder_demo):
                os.makedirs(save_folder_demo)
                
            plt.figure()        
            plt.imshow(binary_image_inverted_cropped_smoothed_array, cmap = plt.cm.gray, origin='upper')
            plt.plot(contours_flattened[:, 1], contours_flattened[:, 0], linewidth=0.2, color='C1')
            plt.scatter(contours_flattened[0, 1], contours_flattened[0, 0], marker = 's', color='r')
            plt.axvline(cm_x_px, color='C0')
            plt.axhline(cm_y_px, color='C0')
            if tilted_images_flag:
                if re.search("_tilted_", filename):
                    plt.axhline(lower_limit_y_tilt_analysis, color='C0', linestyle = ':')
                    plt.axhline(upper_limit_y_tilt_analysis, color='C0', linestyle = ':')
                    plt.plot(contour_tilted_gap_x_left, contour_tilted_gap_y_left, linewidth=1, color='C4')
                    plt.plot(contour_tilted_gap_x_right, contour_tilted_gap_y_right, linewidth=1, color='C5')
            plt.plot(cm_x_px, cm_y_px, 'C3x')
            figure_name = filename[:-4] + '_with_lines'
            figure_path = os.path.join(save_folder_demo, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
            #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
            plt.close()

##############################################################################
    
    # calculate new observables

    center_circle_top_array_x = center_circle_top_array[:, 0]
    center_circle_top_array_y = center_circle_top_array[:, 1]
    center_circle_bottom_array_x = center_circle_bottom_array[:, 0]
    center_circle_bottom_array_y = center_circle_bottom_array[:, 1]

    interhole_distance_array = np.sqrt(
                        (center_circle_top_array_x - center_circle_bottom_array_x)**2 + \
                        (center_circle_top_array_y - center_circle_bottom_array_y)**2
                        )
    gap_width_from_fit = (offset_right_array + amplitude_right_array) - \
                         (offset_left_array + amplitude_left_array)
    
    # calculate tilt tip degree mean
    tilt_tip_degree_left_mean = (tilt_tip_degree_left_array + tilt_tip_degree_right_array)/2

    # create folder to save parameters
    save_folder_stats = os.path.join(initial_folder, 'figures\\step2\\stats')
    if not os.path.exists(save_folder_stats):
        os.makedirs(save_folder_stats)
        
    # save all parameters in one file
    data_to_save = {
        'id'                    :   file_number_array, \
        'label'                 :   list_of_files, \
        'tilted_flag'           :   tilted_flag_array, \
        'gap_width'             :   gap_width_from_fit, \
        'interhole_distance'    :   interhole_distance_array, \
        'radius_top'            :   radius_circle_top_array, \
        'center_top_x'          :   center_circle_top_array_x, \
        'center_top_y'          :   center_circle_top_array_y, \
        'residual_top'          :   residual_err_circle_top_array, \
        'radius_bottom'         :   radius_circle_bottom_array, \
        'center_bottom_x'       :   center_circle_bottom_array_x, \
        'center_bottom_y'       :   center_circle_bottom_array_y, \
        'residual_bottom'       :   residual_err_circle_bottom_array, \
        'amplitude_right'       :   amplitude_right_array, \
        'tip_curvature_right'   :   tip_curvature_right_array, 
        'offset_right'          :   offset_right_array, \
        'goodess_fit_right'     :   gap_r2_right_array, \
        'amplitude_left'        :   amplitude_left_array, \
        'tip_curvature_left'    :   tip_curvature_left_array, \
        'offset_left'           :   offset_left_array, \
        'goodess_fit_left'      :   gap_r2_left_array, \
        'tip_curvature_mean'    :   tip_curvature_mean_array, \
        'tilt_tip_degree_left'  :   tilt_tip_degree_left_array, \
        'tilt_tip_degree_right' :   tilt_tip_degree_right_array, \
        'tilt_tip_degree_mean'  :   tilt_tip_degree_left_mean \
        }

    df = pd.DataFrame(data_to_save)
    new_filename = 'SEM_measurements_%s.csv' % (folder_name)
    new_filepath = os.path.join(save_folder_stats, new_filename)
    df.to_csv(new_filepath, index = False, float_format = '%.6f')

    print('\nSTEP 2 finished.')

    return