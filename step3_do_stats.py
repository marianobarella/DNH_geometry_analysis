# # -*- coding: utf-8 -*-
# """
# Created on Thu Aug  1 23:02:29 2024

# @author: BarellaM
# """

import matplotlib.pyplot as plt
import os
import numpy as np
import re
import pandas as pd

##############################################################################

def do_stats(initial_folder, folder_name, sample_text, tilted_images_flag):   
    
    plt.ioff()
    plt.close('all')
    
    # Check if step 2 was executed on this dataset
    working_folder = os.path.join(initial_folder, 'figures\\stats')
    if not os.path.exists(working_folder):
        print(working_folder, 'DOES NOT EXIST')
        print('Run step 2 or double check input folder')
        return

    list_of_files = os.listdir(working_folder)
    parameters_file = [f for f in list_of_files if f.endswith("%s.csv" % folder_name)]
    
    filepath = os.path.join(working_folder, parameters_file)
    data = pd.read_csv(filepath)
    
    id = data['id'].values
    label = data['label'].values
    tilted_flag_array = data['tilted_flag'].values
    gap_width_array = data['gap_width'].values
    dnh_length_array = data['dnh_length'].values
    radius_circle_top_array = data['radius_top'].values
    center_circle_top_array_x = data['center_top_x'].values
    center_circle_top_array_y = data['center_top_y'].values
    residual_err_circle_top_array = data['residual_top'].values
    radius_circle_bottom_array = data['radius_bottom'].values
    center_circle_bottom_array_x = data['center_bottom_x'].values
    center_circle_bottom_array_y = data['center_bottom_y'].values
    residual_err_circle_bottom_array = data['residual_bottom'].values
    amplitude_right_array = data['amplitude_right'].values
    tip_curvature_right_array = data['tip_curvature_right'].values
    offset_right_array = data['offset_right'].values
    gap_r2_right_array = data['goodess_fit_right'].values
    amplitude_left_array = data['amplitude_left'].values
    tip_curvature_left_array = data['tip_curvature_left'].values
    offset_left_array = data['offset_left'].values
    gap_r2_left_array = data['goodess_fit_left'].values
    tilt_tip_degree_left_array = data['tilt_tip_degree_left'].values
    tilt_tip_degree_right_array = data['tilt_tip_degree_right'].values
    
    # calculate new observables
    interhole_distance = np.sqrt(
                        (center_circle_top_array_x - center_circle_bottom_array_x)**2 + \
                        (center_circle_top_array_y - center_circle_bottom_array_y)**2
                        )
    gap_width_from_fit = (offset_right_array + amplitude_right_array) - \
                         (offset_left_array + amplitude_left_array)
            
    # remove tilted images data
    gap_width_array = gap_width_array[tilted_flag_array == 0]
    dnh_length_array = dnh_length_array[tilted_flag_array == 0]
    radius_circle_top_array = radius_circle_top_array[tilted_flag_array == 0]
    radius_circle_bottom_array = radius_circle_bottom_array[tilted_flag_array == 0]
    tip_curvature_right_array = tip_curvature_right_array[tilted_flag_array == 0]
    tip_curvature_left_array = tip_curvature_left_array[tilted_flag_array == 0]
    interhole_distance = interhole_distance[tilted_flag_array == 0]
    gap_width_from_fit = gap_width_from_fit[tilted_flag_array == 0]
                         
    # keep tilted data
    tilt_tip_degree_left_array = tilt_tip_degree_left_array[tilted_flag_array == 1]
    tilt_tip_degree_right_array = tilt_tip_degree_right_array[tilted_flag_array == 1]
    
    # remove zeroes
    gap_width_array = gap_width_array[gap_width_array != 0]
    dnh_length_array = dnh_length_array[dnh_length_array != 0]
    radius_circle_top_array = radius_circle_top_array[radius_circle_top_array != 0]
    radius_circle_bottom_array = radius_circle_bottom_array[radius_circle_bottom_array != 0]
    tip_curvature_right_array = tip_curvature_right_array[tip_curvature_right_array != 0]
    tip_curvature_left_array = tip_curvature_left_array[tip_curvature_left_array != 0]
    interhole_distance = interhole_distance[interhole_distance != 0]
    gap_width_from_fit = gap_width_from_fit[gap_width_from_fit != 0]
    tilt_tip_degree_left_array = tilt_tip_degree_left_array[tilt_tip_degree_left_array != 0]
    tilt_tip_degree_right_array = tilt_tip_degree_right_array[tilt_tip_degree_right_array != 0]
    
    # group data
    tip_curvatures = np.abs(np.concatenate((tip_curvature_right_array, 
                                            tip_curvature_left_array)))
    radius_circles = np.concatenate((radius_circle_top_array, radius_circle_bottom_array))
    calculated_dnh_length_array = interhole_distance + \
                                  radius_circle_top_array + \
                                  radius_circle_bottom_array
    tilt_tip_degree = np.concatenate((tilt_tip_degree_left_array, \
                                      tilt_tip_degree_right_array))
    
    ##########################################################################
    # MAKE PLOTS
    ##########################################################################
    # plot gap as calculated from profiles
    fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 2, 1)
    ax[1] = plt.subplot(1, 2, 2)
    
    ax[0].violinplot(gap_width_array, 
                      showmeans=False, 
                      showmedians=True,
                      showextrema=True)
    ax[0].set_ylabel('Gap width from profile (nm)')
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_ylim([0,20])

    ax[1].violinplot(gap_width_from_fit, 
                      showmeans=False, 
                      showmedians=True,
                      showextrema=True)
    ax[1].set_ylabel('Gap width from fit (nm)')
    ax[1].get_xaxis().set_visible(False)
    ax[1].set_ylim([0,20])
    
    figure_name = '%s_violin_plot_gap_width' % sample_text
    figure_path = os.path.join(working_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
    #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
    plt.close()
    
    ##########################################################################
    
    fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 2, 1)
    ax[1] = plt.subplot(1, 2, 2)
    
    ax[0].hist(gap_width_array, bins=10, range=[0, 40], 
                edgecolor = 'k', density=False)
    title1 = 'median %d nm, std dev %d nm' % \
                (np.median(gap_width_array), np.std(gap_width_array, ddof=1))
    ax[0].set_title(title1)
    ax[0].axvline(np.median(gap_width_array), color='C3', linestyle = '--')
    ax[0].set_xlabel('Gap width from profile (nm)')
    
    ax[1].hist(gap_width_from_fit, bins=10, range=[0, 40], 
                edgecolor = 'k', density=False)
    title1 = 'median %d nm, std dev %d nm' % \
                (np.median(gap_width_from_fit), np.std(gap_width_from_fit, ddof=1))
    ax[1].set_title(title1)
    ax[1].axvline(np.median(gap_width_from_fit), color='C3', linestyle = '--')
    ax[1].set_xlabel('Gap width from fit (nm)')
    
    figure_name = '%s_histogram_gap_width' % sample_text
    figure_path = os.path.join(working_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
    #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
    plt.close()
    
    ##########################################################################
    # # plot gap as obtained from fits
    # ig, axes = plt.subplots(ncols=2, figsize=(9, 2.5))
    # ax = axes.ravel()
    # ax[0] = plt.subplot(1, 2, 1)
    # ax[1] = plt.subplot(1, 2, 2)
    
    # ax[0].violinplot(dnh_length_array, 
    #                   showmeans=False, 
    #                   showmedians=True,
    #                   showextrema=True)
    # ax[0].set_ylabel('DNH length from profile (nm)')
    # ax[0].get_xaxis().set_visible(False)
    # ax[0].set_ylim([370,390])
    
    # ax[1].violinplot(calculated_dnh_length_array, 
    #                   showmeans=False, 
    #                   showmedians=True,
    #                   showextrema=True)
    # ax[1].set_ylabel('DNH length from fit (nm)')
    # ax[1].get_xaxis().set_visible(False)
    # ax[1].set_ylim([370,390])
    
    # figure_name = '%s_violin_plot_DNH_length' % sample_text
    # figure_path = os.path.join(working_folder, '%s.png' % figure_name)
    # plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    # #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
    # #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
    # plt.close()
    
    ##########################################################################
    
    # fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
    # ax = axes.ravel()
    # ax[0] = plt.subplot(1, 2, 1)
    # ax[1] = plt.subplot(1, 2, 2)

    # ax[0].hist(dnh_length_array, bins=20, range=[350, 450], 
    #             edgecolor = 'k', density=False)
    # title2 = 'median %d nm, std dev %d nm' % \
    #             (np.median(dnh_length_array), np.std(dnh_length_array, ddof=1))
    # ax[0].set_title(title2)
    # ax[0].axvline(np.median(dnh_length_array), color='C3', linestyle = '--')
    # ax[0].set_xlabel('DNH length from profile (nm)')
    
    # ax[1].hist(calculated_dnh_length_array, bins=20, range=[350, 450], 
    #             edgecolor = 'k', density=False)
    # title2 = 'median %d nm, std dev %d nm' % \
    #             (np.median(calculated_dnh_length_array), np.std(calculated_dnh_length_array, ddof=1))
    # ax[1].set_title(title2)
    # ax[1].axvline(np.median(calculated_dnh_length_array), color='C3', linestyle = '--')
    # ax[1].set_xlabel('DNH length from fit (nm)')
    
    # figure_name = '%s_histogram_DNH_length' % sample_text
    # figure_path = os.path.join(working_folder, '%s.png' % figure_name)
    # plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    # #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
    # #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
    # plt.close()
    
    ##########################################################################
    # plot curvature and circle radius as obtained from fits
    fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 2, 1)
    ax[1] = plt.subplot(1, 2, 2)
    
    ax[0].violinplot(tip_curvatures, 
                      showmeans=False, 
                      showmedians=True,
                      showextrema=True)
    ax[0].set_ylabel('Tip curvature (a.u.)')
    ax[0].get_xaxis().set_visible(False)
    
    ax[1].hist(tip_curvatures, bins=10, range=[0, 0.05], 
                edgecolor = 'k', density=False)
    title1 = 'median %.3f nm, std dev %.3f nm' % \
                (np.median(tip_curvatures), np.std(tip_curvatures, ddof=1))
    ax[1].set_title(title1)
    ax[1].axvline(np.median(tip_curvatures), color='C3', linestyle = '--')
    ax[1].set_xlabel('Tip curvatures (a.u.)')
    
    figure_name = '%s_tip_curvature' % sample_text
    figure_path = os.path.join(working_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
    #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
    plt.close()
    
    ##########################################################################
    
    fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 2, 1)
    ax[1] = plt.subplot(1, 2, 2)
    
    ax[0].violinplot(radius_circles, 
                      showmeans=False, 
                      showmedians=True,
                      showextrema=True)
    ax[0].set_ylabel('Holes radius (nm)')
    ax[0].get_xaxis().set_visible(False)
       
    ax[1].hist(radius_circles, bins=20, range=[40, 120], 
                edgecolor = 'k', density=False)
    title2 = 'median %.1f nm, std dev %.1f nm' % \
                (np.median(radius_circles), np.std(radius_circles, ddof=1))
    ax[1].set_title(title2)
    ax[1].axvline(np.median(radius_circles), color='C3', linestyle = '--')
    ax[1].set_xlabel('Holes radius (nm)')
    
    figure_name = '%s_holes_radius' % sample_text
    figure_path = os.path.join(working_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
    #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
    plt.close()

    ##########################################################################
    # plot interhole distance
    fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 2, 1)
    ax[1] = plt.subplot(1, 2, 2)
    
    ax[0].violinplot(interhole_distance, 
                      showmeans=False, 
                      showmedians=True,
                      showextrema=True)
    ax[0].set_ylabel('Interhole distance (nm)')
    ax[0].get_xaxis().set_visible(False)

    ax[1].hist(interhole_distance, bins=25, range=[175, 250], 
                edgecolor = 'k', density=False)
    title2 = 'median %d nm, std dev %d nm' % \
                (np.median(interhole_distance), np.std(interhole_distance, ddof=1))
    ax[1].set_title(title2)
    ax[1].axvline(np.median(interhole_distance), color='C3', linestyle = '--')
    ax[1].set_xlabel('Interhole distance (nm)')
    
    figure_name = '%s_interhole_distance' % sample_text
    figure_path = os.path.join(working_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
    #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
    plt.close()
    
    
    
    ##########################################################################
    if tilted_images_flag:
        # plot tip degree
        fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 2, 1)
        ax[1] = plt.subplot(1, 2, 2)
    
        ax[0].violinplot(tilt_tip_degree, 
                          showmeans=False, 
                          showmedians=True,
                          showextrema=True)
        ax[0].set_ylabel('Tip degree (a.u.)')
        ax[0].get_xaxis().set_visible(False)
    
        ax[1].hist(tilt_tip_degree, bins=20, range=[0, 2], 
                    edgecolor = 'k', density=False)
        title2 = 'median %.2f nm, std dev %.2f nm' % \
                    (np.median(tilt_tip_degree), np.std(tilt_tip_degree, ddof=1))
        ax[1].set_title(title2)
        ax[1].axvline(np.median(tilt_tip_degree), color='C3', linestyle = '--')
        ax[1].set_xlabel('Tip degree (a.u.)')
        
        figure_name = '%s_tip_degree' % sample_text
        figure_path = os.path.join(working_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
        #figure_path = os.path.join(save_folder, '%s.pdf' % figure_name)
        #plt.savefig(figure_path, dpi = 300, bbox_inches='tight', format = 'pdf')
        plt.close()
    
    ##########################################################################
    ##########################################################################
    ##########################################################################
    
    print('\nSTEP 3 finished.')

    return


