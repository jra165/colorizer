#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:49:09 2021

@author: Joshua Atienza, Janet Zhang
@net-id: jra165, jrz42
@project: colorizer
"""

import imageio
from typing import List
import numpy as np
import os
import random
import sys
import math

from PIL import Image
from typing import Tuple, Dict

import matplotlib.pyplot as plt


def k_cluster(rgb):
    # Turn into 2D numpy array, multiply length by width and keep number of items in 3
    rgb2 = rgb.reshape(250000, 3)

    #Obtain initial centroids of image
    #Sample 5 random points from number of image pixels
    img_pixels = random.sample(range(0, 250000), 5)
    #Initialize centroids list
    initial_centroids = []
    # Retrieve pixels using random numbers as indices
    for i in img_pixels:
        initial_centroids.append(rgb2[i])
    centroids = np.array(initial_centroids)

    # Recalculate centroid 10 times until reaches stable state
    for i in range(10):
        new_centroids = np.zeros(shape=(len(centroids), 3))
        element_count = np.zeros(shape=len(centroids))

        for pixel in rgb2:
            min_dist = 1000000
            curr_centroid = 0
            # Determine centroid that pixel is closest to
            for i in range(len(centroids)):
                #Calculate the Euclidean distance between pixel and centroid
                distance = 0
                for j in range(3):
                    distance += (int(pixel[j]) - int(centroids[i][j])) ** 2
                curr_dist = math.sqrt(distance)
                if curr_dist < min_dist:
                    curr_centroid = i
                    min_dist = curr_dist
            # Add the pixel values to the corresponding centroid index
            new_centroids[curr_centroid] += pixel
            element_count[curr_centroid] += 1
    
        # Divide the sum of the pixel values for each pixel for each centroid by the number of pixels for that centroid
        for i in range(len(centroids)):
            new_centroids[i] /= element_count[i]
        centroids = new_centroids.astype(int)
        

    # Assign pixels to a color
    pixel_colors = np.zeros(shape=(250000, 1))
    index = 0
    for pixel in rgb2:
        min_dist = (255 * math.sqrt(3)) + 1
        curr_centroid = 0
        # Determine centroid that pixel is closest to
        for i in range(len(centroids)):
            #Calculate the Euclidean distance between pixel and centroid
            distance = 0
            for j in range(3):
                distance += (int(pixel[j]) - int(centroids[i][j])) ** 2
            curr_dist = math.sqrt(distance)
            if curr_dist < min_dist:
                curr_centroid = i
                min_dist = curr_dist
        # Assign the pixel to a color
        pixel_colors[index] = curr_centroid
        index += 1
    color_arr = np.concatenate((rgb2, pixel_colors), 1)
    unique, counts = np.unique(color_arr[:, 3], return_counts=True)
    # Group all pixels by color
    color_dict = {}
    for i in range(5):
        color_dict['color_' + str(i)] = color_arr[np.where(color_arr[:, 3] == i)]

    # Plot centroids and pixels according to color they correspond to
    ax = plt.axes(projection='3d')
    
    #Slicing list according to r, g, b values respectively
    for i in range(5):
        ax.scatter3D(color_dict['color_' + str(i)][:, 0], color_dict['color_' + str(i)][:, 1],
                     color_dict['color_' + str(i)][:, 2], alpha=0.1,
                     color=centroids[i] / np.array([[255.0, 255.0, 255.0]]))
    ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='black')
    plt.show()

    return centroids.astype('uint8'), color_arr

def color_R_side(grey, rgb_recolor, five_colors, color_arr):

    ###BEGINNING OF GETTING R HAND grey SCORES###
    L_greyness_dict = {}

    # Iterate through L half of grey, skipping edges, use 499 as # pixels - 1, use 249 as # pixels /2 - 1
    for i in range(1, 499):
        for j in range(1, 249):
              
            #Calculates the grey score of a given 3x3 patch, the sum of the grey values of each pixel in the grid
            score = 0
        
            for x in range(-1, 2):
                for y in range(-1, 2):
                    score += int(grey[i + x][j + y])
        
            L_greyness = int(score / 9)   

            if L_greyness in L_greyness_dict:
                L_greyness_dict.get(L_greyness).append((i, j))
                L_greyness_dict.update({L_greyness: L_greyness_dict.get(L_greyness)})
            else:
                L_greyness_dict[L_greyness] = [(i, j)]
    
    print("This is the L grey dictionary")
    print(L_greyness_dict.keys())
    
    ###END OF GETTING L HAND grey SCORES###


    # Fill in R half of rgb_recolor with new colors, ignoring edges
    for i in range(1, 499):
        for j in range(250, 499):
            
            # Retrieve most 6 most similar patch centers and their similarity scores
            score = 0
        
            for x in range(-1, 2):
                for y in range(-1, 2):
                    score += int(grey[i + x][j + y])
        
            #Calculate R grey score
            R_greyness = int(score / 9)    
            
            counter_incr = R_greyness
            counter_decr = counter_incr
        
            # Retrieve 6 similar patches from dictionary, the sum of the grey values of each pixel in the grid
            six_patches_coords = L_greyness_dict.get(R_greyness)
            if six_patches_coords is None:
                six_patches_coords = []
            six_patches = [(0, x) for x in six_patches_coords]
            # print("similar patches", len(six_patches), (R_i, R_j))
        
            # Add more coordinates until you hit 6
            while len(six_patches) < 6:
        
                counter_incr += 1
                counter_decr -= 1
        
                # Go one key up
                six_patches_up_coords = L_greyness_dict.get(counter_incr)
                if six_patches_up_coords is None:
                    six_patches_up_coords = []
                six_patches_up = [(counter_incr - R_greyness, x) for x in six_patches_up_coords]
                six_patches.extend(six_patches_up)
        
                # Go one key down
                six_patches_down_coords = L_greyness_dict.get(counter_decr)
                if six_patches_down_coords is None:
                    six_patches_down_coords = []
                six_patches_down = [(R_greyness - counter_decr, x) for x in
                                             six_patches_down_coords]
                six_patches.extend(six_patches_down)
        
            six_patches = six_patches[:6]


            # Find representative color for each patch and add patch to
            color_clusters_array = [[] for _ in range(5)]
            
            color_cluster = 0
            for patch in six_patches:
                color_i = int(color_arr[(patch[1][0] * 500 + patch[1][1])][3])

                color_clusters_array[color_i].append(patch)

                if len(color_clusters_array[color_i]) > color_cluster:
                    color_cluster = len(color_clusters_array[color_i])

            # Retrieve all color indices with that max frequency
            find_freq_colors = []
            for x in range(len(color_clusters_array)):
                if len(color_clusters_array[x]) == color_cluster:
                    find_freq_colors.append(x)

            # If there is a most represented color, make the corresponding pixel that color
            if len(find_freq_colors) == 1:
                rgb_recolor[i][j] = five_colors[find_freq_colors[0]]

            # Otherwise, break ties based on similarity score
            else:
                patch_list = []

                # Put all patches that map to the most represented colors into patch_list
                for x in range(len(color_clusters_array)):
                    if len(color_clusters_array[x]) == color_cluster:
                        for patch in color_clusters_array[x]:
                            patch_list.append(patch)

                # Select the color that is mapped to the most similar patch
                max_similarity = 100000
                closest_patch = None
                for patch in patch_list:
                    if patch[0] < max_similarity:
                        max_similarity = patch[0]
                        closest_patch = patch

                # Make the original pixel the same color as that of the most similar patch
                rgb_recolor[i][j] = rgb_recolor[closest_patch[1][0]][closest_patch[1][1]]

    return rgb_recolor

def run_basic_agent(rgb_recolor, grey, five_colors, color_arr):
    # Color R hand side
    newest_rgb = color_R_side(grey, rgb_recolor, five_colors, color_arr)
    plt.imshow(newest_rgb.astype('uint8'))
    plt.show()
    print("Done with basic")
    return newest_rgb

def run_improved_agent(rgb, grey, five_colors, color_arr):
    eq_red, eq_green, eq_blue = generate_regression_equations(rgb, grey)

    num_rows = rgb.shape[0]
    num_cols = rgb.shape[1]

    # Fill in R half of rgb_recolor with new colors, ignoring edges
    for i in range(1, num_rows - 1):
        for j in range(int(num_cols / 2), num_cols - 1):
            grey_value = grey[i][j]

            red = int(eq_red[0] * grey_value + eq_red[1])
            green = int(eq_green[0] * grey_value + eq_green[1])
            blue = int(eq_blue[0] * grey_value + eq_blue[1])

            # Map to the closest representative color
            min_dist = 100000000
            closest_color = None
            dist = []

            for color in five_colors:
                curr_dist = 0.21*abs(red-color[0]) + 0.72*abs(green-color[1]) + 0.07*abs(blue-color[2])
                dist.append(curr_dist)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    closest_color = color

            rgb[i][j] = closest_color

    plt.imshow(rgb.astype('uint8'))
    plt.show()
    return rgb


def generate_regression_equations(rgb: np.array, grey: np.array):
    wr = 1
    wg = 1
    wb = 1
    br = 0
    bg = 0
    bb = 0
    alpha = 0.00003


    # Iterate through L half of grey
    for i in range(1, grey.shape[0] - 1):
        for j in range(1, int(grey.shape[1] / 2) - 1):
            probability = random.uniform(0, 1)
            if probability > 0.8:
                continue

            # color_pixel = rgb[i][j]
            grey_value = grey[i][j]

            # Calculate y hat
            y_hat_r = wr * grey_value + br
            y_hat_g = wg * grey_value + bg
            y_hat_b = wb * grey_value + bb

            # color_pixel = five_colors[int(color_arr[i * j][3])]
            color_pixel = rgb[i][j]

            # Calculate loss
            loss_r = (y_hat_r - color_pixel[0]) ** 2
            loss_g = (y_hat_g - color_pixel[1]) ** 2
            loss_b = (y_hat_b - color_pixel[2]) ** 2

            # Calculate new weights
            wr = wr - alpha * (y_hat_r - color_pixel[0]) * grey_value
            wg = wg - alpha * (y_hat_g - color_pixel[1]) * grey_value
            wb = wb - alpha * (y_hat_b - color_pixel[2]) * grey_value

            # Calculate new b values
            br = br - alpha * (y_hat_r - color_pixel[0])
            bg = bg - alpha * (y_hat_g - color_pixel[1])
            bb = bb - alpha * (y_hat_b - color_pixel[2])

            eq_red = 'y = ' + str(wr) + 'x + ' + str(br)
            eq_green = 'y = ' + str(wg) + 'x + ' + str(bg)
            eq_blue = 'y = ' + str(wb) + 'x + ' + str(bb)


    return (wr, br), (wg, bg), (wb, bb)


def calculate_accuracy(base: np.array, recolored: np.array) -> float:
    rows = base.shape[0]
    cols = base.shape[1]
    counted_cells = 0
    correctly_colored = 0

    for i in range (1,rows-1):
        for j in range(int(cols/2), cols-1):
            if np.array_equal(base[i][j], recolored[i][j]):
                correctly_colored += 1
            counted_cells += 1

    accuracy = correctly_colored/counted_cells
    return accuracy


def main():
    print('hello world')

    #Opens image and resizes to 500px x 500px for running purposes
    img = Image.open("palm_tree.jpeg")
    result = img.resize((500,500), resample=Image.BILINEAR)
    result.save("compressed_palm_tree.jpg")

    # Convert to corresponding grey scale 
    rgb = np.array(result)[...,:3]
    grey_list = []

    #Utilize the given equation to convert to corresponding greyscale values
    for i in range(len(rgb)):
        for j in range(len(rgb[i])):
            
            grey_value = 0.21 * int(rgb[i][j][0]) + 0.72 * int(rgb[i][j][1]) + 0.07 * int(rgb[i][j][2])
            grey_list.append(grey_value)

    grey = np.reshape(grey_list, (-1, rgb.shape[1]))

    five_colors, color_arr = k_cluster(rgb)
    # print(rgb.shape)

    print("Representative colors:", five_colors)

    num_rows = rgb.shape[0]
    num_cols = rgb.shape[1]
    rgb_recolor = np.zeros(shape=(num_rows, num_cols, rgb.shape[2]))
    rgb_recolor_labels = np.zeros(shape=(num_rows, num_cols))

    # Fill in L half of rgb_recolor with new colors
    for i in range(num_rows):
        for j in range(num_cols):
            color_i = int(color_arr[(i * num_cols + j)][3])
            # print(color_i)
            rgb_recolor[i][j] = five_colors[color_i]
            rgb_recolor_labels[i][j] = color_i

    np.set_printoptions(threshold=5)
    # print(rgb)
    # print(rgb_recolor)
    plt.imshow(rgb_recolor.astype('uint8'))
    plt.show()

    basic_rgb = np.copy(rgb_recolor)
    improved_rgb = np.copy(rgb_recolor)
    advanced_rgb = np.copy(rgb_recolor)

    basic_recolored = run_basic_agent(basic_rgb, grey, five_colors, color_arr)
    improved_recolored = run_improved_agent(improved_rgb, grey, five_colors, color_arr)

    # L_half_grey = np.delete(grey, [int(num_cols / 2), num_cols-1], axis=1)
    L_half_grey = np.delete(grey, np.s_[int(num_cols / 2): num_cols], axis=1)
    L_half_rgb_recolor_labels = np.delete(rgb_recolor_labels, np.s_[int(num_cols / 2): num_cols], axis=1)

    basic_accuracy = calculate_accuracy(rgb_recolor, basic_recolored)
    print("Basic recoloring accuracy: ", basic_accuracy)

    improved_accuracy = calculate_accuracy(rgb_recolor, improved_recolored)
    print("Improved recoloring accuracy: ", improved_accuracy)

    print('goodbye world')


if __name__ == '__main__':
    main()
