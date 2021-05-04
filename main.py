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


"""
Gets the rgb and grayscale values from our image as two lists respectively.
"""
def retrieve_pixels():
    
    #Opens image and resizes to 500px x 500px for running purposes
    img = Image.open("palm_tree.jpeg")
    result = img.resize((500,500), resample=Image.BILINEAR)
    result.save("compressed_palm_tree.jpg")

    # Convert to corresponding gray scale 
    rgb_list = np.array(result)[...,:3]
    gray_list = []

    #Utilize the given equation to convert to corresponding grayscale values
    for i in range(len(rgb_list)):
        for j in range(len(rgb_list[i])):
            
            gray_value = 0.21 * int(rgb_list[i][j][0]) + 0.72 * int(rgb_list[i][j][1]) + 0.07 * int(rgb_list[i][j][2])
            gray_list.append(gray_value)

    gray = np.reshape(gray_list, (-1, rgb_list.shape[1]))
    return rgb_list, gray


"""
Run k-means clustering on the image and return the 5 representative colors in a list
"""
def cluster_pixels(rgb: np.array) -> Tuple[np.array, np.array]:
    """
    Obtain the 5 representative colors of the image through k-means clustering
    :param rgb: np.array of colored image pixels
    :return: np.array of the 5 representative colors
    """

    # Turn into 2D numpy array, multiply length by width and keep number of items in 3
    flattened_rgb = rgb.reshape(250000, 3)

    
    #Obtain initial centroids of image
    #Sample 5 random points from number of image pixels
    img_pixels = random.sample(range(0, 250000), 5)

    #Initialize centroids list
    initial_centroids = []

    # Retrieve pixels using random numbers as indices
    for i in img_pixels:
        initial_centroids.append(flattened_rgb[i])

    centroids = np.array(initial_centroids)
    
    

    # Recalculate centroid 10 times until reaches stable state
    for i in range(10):
        
        new_centroids = np.zeros(shape=(len(centroids), 3))
        element_count = np.zeros(shape=len(centroids))

        for pixel in flattened_rgb:
    
            min_distance = 1000000
            min_centroid_i = 0
    
            # Determine centroid that pixel is closest to
            for i in range(len(centroids)):
                
                #Calculate the Euclidean distance between pixel and centroid
                distance = 0
                for j in range(3):
                    distance += (int(pixel[j]) - int(centroids[i][j])) ** 2
                curr_distance = math.sqrt(distance)
                
                if curr_distance < min_distance:
                    min_centroid_i = i
                    min_distance = curr_distance
    
            # Add the pixel values to the corresponding centroid index
            new_centroids[min_centroid_i] += pixel
            element_count[min_centroid_i] += 1
    
        print("centroid sums before dividing", new_centroids)
        print("element count: ", element_count)
    
        # Divide the sum of the pixel values for each pixel for each centroid by the number of pixels for that centroid
        for i in range(len(centroids)):
            new_centroids[i] /= element_count[i]
    
        print("centroid rgb values:", new_centroids)
        centroids = new_centroids.astype(int)
    
    
        print("Iteration: ", i)
        print(centroids)
        

    # Assign pixels to a color
    pixel_colors = np.zeros(shape=(250000, 1))
    index = 0

    for pixel in flattened_rgb:

        min_distance = 255 * math.sqrt(3) + 1
        min_centroid_i = 0

        # Determine centroid that pixel is closest to
        for i in range(len(centroids)):
            
            #Calculate the Euclidean distance between pixel and centroid
            distance = 0
            for j in range(3):
                distance += (int(pixel[j]) - int(centroids[i][j])) ** 2
            curr_distance = math.sqrt(distance)
    
            
            if curr_distance < min_distance:
                min_centroid_i = i
                min_distance = curr_distance

        # Assign the pixel to a color
        pixel_colors[index] = min_centroid_i
        index += 1

    pixel_color_array = np.concatenate((flattened_rgb, pixel_colors), 1)
    
    unique, counts = np.unique(pixel_color_array[:, 3], return_counts=True)
    print('Number of pixels per color: ', dict(zip(unique, counts)))

    # Group all pixels by color
    color_dict = {}
    for i in range(5):
        color_dict['color_' + str(i)] = pixel_color_array[np.where(pixel_color_array[:, 3] == i)]

    # Plot centroids and pixels according to color they correspond to
    ax = plt.axes(projection='3d')
    
    #Slicing list according to r, g, b values respectively
    for i in range(5):
        ax.scatter3D(color_dict['color_' + str(i)][:, 0], color_dict['color_' + str(i)][:, 1],
                     color_dict['color_' + str(i)][:, 2], alpha=0.1,
                     color=centroids[i] / np.array([[255.0, 255.0, 255.0]]))
    ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='black')
    plt.show()

    return centroids.astype('uint8'), pixel_color_array



def get_similar_gray_patches(gray: np.array, left_gray_scores_dict: Dict[int, List[Tuple[int, int]]], right_i: int,
                             right_j: int) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Retrieve the 6 most similar gray patches on the left hand side
    :param gray: array of gray pixels
    :param left_gray_scores_dict: gray scores for left hand side
    :param right_i: x coordinate of right pixel
    :param right_j: y coordinate of right pixel
    :return: List of the 6 most similar gray patches
    """
    
    #Calculates the gray score of a given 3x3 patch, on the right side
    score = 0

    for x in range(-1, 2):
        for y in range(-1, 2):
            score += int(gray[right_i + x][right_j + y])

    right_gray_score = int(score / 9)    
    
    
    new_score_key_up = right_gray_score
    new_score_key_down = right_gray_score

    # Retrieve 6 similar patches from dictionary, the sum of the gray values of each pixel in the grid
    similar_gray_patch_coordinates = left_gray_scores_dict.get(right_gray_score)
    if similar_gray_patch_coordinates is None:
        similar_gray_patch_coordinates = []
    similar_gray_patches = [(0, x) for x in similar_gray_patch_coordinates]
    # print("similar patches", len(similar_gray_patches), (right_i, right_j))

    # Add more coordinates until you hit 6
    while len(similar_gray_patches) < 6:

        new_score_key_up += 1
        new_score_key_down -= 1

        # Go one key up
        similar_gray_patch_up_coordinates = left_gray_scores_dict.get(new_score_key_up)
        if similar_gray_patch_up_coordinates is None:
            similar_gray_patch_up_coordinates = []
        similar_gray_patches_up = [(new_score_key_up - right_gray_score, x) for x in similar_gray_patch_up_coordinates]
        similar_gray_patches.extend(similar_gray_patches_up)

        # Go one key down
        similar_gray_patch_down_coordinates = left_gray_scores_dict.get(new_score_key_down)
        if similar_gray_patch_down_coordinates is None:
            similar_gray_patch_down_coordinates = []
        similar_gray_patches_down = [(right_gray_score - new_score_key_down, x) for x in
                                     similar_gray_patch_down_coordinates]
        similar_gray_patches.extend(similar_gray_patches_down)
        # print("papa")

    return similar_gray_patches[:6]


def color_right_side(gray: np.array, new_rgb: np.array, representative_colors: np.array,
                     pixel_color_array: np.array) -> np.array:
    """
    Color the right side of the grayscale image
    :param gray: array of gray pixels
    :param new_rgb: array that will be colored
    :param representative_colors: 5 colors that will color the image
    :param pixel_color_array: the current colors for each pixel
    :return: a fully colored rgb image
    """
    print("starting right side coloring")

    ###BEGINNING OF GETTING RIGHT HAND GRAY SCORES###
    left_gray_scores_dict = {}

    # Iterate through left half of gray, skipping edges, use 499 as # pixels - 1, use 249 as # pixels /2 - 1
    for i in range(1, 499):
        for j in range(1, 249):
              
            #Calculates the gray score of a given 3x3 patch, the sum of the gray values of each pixel in the grid
            score = 0
        
            for x in range(-1, 2):
                for y in range(-1, 2):
                    score += int(gray[i + x][j + y])
        
            left_gray_score = int(score / 9)   

            if left_gray_score in left_gray_scores_dict:
                left_gray_scores_dict.get(left_gray_score).append((i, j))
                left_gray_scores_dict.update({left_gray_score: left_gray_scores_dict.get(left_gray_score)})
            else:
                left_gray_scores_dict[left_gray_score] = [(i, j)]
    
    print("This is the left gray dictionary")
    print(left_gray_scores_dict.keys())
    
    ###END OF GETTING LEFT HAND GRAY SCORES###


    # Fill in right half of new_rgb with new colors, ignoring edges
    for i in range(1, 499):
        for j in range(250, 499):
            
            # Retrieve most 6 most similar patch centers and their similarity scores
            score = 0
        
            for x in range(-1, 2):
                for y in range(-1, 2):
                    score += int(gray[i + x][j + y])
        
            #Calculate right gray score
            right_gray_score = int(score / 9)    
            
            
            new_score_key_up = right_gray_score
            new_score_key_down = right_gray_score
        
            # Retrieve 6 similar patches from dictionary, the sum of the gray values of each pixel in the grid
            similar_gray_patch_coordinates = left_gray_scores_dict.get(right_gray_score)
            if similar_gray_patch_coordinates is None:
                similar_gray_patch_coordinates = []
            similar_gray_patches = [(0, x) for x in similar_gray_patch_coordinates]
            # print("similar patches", len(similar_gray_patches), (right_i, right_j))
        
            # Add more coordinates until you hit 6
            while len(similar_gray_patches) < 6:
        
                new_score_key_up += 1
                new_score_key_down -= 1
        
                # Go one key up
                similar_gray_patch_up_coordinates = left_gray_scores_dict.get(new_score_key_up)
                if similar_gray_patch_up_coordinates is None:
                    similar_gray_patch_up_coordinates = []
                similar_gray_patches_up = [(new_score_key_up - right_gray_score, x) for x in similar_gray_patch_up_coordinates]
                similar_gray_patches.extend(similar_gray_patches_up)
        
                # Go one key down
                similar_gray_patch_down_coordinates = left_gray_scores_dict.get(new_score_key_down)
                if similar_gray_patch_down_coordinates is None:
                    similar_gray_patch_down_coordinates = []
                similar_gray_patches_down = [(right_gray_score - new_score_key_down, x) for x in
                                             similar_gray_patch_down_coordinates]
                similar_gray_patches.extend(similar_gray_patches_down)
                # print("papa")
        
            similar_gray_patches = similar_gray_patches[:6]
            
            
            # print("retrieving 6 patches")
            # print(similar_gray_patches)

            # Find representative color for each patch and add patch to
            patches_for_each_color = [[] for _ in range(5)]
            
            max_color_frequency = 0
            for patch in similar_gray_patches:
                color_index = int(pixel_color_array[(patch[1][0] * 500 + patch[1][1])][3])
                # color_index = new_rgb[patch[1][0]][patch[1][1]][3]
                # color_counts[color_index] += 1

                patches_for_each_color[color_index].append(patch)

                if len(patches_for_each_color[color_index]) > max_color_frequency:
                    max_color_frequency = len(patches_for_each_color[color_index])

            # print("patches for each color", patches_for_each_color, (i, j))
            # Retrieve all color indices with that max frequency
            most_frequent_color_indices = []
            for x in range(len(patches_for_each_color)):
                if len(patches_for_each_color[x]) == max_color_frequency:
                    most_frequent_color_indices.append(x)

            # If there is a most represented color, make the corresponding pixel that color
            if len(most_frequent_color_indices) == 1:
                new_rgb[i][j] = representative_colors[most_frequent_color_indices[0]]
                # print("no tie", (i, j), representative_colors[most_frequent_color_indices[0]])

            # Otherwise, break ties based on similarity score
            else:
                # print("there is a tie", (i, j))
                potential_patches = []

                # Put all patches that map to the most represented colors into potential_patches
                for x in range(len(patches_for_each_color)):
                    if len(patches_for_each_color[x]) == max_color_frequency:
                        for patch in patches_for_each_color[x]:
                            potential_patches.append(patch)

                # Select the color that is mapped to the most similar patch
                best_similarity_score = 100000
                most_similar_patch = None
                for patch in potential_patches:
                    if patch[0] < best_similarity_score:
                        best_similarity_score = patch[0]
                        most_similar_patch = patch

                # Make the original pixel the same color as that of the most similar patch
                new_rgb[i][j] = new_rgb[most_similar_patch[1][0]][most_similar_patch[1][1]]

    return new_rgb


def run_basic_agent(new_rgb: np.array, gray: np.array, representative_colors: np.array, pixel_color_array: np.array) -> np.array:
    # Color right hand side
    newest_rgb = color_right_side(gray, new_rgb, representative_colors, pixel_color_array)
    plt.imshow(newest_rgb.astype('uint8'))
    plt.show()
    print("Done with basic")
    return newest_rgb

def run_improved_agent(rgb: np.array, gray: np.array, representative_colors, pixel_color_array) -> np.array:
    red_equation, green_equation, blue_equation = generate_regression_equations(rgb, gray)

    num_rows = rgb.shape[0]
    num_cols = rgb.shape[1]

    # Fill in right half of new_rgb with new colors, ignoring edges
    for i in range(1, num_rows - 1):
        for j in range(int(num_cols / 2), num_cols - 1):
            gray_pixel_value = gray[i][j]

            new_red_value = int(red_equation[0] * gray_pixel_value + red_equation[1])
            new_green_value = int(green_equation[0] * gray_pixel_value + green_equation[1])
            new_blue_value = int(blue_equation[0] * gray_pixel_value + blue_equation[1])

            # Map to the closest representative color
            closest_color = map_to_closest_color(new_red_value, new_green_value, new_blue_value, representative_colors)
            rgb[i][j] = closest_color
            # rgb[i][j] = [new_red_value, new_green_value, new_blue_value]

    plt.imshow(rgb.astype('uint8'))
    plt.show()

    return rgb


def map_to_closest_color(red: int, green: int, blue: int, representative_colors: np.array):
    min_distance = 100000000
    closest_color = None
    dist = []

    for color in representative_colors:

        # Temporarily find closest distance to green
        red_difference = abs(red - color[0])
        green_difference = abs(green - color[1])
        blue_difference = abs(blue - color[2])

        curr_distance = .21 * red_difference + .72 * green_difference + .07 * blue_difference

        dist.append(curr_distance)
        if curr_distance < min_distance:
            min_distance = curr_distance
            closest_color = color

    # print("Actual color", red, green, blue)
    # print("Representative color", closest_color)
    # print(dist)
    return closest_color


def generate_regression_equations(rgb: np.array, gray: np.array):
    wr = 1
    wg = 1
    wb = 1
    br = 0
    bg = 0
    bb = 0
    alpha = 0.00003

    pixels_counted = 0

    # Iterate through left half of gray
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, int(gray.shape[1] / 2) - 1):
            probability = random.uniform(0, 1)
            if probability > 0.8:
                continue

            # color_pixel = rgb[i][j]
            gray_pixel_value = gray[i][j]

            # Calculate y hat
            y_hat_r = wr * gray_pixel_value + br
            y_hat_g = wg * gray_pixel_value + bg
            y_hat_b = wb * gray_pixel_value + bb

            # color_pixel = representative_colors[int(pixel_color_array[i * j][3])]
            color_pixel = rgb[i][j]

            # Calculate loss
            loss_r = (y_hat_r - color_pixel[0]) ** 2
            loss_g = (y_hat_g - color_pixel[1]) ** 2
            loss_b = (y_hat_b - color_pixel[2]) ** 2

            # Calculate new weights
            wr = wr - alpha * (y_hat_r - color_pixel[0]) * gray_pixel_value
            wg = wg - alpha * (y_hat_g - color_pixel[1]) * gray_pixel_value
            wb = wb - alpha * (y_hat_b - color_pixel[2]) * gray_pixel_value

            # Calculate new b values
            br = br - alpha * (y_hat_r - color_pixel[0])
            bg = bg - alpha * (y_hat_g - color_pixel[1])
            bb = bb - alpha * (y_hat_b - color_pixel[2])

            red_equation = 'y = ' + str(wr) + 'x + ' + str(br)
            green_equation = 'y = ' + str(wg) + 'x + ' + str(bg)
            blue_equation = 'y = ' + str(wb) + 'x + ' + str(bb)

            # print('Cell: ', (i, j))
            # print('Pixel Color: ', color_pixel)
            # print('Red Loss: ', loss_r)
            # print('Green Loss: ', loss_g)
            # print('Blue Loss: ', loss_b)
            # print('Red Equation: ', red_equation)
            # print('Green Equation: ', green_equation)
            # print('Blue Equation: ', blue_equation)
            # print()

            pixels_counted += 1

            # if (max(loss_r, loss_b, loss_g) < 1 and pixels_counted > gray.shape[0]*gray.shape[1]*0.4):
            #     break

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
    rgb, gray = retrieve_pixels()
    representative_colors, pixel_color_array = cluster_pixels(rgb)
    # print(rgb.shape)

    print("Representative colors:", representative_colors)

    num_rows = rgb.shape[0]
    num_cols = rgb.shape[1]
    new_rgb = np.zeros(shape=(num_rows, num_cols, rgb.shape[2]))
    new_rgb_labels = np.zeros(shape=(num_rows, num_cols))

    # Fill in left half of new_rgb with new colors
    for i in range(num_rows):
        for j in range(num_cols):
            color_index = int(pixel_color_array[(i * num_cols + j)][3])
            # print(color_index)
            new_rgb[i][j] = representative_colors[color_index]
            new_rgb_labels[i][j] = color_index

    np.set_printoptions(threshold=5)
    # print(rgb)
    # print(new_rgb)
    plt.imshow(new_rgb.astype('uint8'))
    plt.show()

    basic_rgb = np.copy(new_rgb)
    improved_rgb = np.copy(new_rgb)
    advanced_rgb = np.copy(new_rgb)

    basic_recolored = run_basic_agent(basic_rgb, gray, representative_colors, pixel_color_array)
    improved_recolored = run_improved_agent(improved_rgb, gray, representative_colors, pixel_color_array)

    # left_half_gray = np.delete(gray, [int(num_cols / 2), num_cols-1], axis=1)
    left_half_gray = np.delete(gray, np.s_[int(num_cols / 2): num_cols], axis=1)
    left_half_new_rgb_labels = np.delete(new_rgb_labels, np.s_[int(num_cols / 2): num_cols], axis=1)

    basic_accuracy = calculate_accuracy(new_rgb, basic_recolored)
    print("Basic recoloring accuracy: ", basic_accuracy)

    improved_accuracy = calculate_accuracy(new_rgb, improved_recolored)
    print("Improved recoloring accuracy: ", improved_accuracy)


    # print(gray)
    # print(gray.shape)
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    print('goodbye world')


if __name__ == '__main__':
    main()
