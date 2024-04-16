import cv2
import numpy as np
import collections
from sklearn.cluster import KMeans, MeanShift, DBSCAN
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
import colorsys
from scipy.spatial.distance import euclidean, cdist


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rgb_to_hsv(r, g, b):
    # Normalize RGB values
    r /= 255.0
    g /= 255.0
    b /= 255.0

    # Convert RGB to HSV
    hsv = colorsys.rgb_to_hsv(r, g, b)
    return hsv

def is_similar_to_white(hsv, hue_threshold, saturation_threshold, value_threshold):
    # Define threshold values for similarity to white
    # hue_threshold = 30  # You can adjust this threshold as needed
    # saturation_threshold = 0.2  # You can adjust this threshold as needed
    # value_threshold = 0.9  # You can adjust this threshold as needed

    # Get HSV components
    hue, saturation, value = hsv

    # Check if color is similar to white based on thresholds
    if saturation <= saturation_threshold and \
       value >= value_threshold:
        return True
    else:
        return False
    
def detectColorWithHSV(ori_im2arr, thres_hue=30, thres_saturation= 0.2, thres_value=0.9):
    temp_img = []
    
    for i in range(len(ori_im2arr)):
        for j in range(len(ori_im2arr[i])):
            # getting the RGB pixel value.
            r, g, b = ori_im2arr[i, j]
            hsv_values = rgb_to_hsv(r, g, b)
            similar_to_white = is_similar_to_white(hsv_values, thres_hue, thres_saturation, thres_value)

            if similar_to_white == False:
                temp_img.append((r, g, b))

    # Count occurrences of each unique color tuple
    color_count = collections.Counter(temp_img)

    # Sort unique colors based on their counts
    unique_colors = sorted(set(temp_img), key=lambda x: color_count[x], reverse=True)

    return unique_colors

def detectColorFromHSV(ori_im2arr_hsv, thres_hue=30, thres_saturation= 0.2, thres_value=0.9):
    temp_img = []

    for i, color1 in enumerate(ori_im2arr_hsv):
        h, s, v = color1
        # hsv_values = rgb_to_hsv(r, g, b)
        similar_to_white = is_similar_to_white([h, s, v], thres_hue, thres_saturation, thres_value)

        if similar_to_white == False:
            temp_img.append((h, s, v))
    
    # for i in range(len(ori_im2arr_hsv)):
    #     for j in range(len(ori_im2arr_hsv[i])):
    #         # getting the RGB pixel value.
    #         h, s, v = ori_im2arr_hsv[i, j]
    #         # hsv_values = rgb_to_hsv(r, g, b)
    #         similar_to_white = is_similar_to_white([h, s, v], thres_hue, thres_saturation, thres_value)

    #         if similar_to_white == False:
    #             temp_img.append((h, s, v))

    # Count occurrences of each unique color tuple
    color_count = collections.Counter(temp_img)

    # Sort unique colors based on their counts
    unique_colors = sorted(set(temp_img), key=lambda x: color_count[x], reverse=True)

    return unique_colors

def detectColorWithHSV_replace_similar(hsv_values, thres_hue=30, thres_saturation= 0.2, thres_value=0.9):
    temp_img = []
    
    for i in range(len(hsv_values)):
        # getting the RGB pixel value.
        similar_to_white = is_similar_to_white(hsv_values[i], thres_hue, thres_saturation, thres_value)

        if similar_to_white == False:
            temp_img.append(hsv_values[i])

    # Count occurrences of each unique color tuple
    color_count = collections.Counter(temp_img)

    # Sort unique colors based on their counts
    unique_colors = sorted(set(temp_img), key=lambda x: color_count[x], reverse=True)

    return unique_colors

def convertRGBtoHSV(ori_im2arr):
    ori_im2arr_hsv = []
    for i in range(len(ori_im2arr)):
        for j in range(len(ori_im2arr[i])):
            # getting the RGB pixel value.
            r, g, b = ori_im2arr[i, j]
            hsv_values = rgb_to_hsv(r, g, b)
            ori_im2arr_hsv.append(hsv_values)

    return ori_im2arr_hsv

def replace_similar_with_mean_show_distances_and_unique(image_A_hsv, threshold):
    
    colors = image_A_hsv.reshape(-1, 3)

    # Create an array to hold the modified colors, initially a copy of the original array
    modified_colors = np.copy(colors)

    # Create a list to record distances for similar colors
    similar_color_distances = []

    # Iterate over each color
    for i, color1 in enumerate(colors):
        # Initialize a group for potential merging
        merge_group = [color1]

        for j, color2 in enumerate(colors):
            if i != j:
                distance = euclidean(color1, color2)
                # Add color to merge group and record distance if it's below the threshold
                if distance < threshold:
                    merge_group.append(color2)
                    similar_color_distances.append((color1, color2, distance))

        # If more than one color is in the merge group, calculate their mean
        if len(merge_group) > 1:
            merged_color = np.mean(merge_group, axis=0)
            # Replace the similar colors in the original array with their mean
            for color in merge_group:
                modified_colors[np.all(colors == color, axis=1)] = merged_color

    # Extract unique colors from the modified array
    unique_colors = np.unique(modified_colors, axis=0)

    return modified_colors, similar_color_distances, unique_colors

def replace_similar_with_mean_show_distances_and_unique_optimize(ori_im2arr, threshold):
    # Reshape image array to a 2D array where each row is a color
    colors = ori_im2arr.reshape(-1, 3)

    # Convert to HSV color space if necessary (assuming it's already HSV based on original code)
    # hsv_colors = rgb_to_hsv(colors)  # Uncomment if colors need conversion

    # Calculate the pairwise distances between colors
    distances = cdist(colors, colors, 'euclidean')

    # Identify pairs of colors that are within the threshold
    similar_pairs = distances < threshold

    # To avoid double counting, we mask the upper triangle of the similarity matrix
    np.fill_diagonal(similar_pairs, False)
    similar_pairs = np.triu(similar_pairs)

    # Replace similar colors with their mean
    for i in range(len(colors)):
        similar_colors = colors[similar_pairs[i]]
        if similar_colors.size > 0:
            mean_color = np.mean(np.vstack((colors[i], similar_colors)), axis=0)
            colors[similar_pairs[i]] = mean_color
            colors[i] = mean_color

    # Extract unique colors from the modified array
    unique_colors = np.unique(colors, axis=0)

    return colors.reshape(ori_im2arr.shape), similar_pairs, unique_colors


def detectColorWithHSV_after_filter(ori_im2arr, thres_hue=30, thres_saturation= 0.2, thres_value=0.9):
    temp_img = []
    for i in range(len(ori_im2arr)):
        for j in range(len(ori_im2arr[i])):
            # getting the RGB pixel value.
            r, g, b = ori_im2arr[i]
            hsv_values = rgb_to_hsv(r, g, b)
            similar_to_white = is_similar_to_white(hsv_values, thres_hue, thres_saturation, thres_value)

            if similar_to_white == False:
                temp_img.append((r, g, b))

    # Count occurrences of each unique color tuple
    color_count = collections.Counter(temp_img)

    # Sort unique colors based on their counts
    unique_colors = sorted(set(temp_img), key=lambda x: color_count[x], reverse=True)

    return unique_colors

# Function to multiply images in a directory
def multiply_images_in_directory(directory_path):
    # Get all files in the directory
    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    image_files.sort()

    if len(image_files) < 2:
        print("There should be at least 2 images in the directory to perform multiplication.")
        return

    # Read the first image to initialize the result
    print(image_files[0])
    result = cv2.imread(os.path.join(directory_path, image_files[0]))
    result = result/255
#     result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Loop through remaining images and multiply them with the result
    for i in range(1, len(image_files)):
        print(image_files[i])
        img = cv2.imread(os.path.join(directory_path, image_files[i]))
        img = img/255
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ensure all images are of the same size for multiplication
        if img.shape != result.shape:
            print(f"Images should be of the same size. Skipping multiplication for {image_files[i]}.")
            continue

        # Perform multiplication
#         result = cv2.multiply(result, img)
        result = result * img
    result = result * 255
    
#     result += img

    # Save the resulting multiplied image
    cv2.imwrite(f"{directory_path}/multiplied_image.jpg", result)
    print("Multiplication completed. Resulting image saved as multiplied_image.png")

def show_distribution_graphic_3D(directory_path_ori_A, pixels, non_white_pixels, non_white_labels, cluster_centers, folder_name):
    # Plot cluster distribution using scatter plot
    fig = plt.figure(figsize=(20, 10))

    # Plot original pixels
    # Plot original pixels in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=pixels / 255.0, s=1, marker='o')
    ax1.set_title('Pixel Distribution Before Clustering')
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    # Create custom legend elements for different RGB values
    unique_colors = np.unique((pixels / 255.0), axis=0)  # Get unique RGB colors
    legend_labels = [f'Color ({int(r * 255)}, {int(g * 255)}, {int(b * 255)})' for r, g, b in unique_colors]

    # Limit the legend to display only the first 5 entries
    max_legend_entries = 5
    if len(unique_colors) > max_legend_entries:
        legend_labels = legend_labels[:max_legend_entries]
        unique_colors = unique_colors[:max_legend_entries]

    # Create legend elements for the limited number of entries
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, unique_colors)]

    # Add ellipsis (...) if there are more entries beyond the limit
    if len(unique_colors) < len(pixels):
        legend_elements.append(Line2D([0], [0], marker='None', color='w', label='...', markersize=0))

    # Display legend with custom legend elements
    ax1.legend(handles=legend_elements, loc='upper right', title='Colors')

    # Get colors from original image for scatter plot
    colors = [non_white_pixels[non_white_labels.ravel() == i].mean(axis=0) for i in range(k_value)]
    colors = np.array(colors, dtype=np.uint8)

    # Plot histogram of cluster colors
    # Plot clustered pixels
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Adjust the marker size for cluster centers (you can modify this value)
    cluster_marker_size = 50

    ax2.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
        c='red', marker='x', s=cluster_marker_size, label='Cluster Centers'
    )

    ax2.scatter(non_white_pixels[:, 0], non_white_pixels[:, 1], non_white_pixels[:, 2], c=[colors[label] / 255.0 for label in non_white_labels], s=1, marker='o')
    # ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='x', label='Cluster Centers')
    
    ax2.set_title('Pixel Distribution After Clustering')
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green')
    ax2.set_zlabel('Blue')

    # # Create a legend using proxy artists
    # legend_labels = [f'Cluster {i}' for i in range(k_value)]
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=f'C{i}', markersize=10) for i, label in enumerate(legend_labels)]
    # plt.legend(handles=legend_elements, loc='upper right', title='Clusters')

    # Create custom legend entries using colors from the scatter plot
    
    legend_labels = [f'Cluster {i} {tuple(map(int, color))}' for i, color in enumerate(cluster_centers)]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=colors[i] / 255.0, markersize=10) for i, label in enumerate(legend_labels)]
    ax2.legend(handles=legend_elements, loc='upper right', title='Clusters')

    plt.suptitle("Color Distribution Before and After Clustering")
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{directory_path_ori_A}{folder_name}\\graphic_distribution.png")
    plt.close()

def show_distribution_graphic_3D_using_HSV(directory_path_ori_A, pixels, non_white_pixels, non_white_labels, cluster_centers, folder_name):
    # Plot cluster distribution using scatter plot
    fig = plt.figure(figsize=(20, 10))

    # Plot original pixels
    # Plot original pixels in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # rgb_colors = plt.cm.hsv(pixels)  # Normalize HSV values
    # Initialize an empty array for the RGB values
    rgb_colors = np.zeros_like(pixels, dtype=float)

    # Convert each HSV value to RGB
    for i in range(len(pixels)):
        h, s, v = pixels[i, 0], pixels[i, 1], pixels[i, 2]  # Normalize HSV values
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb_colors[i] = [r, g, b]

    ax1.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=rgb_colors , s=1, marker='o')
    ax1.set_title('Pixel Distribution Before Clustering')
    ax1.set_xlabel('Hue')
    ax1.set_ylabel('Saturation')
    ax1.set_zlabel('Value')
    # Create custom legend elements for different RGB values
    unique_colors = np.unique(pixels, axis=0)  # Get unique RGB colors
    legend_labels = [f'Hue: {h:.3f}, Sat: {s:.3f}, Val: {v:.3f}' for h, s, v in unique_colors]

    # Limit the legend to display only the first 5 entries
    max_legend_entries = 5
    if len(unique_colors) > max_legend_entries:
        legend_labels = legend_labels[:max_legend_entries]
        unique_colors = unique_colors[:max_legend_entries]

    # Create legend elements for the limited number of entries
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, unique_colors)]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, unique_colors)]


    # Add ellipsis (...) if there are more entries beyond the limit
    if len(unique_colors) < len(pixels):
        legend_elements.append(Line2D([0], [0], marker='None', color='w', label='...', markersize=0))

    # Display legend with custom legend elements
    ax1.legend(handles=legend_elements, loc='upper right', title='Colors')

    # Get colors from original image for scatter plot
    _colors = [non_white_pixels[non_white_labels.ravel() == i].mean(axis=0) for i in range(k_value)]
    colors = np.array(_colors, dtype=float)

    # Convert HSV to RGB for visualization
    rgb_colors = np.zeros_like(colors, dtype=float)

    # Convert each HSV value to RGB
    for i in range(len(colors)):
        h, s, v = colors[i, 0], colors[i, 1], colors[i, 2]  # Normalize HSV values
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb_colors[i] = [r, g, b]

    # Plot histogram of cluster colors
    # Plot clustered pixels
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Adjust the marker size for cluster centers (you can modify this value)
    cluster_marker_size = 50

    # Convert cluster_centers from HSV to RGB for visualization
    # rgb_cluster_centers = plt.cm.hsv(cluster_centers)

    rgb_cluster_centers = np.zeros_like(cluster_centers, dtype=float)

    # Convert each HSV value to RGB
    for i in range(len(cluster_centers)):
        h, s, v = cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2]  # Normalize HSV values
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb_cluster_centers[i] = [r, g, b]


    ax2.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
        c='red', marker='x', s=cluster_marker_size, label='Cluster Centers'
    )

    # ax2.scatter(non_white_pixels[:, 0], non_white_pixels[:, 1], non_white_pixels[:, 2], c=[colors[label] / 255.0 for label in non_white_labels], s=1, marker='o')
    # ax2.scatter(non_white_pixels[:, 0], non_white_pixels[:, 1], non_white_pixels[:, 2], c=rgb_colors[non_white_labels], s=1, marker='o')
    ax2.scatter(non_white_pixels[:, 0], non_white_pixels[:, 1], non_white_pixels[:, 2], c=[rgb_colors[label] for label in non_white_labels], s=1, marker='o')

    # ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='x', label='Cluster Centers')
    
    ax2.set_title('Pixel Distribution After Clustering')
    ax2.set_xlabel('Hue')
    ax2.set_ylabel('Saturation')
    ax2.set_zlabel('Value')

    # # Create a legend using proxy artists
    # legend_labels = [f'Cluster {i}' for i in range(k_value)]
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=f'C{i}', markersize=10) for i, label in enumerate(legend_labels)]
    # plt.legend(handles=legend_elements, loc='upper right', title='Clusters')

    # Create custom legend entries using colors from the scatter plot
    
    # legend_labels = [f'Cluster {i} {tuple(map(int, color))}' for i, color in enumerate(cluster_centers)]
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=colors[i] / 255.0, markersize=10) for i, label in enumerate(legend_labels)]
    # ax2.legend(handles=legend_elements, loc='upper right', title='Clusters')

    legend_labels = [f'Cluster {i} (Hue: {h:.3f}, Sat: {s:.3f}, Val: {v:.3f})' for i, (h, s, v) in enumerate(cluster_centers)]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, rgb_cluster_centers)]

    ax2.legend(handles=legend_elements, loc='upper right', title='Clusters')


    plt.suptitle("Color Distribution Before and After Clustering")
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{directory_path_ori_A}{folder_name_create}\\graphic_distribution_AL_{folder_name}_{no_aug}.png")
    plt.close()

def show_distribution_graphic_3D_using_HSV_DBSCAN(directory_path_ori_A, pixels, non_white_pixels, dbscan_labels, folder_name):
    # Plot cluster distribution using scatter plot
    fig = plt.figure(figsize=(20, 10))

    # Plot original pixels
    # Plot original pixels in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # rgb_colors = plt.cm.hsv(pixels)  # Normalize HSV values
    # Initialize an empty array for the RGB values
    rgb_colors = np.zeros_like(pixels, dtype=float)

    # Convert each HSV value to RGB
    for i in range(len(pixels)):
        h, s, v = pixels[i, 0], pixels[i, 1], pixels[i, 2]  # Normalize HSV values
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb_colors[i] = [r, g, b]

    ax1.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=rgb_colors , s=1, marker='o')
    ax1.set_title('Pixel Distribution Before Clustering')
    ax1.set_xlabel('Hue')
    ax1.set_ylabel('Saturation')
    ax1.set_zlabel('Value')
    # Create custom legend elements for different RGB values
    unique_colors = np.unique(pixels, axis=0)  # Get unique RGB colors
    legend_labels = [f'Hue: {h:.3f}, Sat: {s:.3f}, Val: {v:.3f}' for h, s, v in unique_colors]

    # Limit the legend to display only the first 5 entries
    max_legend_entries = 5
    if len(unique_colors) > max_legend_entries:
        legend_labels = legend_labels[:max_legend_entries]
        unique_colors = unique_colors[:max_legend_entries]

    # Create legend elements for the limited number of entries
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, unique_colors)]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, unique_colors)]


    # Add ellipsis (...) if there are more entries beyond the limit
    if len(unique_colors) < len(pixels):
        legend_elements.append(Line2D([0], [0], marker='None', color='w', label='...', markersize=0))

    # Display legend with custom legend elements
    ax1.legend(handles=legend_elements, loc='upper right', title='Colors')

    # Plot histogram of cluster colors
    # Plot clustered pixels
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Exclude noise label (-1) if present
    unique_labels = set(dbscan_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    # Create a legend for the clusters
    legend_elements = []

    # Iterate over clusters (excluding noise)
    for cluster_num in unique_labels:
        cluster_pixels = non_white_pixels[dbscan_labels == cluster_num]
        average_color = np.mean(cluster_pixels, axis=0)
        r, g, b = colorsys.hsv_to_rgb(average_color[0], average_color[1], average_color[2])
        rgb_color = [r, g, b]

        # Plot the cluster
        ax2.scatter(cluster_pixels[:, 0], cluster_pixels[:, 1], cluster_pixels[:, 2], c=[rgb_color], s=1, marker='o')

        # Add to legend
        legend_label = f'Cluster {cluster_num} (Hue: {average_color[0]:.3f}, Sat: {average_color[1]:.3f}, Val: {average_color[2]:.3f})'
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label, markerfacecolor=rgb_color, markersize=10))
    
    ax2.set_title('Pixel Distribution After Clustering')
    ax2.set_xlabel('Hue')
    ax2.set_ylabel('Saturation')
    ax2.set_zlabel('Value')

    ax2.legend(handles=legend_elements, loc='upper right', title='Clusters')


    plt.suptitle("Color Distribution Before and After Clustering")
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{directory_path_ori_A}{folder_name}\\graphic_distribution.png")
    plt.close()


def show_distribution_graphic_3D_mean_shift(directory_path_ori_A, pixels, non_white_pixels, non_white_labels, cluster_centers, folder_name):
    # Plot cluster distribution using scatter plot
    fig = plt.figure(figsize=(20, 10))

    # Plot original pixels
    # Plot original pixels in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], c=pixels / 255.0, s=1, marker='o')
    ax1.set_title('Pixel Distribution Before Clustering')
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    # Create custom legend elements for different RGB values
    unique_colors = np.unique((pixels / 255.0), axis=0)  # Get unique RGB colors
    legend_labels = [f'Color ({int(r * 255)}, {int(g * 255)}, {int(b * 255)})' for r, g, b in unique_colors]

    # Limit the legend to display only the first 5 entries
    max_legend_entries = 5
    if len(unique_colors) > max_legend_entries:
        legend_labels = legend_labels[:max_legend_entries]
        unique_colors = unique_colors[:max_legend_entries]

    # Create legend elements for the limited number of entries
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in zip(legend_labels, unique_colors)]

    # Add ellipsis (...) if there are more entries beyond the limit
    if len(unique_colors) < len(pixels):
        legend_elements.append(Line2D([0], [0], marker='None', color='w', label='...', markersize=0))

    # Display legend with custom legend elements
    ax1.legend(handles=legend_elements, loc='upper right', title='Colors')

    # Get colors from original image for scatter plot
    # colors = [non_white_pixels[non_white_labels.ravel() == i].mean(axis=0) for i in range(k_value)]
    # colors = np.array(colors, dtype=np.uint8)

    unique_labels = np.unique(non_white_labels)
    colors = [non_white_pixels[non_white_labels == label].mean(axis=0) for label in unique_labels]

    # Plot histogram of cluster colors
    # Plot clustered pixels
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Adjust the marker size for cluster centers (you can modify this value)
    cluster_marker_size = 50

    ax2.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
        c='red', marker='x', s=cluster_marker_size, label='Cluster Centers'
    )

    # ax2.scatter(non_white_pixels[:, 0], non_white_pixels[:, 1], non_white_pixels[:, 2], c=[colors[label] / 255.0 for label in non_white_labels], s=1, marker='o')
    # ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='x', label='Cluster Centers')

    # Scatter plot for all pixels, colored by their respective cluster
    for label, color in zip(unique_labels, colors):
        ax2.scatter(non_white_pixels[non_white_labels == label, 0], non_white_pixels[non_white_labels == label, 1], non_white_pixels[non_white_labels == label, 2], c=[color / 255.0], s=1, marker='o')
    
    ax2.set_title('Pixel Distribution After Clustering')
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green')
    ax2.set_zlabel('Blue')

    # # Create a legend using proxy artists
    # legend_labels = [f'Cluster {i}' for i in range(k_value)]
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=f'C{i}', markersize=10) for i, label in enumerate(legend_labels)]
    # plt.legend(handles=legend_elements, loc='upper right', title='Clusters')

    # Create custom legend entries using colors from the scatter plot
    
    # legend_labels = [f'Cluster {i} {tuple(map(int, color))}' for i, color in enumerate(cluster_centers)]
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=colors[i] / 255.0, markersize=10) for i, label in enumerate(legend_labels)]
    # ax2.legend(handles=legend_elements, loc='upper right', title='Clusters')

    # Create custom legend entries using colors from the scatter plot
    legend_labels = [f'Cluster {i} {tuple(map(int, color))}' for i, color in enumerate(colors)]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color / 255.0, markersize=10) for label, color in zip(legend_labels, colors)]

    ax2.legend(handles=legend_elements, loc='upper right', title='Clusters')

    plt.suptitle("Color Distribution Before and After Clustering")
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{directory_path_ori_A}{folder_name}\\graphic_distribution.png")
    plt.close()

def process_kmeans_and_show_distribution_using_HSV_original(directory_path_ori_A, image_A, image_B, dataColor, k_value, folder_name, t_hue, t_saturation, t_values, mask_color = False):

    # Convert the image from BGR to RGB
    # image_rgb = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
    image_rgb = image_A

    # Flatten the image to 2D array of pixels (rows = pixels, columns = RGB)
    pixels = image_rgb.reshape((-1, 3))

    num_colors = k_value  # Change this number as needed

    print(dataColor)
    print(num_colors)
    
    # Calculate row means using rgb_to_hsv function
    row_hsv = []
    for row in pixels:
        hsv_values = rgb_to_hsv(row[0], row[1], row[2])
        row_hsv.append(hsv_values)

    # Convert list to numpy array
    row_hsv = np.array(row_hsv)

    # Check similarity to white using is_similar_to_white function
    white_mask = []
    for _row_hsv in row_hsv:
        hsv_values = _row_hsv
        is_white_similar = is_similar_to_white(hsv_values, t_hue, t_saturation, t_values)
        white_mask.append(is_white_similar)

    # Convert list to numpy array
    white_mask = np.array(white_mask)

    # Create a mask for non-white pixels to keep the size same
    non_white_mask = np.logical_not(white_mask)

    # print(non_white_mask)

    # Apply K-Means clustering to non-white pixels
    non_white_pixels = pixels[non_white_mask]

    dataColor2 = detectColorWithHSV_after_filter(non_white_pixels, t_hue, t_saturation, t_values)
    print(dataColor2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS )
    _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_PP_CENTERS )

    cluster_centers_hsv = []
    for row in cluster_centers:
        cluster_centers_hsv_values = rgb_to_hsv(row[0], row[1], row[2])
        cluster_centers_hsv.append(cluster_centers_hsv_values[2])

    show_distribution_graphic_3D(directory_path_ori_A, pixels, non_white_pixels, non_white_labels, cluster_centers, folder_name)

    # Sort cluster centers and labels based on brightness
    sorted_indices = np.argsort(cluster_centers_hsv)
    # print(sorted_indices)

    labels = np.full_like(white_mask, -1, dtype=np.int32).reshape((-1, 1))  # Initialize with -1 or any value outside cluster labels range
    labels[non_white_mask] = non_white_labels
    labels = labels.reshape((-1))

    # Calculate the average color for each cluster
    average_colors = []
    for i, idx in enumerate(sorted_indices):
        cluster_pixels = pixels[labels == idx]
        average_color = np.mean(cluster_pixels, axis=0)
        average_colors.append(average_color)

    # Reshape the labels to the shape of the original image
    segmented_image = labels.reshape(image_rgb.shape[:2])

    num_image = 0

    for cluster_num in range(len(sorted_indices)):
        # if cluster_num not in clusters_to_eliminate:
        if mask_color == False:
            #background black
            standard_mask = np.ones(segmented_image.shape[:2], dtype=np.uint8) * 255  # Initialize with white mask
            standard_mask[segmented_image == sorted_indices[cluster_num]] = 0  # Set eliminated clusters to black in the mask

            # Save the standard mask
            cv2.imwrite(f'{directory_path_ori_A}{folder_name}\\mask_{cluster_num}_{cluster_centers[sorted_indices[cluster_num]]}.png', standard_mask)
        # else:
            #background black
            color_mask = np.ones((*segmented_image.shape, 3), dtype=np.uint8) * 255
            color_mask[segmented_image == sorted_indices[cluster_num]] = average_colors[cluster_num].astype(np.uint8)

            # Save the color mask
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving as an image
            cv2.imwrite(f'{directory_path_ori_A}{folder_name}\\color_mask_{cluster_num}_{cluster_centers[sorted_indices[cluster_num]]}.png', color_mask)
        
        num_image += 1

def process_kmeans_and_show_distribution_using_HSV_original_not_pixel(directory_path_ori_A, image_A, image_B, dataColor, k_value, folder_name, t_hue, t_saturation, t_values, mask_color = False):

    # Convert the image from BGR to RGB
    # image_rgb = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
    image_rgb = image_A

    # Flatten the image to 2D array of pixels (rows = pixels, columns = RGB)
    # pixels = image_rgb.reshape((-1, 3))

    num_colors = k_value  # Change this number as needed

    print(dataColor)
    print(num_colors)
    
    # Calculate row means using rgb_to_hsv function
    row_hsv = []
    # for row in pixels:
    #     hsv_values = rgb_to_hsv(row[0], row[1], row[2])
    #     row_hsv.append(hsv_values)

    # Convert list to numpy array
    row_hsv = np.array(image_rgb)

    # Check similarity to white using is_similar_to_white function
    white_mask = []
    for _row_hsv in row_hsv:
        hsv_values = _row_hsv
        is_white_similar = is_similar_to_white(hsv_values, t_hue, t_saturation, t_values)
        white_mask.append(is_white_similar)

    # Convert list to numpy array
    white_mask = np.array(white_mask)

    # Create a mask for non-white pixels to keep the size same
    non_white_mask = np.logical_not(white_mask)

    # print(non_white_mask)

    # Apply K-Means clustering to non-white pixels
    non_white_pixels = row_hsv[non_white_mask]
    # non_white_pixels2 = pixels[non_white_mask]

    dataColor2 = detectColorWithHSV_after_filter(non_white_pixels, t_hue, t_saturation, t_values)
    print(dataColor2)

    # Step 2: Define the key array (initial centroids)
    if k_value != len(init_centroid):
        initial_centroids = np.array(init_centroid[:k_value])
    else:
        initial_centroids = np.array(init_centroid)
    

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Reshape initial_centroids to meet OpenCV's requirements
    initial_centroids = initial_centroids.reshape(-1, 1, 3)  # Reshape if necessary

    # try:
    #     _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS, initial_centroids)
    # except Exception as e:
    #     _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # kmeans = KMeans(n_clusters=k_value, init=initial_centroids, n_init=1)
    # kmeans.fit(np.float32(non_white_pixels))

    # cluster_centers = kmeans.cluster_centers_
    # non_white_labels = kmeans.labels_

    cluster_centers_hsv = []
    for row in cluster_centers:
        # cluster_centers_hsv_values = rgb_to_hsv(row[0], row[1], row[2])
        cluster_centers_hsv.append(row[2])

    # Sort cluster centers and labels based on brightness
    sorted_indices = np.argsort(cluster_centers_hsv)
    # print(sorted_indices)

    labels = np.full_like(white_mask, -1, dtype=np.int32).reshape((-1, 1))  # Initialize with -1 or any value outside cluster labels range
    # labels = labels.reshape((-1)) #activate when using sklearn_kmeans
    labels[non_white_mask] = non_white_labels
    labels = labels.reshape((-1)) #activate when using opencv_kmeans

    # Calculate the average color for each cluster
    average_colors = []
    cluster_centers_ordered = []
    for i, idx in enumerate(sorted_indices):
        cluster_centers_ordered.append(cluster_centers[idx])
        cluster_pixels = row_hsv[labels == idx]
        average_color = np.mean(cluster_pixels, axis=0)
        r, g, b = colorsys.hsv_to_rgb(average_color[0], average_color[1], average_color[2])
        average_colors.append([r*255, g*255, b*255])

    cluster_centers_ordered = np.array(cluster_centers_ordered)
    average_colors = np.array(average_colors)

    # print(len(cluster_centers_ordered))
    # print(cluster_centers_ordered)

    show_distribution_graphic_3D_using_HSV(directory_path_ori_A, row_hsv, non_white_pixels, non_white_labels, cluster_centers_ordered, folder_name)

    # Reshape the labels to the shape of the original image
    segmented_image = labels.reshape((1024, 1024))

    num_image = 0

    for cluster_num in range(len(sorted_indices)):
        # if cluster_num not in clusters_to_eliminate:
        if mask_color == False:
            #background black
            standard_mask = np.ones(segmented_image.shape[:2], dtype=np.uint8) * 255  # Initialize with white mask
            standard_mask[segmented_image == sorted_indices[cluster_num]] = 0  # Set eliminated clusters to black in the mask

            # Save the standard mask
            cv2.imwrite(f'{directory_path_ori_A}{folder_name}\\mask_{cluster_num}_{cluster_centers_ordered[cluster_num]}.png', standard_mask)
        # else:
            #background black
            color_mask = np.ones((*segmented_image.shape, 3), dtype=np.uint8) * 255
            color_mask[segmented_image == sorted_indices[cluster_num]] = average_colors[cluster_num].astype(np.uint8)

            # Save the color mask
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving as an image
            cv2.imwrite(f'{directory_path_ori_A}{folder_name}\\color_mask_{cluster_num}_{cluster_centers_ordered[cluster_num]}.png', color_mask)
        
        num_image += 1

def process_kmeans_and_show_distribution_using_HSV_not_pixel(directory_path, image_A, image_B, dataColor, k_value, folder_name, t_hue, t_saturation, t_values, no_aug, mask_color = False):

    # Convert the image from BGR to RGB
    # image_rgb = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
    image_rgb = image_B

    # Flatten the image to 2D array of pixels (rows = pixels, columns = RGB)
    # pixels = image_rgb.reshape((-1, 3))

    num_colors = k_value  # Change this number as needed

    # print(dataColor)
    # print(num_colors)
    
    # Calculate row means using rgb_to_hsv function
    row_hsv = []
    # for row in pixels:
    #     hsv_values = rgb_to_hsv(row[0], row[1], row[2])
    #     row_hsv.append(hsv_values)

    # Convert list to numpy array
    row_hsv = np.array(image_rgb)

    # Check similarity to white using is_similar_to_white function
    white_mask = []
    for _row_hsv in row_hsv:
        hsv_values = _row_hsv
        is_white_similar = is_similar_to_white(hsv_values, t_hue, t_saturation, t_values)
        white_mask.append(is_white_similar)

    # Convert list to numpy array
    white_mask = np.array(white_mask)

    # Create a mask for non-white pixels to keep the size same
    non_white_mask = np.logical_not(white_mask)

    # print(non_white_mask)

    # Apply K-Means clustering to non-white pixels
    non_white_pixels = row_hsv[non_white_mask]
    # non_white_pixels2 = pixels[non_white_mask]

    dataColor2 = detectColorWithHSV_after_filter(non_white_pixels, t_hue, t_saturation, t_values)
    # print(dataColor2)

    # Step 2: Define the key array (initial centroids)
    if k_value != len(init_centroid):
        initial_centroids = np.array(init_centroid[:k_value])
    else:
        initial_centroids = np.array(init_centroid)
    

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Reshape initial_centroids to meet OpenCV's requirements
    initial_centroids = initial_centroids.reshape(-1, 1, 3)  # Reshape if necessary

    # try:
    #     _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS, initial_centroids)
    # except Exception as e:
    #     _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # kmeans = KMeans(n_clusters=k_value, init=initial_centroids, n_init=1)
    # kmeans.fit(np.float32(non_white_pixels))

    # cluster_centers = kmeans.cluster_centers_
    # non_white_labels = kmeans.labels_

    cluster_centers_hsv = []
    for row in cluster_centers:
        # cluster_centers_hsv_values = rgb_to_hsv(row[0], row[1], row[2])
        cluster_centers_hsv.append(row[2])

    # Sort cluster centers and labels based on brightness
    sorted_indices = np.argsort(cluster_centers_hsv)
    # print(sorted_indices)

    labels = np.full_like(white_mask, -1, dtype=np.int32).reshape((-1, 1))  # Initialize with -1 or any value outside cluster labels range
    # labels = labels.reshape((-1)) #activate when using sklearn_kmeans
    labels[non_white_mask] = non_white_labels
    labels = labels.reshape((-1)) #activate when using opencv_kmeans

    # Calculate the average color for each cluster
    average_colors = []
    cluster_centers_ordered = []
    for i, idx in enumerate(sorted_indices):
        cluster_centers_ordered.append(cluster_centers[idx])
        cluster_pixels = row_hsv[labels == idx]
        average_color = np.mean(cluster_pixels, axis=0)
        r, g, b = colorsys.hsv_to_rgb(average_color[0], average_color[1], average_color[2])
        average_colors.append([r*255, g*255, b*255])

    cluster_centers_ordered = np.array(cluster_centers_ordered)
    average_colors = np.array(average_colors)

    # print(len(cluster_centers_ordered))
    # print(cluster_centers_ordered)

    show_distribution_graphic_3D_using_HSV(directory_path, row_hsv, non_white_pixels, non_white_labels, cluster_centers_ordered, folder_name)

    # Reshape the labels to the shape of the original image
    segmented_image = labels.reshape((1024, 1024))

    num_image = 0

    for cluster_num in range(len(sorted_indices)):
        # if cluster_num not in clusters_to_eliminate:
        if mask_color == False:
            #background black
            standard_mask = np.ones(segmented_image.shape[:2], dtype=np.uint8) * 255  # Initialize with white mask
            standard_mask[segmented_image == sorted_indices[cluster_num]] = 0  # Set eliminated clusters to black in the mask

            # Save the standard mask
            cv2.imwrite(f'{directory_path}{folder_name_create}\\mask_AL_{folder_name}_{cluster_num+1}_{no_aug}.png', standard_mask)
        # else:
            #background black
            color_mask = np.ones((*segmented_image.shape, 3), dtype=np.uint8) * 255
            color_mask[segmented_image == sorted_indices[cluster_num]] = average_colors[cluster_num].astype(np.uint8)

            # Save the color mask
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving as an image
            cv2.imwrite(f'{directory_path}{folder_name_create}\\color_mask_AL_{folder_name}_{cluster_num+1}_{no_aug}.png', color_mask)
        
        num_image += 1


def process_DBSCAN_and_show_distribution_using_HSV_original_not_pixel(directory_path_ori_A, image_A, image_B, init_centroid, k_value, folder_name, t_hue, t_saturation, t_values, mask_color = False):

    # Convert the image from BGR to RGB
    # image_rgb = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)
    image_rgb = image_B

    # Flatten the image to 2D array of pixels (rows = pixels, columns = RGB)
    # pixels = image_rgb.reshape((-1, 3))

    num_colors = k_value  # Change this number as needed

    # print(init_centroid)
    print(num_colors)
    
    # Calculate row means using rgb_to_hsv function
    row_hsv = []
    # for row in pixels:
    #     hsv_values = rgb_to_hsv(row[0], row[1], row[2])
    #     row_hsv.append(hsv_values)

    # Convert list to numpy array
    row_hsv = np.array(image_rgb)

    # Check similarity to white using is_similar_to_white function
    white_mask = []
    for _row_hsv in row_hsv:
        hsv_values = _row_hsv
        is_white_similar = is_similar_to_white(hsv_values, t_hue, t_saturation, t_values)
        white_mask.append(is_white_similar)

    # Convert list to numpy array
    white_mask = np.array(white_mask)

    # Create a mask for non-white pixels to keep the size same
    non_white_mask = np.logical_not(white_mask)

    # print(non_white_mask)

    # Apply K-Means clustering to non-white pixels
    non_white_pixels = row_hsv[non_white_mask]
    # non_white_pixels2 = pixels[non_white_mask]

    dataColor2 = detectColorWithHSV_after_filter(non_white_pixels, t_hue, t_saturation, t_values)
    # print(dataColor2)

    # Step 2: Define the key array (initial centroids)
    initial_centroids = np.array(init_centroid)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(np.float32(non_white_pixels))
    n_clusters  = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)


    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # # Reshape initial_centroids to meet OpenCV's requirements
    # initial_centroids = initial_centroids.reshape(-1, 1, 3)  # Reshape if necessary

    # if(k_value>1):
    #     # Apply K-means using the initial centroids
    #     _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS, initial_centroids)
    # else:
    #     _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # kmeans = KMeans(n_clusters=k_value, init=initial_centroids, n_init=1)
    # kmeans.fit(np.float32(non_white_pixels))

    # cluster_centers = kmeans.cluster_centers_
    # non_white_labels = kmeans.labels_

    # cluster_centers_hsv = []
    # for row in cluster_centers:
    #     # cluster_centers_hsv_values = rgb_to_hsv(row[0], row[1], row[2])
    #     cluster_centers_hsv.append(row[2])

    

    # # Sort cluster centers and labels based on brightness
    # sorted_indices = np.argsort(cluster_centers_hsv)
    # # print(sorted_indices)

    # labels = np.full_like(white_mask, -1, dtype=np.int32).reshape((-1, 1))  # Initialize with -1 or any value outside cluster labels range
    # # labels = labels.reshape((-1)) #activate when using sklearn_kmeans
    # labels[non_white_mask] = non_white_labels
    # labels = labels.reshape((-1)) #activate when using opencv_kmeans

    # # Calculate the average color for each cluster
    average_colors = []
    # cluster_centers_ordered = []
    for cluster_id in range(n_clusters):
        cluster_pixels = non_white_pixels[dbscan_labels == cluster_id]
        average_color = np.mean(cluster_pixels, axis=0)
        r, g, b = colorsys.hsv_to_rgb(average_color[0], average_color[1], average_color[2])
        average_colors.append([r*255, g*255, b*255])

    # cluster_centers_ordered = np.array(cluster_centers_ordered)
    average_colors = np.array(average_colors)

    # print(len(cluster_centers_ordered))
    # print(cluster_centers_ordered)

    show_distribution_graphic_3D_using_HSV_DBSCAN(directory_path_ori_A, row_hsv, non_white_pixels, dbscan_labels, folder_name)

    # Reshape the labels to the shape of the original image
    # Exclude noise label (-1) if present

    unique_labels = set(dbscan_labels)

    labels = np.full_like(white_mask, -1, dtype=np.int32).reshape((-1, 1))  # Initialize with -1 or any value outside cluster labels range
    labels = labels.reshape((-1)) #activate when using sklearn_kmeans, dbscan
    labels[non_white_mask] = dbscan_labels
    # labels = labels.reshape((-1)) #activate when using opencv_kmeans

    
    # if -1 in unique_labels:
    #     unique_labels.remove(-1)

    segmented_image = labels.reshape((1024, 1024))

    num_image = 0

    for cluster_num in unique_labels:
        # if cluster_num not in clusters_to_eliminate:
        if mask_color == False:
            #background black
            standard_mask = np.ones(segmented_image.shape[:2], dtype=np.uint8) * 255  # Initialize with white mask
            standard_mask[segmented_image == cluster_num] = 0  # Set eliminated clusters to black in the mask

            # Save the standard mask
            cv2.imwrite(f'{directory_path_ori_A}{folder_name}\\mask_{cluster_num}.png', standard_mask)
        # else:
            #background black
            color_mask = np.ones((*segmented_image.shape, 3), dtype=np.uint8) * 255
            color_mask[segmented_image == cluster_num] = average_colors[cluster_num].astype(np.uint8)

            # Save the color mask
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving as an image
            cv2.imwrite(f'{directory_path_ori_A}{folder_name}\\color_mask_{cluster_num}.png', color_mask)
        
        num_image += 1

def process_kmeans_and_show_distribution_using_HSV(directory_path, image_A, image_B, dataColor, k_value, folder_name, t_hue, t_saturation, t_values, mask_color = False):

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)

    # Flatten the image to 2D array of pixels (rows = pixels, columns = RGB)
    pixels = image_rgb.reshape((-1, 3))

    num_colors = k_value  # Change this number as needed

    print(dataColor)
    print(num_colors)
    
    # Calculate row means using rgb_to_hsv function
    row_hsv = []
    for row in pixels:
        hsv_values = rgb_to_hsv(row[0], row[1], row[2])
        row_hsv.append(hsv_values)

    # Convert list to numpy array
    row_hsv = np.array(row_hsv)

    # Check similarity to white using is_similar_to_white function
    white_mask = []
    for _row_hsv in row_hsv:
        hsv_values = _row_hsv
        is_white_similar = is_similar_to_white(hsv_values, t_hue, t_saturation, t_values)
        white_mask.append(is_white_similar)

    # Convert list to numpy array
    white_mask = np.array(white_mask)

    # Create a mask for non-white pixels to keep the size same
    non_white_mask = np.logical_not(white_mask)

    # print(non_white_mask)

    # Apply K-Means clustering to non-white pixels
    non_white_pixels = pixels[non_white_mask]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS )
    _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_PP_CENTERS )

    cluster_centers_hsv = []
    for row in cluster_centers:
        cluster_centers_hsv_values = rgb_to_hsv(row[0], row[1], row[2])
        cluster_centers_hsv.append(cluster_centers_hsv_values[2])

    show_distribution_graphic_3D(directory_path, pixels, non_white_pixels, non_white_labels, cluster_centers, folder_name)

    # Sort cluster centers and labels based on brightness
    sorted_indices = np.argsort(cluster_centers_hsv)
    # print(sorted_indices)

    labels = np.full_like(white_mask, -1, dtype=np.int32).reshape((-1, 1))  # Initialize with -1 or any value outside cluster labels range
    labels[non_white_mask] = non_white_labels
    labels = labels.reshape((-1))

    # Calculate the average color for each cluster
    average_colors = []
    for i, idx in enumerate(sorted_indices):
        cluster_pixels = pixels[labels == idx]
        average_color = np.mean(cluster_pixels, axis=0)
        average_colors.append(average_color)

    # Reshape the labels to the shape of the original image
    segmented_image = labels.reshape(image_rgb.shape[:2])

    num_image = 0

    for cluster_num in range(len(sorted_indices)):
        # if cluster_num not in clusters_to_eliminate:
        if mask_color == False:
            #background black
            standard_mask = np.ones(segmented_image.shape[:2], dtype=np.uint8) * 255  # Initialize with white mask
            standard_mask[segmented_image == sorted_indices[cluster_num]] = 0  # Set eliminated clusters to black in the mask

            # Save the standard mask
            cv2.imwrite(f'{directory_path}{folder_name}\\mask_{cluster_num}_{cluster_centers[sorted_indices[cluster_num]]}.png', standard_mask)
        # else:
            #background black
            color_mask = np.ones((*segmented_image.shape, 3), dtype=np.uint8) * 255
            color_mask[segmented_image == sorted_indices[cluster_num]] = average_colors[cluster_num].astype(np.uint8)

            # Save the color mask
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving as an image
            cv2.imwrite(f'{directory_path}{folder_name}\\color_mask_{cluster_num}_{cluster_centers[sorted_indices[cluster_num]]}.png', color_mask)
        
        num_image += 1

def process_mean_shift_and_show_distribution_using_HSV(directory_path, image_A, image_B, dataColor, k_value, folder_name, t_hue, t_saturation, t_values, mask_color = False):

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)

    # Flatten the image to 2D array of pixels (rows = pixels, columns = RGB)
    pixels = image_rgb.reshape((-1, 3))

    num_colors = k_value  # Change this number as needed

    print(dataColor)
    print(num_colors)
    
    # Calculate row means using rgb_to_hsv function
    row_hsv = []
    for row in pixels:
        hsv_values = rgb_to_hsv(row[0], row[1], row[2])
        row_hsv.append(hsv_values)

    # Convert list to numpy array
    row_hsv = np.array(row_hsv)

    # Check similarity to white using is_similar_to_white function
    white_mask = []
    for _row_hsv in row_hsv:
        hsv_values = _row_hsv
        is_white_similar = is_similar_to_white(hsv_values, t_hue, t_saturation, t_values)
        white_mask.append(is_white_similar)

    # Convert list to numpy array
    white_mask = np.array(white_mask)

    # Create a mask for non-white pixels to keep the size same
    non_white_mask = np.logical_not(white_mask)

    # print(non_white_mask)

    # Apply Mean-Shift clustering to non-white pixels
    non_white_pixels = pixels[non_white_mask].astype(np.float32)

    # Apply Mean-Shift clustering
    mean_shift = MeanShift(bandwidth=None)  # You can specify bandwidth if needed
    mean_shift.fit(non_white_pixels)
    non_white_labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_
    print(len(cluster_centers))

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # # _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS )
    # _, non_white_labels, cluster_centers = cv2.kmeans(np.float32(non_white_pixels), k_value, None, criteria, 10, cv2.KMEANS_PP_CENTERS )

    cluster_centers_hsv = []
    for row in cluster_centers:
        cluster_centers_hsv_values = rgb_to_hsv(row[0], row[1], row[2])
        cluster_centers_hsv.append(cluster_centers_hsv_values[2])

    show_distribution_graphic_3D_mean_shift(directory_path, pixels, non_white_pixels, non_white_labels, cluster_centers, folder_name)

    # Sort cluster centers and labels based on brightness
    sorted_indices = np.argsort(cluster_centers_hsv)
    # print(sorted_indices)

    labels = np.full_like(white_mask, -1, dtype=np.int32).reshape((-1, 1))  # Initialize with -1 or any value outside cluster labels range
    
    labels = labels.reshape((-1))
    labels[non_white_mask] = non_white_labels.ravel()

    # Calculate the average color for each cluster
    average_colors = []
    for i, idx in enumerate(sorted_indices):
        cluster_pixels = pixels[labels == idx]
        average_color = np.mean(cluster_pixels, axis=0)
        average_colors.append(average_color)

    # Reshape the labels to the shape of the original image
    segmented_image = labels.reshape(image_rgb.shape[:2])

    num_image = 0

    for cluster_num in range(len(sorted_indices)):
        # if cluster_num not in clusters_to_eliminate:
        if mask_color == False:
            #background black
            standard_mask = np.ones(segmented_image.shape[:2], dtype=np.uint8) * 255  # Initialize with white mask
            standard_mask[segmented_image == sorted_indices[cluster_num]] = 0  # Set eliminated clusters to black in the mask

            # Save the standard mask
            cv2.imwrite(f'{directory_path}{folder_name}\\mask_{cluster_num}_{cluster_centers[sorted_indices[cluster_num]]}.png', standard_mask)
        # else:
            #background black
            color_mask = np.ones((*segmented_image.shape, 3), dtype=np.uint8) * 255
            color_mask[segmented_image == sorted_indices[cluster_num]] = average_colors[cluster_num].astype(np.uint8)

            # Save the color mask
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving as an image
            cv2.imwrite(f'{directory_path}{folder_name}\\color_mask_{cluster_num}_{cluster_centers[sorted_indices[cluster_num]]}.png', color_mask)
        
        num_image += 1


if __name__ == "__main__":

    # threshold = 230
    t_hue = 0 #not using
    t_saturation = 0.073
    t_values = 0.879
    # t_saturation = 0.27 #(<=)
    # t_values = 1.94 #(>=)
    # epoch_list = [10, 50, 100, 150, 200, 250] #500
    epoch_list = [100]#, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100] #500

    # method = "Mean_Shift"
    method = "kmeans"
    # method = "dbscan"

    phase = "train"
    model = "trainA2trainB_1024p_aug_scen_1_newDS_conference"

    for epoch in epoch_list:
        # generated_directory = f"D:\\JupyterLab\\Demo\\pix2pixHD\\results\\trainA2trainB_1024p_no_aug_spatial_attention\\{method}\\{phase}_{epoch}_check\\images"
        generated_directory = f"D:\\JupyterLab\\Demo\\Kmeans_SEAPix\\results\\{model}\\{method}\\{phase}_{epoch}\\images"
        source_directory = f"C:\\Baseline_Dataset\\dataset_fix\\new_data_80_20_sharp_color_fold_1024_aug_scenario1_newDS_conference\\{phase}_A\\"
        source_directory_to_get_init_color = f"C:\\Baseline_Dataset\\dataset_fix\\new_data_80_20_sharp_color_fold_1024_aug_scenario1_newDS_conference\\{phase}_A\\"
        directory_path = f"D:\\JupyterLab\\Demo\\Kmeans_SEAPix\\results\\{model}\\{method}\\{phase}_{epoch}\\"
        # directory_path = f"D:\\JupyterLab\\Demo\\pix2pixHD\\results\\trainA2trainB_1024p_no_aug_spatial_attention\\{method}\\{phase}_{epoch}_check\\"
            
        create_dir(directory_path+"mask_generated")
        directory_path = directory_path+"mask_generated\\"

        print(epoch)

        for filename_B in os.listdir(generated_directory):
            pattern = r'.*_synthesized_image\.png$'  # Adjusted regex pattern to match 'synthesized_image'
            match = re.match(pattern, filename_B)
            if match:
                #sample filename = "AL_1-11_LIU0579_0.png"
                print(filename_B)

                #no_augmentation
                # parts = filename_B.split('_')
                # new_filename_A = '_'.join(parts[1:4])  # Extract "1-11_LIU0579_0.png" by joining parts starting from index 1
                # folder_name = '_'.join(parts[1:3])  # Extract "LIU0579" which is at index 2
                # num_color = parts[1].split("-")[1]

                #with augmentation
                parts = filename_B.split('_')
                new_filename_A = '_'.join(parts[1:4])  # Extract "1-11_LIU0579_0.png" by joining parts starting from index 1
                folder_name_create = '_'.join(parts[1:4])  # Extract "LIU0579" which is at index 2
                folder_name = '_'.join(parts[1:3])  # Extract "LIU0579" which is at index 2
                no_aug = parts[3]
                num_color = parts[1].split("-")[1]

                # Full path for the new directory
                full_path = os.path.join(directory_path, folder_name_create)
                # full_path = os.path.join(directory_path_ori_A, folder_name)

                # Check if the directory exists
                if not os.path.exists(full_path):
                    # image_B = cv2.imread('AL_1-11_LIU0579_0.png')  # Replace with your generated image file
                    image_B_gen = cv2.imdecode(np.fromfile(os.path.join(generated_directory, filename_B), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    # image_B_gen = cv2.imdecode(np.fromfile(os.path.join(source_directory, "A_"+ new_filename_A + ".png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # Replace with your image file
                    image_A_ori = cv2.imdecode(np.fromfile(os.path.join(source_directory, "A_"+ new_filename_A + ".png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # Replace with your image file
                    image_A_ori_to_get_init_color = cv2.imdecode(np.fromfile(os.path.join(source_directory_to_get_init_color, "A_"+ new_filename_A + ".png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)  # Replace with your image file

                    image_A_ori = cv2.cvtColor(image_A_ori, cv2.COLOR_BGR2RGB)
                    image_B_gen = cv2.cvtColor(image_B_gen, cv2.COLOR_BGR2RGB)
                    image_A_to_get_init_color_hsv = convertRGBtoHSV(image_A_ori_to_get_init_color)
                    image_A_to_get_init_color_hsv = np.array(image_A_to_get_init_color_hsv)

                    image_B_hsv = convertRGBtoHSV(image_B_gen)
                    image_B_hsv = np.array(image_B_hsv)

                    init_centroid = detectColorFromHSV(image_A_to_get_init_color_hsv, t_hue, t_saturation, t_values)

                    print(folder_name_create)
                    
                    k_value = int(num_color)
                
                    create_dir(full_path)
                    
                    if method == "kmeans":
                        process_kmeans_and_show_distribution_using_HSV_not_pixel(directory_path, image_A_ori, image_B_hsv, init_centroid, k_value, folder_name, t_hue, t_saturation, t_values, no_aug, False)
                    elif method == "dbscan":
                        process_DBSCAN_and_show_distribution_using_HSV_original_not_pixel(directory_path, image_A_ori, image_B_hsv, init_centroid, k_value, folder_name, t_hue, t_saturation, t_values, False)

                else:
                    print(f"Skip: {folder_name}")
        
        print("done!")
# 