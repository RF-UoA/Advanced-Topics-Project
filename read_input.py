import numpy as np
import pandas as pd
import ast
import re
import math
import os

def read_txt_to_dataframe(file_path):
    # Read the text file into a DataFrame
    df = pd.read_csv(file_path, sep=',')
    
    return df

def read_csv_to_dataframe(file_path):
    # Define a conversion function to interpret string representation of lists
    def convert_to_list(value):
        return ast.literal_eval(value)
    
    def convert_to_array_of_arrays(value):
        # Remove the word 'array' from the string
        value = re.sub(r'array\(', '', value)
        value = re.sub(r'\)', '', value)
        
        # Use ast.literal_eval to safely evaluate the string
        list_of_arrays = ast.literal_eval(value)
        
        # Convert each element to a numpy array
        return [np.array(arr) for arr in list_of_arrays]
    
    def convert_space_separated_list(value):
        # Split the string by spaces and convert to a list of floats
        value = value.strip('[]')
        return [float(x) for x in value.split()]
    
    # Read the csv file into a DataFrame, applying the conversion function to relevant columns
    df = pd.read_csv(file_path, converters={
        'Crater Indices': convert_to_list,
        'Centre points 2D coord': convert_to_array_of_arrays,
        'Camera Position': convert_space_separated_list
    })

    return df

def parse_attitudes(attitudes):
    """
    Converts a list of attitude strings into a list of NumPy arrays.
    
    Args:
        attitudes (list of str): List of strings representing 3x4 extrinsic matrices.
        
    Returns:
        list of np.ndarray: List of 3x4 NumPy arrays.
    """
    matrices = []
    for attitude in attitudes:
        # Remove all brackets and split into rows
        rows = attitude.replace('[', '').replace(']', '').split('\n')
        # Convert each row into a list of floats
        matrix = np.array([list(map(float, row.split())) for row in rows if row.strip()])
        matrices.append(matrix)
    return matrices

def get_intrinsics(fov, im_width, im_height):
    fov = fov*math.pi/180
    fx = im_width/(2*math.tan(fov)) # Conversion from fov to focal length
    fy = im_height/(2*math.tan(fov)) # Conversion from fov to focal length
    cx = im_width/2
    cy = im_height/2
    return (np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]))

def get_correspondences():
    # Image dimensions and FOV
    image_width = 1024
    image_height = 1024
    fov = 30  # Field of view in degrees

    # Calculate the focal length
    focal_length = image_width / (2 * np.tan(np.radians(fov) / 2))

    # Create the initial intrinsic matrix
    camera_matrix = get_intrinsics(fov, image_width, image_height)

    camera_matrix = np.array(camera_matrix)
    ground_truth_matrix = camera_matrix

    robbins = read_txt_to_dataframe('data/robbins_navigation_dataset_christians_all.txt')
    t_inst = read_csv_to_dataframe('output/testing_instance.csv')

    w_points_list = []
    i_points_list = []
    attitude_list = []
    ref_vals = []

    for i, (index, row) in enumerate(t_inst.iterrows()):

        i_points_list.append(np.vstack(t_inst['Centre points 2D coord'][i]))
        ref_vals.append(t_inst['Camera Position'][i])
        # Read attitude values (camera extrinsic matrix)
        extrinsic_matrix_str = t_inst['Camera Extrinsic'][i]
        attitude_list.append(extrinsic_matrix_str)


        indices = t_inst['Crater Indices'][i]

        x_vals = robbins.iloc[indices][' X']
        y_vals = robbins.iloc[indices][' Y']
        z_vals = robbins.iloc[indices][' Z']

        w_points = robbins.iloc[indices][[' X', ' Y', ' Z']].to_numpy()
        w_points_list.append(w_points)

    # object_points = [obj.reshape(-1, 1, 3).astype(np.float32) for obj in w_points_list]
    # image_points = [img.reshape(-1, 1, 2).astype(np.float32) for img in i_points_list]

    # print("Object Points: ", object_points)
    # print("Image Points: ", image_points)

    return w_points_list, i_points_list, camera_matrix, parse_attitudes(attitude_list)