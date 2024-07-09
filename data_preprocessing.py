import h5py
import os
import cv2
import numpy as np
import tensorflow as tf

# Define the path to your Downloads folder and the .mat file
desktop_folder = '/Users/benwalsh/Desktop'
mat_file = 'machine_learning/SVHN-data/train/digitStruct.mat'
images_folder = '/Users/benwalsh/Desktop/machine_learning/SVHN-data/train'

# Combine the paths to create the full path to the .mat file
file_path = os.path.join(desktop_folder, mat_file)

# Target size for resizing images
target_size = (640, 640)

# Function to extract string from h5py object
def read_string(string):
    # Convert byte array to string
    return ''.join(chr(c[0]) for c in string if c != 0)

# Function to extract bounding box data
def extract_bbox(f, bbox_ref):
    bbox_data = {}
    for key in f[bbox_ref].keys():
        # Handle the case where the value is a reference to an array of values
        values = f[bbox_ref][key]
        if len(values) > 1:
            bbox_data[key] = [f[values[i][0]][()][0][0] for i in range(len(values))]
        else:
            bbox_data[key] = [values[0][0]]
    return bbox_data

# Function to resize bounding boxes
def resize_bbox(bbox, original_size, target_size):
    x_scale = target_size[1] / original_size[1]
    y_scale = target_size[0] / original_size[0]

    resized_bbox = {}
    
    resized_bbox['height'] = [y * y_scale for y in bbox['height']]
    resized_bbox['width'] = [x * x_scale for x in bbox['width']]
    resized_bbox['left'] = [x * x_scale for x in bbox['left']]
    resized_bbox['top'] = [y * y_scale for y in bbox['top']]
    
    return resized_bbox

# Function to lead the .mat file using h5py
def load_dataset():
    train_set_X = tf.zeros((33402, 640, 640, 3), dtype=tf.uint8)
    train_set_Y = tf.zeros((19, 19, 30))
    with h5py.File(file_path, 'r') as f:
        digitStruct = f['digitStruct']
        names = digitStruct['name']
        bboxes = digitStruct['bbox']

        for i in range(len(names)):
            # Get the file name
            name_ref = names[i][0]
            image_name = f[name_ref][()]
            image_name =  read_string(image_name) 

            image_path = os.path.join(images_folder, image_name)

            # Load image
            image = cv2.imread(image_path)
            original_size = image.shape[:2] # (height, width)

            # Resize image
            resized_image = cv2.resize(image, target_size)
            resized_image_path = os.path.join(images_folder, f"resized_{image_name}")
            if i == 0:
                cv2.imwrite(resized_image_path, resized_image)
            train_set_X[i] = resized_image

            # Get bounding box data
            bbox_ref = bboxes[i][0]
            bbox_data = extract_bbox(f, bbox_ref)

            # Resize bounding box data
            resized_bbox_data = resize_bbox(bbox_data, original_size, target_size)

            if i%1000 == 0:
                print(f"Processed {image_name}")
                print("Original bbox data:", bbox_data)
                print("Resized bbox data:", resized_bbox_data)
    return train_set_X, train_set_Y