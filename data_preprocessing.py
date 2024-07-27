import h5py
import os
import cv2
import numpy as np
import tensorflow as tf

# Function to extract string from h5py object
def read_string(string):
    # Convert byte array to string
    return ''.join(chr(c[0]) for c in string if c != 0)

# Function to extract a list of dictionaries representing
# the bounding box information for a particular image
def extract_bboxes(f, bbox_ref):
    bbox = f[bbox_ref]
    bboxes = []
    for i in range(len(bbox['height'])):
        bboxes.append(extract_bbox(f, bbox_ref, i))
    return bboxes

# Function to extract bounding box data
def extract_bbox(f, bbox_ref, i):
    bbox_data = {}
    for key in f[bbox_ref].keys():
        # Handle the case where the value is a reference to an array of values
        values = f[bbox_ref][key]
        if len(values) > 1:
            bbox_data[key] = f[values[i][0]][()][0][0]
        else:
            bbox_data[key] = values[0][0]
    return bbox_data

# Function to resize bounding boxes
def resize_bbox(bbox, original_size, target_size):
    x_scale = target_size[1] / original_size[1]
    y_scale = target_size[0] / original_size[0]

    resized_bbox = {}
    
    resized_bbox['height'] = y_scale * bbox['height']
    resized_bbox['width'] = x_scale * bbox['width']
    resized_bbox['left'] = x_scale * bbox['left']
    resized_bbox['top'] = y_scale * bbox['top']
    resized_bbox['label'] = bbox['label']
    
    return resized_bbox

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    Each box is defined by [x_min, y_min, x_max, y_max].
    """
    x_min1, y_min1, x_max1, y_max1 = box1[0], box1[1], box1[2], box1[3]
    x_min2, y_min2, x_max2, y_max2 = box2[0], box2[1], box2[2], box2[3]

    # Calculate intersection
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Calculate union
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def find_best_anchor(bbox, anchor_boxes, grid_center):
    """
    Find the best anchor box based on the highest IoU with the given bounding box.
    """
    bbox_width, bbox_height = bbox['width'], bbox['height']
    bbox_x_min, bbox_y_min = bbox['left'], bbox['top']
    bbox_x_max, bbox_y_max = bbox_x_min + bbox_width, bbox_y_min + bbox_height
    bbox_coords = [bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]

    grid_center_x, grid_center_y = grid_center

    best_anchor_idx = 0
    best_iou = 0

    for idx, (anchor_w, anchor_h) in enumerate(anchor_boxes):
        anchor_x_min = grid_center_x - anchor_w / 2
        anchor_y_min = grid_center_y - anchor_h / 2
        anchor_x_max = grid_center_x + anchor_w / 2
        anchor_y_max = grid_center_y + anchor_h / 2
        anchor_coords = [anchor_x_min, anchor_y_min, anchor_x_max, anchor_y_max]

        iou = calculate_iou(bbox_coords, anchor_coords)
        if iou > best_iou:
            best_iou = iou
            best_anchor_idx = idx

    return best_anchor_idx


def get_grid_position(bbox, target_size):
    # Determine the grid position for the bounding box center
    grid_size = 19  # Adjust according to your grid size
    target_h, target_w = target_size
    center_x = bbox['left'] + bbox['width'] / 2
    center_y = bbox['top'] + bbox['height'] / 2

    grid_x = int(center_x / target_w * grid_size)
    grid_y = int(center_y / target_h * grid_size)

    return grid_x, grid_y

def normalize_bbox(resized_bbox_data, grid_x, grid_y, grid_size, img_size):
    # Calculate grid cell size
    grid_width = img_size[0] / grid_size[0]
    grid_height = img_size[1] / grid_size[1]
    
    # Calculate offsets for normalization
    left_x_of_grid_cell = grid_x * grid_width
    top_y_of_grid_cell = grid_y * grid_height

    # Calculate normalized values
    normalized_x = (resized_bbox_data['left'] + resized_bbox_data['width'] / 2 - left_x_of_grid_cell) / grid_width
    normalized_y = (resized_bbox_data['top'] + resized_bbox_data['height'] / 2 - top_y_of_grid_cell) / grid_height
    normalized_w = resized_bbox_data['width'] / img_size[0]
    normalized_h = resized_bbox_data['height'] / img_size[1]

    return normalized_x, normalized_y, normalized_w, normalized_h

# Function to lead the .mat file using h5py
# Note: this is only for training set currently
def load_dataset(file_path, images_folder, target_size=(640, 640)):
    num_samples = 33402
    train_set_X = np.zeros((num_samples, 640, 640, 3), dtype=np.uint8)
    train_set_Y = np.zeros((num_samples, 19, 19, 30))
    # anchor boxes can be improved with K means ...
    # still need to look into what that means & how to do it
    anchor_boxes = [
        [80, 160],  # Tall and thin box (width, height)
        [160, 80]   # Short and wide box (width, height)
    ]
    # open the file with h5py
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
            
            # Assign resized image to location in train_set_X tensor
            train_set_X[i] = resized_image

            # Get bounding box data
            bbox_ref = bboxes[i][0]
            bbox_data_list = extract_bboxes(f, bbox_ref)
            
            # Process each bounding box
            for bbox_data in bbox_data_list:
                # Resize bounding box data
                resized_bbox_data = resize_bbox(bbox_data, original_size, target_size)
                
                # Assign bounding boxes to anchor boxes and store in train_set_Y
                grid_x, grid_y = get_grid_position(resized_bbox_data, target_size)
                best_anchor_idx = find_best_anchor(resized_bbox_data, anchor_boxes)

                # grab normalized values
                normalized_x, normalized_y, normalized_w, normalized_h = normalize_bbox(resized_bbox_data, grid_x, grid_y, (19, 19), (640, 640))

                # extra label from bbox_data
                label = resized_bbox_data['label']
                train_set_Y[i, grid_x, grid_y, best_anchor_idx * 15:(best_anchor_idx + 1) * 15] = [
                    1,  # object confidence
                    normalized_x, normalized_y, normalized_w, normalized_h,
                    int(label==10), int(label==1), int(label==2), int(label==3), int(label==4), int(label==5), int(label==6),
                    int(label==7), int(label==8), int(label==9)
                ]

            if i%1000 == 0:
                print(f"Processed {image_name}")
                print("Original bbox data:", bbox_data)
                print("Resized bbox data:", resized_bbox_data)

    train_set_X = tf.convert_to_tensor(train_set_X, dtype=tf.uint8)
    train_set_Y = tf.convert_to_tensor(train_set_Y)
    
    return train_set_X, train_set_Y

# Define the path to your Downloads folder and the .mat file
desktop_folder = '/Users/benwalsh/Desktop'
mat_file = 'machine_learning/SVHN-data/train/digitStruct.mat'
images_folder = '/Users/benwalsh/Desktop/machine_learning/SVHN-data/train'

# Combine the paths to create the full path to the .mat file
file_path = os.path.join(desktop_folder, mat_file)

train_set_X, train_set_Y = load_dataset(file_path, images_folder)