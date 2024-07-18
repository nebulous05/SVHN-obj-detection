## Information about the SVHN dataset:
- training set consists of 33,402 images in the form of png files
- labels are given in the dataStruct.mat file
- for each image in the dataset, we are provided with the following labels:
    - **'height'**: the height of the bounding box
    - **'width'** : the width of the bounding box
    - **'label'** : the class label of the digits (1-10, where 10 represents the digit '0')
    - **'top'**   : the y-coordinate of the top-left corner of the bounding box
    - **'left'**  : the x-coordinate of the top-left corner of the bounding box
    - Note that each of the keys map to an array of floats corresponding to each digit present in the image
