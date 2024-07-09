import tensorflow as tf
from tensorflow.keras import layers, models

def create_yolo_model(input_shape=(640, 640, 3), grid_size=19, num_anchors=2, num_classes=10):
    # Number of filters in the output layer
    output_filters = num_anchors * (5 + num_classes)

    inputs = tf.keras.Input(shape=input_shape)

    # Define the convolutional layers
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # 320x320x32

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 160x160x64

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 80x80x128

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 40x40x256

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 20x20x512

    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)

    # Add more layers as needed to further reduce the spatial dimensions
    # to achieve the desired grid size of 19x19

    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 10x10x1024

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(output_filters, (1, 1), padding='same', activation='linear')(x)  # 10x10x30

    # Resize the final feature map to 19x19 if needed
    x = layers.UpSampling2D((2, 2))(x)  # 20x20x30

    # Assuming the final desired output is 19x19
    outputs = layers.Cropping2D(((1, 0), (1, 0)))(x)  # 19x19x30

    model = models.Model(inputs, outputs)
    return model

# Create the YOLO model
yolo_model = create_yolo_model()

# Print the model summary
yolo_model.summary()
