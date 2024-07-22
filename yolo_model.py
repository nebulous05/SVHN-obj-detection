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

def yolo_loss(y_true, y_pred, grid_size, num_classes):
    """
    YOLO loss function that only includes bbox_loss and class_loss when object confidence is 1.

    Args:
    - y_true: Tensor of shape (batch_size, grid_size, grid_size, num_anchors * (5 + num_classes))
    - y_pred: Tensor of shape (batch_size, grid_size, grid_size, num_anchors * (5 + num_classes))
    - grid_size: Tuple of integers (height, width) representing the grid size (e.g., (19, 19))
    - num_classes: Integer representing the number of classes

    Returns:
    - total_loss: Tensor representing the total loss
    """
    
    # Unpack ground truth and predictions
    object_mask = y_true[..., 0]
    true_bbox = y_true[..., 1:5]
    true_class = y_true[..., 5:]

    pred_object = y_pred[..., 0]
    pred_bbox = y_pred[..., 1:5]
    pred_class = y_pred[..., 5:]

    # Objectness Loss: Binary cross-entropy between objectness scores
    objectness_loss = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(object_mask, pred_object)
    )

    # Bounding Box Loss: Mean squared error between predicted and true bounding box coordinates
    bbox_loss = tf.reduce_sum(
        object_mask * tf.reduce_sum(tf.square(true_bbox - pred_bbox), axis=-1)
    )

    # Class Loss: Binary cross-entropy between class predictions
    class_loss = tf.reduce_sum(
        object_mask * tf.keras.losses.binary_crossentropy(true_class, pred_class)
    )

    # Total loss: Combining all the losses
    total_loss = objectness_loss + bbox_loss + class_loss

    return total_loss

model = create_yolo_model()
model.summary()

model.compile(optimizer='adam', loss=lambda y_true, y_pred: yolo_loss(y_true, y_pred, grid_size=(19, 19), num_classes=10))

train_set_X = tf.zeros((33402, 640, 640, 3))
train_set_Y = tf.zeros((33402, 19, 19, 30))

# Assuming train_set_X and train_set_Y are your input and target tensors
history = model.fit(
    train_set_X, train_set_Y,
    epochs=10,  # Adjust as needed
    batch_size=32,  # Adjust as needed
    validation_split=0.1  # Adjust as needed
)
