from tensorflow.keras import layers, Sequential

# Data Augmentation layers
data_augmentation = Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.0138), # +- 5 degrees
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05)
])

# Data augmentation method
def add_data_augmentation(image_tensor):
    augmented_image = data_augmentation(image_tensor, training=True)
    return augmented_image