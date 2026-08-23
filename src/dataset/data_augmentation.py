from tensorflow.keras import layers, Sequential

# Data Augmentation layers
data_augmentation = Sequential([
    # Geometric
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.083), # +-30 degrees 
    
    # Intensity
    layers.RandomContrast(0.09),
    layers.RandomBrightness(0.2, value_range=(0, 255))
])

# Data augmentation function
def add_data_augmentation(image_tensor):
    augmented_image = data_augmentation(image_tensor, training=True)
    return augmented_image