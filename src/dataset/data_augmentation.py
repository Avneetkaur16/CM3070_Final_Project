from tensorflow.keras import layers, Sequential

# Data Augmentation layers
data_augmentation = Sequential([
    # Geometric
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2), 
    
    # Intensity
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.09)
])

# Data augmentation function
def add_data_augmentation(image_tensor):
    augmented_image = data_augmentation(image_tensor, training=True)
    return augmented_image