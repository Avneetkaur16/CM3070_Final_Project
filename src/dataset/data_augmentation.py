from tensorflow.keras import layers, Sequential

# Data Augmentation layers
data_augmentation = Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.0138), # +- 5 degrees
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05)
])

# Data augmentation function
def add_data_augmentation(image_tensor):
    augmented_image = data_augmentation(image_tensor, training=True)
    return augmented_image

# Data augmentation function for CC and MLO views for two-view training 
def add_data_augmentation_two_view(image_tensors):
  cc_augmented_image = data_augmentation(image_tensors[0], training=True)
  mlo_augmented_image = data_augmentation(image_tensors[1], training=True)
  return (cc_augmented_image, mlo_augmented_image)