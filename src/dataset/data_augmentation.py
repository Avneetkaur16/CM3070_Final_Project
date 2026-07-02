from tensorflow.keras import layers

# Data augmentation layers
random_flipping_layer = layers.RandomFlip('horizontal_and_vertical')
random_rotations_layer = layers.RandomRotation(0.0138) # +- 5 degrees
random_zoom_layer = layers.RandomZoom(0.05)
random_contrast_layer = layers.RandomContrast(0.05)

# Data augmentation function for training data
def data_augmentation(image_tensor, label_tensor):
  image_tensor = random_flipping_layer(image_tensor)
  image_tensor = random_rotations_layer(image_tensor)
  image_tensor = random_zoom_layer(image_tensor)
  image_tensor = random_contrast_layer(image_tensor)
  return image_tensor, label_tensor