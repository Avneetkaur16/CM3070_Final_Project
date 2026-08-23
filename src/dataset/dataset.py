import numpy as np
import tensorflow as tf
from src.dataset.data_augmentation import add_data_augmentation

# Load and preprocess image path tensors using the selected preprocessor function for a given image path
def load_and_preprocess_images(image_path_tensor, preprocessor, model_type):
  # tf.numpy_function converts image_path_tensor to numpy object which is used by image preprocessing function (wrapper) and the processed resulting image
  # is converted back to an image tensor of float32
  image_tensor = tf.numpy_function(preprocessor, [image_path_tensor, model_type], tf.float32)

  # Prevent shape loss
  image_tensor.set_shape((224, 224, 3))
  return image_tensor

# Generate a tf.data dataset using a dataframe
def generate_dataset(df, batch_size, preprocessor, model_type, training=False):
  # Extract image paths and pathologies from the dataframe
  image_paths = df['new_image_path'].values
  pathologies = df['pathology'].values

  # Create a tf.data Dataset using image paths and pathologies
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, pathologies))

  # Map image tensors to pathology labels using load_and_preprocess_images method
  dataset = dataset.map(lambda x, y: (load_and_preprocess_images(x, preprocessor, model_type), y) , num_parallel_calls=tf.data.AUTOTUNE)

  # Data augmentation and shuffling for training data
  if(training):
    dataset = dataset.map(lambda x, y: (add_data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(df))

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset