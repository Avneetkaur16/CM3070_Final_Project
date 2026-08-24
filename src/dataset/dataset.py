import numpy as np
import tensorflow as tf
from src.dataset.data_augmentation import add_data_augmentation

# Load and preprocess image path tensors using the selected preprocessor function for a given image path
def load_and_preprocess_images(image_path_tensor, preprocessor):
  # tf.numpy_function converts image_path_tensor to numpy object which is used by image preprocessing function (wrapper) and the processed resulting image
  # is converted back to an image tensor of float32
  image_tensor = tf.numpy_function(preprocessor, [image_path_tensor], tf.float32)

  # Prevent shape loss
  image_tensor.set_shape((224, 224, 3))
  return image_tensor

# Shuffled the training dataset based on the buffer size = length of the dataframe (For training set only)
def shuffle_training_dataset(dataset, buffer_size):
  shuffled_dataset = dataset.shuffle(buffer_size)
  return shuffled_dataset

# Adds data augmentation to the images of the dataset provided (For Training set only)
def data_augmented_dataset(dataset):
  augmented_data = dataset.map(lambda x, y: (add_data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
  return augmented_data

# Batch the dataset based on the batch size
def batch_dataset(dataset, batch_size):
  batched_dataset = dataset.batch(batch_size)
  prefetched_dataset = batched_dataset.prefetch(tf.data.AUTOTUNE)
  return prefetched_dataset

# Generate a tf.data dataset using a dataframe
def generate_dataset(df, preprocessor):
  # Extract image paths and pathologies from the dataframe
  image_paths = df['new_image_path'].values
  pathologies = df['pathology'].values

  # Create a tf.data Dataset using image paths and pathologies
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, pathologies))

  # Map image tensors to pathology labels using load_and_preprocess_images method
  dataset = dataset.map(lambda x, y: (load_and_preprocess_images(x, preprocessor), y) , num_parallel_calls=tf.data.AUTOTUNE)

  return dataset

def generate_class_weight_dict(df, classes, total_samples):
  # Count the no. of samples with 0.0 and 1.0 pathologies
  samples_count_0 = len(df[df['pathology'] == 0.0])
  samples_count_1 = len(df[df['pathology'] == 1.0])

  # Compute class weights for 0.0 and 1.0 pathology classes
  weighted_0 = total_samples / (classes * samples_count_0)
  weighted_1 = total_samples / (classes * samples_count_1)

  # Create a class weights dictionary with class weights for 0.0 and 1.0 pathologies
  class_weights_dict = {
    0: weighted_0,
    1: weighted_1
  }

  return class_weights_dict