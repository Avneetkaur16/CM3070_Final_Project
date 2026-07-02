import tensorflow as tf
from data_augmentation import data_augmentation

def load_and_preprocess_images(image_path_tensor, label_tensor, preprocessor):
  # tf.numpy_function converts image_path_tensor to numpy object which is used by image preprocessing function (wrapper) and the processed resulting image
  # is converted back to an image tensor of float32
  image_tensor = tf.numpy_function(preprocessor, [image_path_tensor], tf.float32)

  # Prevent shape loss
  image_tensor.set_shape((224, 224, 3))
  return image_tensor, label_tensor

# Generate a tf.data dataset using a dataframe
def generate_dataset(df, batch_size, preprocessor, training=False):
  image_paths = df['new_image_path'].values
  pathologies = df['pathology'].values

  # Create a tf.data Dataset using image paths and pathologies
  dataset = tf.data.Dataset.from_tensor_slices((image_paths, pathologies))

  # Map image tensors to pathology labels using load_and_preprocess_images method
  dataset = dataset.map(lambda x, y: load_and_preprocess_images(x, y, preprocessor) , num_parallel_calls=tf.data.AUTOTUNE)

  # Data augmentation
  if(training):
    dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset