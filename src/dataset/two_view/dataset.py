import tensorflow as tf
from data_augmentation import data_augmentation

# Merge CC and MLO views of each patient
def merge_views_by_patient(df):
  cc_df = df[df['image view'] == 'CC']
  mlo_df = df[df['image view'] == 'MLO']

  # Merge CC and MLO views by patient_id
  merged_df = cc_df.merge(mlo_df, on='patient_id', how='inner', suffixes=('_cc', '_mlo'))

  # Assign a common pathology based on both views
  merged_df['pathology'] = merged_df.apply(lambda row: 1.0 if row['pathology_cc'] == 1.0 or row['pathology_mlo'] == 1.0 else 0.0, axis=1)

  return merged_df

# Load and preprocess the image path tensors using the selected preprocessors for both CC and MLO views
def load_and_preprocess_images_two_view(image_paths_tensor, preprocessor):
  # Generate CC and MLO view image tensors
  cc_image_tensor = tf.numpy_function(preprocessor, [image_paths_tensor[0]], tf.float32)
  mlo_image_tensor = tf.numpy_function(preprocessor, [image_paths_tensor[1]], tf.float32)

  # Set the shape of CC and MLO image tensors
  cc_image_tensor.set_shape((224, 224, 3))
  mlo_image_tensor.set_shape((224, 224, 3))

  return (cc_image_tensor, mlo_image_tensor)

# Data augmentation method for CC and MLO views
def add_data_augmentation_two_view(image_tensors):
  cc_augmented_image = data_augmentation(image_tensors[0], training=True)
  mlo_augmented_image = data_augmentation(image_tensors[1], training=True)
  return (cc_augmented_image, mlo_augmented_image)

# VGG16 Preprocessing for CC and MLO views separately
def vgg16_preprocess_two_view(image_tensors):
  cc_image_tensor = tf.keras.applications.vgg16.preprocess_input(image_tensors[0])
  mlo_image_tensor = tf.keras.applications.vgg16.preprocess_input(image_tensors[1])
  return (cc_image_tensor, mlo_image_tensor)

# ResNet50 Preprocessing for CC and MLO views separately
def resnet50_preprocess_two_view(image_tensors):
  cc_image_tensor = tf.keras.applications.resnet50.preprocess_input(image_tensors[0])
  mlo_image_tensor = tf.keras.applications.resnet50.preprocess_input(image_tensors[1])
  return (cc_image_tensor, mlo_image_tensor)

# Generate a dataset of CC and MLO views for two-view models
def generate_two_view_dataset(df, batch_size, preprocessor, model_type, training=False):
  # Extract all CC and MLO views with their common pathology label
  image_paths_cc = df['new_image_path_cc'].values
  image_paths_mlo = df['new_image_path_mlo'].values
  pathologies = df['pathology'].values

  # Create a tf.data dataset with CC and MLO views and their common pathologies
  dataset = tf.data.Dataset.from_tensor_slices(((image_paths_cc, image_paths_mlo), pathologies))

  # Load CC and MLO views and preprocess them using the selected preprocessor
  dataset = dataset.map(lambda x, y: (load_and_preprocess_images_two_view(x, preprocessor), y), num_parallel_calls=tf.data.AUTOTUNE)

  if(training):
    dataset = dataset.map(lambda x, y: (add_data_augmentation_two_view(x), y), num_parallel_calls=tf.data.AUTOTUNE)

  # VGG16
  if(model_type == 'vgg16'):
    dataset = dataset.map(lambda x, y: (vgg16_preprocess_two_view(x), y), num_parallel_calls=tf.data.AUTOTUNE)

  # ResNet50
  elif(model_type == 'resnet50'):
    dataset = dataset.map(lambda x, y: (resnet50_preprocess_two_view(x), y), num_parallel_calls=tf.data.AUTOTUNE)

  dataset = dataset.shuffle(buffer_size=len(df))
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset