import pandas as pd
import tensorflow as tf

# CNN model compiler function
def compile_model(model):
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
    metrics=[
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
  ])
  return model

# Function to update root image path of training images
def update_image_path_training(image_path):
  return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_train_data')

# Function to update root image path of validation images
def update_image_path_validation(image_path):
  return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_val_data')

# Function to update root image path of testing images
def update_image_path_testing(image_path):
  return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_test_data')

# Function to compute f1 score from precision and recall
def compute_f1_score(precision, recall):
  f1_score = 0.0

  if(precision + recall != 0.0):
    f1_score = (2 * (precision * recall)) / (precision + recall)

  return f1_score

# Function to compute specificity using true negatives and false positives
def compute_specificity(tn, fp):
  return tn / (tn + fp)

# Function to generate a results dataframe
def generate_results_df(model_name, view, pipeline, eval_metrics, specificity, f1, 
                        training_time, prediction_time, peak_memory, model_params):
  results = pd.DataFrame({
    'model': [model_name],
    'view': [view],
    'image_preprocessing_pipeline': [pipeline],
    'accuracy': [eval_metrics['accuracy']],
    'auc': [eval_metrics['auc']],
    'sensitivity': [eval_metrics['recall']],
    'precision': [eval_metrics['precision']],
    'specificity': [specificity],
    'f1_score': [f1],
    'training_time_mins': [training_time / 60.0],
    'prediction_time_sec': [prediction_time],
    'peak_memory_used_MB': [peak_memory / (1024**2)],
    'parameters': [model_params]
  })

  return results