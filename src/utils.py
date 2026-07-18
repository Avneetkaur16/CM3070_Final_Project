import os
import pandas as pd

# Function to update root image path of training images
def update_image_path_training(image_path):
  return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_train_data')

# Function to update root image path of validation images
def update_image_path_validation(image_path):
  return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_val_data')

# Function to update root image path of testing images
def update_image_path_testing(image_path):
  return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_test_data')

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
    'specificity': [specificity],
    'f1_score': [f1],
    'training_time_mins': [training_time / 60.0],
    'prediction_time_sec': [prediction_time],
    'peak_memory_used_MB': [peak_memory / (1024**2)],
    'parameters': [model_params]
  })

  return results

# Function to store the image preprocessing experimental results in a csv file in the google drive
def store_image_experiment_results(results_file_path, results_df):
  # Store the results of experiment in the image preprocessing results dataframe
  if not os.path.isfile(results_file_path):
    # First instance of results
    image_preprocessing_results = results_df
  else:  
    # Append to the existing results if this is not the first instance
    image_preprocessing_results = pd.read_csv(results_file_path)
    image_preprocessing_results = pd.concat([image_preprocessing_results, results_df], ignore_index=True)

  # Store the updated dataframe in a csv stored in drive
  image_preprocessing_results.to_csv(results_file_path, index=False)
  print(f"Stored results for {results_df['image_preprocessing_pipeline']} pipeline")

# Function to generate a results data dataframe containing true pathologies, prediction probabilities and prediction pathologies
def generate_results_data_df(model_name, view, pipeline, true_pathology, pred_pathology):
  results_data = pd.DataFrame({
    'model': [model_name],
    'view': [view],
    'image_preprocessing_pipeline': [pipeline],
    'true_pathology': [true_pathology],
    'pred_pathology': [pred_pathology.ravel()]
  })

  return results_data

# Function to store image preprocessing experimental results data in csv file in the google drive
def store_image_preprocessing_results_data(results_data_file_path, results_data_df):
  # Store the results data in the image preprocessing results data dataframe
  if not os.path.isfile(results_data_file_path):
    # First instance of results data
    image_preprocessing_results_data = results_data_df
  else:  
    # Append to the existing results data if this is not the first instance
    image_preprocessing_results_data = pd.read_csv(results_data_file_path)
    image_preprocessing_results_data = pd.concat([image_preprocessing_results_data, results_data_df], ignore_index=True)

  # Store the updated dataframe in a csv stored in drive
  image_preprocessing_results_data.to_csv(results_data_file_path, index=False)
  print(f"Stored results for {results_data_df['image_preprocessing_pipeline']} pipeline")
