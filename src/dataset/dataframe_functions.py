import os
import pandas as pd

# Function to generate image preprocessing results dataframe
def generate_image_preprocessing_results_df(model_name, preprocessor, eval_metrics, specificity, f1_score):
  results_df = pd.DataFrame({
    'model': [model_name],
    'image_preprocessing_pipeline': [preprocessor],
    'accuracy': [eval_metrics['accuracy']],
    'auc': [eval_metrics['auc']],
    'sensitivity': [eval_metrics['recall']],
    'precision': [eval_metrics['precision']],
    'specificity': [specificity],
    'f1_score': [f1_score]
  })
  return results_df

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

# Function to generate model-specific list for image preprocessing experiment for the given metric
def generate_metric_list_for_image_experiment_results(model_name, baseline_df, image_preprocess_df, metric_name):
 
  # Baseline metric value
  baseline_metric_value = baseline_df[baseline_df['model'] == model_name][metric_name]

  # Experiment1 metric value (CLAHE + Median blur)
  clahe_median_blur_metric_value = image_preprocess_df[
    (image_preprocess_df['model'] == model_name) 
    & (image_preprocess_df['image_preprocessing_pipeline'] == 'CLAHE + Median Blur')][metric_name]

  # Experiment 2 metric value (Histogram Equalization + Gaussian Blur)
  histo_equalized_guass_blur_metric_value = [
    (image_preprocess_df['model'] == model_name) 
    & (image_preprocess_df['image_preprocessing_pipeline'] == 'Histogram Equalization + Gaussian Blur')][metric_name]

  return [baseline_metric_value, clahe_median_blur_metric_value, histo_equalized_guass_blur_metric_value]