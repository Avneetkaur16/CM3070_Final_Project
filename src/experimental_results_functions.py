import pandas as pd
import os

# Function to generate results metrics dataframe
def generate_results_metrics_df(model_name, experimental_value, eval_metrics, experiment_type):
     # Dataframe with metrics data for image preprocessing experiment 
    if(experiment_type == 'image_preprocessing'):
        results_df = pd.DataFrame({
            'model': [model_name],
            'image_preprocessing_pipeline': [experimental_value],
            'view_type': 'Unified',
            'lesion_type': 'Unified',
            'pr_auc': [eval_metrics['pr_auc']],
            'sensitivity': [eval_metrics['sensitivity']],
            'precision': [eval_metrics['precision']],
            'specificity': [eval_metrics['specificity']],
            'f1_score': [eval_metrics['f1_score']],
        })

    # Dataframe with metrics data for views-specific training experiment
    elif(experiment_type == 'views'):
        results_df = pd.DataFrame({
            'model': [model_name],
            'image_preprocessing_pipeline': 'Baseline',
            'view_type': [experimental_value],
            'lesion_type': 'Unified',
            'pr_auc': [eval_metrics['pr_auc']],
            'sensitivity': [eval_metrics['sensitivity']],
            'precision': [eval_metrics['precision']],
            'specificity': [eval_metrics['specificity']],
            'f1_score': [eval_metrics['f1_score']],
        })

    # Dataframe with metrics data for lesion-specific training experiment
    elif(experiment_type == 'lesions'):
        results_df = pd.DataFrame({
            'model': [model_name],
            'image_preprocessing_pipeline': 'CLAHE + Median Blur', # From image preprocessing experiment results
            'view_type': 'Unified',
            'lesion_type': [experimental_value],
            'pr_auc': [eval_metrics['pr_auc']],
            'sensitivity': [eval_metrics['sensitivity']],
            'precision': [eval_metrics['precision']],
            'specificity': [eval_metrics['specificity']],
            'f1_score': [eval_metrics['f1_score']],
        })

    elif(experiment_type == 'baseline'):
        results_df = pd.DataFrame({
            'model': [model_name],
            'image_preprocessing_pipeline': 'Baseline',
            'view_type': 'Unified',
            'lesion_type': 'Unified',
            'pr_auc': [eval_metrics['pr_auc']],
            'sensitivity': [eval_metrics['sensitivity']],
            'precision': [eval_metrics['precision']],
            'specificity': [eval_metrics['specificity']],
            'f1_score': [eval_metrics['f1_score']],
        })
        
    return results_df

# Function to store the experimental results in a csv file in the google drive
def store_experiment_results(results_file_path, results_df):
    # Store the results of experiment in the results dataframe
    if not os.path.isfile(results_file_path):
      # First instance of results
      experiment_results = results_df
    else:  
      # Append to the existing results if this is not the first instance
      experiment_results = pd.read_csv(results_file_path)
      experiment_results = pd.concat([experiment_results, results_df], ignore_index=True)

    # Store the updated dataframe in a csv stored in drive
    experiment_results.to_csv(results_file_path, index=False)