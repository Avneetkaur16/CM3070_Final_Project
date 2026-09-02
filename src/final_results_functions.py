import pandas as pd
import os

# Function to generate results dataframe for final configuration models
def generate_final_results_df(model_name, view, lesion, eval_metrics, train_time, infer_time, peak_memory, 
                              ece_before_scaling, ece_after_scaling, brier_score_before_scaling, brier_score_after_scaling):
    # Create a result dataframe for the given data
    result_df = pd.DataFrame({
        'model': model_name,
        'image_preprocessing_pipeline': 'CLAHE + Median Blur', # from image preprocessing experiment results
        'view_type': [view], 
        'lesion_type': [lesion],
        'accuracy': [eval_metrics['accuracy']],
        'auc': [eval_metrics['auc']],
        'sensitivity': [eval_metrics['sensitivity']],
        'precision': [eval_metrics['precision']],
        'specificity': [eval_metrics['specificity']],
        'f1_score': [eval_metrics['f1_score']],
        'training_time_mins': [train_time / 60.0],
        'inference_time_sec': [infer_time],
        'peak_memory_used_MB': [peak_memory / (1024**2)],
        'ece_before_scaling': [ece_before_scaling],
        'ece_after_scaling': [ece_after_scaling],
        'brier_score_before_scaling': [brier_score_before_scaling],
        'brier_score_after_scaling': [brier_score_after_scaling]
    })
    return result_df

# Function to save the final results in the given results file
def store_final_results(results_file_path, results_df):
    # If this is the first instance of the results file
    if not os.path.isfile(results_file_path):
        final_results = results_df
    else:
        # Otherwise, if this is a successive instance of the final results, then read the file and append to it
        final_results = pd.read_csv(results_file_path)
        final_results = pd.concat([final_results, results_df], ignore_index=True)

    # Store the final results in the given csv file
    final_results.to_csv(results_file_path, index=False)

    print("Saved final results")

# Function to generate a list containing all metrics values for the given metrics and model for final configurations
def generate_metrics_list_from_final_results(model_name, baseline_df, calci_df, mass_df, metric_name):
    # Baseline metric value
    baseline_metric = baseline_df.loc[baseline_df['model'] == model_name, metric_name].iloc[0]

    # CC-Only-Calci-Only metric value
    cc_calci_metric = calci_df.loc[(calci_df['model'] == model_name) 
                               & (calci_df['view_type'] == 'CC-Only-View') 
                               & (calci_df['lesion_type'] == 'Calcification-Only-Lesion'), metric_name].iloc[0]

    # CC-Only-Mass-Only metric
    cc_mass_metric = mass_df.loc[(mass_df['model'] == model_name)
                             & (mass_df['view_type'] == 'CC-Only-View')
                             & (mass_df['lesion_type'] == 'Mass-Only-Lesion'), metric_name].iloc[0]

    # Unified-Calci-Only metric
    uni_calci_metric = calci_df.loc[(calci_df['model'] == model_name)
                         & (calci_df['view_type'] == 'Unified-View')
                         & (calci_df['lesion_type'] == 'Calcification-Only-Lesion'), metric_name].iloc[0]

    # Unified-Mass-Only metric
    uni_mass_metric = mass_df.loc[(mass_df['model'] == model_name)
                       & (mass_df['view_type'] == 'Unified-View')
                       & (mass_df['lesion_type'] == 'Mass-Only-Lesion'), metric_name].iloc[0]

    return [baseline_metric, cc_calci_metric, cc_mass_metric, uni_calci_metric, uni_mass_metric]