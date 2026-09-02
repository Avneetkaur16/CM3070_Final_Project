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

def generate_grouped_dataframe_for_final_results(data_dict):
    # Extract all metric lists from data dict
    vgg16 = data_dict['vgg16']
    resnet50 = data_dict['resnet50']
    densenet121 = data_dict['densenet121']
    mobilenetv2 = data_dict['mobilenetv2']
    efficientnetb0 = data_dict['efficientnetb0']

    # Create a grouped dataframe
    grouped_df = pd.DataFrame({
        'model': ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'EfficientNet-B0'],
        'Baseline': [vgg16[0], resnet50[0], densenet121[0], mobilenetv2[0], efficientnetb0[0]],
        'CC-Only-Calci-Only': [vgg16[1], resnet50[1], densenet121[1], mobilenetv2[1], efficientnetb0[1]],
        'CC-Only-Mass-Only': [vgg16[2], resnet50[2], densenet121[2], mobilenetv2[2], efficientnetb0[2]],
        'Unified-Calci-Only': [vgg16[3], resnet50[3], densenet121[3], mobilenetv2[3], efficientnetb0[3]],
        'Unified-Mass-Only': [vgg16[4], resnet50[4], densenet121[4], mobilenetv2[4], efficientnetb0[4]]
    })

    return grouped_df