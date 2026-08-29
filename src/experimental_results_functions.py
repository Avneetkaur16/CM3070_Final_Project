import pandas as pd
import os

# Function to generate results metrics dataframe
def generate_results_metrics_df(model_name, experimental_value, eval_metrics, train_time, infer_time, peak_memory, experiment_type):
    # Adjust training time and peak memory usage
    training_time = train_time / 60.0
    peak_memory_used = peak_memory / (1024**2)

     # Dataframe with metrics data for image preprocessing experiment 
    if(experiment_type == 'image_preprocessing'):
        results_df = pd.DataFrame({
            'model': [model_name],
            'image_preprocessing_pipeline': [experimental_value],
            'view_type': 'Unified',
            'lesion_type': 'Unified',
            'accuracy': [eval_metrics['accuracy']],
            'auc': [eval_metrics['auc']],
            'sensitivity': [eval_metrics['sensitivity']],
            'precision': [eval_metrics['precision']],
            'specificity': [eval_metrics['specificity']],
            'f1_score': [eval_metrics['f1_score']],
            'training_time_mins': [training_time],
            'inference_time_sec': [infer_time],
            'peak_memory_used_MB': [peak_memory_used]
        })

    # Dataframe with metrics data for views-specific training experiment
    elif(experiment_type == 'views'):
        results_df = pd.DataFrame({
            'model': [model_name],
            'image_preprocessing_pipeline': 'Baseline',
            'view_type': [experimental_value],
            'lesion_type': 'Unified',
            'accuracy': [eval_metrics['accuracy']],
            'auc': [eval_metrics['auc']],
            'sensitivity': [eval_metrics['sensitivity']],
            'precision': [eval_metrics['precision']],
            'specificity': [eval_metrics['specificity']],
            'f1_score': [eval_metrics['f1_score']],
            'training_time_mins': [training_time],
            'inference_time_sec': [infer_time],
            'peak_memory_used_MB': [peak_memory_used]
        })

    # Dataframe with metrics data for lesion-specific training experiment
    elif(experiment_type == 'lesions'):
        results_df = pd.DataFrame({
            'model': [model_name],
            'image_preprocessing_pipeline': 'Baseline',
            'view_type': 'Unified',
            'lesion_type': [experimental_value],
            'accuracy': [eval_metrics['accuracy']],
            'auc': [eval_metrics['auc']],
            'sensitivity': [eval_metrics['sensitivity']],
            'precision': [eval_metrics['precision']],
            'specificity': [eval_metrics['specificity']],
            'f1_score': [eval_metrics['f1_score']],
            'training_time_mins': [training_time],
            'inference_time_sec': [infer_time],
            'peak_memory_used_MB': [peak_memory_used]
        })

    elif(experiment_type == 'baseline'):
        results_df = pd.DataFrame({
            'model': [model_name],
            'image_preprocessing_pipeline': 'Baseline',
            'view_type': 'Unified',
            'lesion_type': 'Unified',
            'accuracy': [eval_metrics['accuracy']],
            'auc': [eval_metrics['auc']],
            'sensitivity': [eval_metrics['sensitivity']],
            'precision': [eval_metrics['precision']],
            'specificity': [eval_metrics['specificity']],
            'f1_score': [eval_metrics['f1_score']],
            'training_time_mins': [training_time],
            'inference_time_sec': [infer_time],
            'peak_memory_used_MB': [peak_memory_used]
        })
        
    return results_df

# Function to store the experimental results in a csv file in the google drive
def store_experiment_results(results_file_path, results_df, experiment_type):
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

    if(experiment_type == 'image_preprocessing'):
       print(f"Stored results for {results_df['image_preprocessing_pipeline']} pipeline")
    elif(experiment_type == 'views'):
       print(f"Stored results for {results_df['view_type']} views")
    elif(experiment_type == 'lesions'):
       print(f"Stored results for {results_df['lesion_type']} lesions")
    elif(experiment_type == 'baseline'):
        print(f"Stored results for the baseline")

# Function to generate model-specific list for baseline, given experiment and the given metric
def generate_metric_list_for_experiment_results(model_name, baseline_df, experimental_df, metric_name, experiment_type):
    # Baseline metric value
    baseline_metric_value = baseline_df.loc[baseline_df['model'] == model_name, metric_name].iloc[0]

    # Image Preprocessing Experiment
    if(experiment_type == 'image_preprocessing'):
        # Experiment1 metric value (CLAHE + Median blur)
        clahe_median_blur_metric_value = experimental_df.loc[(experimental_df['model'] == model_name) 
            & (experimental_df['image_preprocessing_pipeline'] == 'CLAHE + Median Blur'), metric_name].iloc[0]
    
        # Experiment 2 metric value (Histogram Equalization + Gaussian Blur)
        histo_equalized_guass_blur_metric_value = experimental_df.loc[(experimental_df['model'] == model_name) 
            & (experimental_df['image_preprocessing_pipeline'] == 'Histogram Equalization + Gaussian Blur'), metric_name].iloc[0]

        return [baseline_metric_value, clahe_median_blur_metric_value, histo_equalized_guass_blur_metric_value]

    # View-specific training Experiment
    elif(experiment_type == 'views'):
        # Experiment1 metric value (CC only)
        cc_only_metric_value = experimental_df.loc[(experimental_df['model'] == model_name) 
            & (experimental_df['view_type'] == 'CC-Only-View'), metric_name].iloc[0]
    
        # Experiment 2 metric value (MLO only)
        mlo_only_metric_value = experimental_df.loc[(experimental_df['model'] == model_name) 
            & (experimental_df['view_type'] == 'MLO-Only-View'), metric_name].iloc[0]

        return [baseline_metric_value, cc_only_metric_value, mlo_only_metric_value]

    # Lesion-specific training Experiment
    elif(experiment_type == 'lesions'):
        # Experiment 1 metric value (Calcification only)
        calcifications_only_metric_value = experimental_df.loc[(experimental_df['model'] == model_name) 
            & (experimental_df['lesion_type'] == 'Calcification-Only-Lesion'), metric_name].iloc[0]

        # Experiment2 metric value (Masses only)
        masses_only_metric_value = experimental_df.loc[(experimental_df['model'] == model_name) 
            & (experimental_df['lesion_type'] == 'Mass-Only-Lesion'), metric_name].iloc[0]
        
        return [baseline_metric_value, calcifications_only_metric_value, masses_only_metric_value]
    else:
        return None

# Function to generate a dataframe for grouped bar chart for baseline, given experiment's evaluations and the given metric
def generate_grouped_dataframe_for_experiment_results(data_dict, experiment_type):
    # Extract all model lists from the data dict containing metric values for the given metric
    vgg16_list = data_dict['vgg16']
    resnet50_list = data_dict['resnet50']
    densenet121_list = data_dict['densenet121']
    mobilenetv2_list = data_dict['mobilenetv2']
    efficientnetb0_list = data_dict['efficientnetb0']

    if(experiment_type == 'image_preprocessing'):
        # Experiment: Image Preprocessing
        grouped_models_data = {
            'model': ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'EfficientNet-B0'],
            'Baseline': [vgg16_list[0], resnet50_list[0], densenet121_list[0], mobilenetv2_list[0], efficientnetb0_list[0]],
            'CLAHE + Median Blur': [vgg16_list[1], resnet50_list[1], densenet121_list[1], mobilenetv2_list[1], efficientnetb0_list[1]],
            'Histogram Equalization + Gaussian Blur': [vgg16_list[2], resnet50_list[2], densenet121_list[2], mobilenetv2_list[2], efficientnetb0_list[2]]
        }
        
    elif(experiment_type == 'views'):
        # Experiment: View-specific training
        grouped_models_data = {
            'model': ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'EfficientNet-B0'],
            'Baseline (Unified)': [vgg16_list[0], resnet50_list[0], densenet121_list[0], mobilenetv2_list[0], efficientnetb0_list[0]],
            'CC-Only-View': [vgg16_list[1], resnet50_list[1], densenet121_list[1], mobilenetv2_list[1], efficientnetb0_list[1]],
            'MLO-Only-View': [vgg16_list[2], resnet50_list[2], densenet121_list[2], mobilenetv2_list[2], efficientnetb0_list[2]]
        }
    elif(experiment_type == 'lesions'):
        # Experiment: Lesion-specific training
        grouped_models_data = {
            'model': ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'EfficientNet-B0'],
            'Baseline (Unified)': [vgg16_list[0], resnet50_list[0], densenet121_list[0], mobilenetv2_list[0], efficientnetb0_list[0]],
            'Calcification-Only-Lesion': [vgg16_list[1], resnet50_list[1], densenet121_list[1], mobilenetv2_list[1], efficientnetb0_list[1]],
            'Mass-Only-Lesion': [vgg16_list[2], resnet50_list[2], densenet121_list[2], mobilenetv2_list[2], efficientnetb0_list[2]]
        }

    # Create a dataframe using grouped models data
    grouped_models_df = pd.DataFrame(grouped_models_data)
    return grouped_models_df