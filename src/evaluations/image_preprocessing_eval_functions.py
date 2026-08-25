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

# Function to generate a dataframe for grouped bar chart for the image preprocessing experiment's evaluations for the given metric
def generate_grouped_dataframe_for_image_experiment_results(data_dict):
    # Extract all model lists from the data dict containing metric values for the given metric
    vgg16_list = data_dict['vgg16']
    resnet50_list = data_dict['resnet50']
    densenet121_list = data_dict['densenet121']
    mobilenetv2_list = data_dict['mobilenetv2']
    efficientnetb0_list = data_dict['efficientnetb0']

    # Experiment: Image Preprocessing
    grouped_models_data = {
        'model': ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'EfficientNet-B0'],
        'Baseline': [vgg16_list[0], resnet50_list[0], densenet121_list[0], mobilenetv2_list[0], efficientnetb0_list[0]],
        'CLAHE + Median Blur': [vgg16_list[1], resnet50_list[1], densenet121_list[1], mobilenetv2_list[1], efficientnetb0_list[1]],
        'Histogram Equalization + Gaussian Blur': [vgg16_list[2], resnet50_list[2], densenet121_list[2], mobilenetv2_list[2], efficientnetb0_list[2]]
    }
    # Create a dataframe using grouped models data
    grouped_models_df = pd.DataFrame(grouped_models_data)
    return grouped_models_df