import pandas as pd

# Function to generate view-specific training results dataframe
def generate_views_results_df(model_name, view_type, eval_metrics, specificity, f1_score):
    results_df = pd.DataFrame({
        'model': [model_name],
        'view_type': [view_type],
        'accuracy': [eval_metrics['accuracy']],
        'auc': [eval_metrics['auc']],
        'sensitivity': [eval_metrics['recall']],
        'precision': [eval_metrics['precision']],
        'specificity': [specificity],
        'f1_score': [f1_score]
    })
    return results_df

# Function to generate model-specific list for view-specific training experiment for the given metric
def generate_metric_list_for_views_experiment_results(model_name, baseline_df, views_df, metric_name):
    # Baseline metric value
    baseline_metric_value = baseline_df.loc[baseline_df['model'] == model_name, metric_name].iloc[0]

    # Experiment1 metric value (CC only)
    cc_only_metric_value = views_df.loc[
      (views_df['model'] == model_name) 
      & (views_df['view_type'] == 'CC'), metric_name].iloc[0]

    # Experiment 2 metric value (MLO only)
    mlo_only_metric_value = views_df.loc[
      (views_df['model'] == model_name) 
      & (views_df['lesion_type'] == 'MLO'), metric_name].iloc[0]

    return [baseline_metric_value, cc_only_metric_value, mlo_only_metric_value]

# Function to generate a dataframe for grouped bar chart for the views experiment's evaluations for the given metric
def generate_grouped_dataframe_for_views_experiment_results(data_dict):
    # Extract all model lists from the data dict containing metric values for the given metric
    vgg16_list = data_dict['vgg16']
    resnet50_list = data_dict['resnet50']
    densenet121_list = data_dict['densenet121']
    mobilenetv2_list = data_dict['mobilenetv2']
    efficientnetb0_list = data_dict['efficientnetb0']

    # Experiment: View-specific training
    grouped_models_data = {
        'model': ['VGG16', 'ResNet50', 'DenseNet121', 'MobileNetV2', 'EfficientNet-B0'],
        'Baseline': [vgg16_list[0], resnet50_list[0], densenet121_list[0], mobilenetv2_list[0], efficientnetb0_list[0]],
        'CC-Only': [vgg16_list[1], resnet50_list[1], densenet121_list[1], mobilenetv2_list[1], efficientnetb0_list[1]],
        'MLO-Only': [vgg16_list[2], resnet50_list[2], densenet121_list[2], mobilenetv2_list[2], efficientnetb0_list[2]]
    }

    # Create a dataframe using grouped models data
    grouped_models_df = pd.DataFrame(grouped_models_data)
    return grouped_models_df