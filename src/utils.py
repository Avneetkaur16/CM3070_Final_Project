import pandas as pd
import tensorflow as tf
import os

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
    print(f"Stored results for {results_df['image_preprocessing_pipeline']} pipeline")

# Function to compute f1 score from precision and recall
def compute_f1_score(precision, recall):
    f1_score = 0.0

    if(precision + recall != 0.0):
      f1_score = (2 * (precision * recall)) / (precision + recall)

    return f1_score

# Function to compute specificity using true negatives and false positives
def compute_specificity(tn, fp):
    return tn / (tn + fp)