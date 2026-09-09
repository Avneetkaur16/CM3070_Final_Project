import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve
from src.utils import compute_accuracy, compute_f1_score, compute_precision, compute_sensitivity, compute_specificity

# Function to generate results dataframe for final configuration models
def generate_final_results_df(model_name, lesion, eval_metrics, ece_before_scaling, ece_after_scaling):
    # Create a result dataframe for the given data
    result_df = pd.DataFrame({
        'model': model_name,
        'image_preprocessing_pipeline': 'Baseline', # from image preprocessing experiment results
        'view_type': 'CC-Only-View', # from view-specific experiment results 
        'lesion_type': [lesion],
        'pr_auc': [eval_metrics['auc']],
        'sensitivity': [eval_metrics['sensitivity']],
        'precision': [eval_metrics['precision']],
        'specificity': [eval_metrics['specificity']],
        'f1_score': [eval_metrics['f1_score']],
        'accuracy': [eval_metrics['accuracy']],
        'ece_before_scaling': [ece_before_scaling],
        'ece_after_scaling': [ece_after_scaling],
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

# Function for patient level evaluations of a model for a given dataset
def patient_level_evaluation_metrics(model, df, dataset, classification_threshold):
    # Compute logits from the dataset
    logits = model.predict(dataset)
    # Calculate probabilities from logits
    probs = tf.nn.sigmoid(logits)
    # Compute predictions from prediction probabilities
    preds = tf.cast(probs >= classification_threshold, tf.float32)

    # Create a copy of the original dataframe
    new_df = df.copy()
    # Add prediction column to the new df with prediction values calculated above
    new_df['prediction'] = preds.numpy()

    # Group the new dataframe on 'patient_id' based on max() values of pathology and prediction
    patient_level_df = new_df.groupby('patient_id')[['pathology', 'prediction']].max().reset_index()

    # Get confusion matrix values from the patient level dataframe
    tn, fp, fn, tp = confusion_matrix(patient_level_df['pathology'], patient_level_df['prediction']).ravel()

    # Calculate evaluation metrics
    accuracy = compute_accuracy(tp, tn, fp, fn)
    sensitivity = compute_sensitivity(tp, fn)
    specificity = compute_specificity(tn, fp)
    precision = compute_precision(tp, fp)
    f1_score = compute_f1_score(precision, sensitivity)

    # Create a patient-level evaluation metrics dictionary
    patient_level_metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score   
    }

    return patient_level_metrics

# Function to compute Youden's Index for validation-set based classification threshold optimization
def compute_youdens_index_and_optimal_threshold(prediction_probs, true_labels):
    # Compute false-positive-rates and true-positive-rates for different thresholds using true labels and prediction probabilities
    fpr, tpr, thresholds = roc_curve(true_labels, prediction_probs)

    # Youden's Index J = (sensitivity + specificity - 1) OR TPR - FPR (sensitivity - (1 - specificity))
    j = tpr - fpr

    # Max Youden's Index
    max_j = np.argmax(j)

    # Optimal threshold based on max Youden's Index
    optimal_threshold = thresholds[max_j]

    return max_j, j, optimal_threshold, thresholds, tpr, fpr