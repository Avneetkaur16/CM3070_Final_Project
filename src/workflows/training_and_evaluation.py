import tensorflow as tf
import time
from src.common_model_functions import compile_model
from sklearn.metrics import confusion_matrix
from src.utils import compute_f1_score, compute_specificity

# Function to compile and train the selected model
def train_selected_model(model, train_dataset, val_dataset, epochs, earlystopping, class_weight_dict):
    # Compile the selected model
    model = compile_model(model)

    # GPU Memory usage computation start
    tf.config.experimental.reset_memory_stats('GPU:0')
    # Training Start time
    train_start = time.time()

    # Fit the model with training data, validation data. Use early stopping for early stop and class_weight_dict for class-imbalance
    model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=epochs, 
        callbacks=[earlystopping], 
        class_weight=class_weight_dict
    )

    # Compute peak memory usage
    peak_memory = tf.config.experimental.get_memory_info('GPU:0')['peak']
    # Training End time
    train_end = time.time()

    # Total Training time
    training_time = train_end - train_start

    # Return the trained model
    return model, training_time, peak_memory

# Function to get logits, predictions and evaluation metrics from the trained model using test/validation dataset
def generate_predictions_and_evaluations(trained_model, dataset, true_labels, prediction_threshold):
    # Predictions

    # No. of samples in the dataset
    num_samples = dataset.cardinality().numpy()

    # Prediction start time
    prediction_start = time.time()
    # Get logits from the trained model using the given dataset
    logits = trained_model.predict(dataset)
    # Prediction end time
    prediction_end = time.time()

    # Get probabilities from the logits using sigmoid activation function
    probabilities = tf.nn.sigmoid(logits)
    # Get predictions from the probabilities based on prediction threshold value
    predictions = tf.cast(probabilities >= prediction_threshold, tf.float32)

    # Total prediction time
    prediction_time = prediction_end - prediction_start
    # Inference time
    inference_time = prediction_time / num_samples

    # Evaluations
    # Get evaluation metrics (AUC, Accuracy, Precision, Recall) from the trained model using the given dataset
    eval_metrics = trained_model.evaluate(dataset, return_dict=True)
    # Compute F1 score
    f1_score = compute_f1_score(eval_metrics['precision'], eval_metrics['recall'])
    # Get confusion matrix values
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    # Compute Specificity
    specificity = compute_specificity(tn, fp)

    # Create an evaluation metrics dictionary containing ALL performance metrics
    eval_metrics_dict = {
        'auc': eval_metrics['auc'],
        'accuracy': eval_metrics['accuracy'],
        'precision': eval_metrics['precision'],
        'sensitivity': eval_metrics['recall'],
        'specificity': specificity,
        'f1_score': f1_score
    }

    return logits, predictions, eval_metrics_dict, inference_time