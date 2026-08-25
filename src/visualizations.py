import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to plot training-validation loss for a given model
def plot_training_validation_loss(history, model_name, preprocessor):
    model_history = history.history
    training_loss = model_history['loss']
    validation_loss = model_history['val_loss']
    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(5, 5))
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title(f"{model_name} with {preprocessor}: Training and Validation Loss Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to plot training-validation accuracy for a given model
def plot_training_validation_accuracy(history, model_name, preprocessor):
    model_history = history.history
    training_accuracy = model_history['accuracy']
    validation_accuracy = model_history['val_accuracy']
    epochs = range(1, len(training_accuracy) + 1)

    plt.figure(figsize=(5, 5))
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.title(f"{model_name} with {preprocessor}: Training and Validation Accuracy Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Confusion matrix for a given model
def plot_confusion_matrix(true_pathology, predicted_pathology, model_name, preprocessor):
    cm = confusion_matrix(true_pathology, predicted_pathology)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BENIGN', 'MALIGNANT'])
    cm_disp.plot(cmap=plt.cm.Purples)

    plt.title(f"{model_name} with {preprocessor}: Confusion Matrix")
    plt.show()

# Grouped models bar chart for the given metric
def plot_grouped_bar_char_per_metric(grouped_df, legend_list, metric_name):
    grouped_df.plot(x='model', y=legend_list, kind='bar', figsize=(10, 8), width=0.7)

    plt.title(f"{metric_name}-per model and experiment grouped bar chart")
    plt.xlabel(f"Model", fontsize=12)
    plt.ylabel(f"{metric_name}", fontsize=12)
    plt.xticks(rotation=0)

    plt.grid(axis='y', linestyle='-', alpha=0.6)
    plt.tight_layout()
    plt.show()

# COMPUTATIONAL COSTS GRAPHS

# Bar chart for performance metrics for a given model
def plot_metrics_bar_chart(eval_metrics, model_name, preprocessor):
    auc = eval_metrics['auc']
    sensitivity = eval_metrics['recall']
    precision = eval_metrics['precision']
    f1 = eval_metrics['f1_score']
    accuracy = eval_metrics['accuracy']
    specificity = eval_metrics['specificity']

    metrics = ['AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score', 'Accuracy']
    metric_values = [auc, sensitivity, specificity, precision, f1, accuracy]

    plt.bar(metrics, metric_values)
    plt.title(f"{model_name} with {preprocessor}: Performance Metrics")
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.show()

# Horizontal bar chart for training time of all image preprocessing pipelines for a model
def plot_training_time(training_times, model_name):
    models = list(training_times.keys())
    training_time_values = list(training_times.values())

    plt.barh(models, training_time_values)
    plt.title(f"Training times for pipelines in {model_name}")
    plt.ylabel('Image Preprocessing Pipeline')
    plt.xlabel('Training Time (minutes)')
    plt.show()

# Horizontal bar chart for inference time of all image preprocessing pipelines for a model
def plot_inference_time(prediction_times, model_name, image_samples):
    models = list(prediction_times.keys())
    inference_time_values = list(prediction_times.values())

    for i in range(len(inference_time_values)):
      inference_time_values[i] = inference_time_values[i] / float(image_samples)

    plt.barh(models, inference_time_values)
    plt.title(f"Inference times for pipelines in {model_name}")
    plt.ylabel('Image Preprocessing Pipeline')
    plt.xlabel('Inference Time (ms)')
    plt.show()
