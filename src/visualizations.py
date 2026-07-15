import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to plot training-validation loss for a given model
def plot_training_validation_loss(history, model_name):
  model_history = history.history
  training_loss = model_history['loss']
  validation_loss = model_history['val_loss']
  epochs = range(1, len(training_loss) + 1)

  plt.plot(epochs, training_loss, label='Training Loss')
  plt.plot(epochs, validation_loss, label='Validation Loss')
  plt.title(f"{model_name}: Training and Validation Loss Curve")
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

  plt.plot(epochs, training_accuracy, label='Training Accuracy')
  plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
  plt.title(f"{model_name} with {preprocessor}: Training and Validation Accuracy Curve")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

# Function to plot precision-recall curve for a given model
def plot_precision_recall_curve(history, model_name, preprocessor):
  model_history = history.history
  precision = model_history['precision']
  recall = model_history['recall']
  epochs = range(1, len(precision) + 1)

  plt.plot(epochs, precision, label='Precision')
  plt.plot(epochs, recall, label='Recall')
  plt.title(f"{model_name} with {preprocessor}: Precision-Recall Curve")
  plt.xlabel('Epochs')
  plt.ylabel('Precision/Recall')
  plt.legend()
  plt.show()

# Confusion matrix for a given model
def plot_confusion_matrix(true_pathology, predicted_pathology, model_name, preprocessor):
  cm = confusion_matrix(true_pathology, predicted_pathology)
  cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BENIGN', 'MALIGNANT'])
  cm_disp.plot(cmap=plt.cm.Purples)
  plt.title(f"{model_name} with {preprocessor}: Confusion Matrix")
  plt.show()

# Bar chart for performance metrics for a given model
def plot_metrics_bar_chart(true_pathology, predicted_pathology, eval_metrics, model_name, preprocessor):
  auc = eval_metrics['auc']
  sensitivity = eval_metrics['recall']
  precision = eval_metrics['precision']
  f1 = 0.0
  if(not (precision == 0.0 and sensitivity == 0.0)):
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
  accuracy = eval_metrics['accuracy']

  tn, fp, fn, tp = confusion_matrix(true_pathology, predicted_pathology).ravel()
  specificity = tn / (tn + fp)

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

  for i in range(len(training_time_values)):
    training_time_values[i] = training_time_values[i] / 60.0

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