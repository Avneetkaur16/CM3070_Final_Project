import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to plot training-validation loss for a given model
def plot_training_validation_loss(history, model_name, experiment_variable):
    model_history = history.history
    training_loss = model_history['loss']
    validation_loss = model_history['val_loss']
    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(5, 5))
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title(f"{model_name} with {experiment_variable}: Training and Validation Loss Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Confusion matrix for a given model
def plot_confusion_matrix(true_pathology, predicted_pathology, model_name, experiment_variable):
    cm = confusion_matrix(true_pathology, predicted_pathology)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BENIGN', 'MALIGNANT'])
    cm_disp.plot(cmap=plt.cm.Purples)

    plt.title(f"{model_name} with {experiment_variable}: Confusion Matrix")
    plt.show()

# Receiver Operating Characteristic ROC Curve
def plot_roc_curve(tpr, fpr, final_config_name):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="limegreen")
    plt.scatter(fpr, tpr, color="green", marker="o")
    plt.xlabel('False-Positive-Rate')
    plt.ylabel('True-Positive-Rate')
    plt.title(f"ROC Curve for {final_config_name}")
    plt.show()

# Youden's Index 
def plot_youdens_index(thresholds, j, final_config_name):
    plt.figure(figsize=(5, 5))
    plt.plot(thresholds, j, color="coral")
    plt.scatter(thresholds, j, color="firebrick", marker="o")
    plt.xlabel('Classification Thresholds')
    plt.ylabel("Youden's Index")
    plt.title(f"Classification Thresholds and Youden's Index for {final_config_name}")
    plt.show()

# Plot all metrics in a bar graph
def plot_metrics(metrics, final_config_name):
    metric_names = ['PR-AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'Accuracy']
    metric_colors = ['crimson', 'blueviolet', 'forestgreen', 'lightseagreen', 'tomato']
    plt.figure(figsize=(5, 7))
    ax = plt.bar(x=metric_names, y=metrics, bar_colors=metric_colors)

    for container in ax.containers:
        container.bar_label(container, fmt='%.2f')

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title(f"Metric scores for {final_config_name}")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()