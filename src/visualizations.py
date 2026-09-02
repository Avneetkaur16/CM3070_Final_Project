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

# Function to plot training-validation accuracy for a given model
def plot_training_validation_accuracy(history, model_name, experiment_variable):
    model_history = history.history
    training_accuracy = model_history['accuracy']
    validation_accuracy = model_history['val_accuracy']
    epochs = range(1, len(training_accuracy) + 1)

    plt.figure(figsize=(5, 5))
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.title(f"{model_name} with {experiment_variable}: Training and Validation Accuracy Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Confusion matrix for a given model
def plot_confusion_matrix(true_pathology, predicted_pathology, model_name, experiment_variable):
    cm = confusion_matrix(true_pathology, predicted_pathology)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BENIGN', 'MALIGNANT'])
    cm_disp.plot(cmap=plt.cm.Purples)

    plt.title(f"{model_name} with {experiment_variable}: Confusion Matrix")
    plt.show()

# Grouped models bar chart for the given metric
def plot_grouped_bar_chart_per_metric(grouped_df, legend_list, metric_name):
    grouped_df.plot(x='model', y=legend_list, kind='bar', figsize=(12, 6), width=0.7)

    plt.title(f"{metric_name}-per model and experiment grouped bar chart")
    plt.xlabel(f"Model", fontsize=12)
    plt.ylabel(f"{metric_name}", fontsize=12)
    plt.xticks(rotation=0)

    plt.grid(axis='y', linestyle='-', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Grouped models bar chart for final configurations and given metric
def plot_grouped_bar_chart_per_metric_final(grouped_df, legend_list, metric_name):
    grouped_df.plot(x='model', y=legend_list, kind='bar', figsize=(14, 6), width=0.5)
    
    plt.title(f"{metric_name}-per model for final configuration grouped bar chart")
    plt.xlabel(f"Model", fontsize=12)
    plt.ylabel(f"{metric_name}", fontsize=12)
    plt.xticks(rotation=0)

    plt.grid(axis='y', linestyle='-', alpha=0.6)
    plt.tight_layout()
    plt.show()


# COMPUTATIONAL COSTS GRAPHS

# Grouped models horizontal bar chart for final configurations and given computational metric
def plot_computational_grouped_bar_chart_per_metric_final(grouped_df, legend_list, metric_name):
    grouped_df.plot(x=legend_list, y='model', kind='barh', figsize=(6, 10), width=0.5)
    
    plt.title(f"{metric_name}-per model for final configuration grouped bar chart")
    plt.ylabel(f"Model", fontsize=12)
    plt.xlabel(f"{metric_name}", fontsize=12)
    plt.yticks(rotation=0)

    plt.grid(axis='y', linestyle='-', alpha=0.6)
    plt.tight_layout()
    plt.show()