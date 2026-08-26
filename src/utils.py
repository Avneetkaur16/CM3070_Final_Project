# Function to update root image path of training images
def update_image_path_training(image_path):
    return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_train_data')

# Function to update root image path of validation images
def update_image_path_validation(image_path):
    return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_val_data')

# Function to update root image path of testing images
def update_image_path_testing(image_path):
    return image_path.replace('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset', '/content/extracted_test_data')

# Function to compute f1 score from precision and recall
def compute_f1_score(precision, recall):
    f1_score = 0.0
    if(precision + recall != 0.0):
      f1_score = (2 * (precision * recall)) / (precision + recall)
    return f1_score

# Function to compute specificity using true negatives and false positives
def compute_specificity(tn, fp):
    return tn / (tn + fp)