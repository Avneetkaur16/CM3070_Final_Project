from src.dataset.dataset import generate_dataset, shuffle_training_dataset, data_augmented_dataset, batch_dataset

# Function to produce tf.data dataset for the given dataframe, preprocessor, batch size and shuffle(if training)
def dataset_preparation(df, image_preprocessor, buffer_size, batch_size, training=False):
    # Generate a tf.data dataset using the given image preprocessor
    dataset = generate_dataset(df, image_preprocessor)
    # For training data, add shuffling and data augmentation
    if(training):
        # Shuffle
        dataset = shuffle_training_dataset(dataset, buffer_size)
        # Data Augment
        dataset = data_augmented_dataset(dataset)
    # Batch and prefetch the dataset
    dataset = batch_dataset(dataset, batch_size)
    return dataset