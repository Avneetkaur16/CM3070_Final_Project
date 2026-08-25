import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def create_resnet50_model():
    # Load the ResNet50 model with ImageNet weights
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers
    resnet50.trainable = False

    # Input layer
    input = layers.Input(shape=(224, 224, 3), name="input")

    # ResNet50-specific image preprocessing
    x = tf.keras.applications.resnet50.preprocess_input(input)

    # Train the ResNet50 model using imagenet weights
    x = resnet50(input, training=False)

    # Classification
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(256, activation="relu", name="dense256")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    output = layers.Dense(1, name="dense1")(x)

    model = Model(inputs=input, outputs=output)
    return model