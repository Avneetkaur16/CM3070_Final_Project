import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def fine_tune_resnet50_model():
    # Load ResNet50 with imagenet weights
    resnet50 = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    resnet50.trainable = True

    # Freeze all layers excluding the last 2 layers
    for frozen_layer in resnet50.layers[:-2]:
        frozen_layer.trainable = False

    # Unfreeze the last 2 layers
    for layer in resnet50.layers[-2:]:
        layer.trainable = True

    # Input layer
    input = layers.Input(shape=(224, 224, 3), name="input")

    # ResNet50-specific image preprocessing
    x = tf.keras.applications.resnet50.preprocess_input(input)

    # Pass the preprocessed input to Resnet50 model
    x = resnet50(x, training=False)

    # Classifier layers
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(256, activation="relu", name="dense256")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    output = layers.Dense(1, name="dense1")(x)
    
    model = Model(inputs=input, outputs=output)
    return model