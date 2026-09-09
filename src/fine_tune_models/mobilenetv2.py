import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

def fine_tune_mobilenetv2_model():
    # Load MobileNetV2 model with imagenet weights
    mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    mobilenetv2.trainable = True

    # Freeze all layers excluding the last two layers
    for frozen_layer in mobilenetv2.layers[:-2]:
        frozen_layer.trainable = False

    # Unfreeze the last two layers
    for layer in mobilenetv2.layers[-2:]:
        layer.trainable = True

    # Input layer
    input = layers.Input(shape=(224, 224, 3), name="input")

    # MobileNetV2-specific image preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(input)

    # Pass the preprocessed input to mobilenetv2 model
    x = mobilenetv2(x, training=False)

    # Classifier layers
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(256, activation="relu", name="dense256")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    output = layers.Dense(1, name="dense1")(x)

    model = Model(inputs=input, outputs=output)
    return model   