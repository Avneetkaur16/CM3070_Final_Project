import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

def create_mobilenetv2_model():
    # Load the MobileNetV2 model with ImageNet weights
    mobilenetv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers
    mobilenetv2.trainable = False

    # Input layer
    input = layers.Input(shape=(224, 224, 3), name="input")

    # MobileNetV2 specific image preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(input)

    # Train the model with imagenet weights
    x = mobilenetv2(input, training=False)

    # Classification
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(256, activation="relu", name="dense256")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    output = layers.Dense(1, name="dense1")(x)

    model = Model(inputs=input, outputs=output)
    return model