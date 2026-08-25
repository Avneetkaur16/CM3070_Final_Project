import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

def create_vgg16_model():
    # Load the VGG16 model with ImageNet weights
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers
    vgg16.trainable = False

    # Input layer
    input = layers.Input(shape=(224, 224, 3), name="input")

    # VGG16-specific image preprocessing
    x = tf.keras.applications.vgg16.preprocess_input(input)

    # Train the VGG16 model using imagenet weights
    x = vgg16(input, training=False)

    # Classification
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(256, activation="relu", name="dense256")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    output = layers.Dense(1, name="dense1")(x)

    model = Model(inputs=input, outputs=output)
    return model