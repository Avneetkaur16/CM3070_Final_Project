from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def create_efficientnetb0_model():
    # Load the ResNet50 model with ImageNet weights
    efficientnetb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers
    efficientnetb0.trainable = False

    # Input layer
    input = layers.Input(shape=(224, 224, 3), name="input")

    # Train the EfficientNetB0 model using imagenet weights
    x = efficientnetb0(input, training=False)

    # Classification
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(256, activation="relu", name="dense256")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    output = layers.Dense(1, name="dense1")(x)

    model = Model(inputs=input, outputs=output)
    return model