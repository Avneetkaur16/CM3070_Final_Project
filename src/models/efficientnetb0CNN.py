from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def create_efficientnetb0_model():
  # Load the ResNet50 model with ImageNet weights
  efficientnetb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

  # Freeze the layers
  efficientnetb0.trainable = False

  # Input layer
  input = layers.Input(shape=(224, 224, 3))

  # Train the EfficientNetB0 model using imagenet weights
  x = efficientnetb0(input, training=False)

  # Classification
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(256, activation='relu')(x)
  x = layers.Dropout(0.3)(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  model = Model(inputs=input, outputs=output)
  return model
