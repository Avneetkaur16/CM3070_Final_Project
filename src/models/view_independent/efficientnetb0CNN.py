from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def create_efficientnetb0_model():
  efficientnetb0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  efficientnetb0.trainable = False

  input = layers.Input(shape=(224, 224, 3))
  x = efficientnetb0(input, training=False)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.3)(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  model = Model(inputs=input, outputs=output)
  return model