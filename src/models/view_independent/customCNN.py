from tensorflow.keras import layers, Model

def create_custom_cnn_model(input_shape=(224, 224, 3)):
  input = layers.Input(shape=input_shape)
  x = layers.Conv2D(32, (3, 3), activation='relu')(input)
  x = layers.Conv2D(32, (3, 3), activation='relu')(x)
  x = layers.MaxPooling2D((2, 2))(x)

  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.MaxPooling2D((2, 2))(x)

  x = layers.Conv2D(128, (3, 3), activation='relu')(x)
  x = layers.Conv2D(128, (3, 3), activation='relu')(x)
  x = layers.MaxPooling2D((2, 2))(x)

  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.3)(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  model = Model(inputs=input, outputs=output)
  return model