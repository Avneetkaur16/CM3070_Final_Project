from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

def create_vgg16_model():
  vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  vgg16.trainable = False

  input = layers.Input(shape=(224, 224, 3))
  x = vgg16(input, training=False)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.2)(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  model = Model(inputs=input, outputs=output)
  return model