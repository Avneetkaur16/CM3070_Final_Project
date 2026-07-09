from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def create_resnet50_model():
  resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
  resnet50.trainable = False

  input = layers.Input(shape=(224, 224, 3))
  x = resnet50(input, training=False)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.3)(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  model = Model(inputs=input, outputs=output)
  return model