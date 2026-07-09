from tensorflow.keras import layers, Model
from view_independent.customCNN import create_custom_cnn_model

def create_custom_cnn_two_view_model(input_shape=(224, 224, 3)):
  # Create two input layers for CC and MLO views
  cc_input = layers.Input(shape=input_shape)  # CC
  mlo_input = layers.Input(shape=input_shape)  # MLO

  # Create two branches of custom CNN model for CC and MLO views
  customCNN_model_CC_branch = create_custom_cnn_model(input_shape=(224, 224, 3))  # CC
  customCNN_model_MLO_branch = create_custom_cnn_model(input_shape=(224, 224, 3))  # MLO

  # Extract feature vectors from both views
  cc_features = customCNN_model_CC_branch(cc_input)  # CC feature vector
  mlo_features = customCNN_model_MLO_branch(mlo_input)  # MLO feature vector

  # Concatenate both feature vectors of CC and MLO views
  features_combined = layers.concatenate([cc_features, mlo_features], axis=1)

  # Layers for processing concatenated features
  x = layers.Dense(128, activation='relu')(features_combined)
  x = layers.Dropout(0.3)(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  # Model with two inputs (CC views, MLO views) and one output (pathology label)
  model = Model(inputs=[cc_input, mlo_input], outputs=output)
  return model