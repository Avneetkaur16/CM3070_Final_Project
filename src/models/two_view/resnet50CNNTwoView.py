from tensorflow.keras import layers, Model
from view_independent.resnet50CNN import create_resnet50_model

def create_resnet50_two_view_model(input_shape=(224, 224, 3)):
  # Create two input layers for CC and MLO views
  cc_input = layers.Input(shape=input_shape)
  mlo_input = layers.Input(shape=input_shape)

  # Create two branches of ResNet50 CNN model for CC and MLO views
  resnet50_model_CC_branch = create_resnet50_model(input_shape=input_shape)
  resnet50_model_MLO_branch = create_resnet50_model(input_shape=input_shape)

  # Extract feature vectors from both views
  cc_features = resnet50_model_CC_branch(cc_input)
  mlo_features = resnet50_model_MLO_branch(mlo_input)

  # Concatenate both feature vectors of CC and MLO views
  features_combined = layers.concatenate([cc_features, mlo_features], axis=1)

  # Layers for processing concatenated features
  x = layers.Dense(128, activation='relu')(features_combined)
  x = layers.Dropout(0.3)(x)
  output = layers.Dense(1, activation='sigmoid')(x)

  # Model with two inputs (CC views, MLO views) and one output (pathology label)
  model = Model(inputs=[cc_input, mlo_input], outputs=output)
  return model