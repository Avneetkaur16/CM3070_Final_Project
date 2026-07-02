import tensorflow as tf

def compile_model(model):
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
  ])
  return model