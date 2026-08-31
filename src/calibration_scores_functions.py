import tensorflow as tf
from src.constants import TEMPERATURE_OPTIMIZATION_STEPS, TEMPERATURE_OPTIMIZATION_LEARNING_RATE

# Function to calculate Negative Log-Like Loss for temperature scaling
def compute_nll(temperature, original_logits, true_labels):
  # Compute logites scaled by temperature
  scaled_logits = original_logits / temperature
  
  # Compute Negative Log Likelihood Loss
  nll_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=scaled_logits)
  return tf.reduce_mean(nll_loss)

# Function to perform temperature scaling
def temperature_scaling(original_logits, true_labels, temperature):
  # Define an optimizer for temperature optimization with learning rate 1e-2
  temp_optimizer = tf.keras.optimizers.Adam(learning_rate=TEMPERATURE_OPTIMIZATION_LEARNING_RATE)

  for i in range(TEMPERATURE_OPTIMIZATION_STEPS):
    # Record all computations of NLL loss
    with tf.GradientTape() as tape:
      nll_loss = compute_nll(temperature, original_logits, true_labels)

    # Compute gradients of NLL loss with respect to temperature
    gradients = tape.gradient(nll_loss, [temperature])

    # Optimize the temperature 
    temp_optimizer.apply_gradients(zip(gradients, [temperature]))

  return temperature

# Function to calculate Brier Score
def compute_brier_score(pred_probabilities, true_labels_tensor):
  # Compute squared difference between predicted probabilities and true labels
  squared_diff = tf.math.squared_difference(pred_probabilities, true_labels_tensor)

  # Compute the average of squared difference
  brier_score = tf.reduce_mean(squared_diff)
  return brier_score