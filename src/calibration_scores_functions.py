import tensorflow as tf
import tensorflow_probability as tfp
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

# Function to calculate ECE Score
def compute_ece_score(data_logits, true_labels_tensor, bins):
    # Squeeze true labels tensor
    true_labels_squeezed = tf.squeeze(true_labels_tensor)

    # Create zeros with data_logits shape
    zeros = tf.zeros_like(data_logits)

    # Generate nlabel logits with [zeros, original logits] for ece function
    logits = tf.concat([zeros, data_logits], axis=1)
    
    # Compute ECE Score
    ece = tfp.stats.expected_calibration_error(bins, logits, tf.cast(true_labels_squeezed, tf.int32))
    return ece.numpy()