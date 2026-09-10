import tensorflow as tf
import numpy as np
import matplotlib as mpl

def generate_image_array(image_path):
    # Load the image using the image path
    image = tf.keras.utils.load_img(image_path)
    # Resize image to 224x244x3
    image = image.resize((224, 224))
    # Convert the image to an image array
    image_array = tf.keras.utils.img_to_array(image)
    # Add a batch dimension to the image array (for model input)
    image_array = tf.expand_dims(image_array, axis=0)
    return image_array

def generate_grad_cam_heatmap(image_array, model, base_model, last_conv_layer):
    # Create the base model for grad cam
    grad_model = tf.keras.models.Model(
        inputs = base_model.input,
        outputs = base_model.get_layer(last_conv_layer).output
    )

    # Add classification layers from the given model (using their weights)
    global_average_pooling = model.get_layer('global_average_pooling')
    dense256 = model.get_layer('dense256')
    dropout = model.get_layer('dropout')
    dense1 = model.get_layer('dense1')

    # Record feature maps and get logits from it
    with tf.GradientTape() as tape:
        feature_maps = grad_model(image_array, training=False)
        gap_output = global_average_pooling(feature_maps)
        dense256_output = dense256(gap_output)
        dropout_output = dropout(dense256_output)
        final_output = dense1(dropout_output)

        logit = final_output[:, 0]

    # Get gradients for each feature map (determine which activations in each feature map affects prediction)
    gradients = tape.gradient(logit, feature_maps)

    # Importance score of each feature map of the image
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))

    # Get the feature maps produced by the last conv layer
    feature_maps = feature_maps[0]
    # Get the importance scores of all feature maps
    pooled_gradients = pooled_gradients[0]

    # Create a heatmap by multiplying each feature map with its importance score and adding all up
    heatmap = tf.reduce_sum(tf.multiply(feature_maps, pooled_gradients), axis=-1)
    # Get positive-only activations
    heatmap = tf.maximum(heatmap, 0)
    # Normalize heatmap
    if(tf.math.reduce_max(heatmap) != 0):
        heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy(), tf.nn.sigmoid(logit)

def superimpose_heatmap_on_image(image_path, heatmap, alpha=4.0):
    # Load the image using the image path
    image = tf.keras.utils.load_img(image_path)
    image = image.resize((224, 224))
    # Convert the loaded image to image array
    image_array = tf.keras.utils.img_to_array(image)

    # Rescale heatmap to 0-255 range
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize the heatmap
    jet = mpl.colormaps.get('jet')
    # RGB Colors
    jet_colors = jet(np.arange(256))[:, :3]
    # Create RGB jet heatmap
    jet_heatmap = jet_colors[heatmap]

    # Generate an RGB image
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    # Resize the jey RGB heatmap to the original image size
    jet_heatmap = jet_heatmap.resize((image_array.shape[1], image_array.shape[0]))
    # Convert the heatmap image to image array
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the jet heatmap image-array on the original image array
    superimposed_image_array = jet_heatmap * alpha + image_array
    # Convert the superimposed image array to an image
    superimposed_image = tf.keras.utils.array_to_img(superimposed_image_array)

    return superimposed_image

def generate_grad_cam_image(image_path, model, base_model, last_conv_layer):
    # Generate an image array using the given image path
    image_array = generate_image_array(image_path)
    # Generate heatmap and predictions using image array, model and last conv layer
    heatmap, prediction = generate_grad_cam_heatmap(image_array, model, base_model, last_conv_layer)
    # Superimpose the heatmap on the original image
    superimposed_image = superimpose_heatmap_on_image(image_path, heatmap)
    return superimposed_image, prediction