import numpy as np
import cv2

# Resize an image to 224 x 224
def resize_image(image):
  image = cv2.resize(image, (224, 224))
  return image

# Normalize an image per channel
def normalize_image(image):
  # Extract all channels (BGR)
  blue_channel = image[:, :, 0]
  green_channel = image[:, :, 1]
  red_channel = image[:, :, 2]

  # Normalize each channel
  blue_channel = (blue_channel / 255.0).astype(np.float32)
  green_channel = (green_channel / 255.0).astype(np.float32)
  red_channel = (red_channel / 255.0).astype(np.float32)

  # Merge all normalized channels
  return cv2.merge([blue_channel, green_channel, red_channel])

# Grayscale Image
def grayscale_image(image):
  # Convert BGR image to 1 channel Grayscaled
  gray_image_1_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Convert 1 channel grayscaled imaged to 3 channel grayscaled image
  gray_image_3_channel = cv2.cvtColor(gray_image_1_channel, cv2.COLOR_GRAY2BGR)

  # Return 3 channel grayscaled image
  return gray_image_3_channel

# CLAHE Image technique with grayscaling
def clahe_image(image):
  # Convert a BGR image to 1 channel grayscale image
  gray_image_1_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply CLAHE to 1 channel grayscaled image
  clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
  clahed_image = clahe.apply(gray_image_1_channel)

  # Convert 1 channel CLAHE image to 3 channel image
  clahed_image = cv2.cvtColor(clahed_image, cv2.COLOR_GRAY2BGR)

  return clahed_image

# Gamma Correction for image (Gamma = 2.0)
def gamma_correct_image(image, gamma):
  # Create a mapping table using gamma value (2.0)
  inv_gamma = 1.0 / gamma
  table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

  # Apply Gamma Correction to the image
  gamma_corrected_image = cv2.LUT(image, table)

  return gamma_corrected_image

# Median Blur with kernel size 3
def median_blur_image(image):
  # Apply median blur on the image with kernel 3 x 3
  median_blurred_image = cv2.medianBlur(image, 3)

  return median_blurred_image