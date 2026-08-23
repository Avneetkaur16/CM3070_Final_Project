import cv2
import numpy as np
from src.image_preprocessing.image_preprocessing_functions import resize_image, clahe_image, median_blur_image

# Baseline image preprocessing
def baseline_preprocessing(image_path):
  # Read, resize and normalize image
  image = cv2.imread(image_path)
  image = resize_image(image)

  image = image.astype(np.float32)
  return image

# CLAHE(with grayscaling) + Median Blue + Baseline image preprocessing
def clahe_median_blur_preprocessing(image_path):
  # Read and resize the image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # CLAHE the image with 1 channel grayscaling
  clahed_image = clahe_image(image)

  # Apply median blur to the CLAHE image
  median_blurred_image = median_blur_image(clahed_image)

  median_blurred_image = median_blurred_image.astype(np.float32)
  return median_blurred_image

