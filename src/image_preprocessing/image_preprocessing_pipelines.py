import cv2
import numpy as np
from src.image_preprocessing.image_preprocessing_functions import resize_image, clahe_image, median_blur_image 
from src.image_preprocessing.image_preprocessing_functions import histogram_equalization_image, gaussian_blur_image

# Baseline image preprocessing
def baseline_preprocessing(image_path):
  # Read and resize the image
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
  clahe_and_median_blurred_image = median_blur_image(clahed_image)

  clahe_and_median_blurred_image = clahe_and_median_blurred_image.astype(np.float32)
  return clahe_and_median_blurred_image

# Histogram Equalization(with grayscaling) + Gaussian Blue + Baseline image preprocessing
def histogram_equalization_gaussian_blur_preprocessing(image_path):
  # Read and resize the image using image path
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Histogram Equalization
  histo_equalized = histogram_equalization_image(image)

  # Gaussian Blur
  histo_equalized_and_gauss_blurred_image = gaussian_blur_image(histo_equalized)

  histo_equalized_and_gauss_blurred_image = histo_equalized_and_gauss_blurred_image.astype(np.float32)
  return histo_equalized_and_gauss_blurred_image