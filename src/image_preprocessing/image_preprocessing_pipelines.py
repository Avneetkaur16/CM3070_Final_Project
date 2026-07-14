import cv2
from .image_functions import resize_image, normalize_image, grayscale_image, clahe_image, gamma_correct_image, median_blur_image

# Baseline image preprocessing
def baseline_preprocessing(image_path):
  # Read, resize and normalize image
  image = cv2.imread(image_path)
  image = resize_image(image)
  image = normalize_image(image)
  return image

# Grayscale + Baseline image preprocessing
def grayscale_preprocessing(image_path):
  # Read and resize image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply grayscaling with 3 channels
  gray_image_3_channel = grayscale_image(image)

  # Normalize grayscaled image
  gray_image_3_channel = normalize_image(gray_image_3_channel)

  return gray_image_3_channel

# CLAHE (with 1 channel grayscaling) + Baseline image preprocessing
def clahe_preprocessing(image_path):
  # Read and resize image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply CLAHE preprocessing on the image
  image_clahed = clahe_image(image)

  # Normalize CLAHE image
  image_clahed = normalize_image(image_clahed)

  return image_clahed

# Gamma Correction + Baseline image preprocessing
def gamma_correction_preprocessing(image_path, gamma=2.0):
  # Read and resize image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply gamma correction to the image
  gamma_corrected_image = gamma_correct_image(image, gamma)

  # Normalize gamma corrected image
  gamma_corrected_image = normalize_image(gamma_corrected_image)

  return gamma_corrected_image

# Median Blur + Baseline image preprocessing
def median_blur_preprocessing(image_path):
  # Read and resize the image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply median blurring on the image
  median_blurred_image = median_blur_image(image)

  # Normalize the median blurred image
  median_blurred_image = normalize_image(median_blurred_image)

  return median_blurred_image

# Grayscaling + Gamma Correction + Baseline image preprocessing
def grayscale_gamma_correction_preprocessing(image_path, gamma=2.0):
  # Read and resize the image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply grayscaling (3-channel) to the image
  grayscaled_image = grayscale_image(image)

  # Apply gamma correction to the grayscaled image
  gamma_corrected_image = gamma_correct_image(grayscaled_image, gamma)

  # Normalize the grayscaled + gamma corrected image
  gamma_corrected_image = normalize_image(gamma_corrected_image)

  return gamma_corrected_image

# Grayscaling + Median blue + Baseline image preprocessing
def grayscale_median_blur_preprocessing(image_path):
  # Read and resize the image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply grayscaling (3-channel) to the image
  grayscaled_image = grayscale_image(image)

  # Apply median blur to the grayscaled image
  median_blurred_image = median_blur_image(grayscaled_image)

  # Normalize the grayscaled + median blurred image
  median_blurred_image = normalize_image(median_blurred_image)

  return median_blurred_image

# CLAHE(with grayscaling) + Gamma Correction + Baseline image preprocessing
def clahe_gamma_correction_preprocessing(image_path, gamma=2.0):
  # Read and resize the image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply CLAHE (with 1 channel grayscaling) to the image
  clahed_image = clahe_image(image)

  # Apply gamma correction to the CLAHE image
  gamma_corrected_image = gamma_correct_image(clahed_image, gamma)

  # Normalize the CLAHE + Gamma Corrected image
  gamma_corrected_image = normalize_image(gamma_corrected_image)

  return gamma_corrected_image

# CLAHE(with grayscaling) + Median Blue + Baseline image preprocessing
def clahe_median_blur_preprocessing(image_path):
  # Read and resize the image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # CLAHE the image with 1 channel grayscaling
  clahed_image = clahe_image(image)

  # Apply median blur to the CLAHE image
  median_blurred_image = median_blur_image(clahed_image)

  # Normalize the CLAHE + Median blurred image
  median_blurred_image = normalize_image(median_blurred_image)

  return median_blurred_image

# Gamma correction + Median blue + baseline image preprocessing
def gamma_correction_median_blur_preprocessing(image_path):
  # Read and resize the image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply gamma correction to the image
  gamma_corrected_image = gamma_correct_image(image, gamma=2.0)

  # Apply median blurring to the gamma corrected image
  median_blurred_image = median_blur_image(gamma_corrected_image)

  # Normalize the gamma corrected + median blurred image
  median_blurred_image = normalize_image(median_blurred_image)

  return median_blurred_image

# CLAHE (with grayscaling) + Gamma Correction + Median blur + Baseline Image preprocessing
def grayscale_clahe_gamma_correction_median_blur_preprocessing(image_path):
  # Read and resize the image
  image = cv2.imread(image_path)
  image = resize_image(image)

  # Apply CLAHE (with 1-channel grayscaling) to the image
  grayscale_clahed_image = clahe_image(image)

  # Apply gamma correction to the CLAHE image
  gamma_corrected_image = gamma_correct_image(grayscale_clahed_image, gamma=2.0)

  # Apply median blurring to the CLAHE + Gamma corrected image
  median_blurred_image = median_blur_image(gamma_corrected_image)

  # Apply normalization to CLAHE + Gamma Corrected + Median blurred image
  median_blurred_image = normalize_image(median_blurred_image)
  
  return median_blurred_image