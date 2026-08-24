import cv2

# Resize an image to 224 x 224
def resize_image(image):
  image = cv2.resize(image, (224, 224))
  return image

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

# Median Blur with kernel size 3
def median_blur_image(image):
  # Apply median blur on the image with kernel 3 x 3
  median_blurred_image = cv2.medianBlur(image, 3)
  return median_blurred_image

# Histogram Equalization technique
def histogram_equalization_image(image):
  # Convert BGR image to 1-channel Grayscale
  gray_image_1_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply histogram equalization to the grascaled image
  histogram_equalized = cv2.equalizeHist(gray_image_1_channel)

  # Convert 1 channel Histogram Equalized image to 3 channel image
  histogram_equalized = cv2.cvtColor(histogram_equalized, cv2.COLOR_GRAY2BGR)

  return histogram_equalized

# Gaussian Blur technique
def gaussian_blur_image(image):
  # Apply Gaussian blur on the image with kernel 3x3
  gauss_blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
  return gauss_blurred_image