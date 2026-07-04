import cv2
from image_methods import resize_image, normalize_image

# Baseline image preprocessing
def baseline_preprocessing(image_path):
  image = cv2.imread(image_path)
  image = resize_image(image)
  image = normalize_image(image)
  return image

# Grayscale + baseline image preprocessing
def grayscale_preprocessing(image_path):
  image = cv2.imread(image_path)
  image = resize_image(image)
  gray_image_1_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_image_3_channel = cv2.cvtColor(gray_image_1_channel, cv2.COLOR_GRAY2BGR)
  gray_image_3_channel = normalize_image(gray_image_3_channel)
  return gray_image_3_channel