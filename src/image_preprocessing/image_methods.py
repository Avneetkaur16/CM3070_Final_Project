import numpy as np
import cv2

# Resize an image to 224 x 224
def resize_image(image):
  image = cv2.resize(image, (224, 224))
  return image

# Normalize an image
def normalize_image(image):
  image = (image / 255.0).astype(np.float32)
  return image