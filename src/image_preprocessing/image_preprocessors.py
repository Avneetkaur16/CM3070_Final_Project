from src.image_preprocessing.image_preprocessing_pipelines import baseline_preprocessing, grayscale_preprocessing, clahe_preprocessing
from src.image_preprocessing.image_preprocessing_pipelines import gamma_correction_preprocessing, median_blur_preprocessing
from src.image_preprocessing.image_preprocessing_pipelines import grayscale_gamma_correction_preprocessing, grayscale_median_blur_preprocessing
from src.image_preprocessing.image_preprocessing_pipelines import clahe_gamma_correction_preprocessing, clahe_median_blur_preprocessing
from src.image_preprocessing.image_preprocessing_pipelines import gamma_correction_median_blur_preprocessing, grayscale_clahe_gamma_correction_median_blur_preprocessing

IMAGE_PREPROCESSORS = {
    'P0': baseline_preprocessing,
    'P1': grayscale_preprocessing,
    'P2': clahe_preprocessing,
    'P3': gamma_correction_preprocessing,
    'P4': median_blur_preprocessing,
    'P5': grayscale_gamma_correction_preprocessing,
    'P6': grayscale_median_blur_preprocessing,
    'P7': clahe_gamma_correction_preprocessing,
    'P8': clahe_median_blur_preprocessing,
    'P9': gamma_correction_median_blur_preprocessing,
    'P10': grayscale_clahe_gamma_correction_median_blur_preprocessing
}