from src.image_preprocessing.image_preprocessing_pipelines import baseline_preprocessing, grayscale_preprocessing, clahe_preprocessing
from src.image_preprocessing.image_preprocessing_pipelines import gamma_correction_preprocessing, median_blur_preprocessing
from src.image_preprocessing.image_preprocessing_pipelines import grayscale_gamma_correction_preprocessing, grayscale_median_blur_preprocessing
from src.image_preprocessing.image_preprocessing_pipelines import clahe_gamma_correction_preprocessing, clahe_median_blur_preprocessing
from src.image_preprocessing.image_preprocessing_pipelines import gamma_correction_median_blur_preprocessing, grayscale_clahe_gamma_correction_median_blur_preprocessing

IMAGE_PREPROCESSORS = {
    'baseline_preprocessor': baseline_preprocessing,
    'grayscale_preprocessor': grayscale_preprocessing,
    'clahe_preprocessor': clahe_preprocessing,
    'gamma_correction_preprocessor': gamma_correction_preprocessing,
    'median_blur_preprocessor': median_blur_preprocessing,
    'grayscale_gamma_correction_preprocessor': grayscale_gamma_correction_preprocessing,
    'grayscale_median_blur_preprocessor': grayscale_median_blur_preprocessing,
    'clahe_gamma_correction_preprocessor': clahe_gamma_correction_preprocessing,
    'clahe_median_blur_preprocessor': clahe_median_blur_preprocessing,
    'gamma_correction_median_blur_preprocessor': gamma_correction_median_blur_preprocessing,
    'grayscale_clahe_gamma_correction_median_blur_preprocessor': grayscale_clahe_gamma_correction_median_blur_preprocessing
}