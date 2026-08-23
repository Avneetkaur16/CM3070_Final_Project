from src.image_preprocessing.image_preprocessing_pipelines import baseline_preprocessing, clahe_median_blur_preprocessing

IMAGE_PREPROCESSORS = {
    'baseline': baseline_preprocessing,
    'clahe_median_blur': clahe_median_blur_preprocessing,
}