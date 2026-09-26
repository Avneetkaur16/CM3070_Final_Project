# Comparative Analyses of CNN Architectures for Breast Cancer Classification

## Project Folder Structure

* src
  * dataset
    * data_augmentation.py
    * dataset.py
  * image_preprocessing
    * image_preprocessing_functions.py
    * image_preprocessing_pipelines.py
    * image_preprocessors.py
  * models
    * densenet121CNN.py
    * efficientnetb0CNN.py
    * mobilenetv2CNN.py
    * resnet50CNN.py
    * vgg16CNN.py
  * notebooks
    * final-project-cbis-ddsm-dataset-preparation.ipynb
    * Baselines.ipynb
    * experiments
      * Image_Preprocessing_Experiments.ipynb
      * View_Specific_Training_Experiments.ipynb
      * Lesion_Specific_Training_Experiments.ipynb
      * Experimental_Analysis.ipynb
    * Image_Preprocessing_And_Data_Augmentation_Visualization.ipynb
    * final_configurations
      * MobileNetV2_Calcification_Classification.ipynb
      * ResNet50_Mass_Classification.ipynb
  * results
    * baseline_results.csv
    * image_preprocessing_experiment_results.csv
    * views_experiment_results.csv
    * lesions_experiment_results.csv
  * workflows
    * dataset_generation.py
    * training_and_evaluation.py
  * calibration_scores_functions.py
  * common_model_functions.py
  * constants.py
  * experimental_results_functions.py
  * final_results_functions.py
  * grad_cam_functions.py
  * utils.py
  * visualizations.py
  * Project_Demo.ipynb


## Python Files and their tasks

1. **dataset.py:** Functions pertaining to generating tf.data Dataset, shuffling, batching, augmenting data and adding class weights.

2. **data_augmentation.py:** Data augmentation sequential model and a function that applies augmentation to an image tensor.

3. **image_preprocessing_functions.py:** Functions pertaining to image transformations (resize, CLAHE, Histogram Equalization, Median Blur, Gaussian Blur)

4. **image_preprocessing_pipelines.py:** Pipeline functions pertaining to combinations of image preprocessing functions (baseline, CLAHE + Median Blur, Histogram Equalization + Gaussian Blur)

5. **image_preprocessors.py:** Dictionary containing name and function of the image preprocessing pipelines for easier access.

6. **vgg16CNN.py:** VGG16 classifier model

7. **resnet50CNN.py:** ResNet50 classifier model

8. **densenet121CNN.py:** DenseNet121 classifier model

9. **mobilenetv2CNN.py:** MobileNetV2 classifier model

10. **efficientnetb0CNN.py:** EfficientNet-B0 classifier model

11. **final-project-cbis-ddsm-dataset-preparation.ipynb:** CBIS-DDSM Dataset (JPEG version) loading and preparation Kaggle Notebook.

12. **Baselines.ipynb:** All baseline models

13. **Image_Preprocessing_And_Data_Augmentation_Visualization.ipynb:** Visualizations of all image preprocessing combinations (pipelines) and data augmentation

14. **Image_Preprocessing_Experiments.ipynb:** Notebook containing all Image Preprocessing Experiments using all CNN models

15. **View_Specific_Training_Experiments.ipynb:** Notebook containing all View-Specific Training Experiments using all CNN models

16. **Lesion_Specific_Training_Experiments.ipynb:** Notebook containing all Lesion-Specific Training Experiments using all CNN models

17. **Experimental_Analysis.ipynb:** Notebook containing graphical + tabular analysis and inference of 'Baselines' with 'Image-Preprocessing', 'View-Specific Training', 'Lesion-Specific Training' experiments.

18. **MobileNetV2_Calcification_Classification.ipynb:** Calcification-only final configuration with MobileNetV2 notebook (with further training, optimal threshold calculations using Youden's Index and ROC Curve, Temperature Scaling, ECE Scores, Image-Level and Patient-Level Evaluations and Grad-CAM)

19. **ResNet50_Mass_Classification.ipynb:** Mass-only final configuration with ResNet50 notebook (with further training, optimal threshold calculations using Youden's Index and ROC Curve, Temperature Scaling, ECE Scores, Image-Level and Patient-Level Evaluations and Grad-CAM)

20. **baseline_results.csv:** Baseline results CSV

21. **image_preprocessing_experiment_results:** Image Preprocessing Experiment Results CSV

22. **views_experiment_results:** View-Specific Training Experiment Results CSV

23. **lesions_experiment_results:** Lesion-Specific Training Experiment Results CSV

24. **dataset_generation.py:** Full pipeline for generating tf.data Dataset (using functions from dataset.py)

25. **training_and_evaluation.py:** Full pipelines for training a model and evaluating (image-level) a trained model

26. **calibration_scores_functions.py:** Functions pertaining to Temperature Scaling, NLL loss and ECE Score calculation

27. **common_model_functions.py:** Model selection by name and model compilation functions commonly used for all models

28. **constants.py:** All constants used in this project

29. **experimental_results_functions.py:** Functions to store experimental results in CSV files for all 3 experiments.

30. **final_results_functions.py:** Functions for Patient-Level evaluations and Youden's Index computation

31. **grad_cam_functions.py:** Grad-CAM functions for heatmap generation and superimposing heatmaps on images

32. **utils.py:** Utility functions

33. **visualizations.py:** Visualization functions for generating graphs

34. **Project_Demo.ipynb:** Project Demo Notebook (for video demo)
