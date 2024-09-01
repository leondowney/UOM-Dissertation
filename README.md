# UOM Dissertation: Variability in Predictions from Random Forest Models

## Overview

This repository contains the code and analysis conducted for the dissertation titled **"Variability in Predictions from Random Forest Models"**. The research explores the prediction variability inherent in Random Forest models, with a specific focus on healthcare datasets. The project compares various Random Forest variations, including Extra Trees, Totally Random Forest, and Adaboost Random Forest, and benchmarks them against traditional machine learning models.

## Files and Directories

- **DataframeToSklearn.py**: 
  - Utility script for converting data frames into a format compatible with scikit-learn models. It prepares datasets for training and testing by the different machine learning models used in this project.

- **Hyperparameter Sensitivity Analysis One.py**: 
  - This script evaluates the impact of different numbers of estimators (`n_estimators`) on the prediction variability of various Random Forest models. The analysis focuses on how the number of trees in a forest affects model performance across several healthcare datasets.

- **Hyperparameter Sensitivity Analysis Two.py**: 
  - This script explores the effect of varying the maximum depth (`max_depth`) of decision trees within Random Forest models. The analysis aims to understand how tree depth influences the model's ability to capture data patterns and its potential for overfitting.

- **Performance Comparison.py**: 
  - A script that compares the performance of different Random Forest variations against baseline machine learning models such as Support Vector Machines (SVM) and Multi-Layer Perceptron (MLP). This comparison is conducted using various evaluation metrics such as accuracy, precision, recall, and F1-score.

- **utility.py**: 
  - Contains utility functions used across the project, including model evaluation metrics, data loading functions, and other helper methods to streamline the analysis process.

## Datasets

The project uses several well-known healthcare datasets:
- **Breast Cancer Wisconsin (Diagnostic) Dataset**
- **Pima Indians Diabetes Dataset**
- **Heart Disease Dataset**

These datasets are used to evaluate the prediction variability and overall performance of the models.

## Running the Code

1. Clone the repository:

   ```bash
   git clone https://github.com/leondowney/UOM-Dissertation.git
   cd UOM-Dissertation
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts in the following order to reproduce the analysis:

   - `python Performance Comparison.py`
   - `python Hyperparameter Sensitivity Analysis One.py`
   - `python Hyperparameter Sensitivity Analysis Two.py`

## Results

The results of the analysis are documented within the dissertation. Key findings include the trade-offs between model complexity and prediction stability, particularly when varying hyperparameters like `n_estimators` and `max_depth`.

## Citation

If you use this code or the findings from this research, please cite the dissertation:

```
@thesis{leondowney2024,
  title={Variability in Predictions from Random Forest Models},
  author={Dongyu Liang},
  year={2024},
  school={The University of Manchester}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
