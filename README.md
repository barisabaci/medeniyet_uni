# Machine Learning Classification Project

## Description
This project demonstrates and compares various machine learning classification techniques, including Decision Trees, Random Forests, and Stacking ensemble methods. It utilizes two different datasets for these purposes: `Bank.csv` for credit scoring and `Credit.csv` for a general classification task.

## Files and Structure
*   **`DTree.py`**: This script performs credit scoring using Decision Tree and Random Forest classifiers on the `files/Bank.csv` dataset. It includes preprocessing steps such as Label Encoding for categorical features and evaluates the models using classification reports, confusion matrices, and feature importance analysis. It also visualizes the confusion matrices.
*   **`Stacking.py`**: This script demonstrates the Stacking ensemble method for classification using the `files/Credit.csv` dataset. It addresses class imbalance using the SMOTETomek technique and compares the performance of Decision Tree, Random Forest, and a Stacking classifier (which uses Logistic Regression as a meta-classifier). Model evaluation is done using cross-validation, and results are visualized.
*   **`files/Bank.csv`**: Dataset used by `DTree.py` for credit scoring tasks.
*   **`files/Credit.csv`**: Dataset used by `Stacking.py` for classification tasks.
*   **`README.md`**: This file, providing an overview and instructions for the project.

## How to Run
To execute the scripts, navigate to the project's root directory and run the following commands in your terminal:

```bash
python DTree.py
python Stacking.py
```

## Dependencies
* pandas
* scikit-learn
* matplotlib
* seaborn
* numpy
* joblib
* xgboost
* imbalanced-learn
* mlxtend
