# knearest
# Classified Data Analysis

This repository contains Python code for analyzing the "Classified Data.csv" dataset. The dataset consists of classified data where the target class is not explicitly described. The code utilizes various libraries such as `numpy`, `pandas`, `matplotlib`, `seaborn`, and machine learning models from `scikit-learn` for analysis and prediction.

## Overview

The provided Python script conducts exploratory data analysis (EDA) and builds a K-Nearest Neighbors (KNN) classification model to predict the target class based on the features from the dataset.

## Steps

1. **Data Loading and Exploration**:
   - Loads the dataset using `pandas`.
   - Displays the first few rows of the dataset using `df.head()`.
   - Displays the column names using `df.columns`.
   - Drops the 'Unnamed: 0' column from the dataset using `df.drop()`.

2. **Feature Scaling**:
   - Standardizes the features using `StandardScaler` from `sklearn.preprocessing`.
   - Transforms the dataset using the scaler.

3. **Data Splitting**:
   - Splits the dataset into features (X) and the target variable (y).
   - Splits the data into training and testing sets using `train_test_split` from `sklearn.model_selection`.

4. **Model Building**:
   - Instantiates a KNN classifier model using `KNeighborsClassifier` from `sklearn.neighbors`.
   - Fits the model to the training data using `k.fit(X_train, y_train)`.

5. **Model Evaluation**:
   - Predicts the target class labels for the test set using `k.predict(X_test)`.
   - Prints a classification report using `classification_report()` from `sklearn.metrics`.
   - Displays a confusion matrix using `confusion_matrix()` from `sklearn.metrics`.

## Prerequisites

Ensure that you have the following libraries installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Ensure that you have the "Classified Data.csv" file available at the specified location.
2. Execute the Python script.
3. Analyze the classification report and confusion matrix to evaluate the model's performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact the repository owner:
- Name: kavipriya
- Email: kavipriyapanneerselvam22@gmail.com
