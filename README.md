

# Bank Marketing Classification

## Project Overview

This project applies five machine learning classification algorithms to the Bank Marketing dataset to predict whether a customer will subscribe to a term deposit.

The project includes data preprocessing, model training, evaluation using multiple performance metrics, and a Streamlit web application for model evaluation and prediction.

## Dataset

- **Dataset:** Bank Marketing Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank)
- **Instances:** 4,521
- **Input Features:** 16
- **Target Variable:** `y` (`yes` / `no`)
- **Problem Type:** Binary Classification

## Machine Learning Models

The following five classification models are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Data Preprocessing

The preprocessing steps include:

- Separating input features and the target variable
- Identifying numerical and categorical features
- Applying `StandardScaler` to numerical features
- Applying `OneHotEncoder` to categorical features
- Splitting the dataset into training and testing sets
- Using a preprocessing pipeline for consistent data transformation

## Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

These metrics help compare model performance, particularly when the target classes are imbalanced.

## Streamlit Application

The Streamlit application provides an interactive interface for model evaluation and prediction.

Users can:

- Select a trained classification model
- Upload a test CSV file
- View evaluation metrics
- View the confusion matrix
- View the classification report
- Generate predictions
- Download prediction results

## Project Structure

```text
Bank-Marketing-Classification/
│
├── app.py
├── ML Ass 2.ipynb
├── bank.csv
├── test_data.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── gaussian_naive_bayes.pkl
│   └── svm.pkl
│
└── README.md
```

## Installation

Make sure Python is installed on your system.

Install the required Python packages:

```bash
python -m pip install streamlit pandas scikit-learn joblib
```

## How to Run the Application

Navigate to the project directory:

```bash
cd Bank-Marketing-Classification
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

## Repository

[Bank-Marketing-Classification on GitHub](https://github.com/gvrajan97/Bank-Marketing-Classification)

## Dataset Reference

UCI Machine Learning Repository: [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank)

## Conclusion

This project demonstrates the implementation and comparison of five machine learning classification algorithms for predicting term deposit subscriptions.

The Streamlit application provides an interactive way to evaluate trained models, view performance metrics, and generate predictions.

