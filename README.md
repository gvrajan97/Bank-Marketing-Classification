

# Bank Marketing Classification

## a. Problem Statement

The objective of this project is to develop and compare five machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on customer and campaign-related information.

The project includes data preprocessing, model training, evaluation using multiple performance metrics, and the development of an interactive Streamlit web application for model evaluation and prediction.

## b. Dataset Description

- **Dataset:** Bank Marketing Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank)
- **Number of Instances:** 4,521
- **Number of Input Features:** 16
- **Target Variable:** `y` (`yes` / `no`)
- **Problem Type:** Binary Classification

The dataset contains customer information and details related to direct marketing campaigns conducted by a bank. The target variable indicates whether the customer subscribed to a term deposit.

## c. GitHub Repository Link

[Bank-Marketing-Classification on GitHub](https://github.com/gvrajan97/Bank-Marketing-Classification)

The repository contains the source code, Jupyter notebook, dataset, test data, trained models, README, and application files.

## d. Models Used

The following five classification algorithms were implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Data Preprocessing

The following preprocessing steps were performed:

- Separated input features and target variable.
- Identified numerical and categorical features.
- Applied `StandardScaler` to numerical features.
- Applied `OneHotEncoder` to categorical features.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used a preprocessing pipeline to ensure consistent transformation.

### Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8928 | 0.8913 | 0.5636 | 0.2981 | 0.3899 | 0.3579 |
| Decision Tree | 0.8586 | 0.6942 | 0.4032 | 0.4808 | 0.4386 | 0.3602 |
| KNN | 0.8840 | 0.7692 | 0.4872 | 0.1827 | 0.2657 | 0.2477 |
| Gaussian Naive Bayes | 0.8210 | 0.7881 | 0.3092 | 0.4519 | 0.3672 | 0.2737 |
| Support Vector Machine | 0.8939 | 0.8830 | 0.5909 | 0.2500 | 0.3514 | 0.3374 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 89.28% and the highest AUC score of 0.8913. It provides good overall discrimination between the two classes, but its recall is relatively low. |
| Decision Tree | Achieved a recall of 48.08%, the highest among the five models. It identifies more positive cases but has lower accuracy and AUC compared to Logistic Regression and SVM. |
| KNN | Achieved an accuracy of 88.40%, but its recall of 18.27% is low. It misses a considerable number of actual positive cases. |
| Gaussian Naive Bayes | Achieved a recall of 45.19%, but its accuracy and precision are comparatively low. It identifies positive cases reasonably well but produces more false positives. |
| Support Vector Machine | Achieved the highest accuracy of 89.39% and precision of 59.09%. It provides strong overall accuracy, but its recall is relatively low. |

### Overall Winner for the Dataset

**Logistic Regression is selected as the overall winner based on its highest AUC score of 0.8913.**

It provides strong overall discrimination between customers who subscribe and those who do not. However, the Support Vector Machine achieves the highest accuracy of 89.39%, while the Decision Tree achieves the highest recall of 48.08%.

The best model depends on the evaluation objective. Logistic Regression is selected here based on AUC, which measures the model's ability to distinguish between the two classes.

## Streamlit Application

The Streamlit application provides an interactive interface for model evaluation and prediction.

Users can:

- Upload a test CSV file
- Select a trained classification model
- View evaluation metrics
- View the confusion matrix
- View the classification report
- Generate predictions
- Download prediction results

The application loads the saved models and evaluates the uploaded test data without retraining the models.

### How to Run the Application

Install the required packages:

```bash
python -m pip install streamlit pandas scikit-learn numpy joblib matplotlib seaborn
```

Navigate to the project directory:

```bash
cd Bank-Marketing-Classification
```

Run the application:

```bash
python -m streamlit run app.py
```

The application will open in the browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

## Project Structure

```text
Bank-Marketing-Classification/
├── app.py
├── ML Ass 2.ipynb
├── bank.csv
├── test_data.csv
├── requirements.txt
├── README.md
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── gaussian_naive_bayes.pkl
    └── svm.pkl
```

## Dataset Reference

UCI Machine Learning Repository: [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank)

## Conclusion

This project demonstrates the implementation and comparison of five machine learning classification algorithms for predicting term deposit subscriptions.

The results show that Logistic Regression provides the highest AUC, Support Vector Machine achieves the highest accuracy, and Decision Tree achieves the highest recall. The Streamlit application enables interactive evaluation and prediction using the trained models.

