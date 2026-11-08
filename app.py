
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Classification")
st.write(
    "Upload test data and evaluate a saved machine learning model."
)

# Model file paths
MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Gaussian Naive Bayes": "models/gaussian_naive_bayes.pkl",
    "SVM": "models/svm.pkl"
}

# Model selection
selected_model = st.selectbox(
    "Select a Classification Model",
    list(MODEL_PATHS.keys())
)

# CSV upload
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(test_data.head())

    if "y" not in test_data.columns:
        st.error(
            "The uploaded CSV must contain the target column 'y' "
            "to calculate evaluation metrics."
        )

    else:
        X_test = test_data.drop("y", axis=1)
        y_test = test_data["y"]

        model_path = MODEL_PATHS[selected_model]

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")

        else:
            model = joblib.load(model_path)

            # Predict using the saved model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            precision = precision_score(
                y_test, y_pred, pos_label="yes", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, pos_label="yes", zero_division=0
            )
            f1 = f1_score(
                y_test, y_pred, pos_label="yes", zero_division=0
            )
            mcc = matthews_corrcoef(y_test, y_pred)

            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=["no", "yes"]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual no", "Actual yes"],
                columns=["Predicted no", "Predicted yes"]
            )

            st.dataframe(cm_df)

            # Classification report
            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                labels=["no", "yes"],
                target_names=["No", "Yes"],
                zero_division=0,
                output_dict=True
            )

            st.dataframe(pd.DataFrame(report).transpose())

            # Predictions
            st.subheader("Predictions")

            prediction_data = test_data.copy()
            prediction_data["Predicted y"] = y_pred

            st.dataframe(prediction_data)

            # Download predictions
            csv_output = prediction_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=csv_output,
                file_name="bank_marketing_predictions.csv",
                mime="text/csv"
            )
