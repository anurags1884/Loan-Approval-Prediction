# Loan Approval Prediction Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to predict whether a loan application is likely to be approved or rejected based on applicant information, financial details, asset values, and CIBIL score.

The objective is to analyze historical loan application data, identify relevant patterns, train machine learning models, and evaluate their predictive performance.

## Objectives

- Clean and preprocess the dataset.
- Perform Exploratory Data Analysis (EDA).
- Engineer relevant features, including the loan-to-income ratio.
- Train and evaluate Logistic Regression and Random Forest classifiers.
- Compare model performance using classification metrics.
- Save and reload the selected model.
- Test the model on a sample loan application.

## Dataset

The dataset contains 4,269 loan application records and 13 original columns.

Key features include:

- Number of dependents
- Education
- Self-employment status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential, commercial, and luxury asset values
- Bank asset value
- Loan status (target variable)

The `loan_id` column was excluded from model training because it serves as an identifier.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Google Colab
- GitHub

## Project Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Categorical variable encoding
6. Train-test split using stratification
7. Feature scaling for Logistic Regression
8. Model training using Logistic Regression and Random Forest
9. Model evaluation and comparison
10. Five-fold stratified cross-validation
11. Confusion matrix analysis
12. Model saving and reloading
13. Sample loan application prediction

## Feature Engineering

A loan-to-income ratio was created using:

`loan_to_income_ratio = loan_amount / income_annum`

A flag was also created to identify negative residential asset values while preserving the original values.

Categorical variables were converted into numerical features using one-hot encoding.

## Model Performance

The dataset was split into 80% training data and 20% test data.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 91.22% | 99.53% |
| Precision | 92.22% | 99.25% |
| Recall | 93.79% | 100.00% |
| F1-score | 93.00% | 99.62% |

Random Forest achieved the best performance on the test set and was selected as the final model.

Five-fold stratified cross-validation was also performed on the training data. The Random Forest achieved a mean accuracy of approximately 99.56%.

These results reflect performance on the available dataset. Independent validation is necessary before real-world use.

## Confusion Matrix

The Random Forest confusion matrix on the test set was:

| Actual / Predicted | Rejected | Approved |
|---|---:|---:|
| Rejected | 319 | 4 |
| Approved | 0 | 531 |

The model correctly classified 850 out of 854 test records.

## Repository Files

- `loan_approval_prediction.ipynb` — notebook containing the analysis, preprocessing, training, and evaluation.
- `loan_approval_rf_model.pkl` — saved Random Forest classifier.
- `loan_approval_features.pkl` — ordered list of features expected by the model.
- `README.md` — project documentation.

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Loan-Approval-Prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your GitHub repository.

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

### 3. Open the notebook

```bash
jupyter notebook loan_approval_prediction.ipynb
```

Run the notebook cells in order. If the original dataset is not included in the repository, download it from its authorized source and update the notebook's dataset path accordingly.

## Limitations

- The model predicts historical loan approval status, not loan repayment or default.
- The high test accuracy may reflect strong patterns specific to this dataset.
- Performance may differ on new data or real-world loan applications.
- The model should not be used as the sole basis for actual lending decisions without further validation, fairness assessment, and appropriate governance.

## Future Improvements

- Validate the model on an independent dataset.
- Perform further checks for data leakage and overfitting.
- Add explainability using feature importance or SHAP.
- Build an interactive prediction interface using Streamlit.
- Deploy the application after appropriate testing.

## Author

**Anurag Shinde**

Aspiring Data Analyst | Machine Learning Enthusiast

## Disclaimer

This project is intended for educational and demonstration purposes only. Predictions are not financial advice and do not guarantee loan approval or rejection.
