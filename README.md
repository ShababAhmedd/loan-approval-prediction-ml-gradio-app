# Loan Approval Prediction System

This project is a machine learning-based **Loan Approval Prediction System** that predicts whether a loan application will be approved or not based on applicant details. It includes full data preprocessing, model training, evaluation, and deployment using a Gradio web interface hosted on Hugging Face Spaces.

Dataset used:  
https://www.kaggle.com/datasets/ninzaami/loan-predication

---

##  Dataset Information

The dataset contains information such as:

- Gender
- Marital Status
- Education
- Income details
- Loan amount
- Credit history
- Property area
- Loan status (Target variable)

Target Variable:
- `Loan_Status` (Y = Approved, N = Not Approved)

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Gradio (for UI)
- Hugging Face Spaces (deployment)
- Pickle (model saving)

---

## Data Preprocessing

The following preprocessing steps were performed:

- Handling missing values using **median, and mode imputation**
- Encoding categorical variables using **Label Encoding and One-Hot Encoding**
- Feature transformation using **log transformation**
- Feature scaling using **StandardScaler**
- Outlier detection using **IQR method**
- Feature engineering (e.g., Family Size creation)
- Removal of irrelevant features (`Loan_ID`)

---

## Pipeline Creation

A complete ML pipeline was created using `ColumnTransformer` and `Pipeline` to ensure:

- Clean preprocessing flow
- No data leakage
- Easy model training and deployment

---

## Models Used

The following models were trained and compared:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Voting Classifier (Ensemble)
- Stacking Classifier (Ensemble)

---

## Best Model Selection

After evaluation, **Logistic Regression** performed the best based on:

- Accuracy
- Precision
- Recall
- F1 Score

It was further optimized using **GridSearchCV**.

Best Parameters:
```python
C = 1
max_iter = 100
penalty = 'l2'
solver = 'lbfgs'
```

---

## Model Evaluation

Final performance on test data:

- Accuracy: ~0.85
- F1 Score: ~0.90
- High Recall for approved loans (important for business use case)

## Cross Validation
- 10-Fold Stratified Cross Validation was used
- Mean F1 Score: ~0.87
- Standard deviation: ~0.03

---

## Web Application (Gradio)

A user-friendly web interface was built using Gradio where users can input the features.

And get instant prediction: Loan Approved / Not Approved

## Deployment

The application is deployed using Hugging Face Spaces.

[Live App Link](https://huggingface.co/spaces/ShababAhmed0/loan-approval-prediction-ml-gradio-app)


## Project Structure
```
.
├── loan_approval_prediction.csv (Dataset)
├── Untitled0.ipynb (notebook)
├── best_lr_model.pkl (best saved model)
├── app.py (Gradio)
├── requirements.txt
├── ydata.html
├── README.md
```
