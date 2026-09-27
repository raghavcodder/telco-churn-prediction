# Telco Customer Churn Prediction

Predicting customer churn for a telecom company using exploratory data analysis and classification models.

## 📌 Overview

Customer churn — when a customer stops using a company's service — is costly for subscription-based businesses like telecoms. This project analyzes the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to understand what drives churn and builds a model to predict it.

## 📂 Project Structure

```
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│   └── tel_churn.csv                          # Cleaned/processed dataset used for modeling
├── notebooks/
│   ├── EDA_telcom.ipynb                       # Exploratory data analysis
│   └── Model_building.ipynb                   # Model training & evaluation
├── requirements.txt
└── README.md
```

## 🔍 Exploratory Data Analysis

`notebooks/EDA_telcom.ipynb` covers:
- Data cleaning (handling missing values, type conversions)
- Univariate and bivariate analysis of customer attributes (tenure, contract type, monthly charges, etc.)
- Visualizing churn distribution and its relationship with other features

## 🤖 Model Building

`notebooks/Model_building.ipynb` covers:
- Train/test split
- Handling class imbalance with **SMOTEENN** (SMOTE + Edited Nearest Neighbours)
- Training and evaluating:
  - Decision Tree Classifier
  - Random Forest Classifier
- Comparing accuracy, precision, recall, and F1-score (accuracy alone is misleading here since the dataset is imbalanced)
- Saving the final trained model with `pickle`

## ✅ Results

Both Decision Tree and Random Forest reached ~94% accuracy after SMOTEENN resampling, with strong precision/recall/F1 on the minority (churn) class. Decision Trees was chosen as the final model as it performed slightly better with: 93.81% accuracy

              precision    recall  f1-score   support

           0       0.95      0.92      0.93       524
           1       0.93      0.96      0.94       624

    accuracy                           0.94      1148
   macro avg       0.94      0.94      0.94      1148
weighted avg       0.94      0.94      0.94      1148

## 🛠️ Tech Stack

- Python, pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTEENN)
- matplotlib, seaborn
- Jupyter Notebook

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook
```

## 📈 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Try additional models (XGBoost, Logistic Regression baseline)
- Deploy as a simple web app (Flask/Streamlit) for interactive churn prediction
