# Devixo Solutions - AI/ML Internship | Task 04: End-to-End ML Pipeline
# Author: Kazim Haider Syed

import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

def main():
    # 1. Load Dataset
    df = pd.read_csv("customer_churn_dataset.csv")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Data Cleaning: Duplicates & Outlier Detection (IQR Capping)
    df.drop_duplicates(inplace=True)
    for col in ["Monthly_Charges", "Support_Calls"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # 3. Feature Engineering
    df["Charges_Per_Tenure"] = np.round(df["Total_Charges"].fillna(df["Monthly_Charges"]) / (df["Tenure_Months"] + 1), 2)
    df["High_Support_Risk"] = (df["Support_Calls"] >= 3).astype(int)
    df["Is_Month_To_Month"] = (df["Contract_Type"] == "Month-to-month").astype(int)

    num_cols = ["Tenure_Months", "Monthly_Charges", "Total_Charges", "Support_Calls", "Charges_Per_Tenure", "High_Support_Risk", "Is_Month_To_Month"]
    cat_cols = ["Contract_Type", "Internet_Service", "Payment_Method"]

    X = df.drop(["CustomerID", "Churn"], axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 4. Preprocessing Pipeline (Imputation + Scaling + OneHotEncoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))]), cat_cols)
        ]
    )

    # 5. Train & Evaluate 6 ML Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
    }

    results = []
    pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = round(time.time() - t0, 4)
        pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "Model Name": name,
            "Training Time (s)": train_time,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
        })

    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    results_df.to_csv("task04_model_comparison.csv", index=False)
    print("\nModel Evaluation Results:\n", results_df.to_string(index=False))

    # 6. Hyperparameter Tuning (GridSearchCV on Gradient Boosting)
    gb_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", GradientBoostingClassifier(random_state=42))])
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 4]
    }
    grid_search = GridSearchCV(gb_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 7. Save Models with Joblib
    joblib.dump(best_model, "best_churn_model.joblib")
    joblib.dump(pipelines, "all_churn_models.joblib")
    print("Saved best_churn_model.joblib and all_churn_models.joblib successfully.")

if __name__ == "__main__":
    main()
