# Customer Churn Prediction System — End-to-End ML Application

**Devixo Solutions — AI/ML Internship (Week 4 / Task 04)**  
**Author:** Kazim Haider Syed

An end-to-end Machine Learning web application built with **Scikit-learn** and **Streamlit** that predicts telecom customer churn, outputs churn probabilities, surfaces actionable retention insights, and allows interactive comparison across 6 trained classification models.

---

## Project Structure

```text
Devixo-Task04-Customer-Churn-App/
├── app.py                           # Multi-page interactive Streamlit web application
├── train_pipeline.py                # End-to-end preprocessing, training, GridSearchCV tuning & export
├── requirements.txt                 # Python dependencies for local & Streamlit Cloud deployment
├── customer_churn_dataset.csv       # Telecom customer dataset (1,500 records)
├── best_churn_model.joblib          # Serialized best model (Tuned Gradient Boosting pipeline)
├── all_churn_models.joblib          # All 6 trained model pipelines for live model switching
├── task04_model_comparison.csv      # Evaluation metrics table across all 6 models
├── feature_importances.csv          # Feature importance rankings
├── task04_confusion_matrices.png    # Confusion matrices across all 6 models
├── task04_roc_curves.png            # ROC-AUC comparison curves
├── task04_feature_importance.png    # Feature importance bar chart
├── .gitignore                       # Standard Python & environment ignores
└── README.md                        # Project documentation
```

---

## Key Features Implemented

### 1. Data Processing & Feature Engineering (`train_pipeline.py`)
* **Missing Value Imputation:** Median imputation for numerical columns and most-frequent imputation for categorical columns inside a leak-free Scikit-learn `ColumnTransformer`.
* **Duplicate Removal & Outlier Handling:** Automatic deduplication and IQR-based capping on `Monthly_Charges` and `Support_Calls`.
* **Feature Engineering:** Created `Charges_Per_Tenure`, `High_Support_Risk`, and `Is_Month_To_Month`.
* **Encoding & Scaling:** `OneHotEncoder(drop="first")` for categorical features and `StandardScaler` for numerical features.

### 2. Model Comparison & Evaluation
Evaluated 6 classification models on a stratified 80/20 train-test split:

| Model Name | Training Time (s) | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Best)** | 0.3699 | **0.9367** | 0.9250 | **0.8506** | **0.8862** | **0.9793** |
| Random Forest | 0.3455 | 0.9300 | **0.9342** | 0.8161 | 0.8712 | 0.9709 |
| Decision Tree | 0.0228 | 0.9200 | 0.9200 | 0.7931 | 0.8519 | 0.9421 |
| SVM (RBF) | 0.2351 | 0.9100 | 0.9054 | 0.7701 | 0.8323 | 0.9651 |
| KNN | 0.0307 | 0.8933 | 0.8481 | 0.7701 | 0.8072 | 0.9354 |
| Logistic Regression | 0.0498 | 0.8833 | 0.8095 | 0.7816 | 0.7953 | 0.9391 |

### 3. Hyperparameter Tuning & Serialization
* Applied **5-fold `GridSearchCV`** on the top-performing Gradient Boosting pipeline (`learning_rate=0.1`, `max_depth=3`, `n_estimators=100`), confirming optimal generalization at **93.67% Accuracy**, **0.8862 F1 Score**, and **0.9793 ROC-AUC**.
* Exported the complete preprocessing + model pipeline using `joblib`.

### 4. Interactive Streamlit App (`app.py`) — Core + Bonus Features
* **Churn Prediction Engine:** Real-time prediction button, probability progress bar, tailored retention recommendations, and **live multi-model selector**.
* **Data Visualization & EDA Page:** KPI summary metrics, contract-level churn rates, billing distribution boxplots, and interactive data explorer.
* **Model Performance & Tuning Page:** Highlighted metrics leaderboard, GridSearchCV summary, Confusion Matrices, ROC-AUC curves, and Feature Importance plots.

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Devixo-Task04-Customer-Churn-App.git
cd Devixo-Task04-Customer-Churn-App

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain models from scratch
python train_pipeline.py

# 4. Launch the Streamlit web app
streamlit run app.py
```

---

## Deploying to Streamlit Community Cloud

1. Push all files in this folder to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud will automatically install `requirements.txt` and launch the live app.
