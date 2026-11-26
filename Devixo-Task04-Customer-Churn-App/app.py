# Devixo Solutions - AI/ML Internship | Task 04: End-to-End Streamlit ML Application
# Author: Kazim Haider Syed

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer Churn Prediction System | Devixo AI",
    page_icon="📊",
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_artifacts():
    best_model = joblib.load("best_churn_model.joblib")
    all_models = joblib.load("all_churn_models.joblib")
    df = pd.read_csv("customer_churn_dataset.csv")
    metrics_df = pd.read_csv("task04_model_comparison.csv")
    fi_df = pd.read_csv("feature_importances.csv")
    return best_model, all_models, df, metrics_df, fi_df

best_model, all_models, df, metrics_df, fi_df = load_artifacts()

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["🔮 Churn Prediction Engine", "📈 Data Visualization & EDA", "🏆 Model Performance & Tuning"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Devixo Solutions — Task 04**\n\n"
    "End-to-End Customer Churn Prediction System with Multi-Model Selection, "
    "Feature Engineering, and Hyperparameter Tuning."
)

# ==============================================================================
# PAGE 1: INTERACTIVE PREDICTION ENGINE (With Bonus Multiple Model Selection)
# ==============================================================================
if page == "🔮 Churn Prediction Engine":
    st.title("🔮 Customer Churn Prediction System")
    st.markdown("Enter customer account details below to predict churn risk, view probability scores, and inspect key risk drivers.")

    col_model, _ = st.columns([1, 2])
    with col_model:
        selected_model_name = st.selectbox(
            "Select ML Model for Inference (Bonus):",
            ["Tuned Gradient Boosting (Best)"] + list(all_models.keys())
        )

    active_model = best_model if selected_model_name == "Tuned Gradient Boosting (Best)" else all_models[selected_model_name]

    st.subheader("📋 Customer Profile & Billing Inputs")
    c1, c2, c3 = st.columns(3)

    with c1:
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with c2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=85.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=10.0, max_value=15000.0, value=float(tenure * monthly_charges), step=10.0)

    with c3:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        support_calls = st.number_input("Customer Support Calls", min_value=0, max_value=15, value=2, step=1)

    if st.button("🚀 Predict Customer Churn", type="primary", use_container_width=True):
        # Apply identical feature engineering as training pipeline
        charges_per_tenure = round(total_charges / (tenure + 1), 2)
        high_support_risk = int(support_calls >= 3)
        is_month_to_month = int(contract == "Month-to-month")

        input_df = pd.DataFrame([{
            "Tenure_Months": tenure,
            "Monthly_Charges": monthly_charges,
            "Total_Charges": total_charges,
            "Support_Calls": support_calls,
            "Charges_Per_Tenure": charges_per_tenure,
            "High_Support_Risk": high_support_risk,
            "Is_Month_To_Month": is_month_to_month,
            "Contract_Type": contract,
            "Internet_Service": internet,
            "Payment_Method": payment
        }])

        pred = active_model.predict(input_df)[0]
        prob = active_model.predict_proba(input_df)[0][1]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred == 1:
                st.error(f"### ⚠️ High Churn Risk Detected (Churn = Yes)")
                st.metric("Churn Probability", f"{prob * 100:.2f}%", delta="High Risk", delta_color="inverse")
            else:
                st.success(f"### ✅ Customer Likely to Retain (Churn = No)")
                st.metric("Churn Probability", f"{prob * 100:.2f}%", delta="Low Risk", delta_color="normal")

            st.progress(float(prob))
            st.caption(f"Model Used: **{selected_model_name}**")

        with res_col2:
            st.subheader("💡 Actionable Retention Insights")
            if contract == "Month-to-month":
                st.warning("• **Contract Risk:** Month-to-month subscribers churn significantly more. Offer a 15% discount to switch to a 1-year contract.")
            if support_calls >= 3:
                st.warning(f"• **Service Friction:** {support_calls} support calls logged. Trigger priority technical callback.")
            if internet == "Fiber optic" and monthly_charges > 70:
                st.info("• **Billing Sensitivity:** High Fiber Optic monthly bill detected. Bundle loyalty perks.")
            if pred == 0:
                st.info("• **Healthy Account:** Maintain standard engagement cadence.")

    # Feature Importance Section on Main Page
    st.markdown("---")
    st.subheader("🔍 Global Feature Importance (What Drives Churn?)")
    fig_fi, ax_fi = plt.subplots(figsize=(10, 4))
    sns.barplot(data=fi_df.head(8), x="Importance", y="Feature", palette="viridis", ax=ax_fi)
    ax_fi.set_title("Top 8 Most Predictive Features")
    st.pyplot(fig_fi)

# ==============================================================================
# PAGE 2: DATA VISUALIZATION & EDA
# ==============================================================================
elif page == "📈 Data Visualization & EDA":
    st.title("📈 Dataset Overview & Exploratory Data Analysis")
    st.markdown(f"Analyzing **{len(df):,} customer records** across billing, contract, and support dimensions.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{len(df):,}")
    m2.metric("Overall Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")
    m3.metric("Avg Monthly Charge", f"${df['Monthly_Charges'].mean():.2f}")
    m4.metric("Avg Tenure", f"{df['Tenure_Months'].mean():.1f} mos")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Rate by Contract Type")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x="Contract_Type", y="Churn", palette="magma", ax=ax1)
        ax1.set_ylabel("Churn Proportion")
        st.pyplot(fig1)

    with c2:
        st.subheader("Monthly Charges Distribution by Churn")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Churn", y="Monthly_Charges", palette="coolwarm", ax=ax2)
        ax2.set_xticklabels(["Retained (0)", "Churned (1)"])
        st.pyplot(fig2)

    st.subheader("Raw Dataset Sample")
    st.dataframe(df.head(25), use_container_width=True)

# ==============================================================================
# PAGE 3: MODEL PERFORMANCE & HYPERPARAMETER TUNING
# ==============================================================================
elif page == "🏆 Model Performance & Tuning":
    st.title("🏆 Model Comparison & Hyperparameter Tuning")
    st.markdown("Comparison across **6 Machine Learning algorithms** evaluated on a 20% stratified hold-out test set.")

    st.subheader("📊 Evaluation Metrics Table")
    st.dataframe(
        metrics_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], color="#d1fae5"),
        use_container_width=True
    )

    st.subheader("⚡ GridSearchCV Hyperparameter Tuning (Gradient Boosting)")
    t1, t2, t3 = st.columns(3)
    t1.metric("Best CV Parameters", "lr=0.1, depth=3, n_est=100")
    t2.metric("Tuned F1-Score", "0.8862", delta="Optimal")
    t3.metric("Tuned ROC-AUC", "0.9793", delta="0.98 AUC")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image("task04_confusion_matrices.png", caption="Confusion Matrices Across All 6 Models", use_container_width=True)
    with col_img2:
        st.image("task04_roc_curves.png", caption="ROC-AUC Curves Comparison", use_container_width=True)
