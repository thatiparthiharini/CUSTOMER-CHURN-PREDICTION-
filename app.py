import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

# KPI Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Model Algorithm", "Random Forest")
col2.metric("Total Features", "9")
col3.metric("Project Domain", "Telecom")

st.markdown("---")

# -----------------------------------
# Sidebar Navigation
# -----------------------------------
page = st.sidebar.selectbox(
    "Navigation",
    ["Home",
     "Predict Customer Churn",
     "Feature Importance",
     "Model Performance",
     "About"]
)

# -----------------------------------
# Session State
# -----------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# -----------------------------------
# HOME PAGE
# -----------------------------------
if page == "Home":

    st.subheader("Welcome to Customer Churn Prediction System")

    st.info(
        "📌 This system predicts telecom customer churn using machine learning."
    )

    st.write("""
Customer churn means a customer **stops using the company's telecom service**.

Businesses use churn prediction to:

• Identify customers likely to leave  
• Improve retention strategies  
• Reduce revenue loss
""")

# -----------------------------------
# PREDICTION PAGE
# -----------------------------------
elif page == "Predict Customer Churn":

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 18, 80, 30)

        monthly_charges = st.number_input(
            "Monthly Charges", 0.0, 5000.0, 500.0
        )

        tenure = st.slider("Tenure (Months)", 0, 72, 12)

        senior = st.selectbox("Senior Citizen", [0, 1])

    with col2:

        gender = st.selectbox("Gender", ["Male", "Female"])

        dependents = st.selectbox("Dependents", ["No", "Yes"])

        internet = st.selectbox("Internet Service", ["DSL", "Fiber"])

        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        payment = st.selectbox(
            "Payment Method",
            ["CreditCard", "BankTransfer",
             "ElectronicCheck", "MailedCheck"]
        )

    if st.button("Predict"):

        input_df = pd.DataFrame({
            "Age": [age],
            "MonthlyCharges": [monthly_charges],
            "Tenure": [tenure],
            "Gender": [gender],
            "SeniorCitizen": [senior],
            "Dependents": [dependents],
            "InternetService": [internet],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment]
        })

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("## Prediction Result")

        if probability > 0.7:
            st.error("⚠️ High Risk Customer (Likely to Churn)")
        elif probability > 0.4:
            st.warning("⚡ Medium Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

        st.write(f"### Churn Probability: {round(probability*100,2)}%")

        # Risk Indicator
        st.subheader("Customer Risk Level")

        st.progress(int(probability * 100))

        # Probability Chart
        fig, ax = plt.subplots()

        ax.bar(["Churn Probability"], [probability*100])

        ax.set_ylim(0, 100)

        ax.set_ylabel("Percentage")

        st.pyplot(fig)

        # Prediction History
        st.session_state.history.append(round(probability*100,2))

        st.subheader("Prediction History")

        st.write(st.session_state.history)

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------
elif page == "Feature Importance":

    st.subheader("Feature Importance Analysis")

    st.write("This chart shows which features influence churn prediction the most.")

    try:

        importances = model.named_steps["classifier"].feature_importances_

        feature_names = [
            "Age",
            "MonthlyCharges",
            "Tenure",
            "SeniorCitizen",
            "Gender",
            "Dependents",
            "InternetService",
            "PaperlessBilling",
            "PaymentMethod"
        ]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances[:len(feature_names)]
        })

        st.bar_chart(importance_df.set_index("Feature"))

    except:

        st.write("Feature importance not available.")

# -----------------------------------
# MODEL PERFORMANCE
# -----------------------------------
elif page == "Model Performance":

    st.subheader("Model Performance Evaluation")

    st.write("""
This section shows model performance using a confusion matrix.
""")

    # Example confusion matrix
    y_true = [0,0,0,1,1,1,0,1,0,1]
    y_pred = [0,0,1,1,1,0,0,1,0,1]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(ax=ax)

    st.pyplot(fig)

# -----------------------------------
# ABOUT PAGE
# -----------------------------------
elif page == "About":

    st.subheader("About Project")

    st.write("""
Customer Churn Prediction System developed using:

• Python  
• Streamlit  
• Scikit-learn  

Machine Learning Algorithm Used:

Random Forest Classifier

Project Domain:

Telecom Industry

Features Used:

• Age  
• Monthly Charges  
• Tenure  
• Gender  
• Dependents  
• Internet Service  
• Paperless Billing  
• Payment Method  
• Senior Citizen  

Objective:

To predict customers who are likely to stop using telecom services so companies can take actions to retain them.
""")
