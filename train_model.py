import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# Sample Expanded Dataset (matching app.py features)
# -------------------------------------------------
data = pd.DataFrame({
    "Age": [25, 40, 30, 50, 28, 45, 35, 60],
    "MonthlyCharges": [500, 800, 600, 1000, 450, 900, 700, 1100],
    "Tenure": [5, 24, 12, 36, 6, 30, 15, 40],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "SeniorCitizen": [0, 1, 0, 1, 0, 1, 0, 1],
    "Dependents": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
    "InternetService": ["DSL", "Fiber", "DSL", "Fiber", "DSL", "Fiber", "DSL", "Fiber"],
    "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    "PaymentMethod": [
        "CreditCard","BankTransfer","ElectronicCheck","MailedCheck",
        "CreditCard","BankTransfer","ElectronicCheck","MailedCheck"
    ],
    "Churn": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
})

# -------------------------------------------------
# Split Features & Target
# -------------------------------------------------
X = data.drop("Churn", axis=1)
y = data["Churn"]

# -------------------------------------------------
# Define columns
# -------------------------------------------------
categorical_cols = ["Gender", "Dependents", "InternetService", "PaperlessBilling", "PaymentMethod"]
numeric_cols = ["Age", "MonthlyCharges", "Tenure", "SeniorCitizen"]

# -------------------------------------------------
# Preprocessing & Pipeline
# -------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Train Model
# -------------------------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------------------
# Save Pipeline
# -------------------------------------------------
pickle.dump(pipeline, open("model.pkl", "wb"))
print("Pipeline saved successfully!")