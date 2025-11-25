import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

try:
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
except:
    st.error("❌ Failed to load dataset. Check your internet connection.")
    st.stop()

# Check dataset is not empty
if df.empty:
    st.error("❌ Dataset is empty. Unable to proceed.")
    st.stop()

st.title("Diabetes Profiling & Prediction System")
st.write("Dataset: **Pima Indians Diabetes Dataset**")

# ---------------- Dataset Overview ----------------
st.subheader("Dataset Overview")
st.write(df.head())
st.write(df.describe())
st.write("Missing values per column:")
st.write(df.isnull().sum())
# ---------------- Data Cleaning ----------------
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

# Show NaN count before filling
st.write("NaN count after replacing zeros:")
st.write(df.isnull().sum())

df.fillna(df.mean(), inplace=True)

# ---------------- Correlation Heatmap ----------------
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
st.pyplot(fig)

# ---------------- Feature Selection ----------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

st.subheader("Feature Importance using Mutual Information")

X_temp = df.drop("Outcome", axis=1)
y_temp = df["Outcome"]

mi = mutual_info_classif(X_temp, y_temp)
mi_series = pd.Series(mi, index=X_temp.columns).sort_values(ascending=False)

st.bar_chart(mi_series)
st.write(mi_series)
st.info("Higher score means stronger relationship with diabetes outcome")


# ---------------- Train-test Split ----------------
# Check if dataset is valid
st.write("Unique target values:", y.unique())
st.write("Counts:", y.value_counts())

st.write(f"Dataset Shape: {df.shape}")
st.write(f"Feature Matrix Shape: {X.shape}")
st.write(f"Target Shape: {y.shape}")
st.write("Class Distribution:")
st.bar_chart(y.value_counts())

if len(df) == 0:
    st.error("❌ Dataset is empty after preprocessing.")
    st.stop()

if y.nunique() < 2:
    st.error("❌ Not enough class variety in target. Stratified split requires both classes (0 & 1).")
    st.stop()

# Split Train-Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Scaling ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Modeling ----------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------- Performance Metrics ----------------
st.subheader("Model Performance")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.4f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
st.write(f"**AUC Score:** {auc:.4f}")

st.write("---")

# ---------------- Prediction Interface ----------------
st.header("Diabetes Prediction")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 250, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
scaled = scaler.transform(input_data)

if st.button("Predict"):
    result = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.subheader("Prediction Result")
    if result == 1:
        st.error("🔴 Patient is likely to have Diabetes")
    else:
        st.success("🟢 Patient is NOT likely to have Diabetes")

    st.write(f"Probability of Diabetes: **{prob*100:.2f}%**")

    st.write("---")
    st.subheader("📍 Risk Factor Observations")
    
    if glucose > df["Glucose"].mean():
        st.warning("⚠ Higher-than-average glucose level")
    if bmi > df["BMI"].mean():
        st.warning("⚠ Higher-than-average BMI increases risk")
    if age > df["Age"].mean():
        st.warning("⚠ Older age increases diabetes risk")
    if bp < 70:
        st.info("⚠ Very low BP detected")

