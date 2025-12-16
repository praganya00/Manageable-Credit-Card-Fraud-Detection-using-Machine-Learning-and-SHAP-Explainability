# ============================================
# Credit Card Customer Attrition Prediction
# ============================================

# ---------- Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import SMOTE
import shap

# ---------- Load Dataset ----------
df = pd.read_csv("creditcard.csv.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# ---------- Target Variable ----------
# Attrition_Flag: Existing Customer -> 0, Attrited Customer -> 1
df["Attrition_Flag"] = df["Attrition_Flag"].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})

print("\nClass Distribution:\n", df["Attrition_Flag"].value_counts())

# ---------- Drop Unnecessary Columns ----------
df.drop([
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
], axis=1, inplace=True)

# ---------- Check Missing Values ----------
print("\nMissing Values:\n", df.isnull().sum())

# ---------- Feature & Target Split ----------
X = df.drop("Attrition_Flag", axis=1)
y = df["Attrition_Flag"]

# ---------- Encode Categorical Variables ----------
X = pd.get_dummies(X, drop_first=True)

# ---------- Feature Scaling ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------- Handle Class Imbalance ----------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ---------- Logistic Regression ----------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_bal, y_train_bal)

y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Results\n")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))

# ---------- Random Forest ----------
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train_bal, y_train_bal)

y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Results\n")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# ---------- SHAP Explainability ----------
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train_bal)

shap.summary_plot(
    shap_values[1],
    X_train_bal,
    feature_names=X.columns
)

# ---------- Visualization ----------
plt.figure(figsize=(6,4))
sns.countplot(x="Attrition_Flag", data=df)
plt.title("Customer Attrition Distribution")
plt.xlabel("0 = Existing Customer, 1 = Attrited Customer")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="Attrition_Flag", y="Credit_Limit", data=df)
plt.title("Credit Limit vs Customer Attrition")
plt.show()

print("\nPROGRAM COMPLETED SUCCESSFULLY")
