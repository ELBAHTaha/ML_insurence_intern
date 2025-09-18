# level_one.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

# ------------------------------
# 1. Load and preprocess dataset
# ------------------------------
data = pd.read_csv("insurance.csv")

# Encode categorical variables
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])

# Separate features for regression (predict charges)
X_reg = data.drop("charges", axis=1)
y_reg = data["charges"]

# Standardize numerical columns
scaler = StandardScaler()
X_reg[['age', 'bmi', 'children']] = scaler.fit_transform(X_reg[['age', 'bmi', 'children']])

# Train-test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("Dataset loaded and preprocessed successfully!")
print("Training set size:", X_train_reg.shape[0])
print("Testing set size:", X_test_reg.shape[0])
print(X_train_reg.head())

# ------------------------------
# 2. Linear Regression
# ------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)

y_pred_reg = lr_model.predict(X_test_reg)

print("R^2 Score:", r2_score(y_test_reg, y_pred_reg))
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))

coefficients = pd.DataFrame({
    "Feature": X_train_reg.columns,
    "Coefficient": lr_model.coef_
})
print(coefficients)

# ✅ Plot predicted vs actual charges
plt.figure(figsize=(6, 5))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Linear Regression: Actual vs Predicted")
plt.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.show()

# ✅ Plot feature importance (coefficients)
plt.figure(figsize=(6, 4))
sns.barplot(x="Coefficient", y="Feature", data=coefficients)
plt.title("Feature Importance - Linear Regression")
plt.show()

# ------------------------------
# 3. Logistic Regression (Balanced)
# ------------------------------
# Predicting smoker (binary)
X_clf = data.drop("smoker", axis=1)
y_clf = data["smoker"]

# Train-test split for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# ✅ Balance dataset with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_clf, y_train_clf)

print("\nBefore SMOTE:", y_train_clf.value_counts().to_dict())
print("After SMOTE:", y_train_res.value_counts().to_dict())

# Train logistic regression with balanced classes
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_res, y_train_res)

y_pred_clf = log_reg.predict(X_test_clf)
cm_clf = confusion_matrix(y_test_clf, y_pred_clf)

print("\nAccuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("Confusion Matrix:\n", cm_clf)
print("Classification Report:\n", classification_report(y_test_clf, y_pred_clf))

# ✅ Plot confusion matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm_clf, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
