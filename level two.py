# level_two.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE

# ------------------------------
# 1. Load and preprocess dataset
# ------------------------------
data = pd.read_csv("insurance.csv")

# Encode categorical variables
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])

X = data.drop("smoker", axis=1)
y = data["smoker"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Balance dataset with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_res.value_counts().to_dict())

# ------------------------------
# 2. Logistic Regression
# ------------------------------
print("\n🔹 Logistic Regression (Balanced)")
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_res, y_train_res)

y_pred_lr = log_reg.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", cm_lr)
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# ✅ Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ✅ ROC Curve
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ------------------------------
# 3. Decision Tree
# ------------------------------
print("\n🔹 Decision Tree (Balanced)")
tree = DecisionTreeClassifier(class_weight="balanced", random_state=42)
tree.fit(X_train_res, y_train_res)

y_pred_tree = tree.predict(X_test)
cm_tree = confusion_matrix(y_test, y_pred_tree)

print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("F1 Score:", f1_score(y_test, y_pred_tree, average="weighted"))
print("Confusion Matrix:\n", cm_tree)
print("Classification Report:\n", classification_report(y_test, y_pred_tree))

# ✅ Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Oranges", xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ✅ Visualize the Tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=["Non-Smoker", "Smoker"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# ------------------------------
# 4. K-Means Clustering
# ------------------------------
print("\n🔹 K-Means Clustering")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

data["Cluster"] = clusters
print(data.groupby("Cluster").mean())

# ✅ Scatter plot for 2D visualization (age vs. charges)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=data["age"], y=data["charges"], hue=data["Cluster"], palette="Set1")
plt.title("K-Means Clustering (Age vs Charges)")
plt.show()
