# level_three.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# 1. Load & preprocess dataset
# -------------------------------
data = pd.read_csv("insurance.csv")

# Convert categorical columns
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data['region'] = data['region'].map({'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3})

X = data.drop("charges", axis=1)
y = (data['charges'] > data['charges'].median()).astype(int)  # Binary target for classification

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Balance dataset
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_bal).value_counts().to_dict())

# -------------------------------
# 2. Random Forest Classifier
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bal, y_train_bal)

y_pred_rf = rf.predict(X_test)
print("\n🔹 Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature importance plot
importances = rf.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(X.columns, importances)
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.show()

# ROC curve
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("Random Forest ROC Curve")
plt.show()

# -------------------------------
# 3. SVM Classifier
# -------------------------------
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_bal, y_train_bal)

y_pred_svm = svm.predict(X_test)
print("\n🔹 SVM Classifier")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# ROC curve
RocCurveDisplay.from_estimator(svm, X_test, y_test)
plt.title("SVM ROC Curve")
plt.show()

# -------------------------------
# 4. Neural Network with Keras
# -------------------------------
nn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_bal.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = nn.fit(X_train_bal, y_train_bal, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Evaluate NN
loss, acc = nn.evaluate(X_test, y_test, verbose=0)
print("\n🔹 Neural Network")
print("Test Accuracy:", acc)

# Plot training history
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
