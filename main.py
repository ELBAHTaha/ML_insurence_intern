# scripts/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_preprocess():
    """
    Loads the insurance dataset, encodes categorical variables,
    scales numerical features, and splits into training/testing sets.

    Returns:
        X_train, X_test, y_train, y_test: Split features and target
        data: Full preprocessed dataframe (optional)
    """

    # 1️⃣ Load dataset (make sure insurance.csv is in the same folder as your script or provide correct path)
    data = pd.read_csv("insurance.csv")

    # 2️⃣ Encode categorical variables
    for col in ["sex", "smoker", "region"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # 3️⃣ Scale numerical features (age, bmi, children)
    scaler = StandardScaler()
    data[["age", "bmi", "children"]] = scaler.fit_transform(data[["age", "bmi", "children"]])

    # 4️⃣ Split features and target
    X = data.drop("charges", axis=1)
    y = data["charges"]

    # 5️⃣ Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Dataset loaded and preprocessed successfully!")
    print("Training set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])

    return X_train, X_test, y_train, y_test, data


# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, data = load_and_preprocess()
    print(data.head())

# lin regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Features (X) and Target (y)
X = data.drop("charges", axis=1)
y = data["charges"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_pred = lin_reg.predict(X_test)

# Evaluation
print("R^2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Show coefficients
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": lin_reg.coef_})
print(coefficients)
#knn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Features: remove "smoker" (target for classification)
X = data.drop(["smoker", "charges"], axis=1)
y = data["smoker"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
