# Insurance ML Internship Project

This repository contains the implementation of an **Insurance Machine Learning Project** for a data science internship. The project is structured into **three levels**, each building on different machine learning techniques, from basic regression to advanced models and neural networks.

## 📋 Project Overview

The goal of this project is to **analyze insurance data** and **predict insurance charges** or classify customers using different ML models. The dataset includes features such as:

- Age
- Sex
- BMI (Body Mass Index)
- Number of children
- Smoking status
- Region

**Dataset source:** [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

**Technologies used:** Python, Pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn, TensorFlow/Keras

## 🔧 Data Preprocessing

All levels use the **same preprocessing pipeline**:

1. **Load the dataset** using pandas:
   ```python
   data = pd.read_csv("insurance.csv")
   ```

2. **Handle categorical variables**:
   - Encode sex, smoker, and region using label encoding or one-hot encoding.

3. **Scale numerical columns**:
   - Apply standardization to age, bmi, and children so that values are centered around 0 with a standard deviation of 1.

4. **Split the dataset** into training and testing sets (80/20 split).

5. **Handle imbalanced data** for classification tasks:
   - Apply SMOTE (Synthetic Minority Oversampling Technique) to balance classes for logistic regression, decision tree, SVM, and neural network models.

## 📊 Level 1 – Linear and Logistic Regression

### Tasks:
- Train a Linear Regression model to predict insurance charges
- Train a Logistic Regression model for binary classification (charges above median)
- Evaluate models using:
  - R² Score & MSE for regression
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix for classification
- Use SMOTE to balance classes for better classification performance

### Key outputs:
- Regression coefficients interpretation
- Confusion matrix and classification report
- Improved performance using balanced Logistic Regression

## 🌳 Level 2 – Decision Trees and K-Means Clustering

### Tasks:
- Train a Decision Tree Classifier to predict high vs low charges
- Visualize and prune the tree to avoid overfitting
- Implement K-Means Clustering for customer segmentation
- Determine the optimal number of clusters using the elbow method
- Evaluate models using accuracy, F1-score, and cluster analysis
- Data preprocessing (scaling and encoding) applied before model training

### Key outputs:
- Confusion matrix and classification report for Decision Tree
- Cluster centroids for K-Means
- Optional visualizations for tree structure and cluster distribution

## 🚀 Level 3 – Advanced ML Models

### Tasks:
- **Random Forest Classifier**
  - Train with hyperparameter tuning
  - Feature importance analysis
  - Evaluate using accuracy, confusion matrix, and ROC curve

- **Support Vector Machine (SVM)**
  - Train SVM with RBF kernel
  - Evaluate performance metrics and ROC curve

- **Neural Network with TensorFlow/Keras**
  - Build a feed-forward network for binary classification
  - Train and validate using backpropagation
  - Plot training & validation accuracy and loss
  - Preprocessing steps applied (scaling, encoding, SMOTE) before model training

### Key outputs:
- Feature importance plots for Random Forest
- ROC curves for Random Forest and SVM
- Accuracy and loss curves for Neural Network
- Confusion matrices and classification reports

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ELBAHTaha/ML_insurence_intern.git
   cd ML_insurence_intern
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run each level**:
   ```bash
   python level_one.py
   python level_two.py
   python level_three.py
   ```

**Note:** Ensure `insurance.csv` is in the same directory as the scripts.

## 📦 Dependencies

The project requires the following Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn
- tensorflow
- keras


## 👨‍💻 Author

**Taha El Bah** – [GitHub](https://github.com/ELBAHTaha)

*Internship Project: Insurance ML Models*
