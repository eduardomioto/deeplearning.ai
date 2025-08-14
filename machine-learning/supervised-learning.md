# Supervised Learning - Complete Guide

Comprehensive coverage of supervised learning algorithms and techniques for machine learning practitioners.

## 📚 Table of Contents

- [Overview](#overview)
- [What is Supervised Learning?](#what-is-supervised-learning)
- [Types of Supervised Learning](#types-of-supervised-learning)
- [Linear Models](#linear-models)
- [Tree-Based Models](#tree-based-models)
- [Support Vector Machines](#support-vector-machines)
- [Neural Networks](#neural-networks)
- [Model Evaluation](#model-evaluation)
- [Feature Engineering](#feature-engineering)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)
- [Resources](#resources)

## 🎯 Overview

Supervised learning is a type of machine learning where the algorithm learns from labeled training data to make predictions on new, unseen data. It's the most common and well-understood approach to machine learning, forming the foundation for many real-world applications.

## 🤖 What is Supervised Learning?

Supervised learning involves training a model on a dataset where each example has:
- **Input features** (X) - The data we use to make predictions
- **Output labels** (y) - The correct answers we want to predict

The goal is to learn a mapping function f: X → y that can accurately predict outputs for new inputs.

### Key Characteristics
- **Labeled data** - Training examples with known correct answers
- **Predictive modeling** - Learning to predict future outcomes
- **Generalization** - Performing well on unseen data
- **Feedback loop** - Model performance guides learning

## 🏷️ Types of Supervised Learning

### **1. Classification**
Predicting discrete categories or classes.

**Examples:**
- Spam vs. legitimate email
- Image classification (cat, dog, bird)
- Disease diagnosis (healthy, mild, severe)
- Customer churn prediction (stay, leave)

**Key Metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve, AUC

### **2. Regression**
Predicting continuous numerical values.

**Examples:**
- House price prediction
- Stock price forecasting
- Temperature prediction
- Sales forecasting

**Key Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² (Coefficient of determination)

## 📈 Linear Models

### **1. Linear Regression**
Models the relationship between features and target as a linear combination.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 100)

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Model coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")
```

**Mathematical Form:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

**Assumptions:**
- Linear relationship between features and target
- Independent and identically distributed errors
- Homoscedasticity (constant variance)
- Normal distribution of errors

### **2. Logistic Regression**
Models the probability of belonging to a class using a sigmoid function.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix

# Generate classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nFeature importance (top 5):")
feature_importance = np.abs(model.coef_[0])
top_features = np.argsort(feature_importance)[-5:]
for i in top_features:
    print(f"Feature {i}: {feature_importance[i]:.3f}")
```

**Mathematical Form:**
```
P(y=1|x) = 1 / (1 + e^(-z))
where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Applications:**
- Medical diagnosis
- Credit scoring
- Marketing response prediction
- Risk assessment

### **3. Ridge & Lasso Regression**
Regularized versions of linear regression to prevent overfitting.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_score = ridge.score(X_test_scaled, y_test)

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_score = lasso.score(X_test_scaled, y_test)

print(f"Ridge R² Score: {ridge_score:.3f}")
print(f"Lasso R² Score: {lasso_score:.3f}")

# Compare coefficients
print(f"\nRidge coefficients (non-zero): {np.sum(ridge.coef_ != 0)}")
print(f"Lasso coefficients (non-zero): {np.sum(lasso.coef_ != 0)}")
```

**Regularization Effects:**
- **Ridge (L2)**: Shrinks coefficients toward zero, prevents overfitting
- **Lasso (L1)**: Sets some coefficients to exactly zero, performs feature selection

## 🌳 Tree-Based Models

### **1. Decision Trees**
Non-linear models that make decisions based on feature thresholds.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create decision tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=[f'Feature_{i}' for i in range(X_train.shape[1])],
          class_names=['Class 0', 'Class 1'], filled=True, rounded=True)
plt.show()

# Feature importance
feature_importance = tree.feature_importances_
print("Feature importance:")
for i, importance in enumerate(feature_importance):
    if importance > 0.01:  # Only show important features
        print(f"Feature {i}: {importance:.3f}")
```

**Advantages:**
- Easy to interpret and visualize
- Handles non-linear relationships
- No assumptions about data distribution
- Can handle mixed data types

**Disadvantages:**
- Prone to overfitting
- Unstable (small changes in data can create very different trees)
- Can create overly complex trees

### **2. Random Forest**
Ensemble method that combines multiple decision trees.

```python
from sklearn.ensemble import RandomForestClassifier

# Create random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# Evaluate model
print("Random Forest Performance:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = rf.feature_importances_
indices = np.argsort(feature_importance)[::-1]

print("\nFeature ranking:")
for f in range(min(10, X_train.shape[1])):
    print(f"{f+1}. Feature {indices[f]} ({feature_importance[indices[f]]:.3f})")

# Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

**Key Parameters:**
- **n_estimators**: Number of trees in the forest
- **max_depth**: Maximum depth of each tree
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required in a leaf node

### **3. Gradient Boosting**
Sequentially builds trees to correct errors of previous trees.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Create gradient boosting model
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                               max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# Make predictions
y_pred = gb.predict(X_test)

# Evaluate model
print("Gradient Boosting Performance:")
print(classification_report(y_test, y_pred))

# Learning curves
train_scores = []
test_scores = []

for i in range(1, 101, 10):
    gb_partial = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1,
                                           max_depth=3, random_state=42)
    gb_partial.fit(X_train, y_train)
    train_scores.append(gb_partial.score(X_train, y_train))
    test_scores.append(gb_partial.score(X_test, y_test))

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101, 10), train_scores, label='Training Score')
plt.plot(range(1, 101, 10), test_scores, label='Test Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

## 🎯 Support Vector Machines

SVMs find the optimal hyperplane that separates classes with maximum margin.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Standardize features (important for SVMs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SVM model
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Evaluate model
print("SVM Performance:")
print(classification_report(y_test, y_pred))

# Different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_scores = []

for kernel in kernels:
    svm_kernel = SVC(kernel=kernel, random_state=42)
    svm_kernel.fit(X_train_scaled, y_train)
    score = svm_kernel.score(X_test_scaled, y_test)
    kernel_scores.append(score)
    print(f"{kernel} kernel: {score:.3f}")

# Plot kernel comparison
plt.figure(figsize=(8, 6))
plt.bar(kernels, kernel_scores)
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy')
plt.title('SVM Performance by Kernel')
plt.ylim(0, 1)
for i, v in enumerate(kernel_scores):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
plt.show()
```

**Kernel Types:**
- **Linear**: Linear decision boundary
- **Polynomial**: Non-linear decision boundary
- **RBF (Radial Basis Function)**: Most commonly used, handles non-linear data
- **Sigmoid**: Similar to neural network activation function

## 🧠 Neural Networks

Multi-layer perceptrons for complex non-linear relationships.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create neural network
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                    activation='relu', solver='adam', random_state=42)
mlp.fit(X_train_scaled, y_train)

# Make predictions
y_pred = mlp.predict(X_test_scaled)

# Evaluate model
print("Neural Network Performance:")
print(classification_report(y_test, y_pred))

# Learning curves
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Neural Network Learning Curve')
plt.grid(True)
plt.show()

# Feature importance (for single hidden layer)
if len(mlp.coefs_) == 2:  # Single hidden layer
    feature_importance = np.mean(np.abs(mlp.coefs_[0]), axis=1)
    print("\nFeature importance (first hidden layer):")
    for i, importance in enumerate(feature_importance):
        if importance > 0.01:
            print(f"Feature {i}: {importance:.3f}")
```

**Architecture Considerations:**
- **Hidden layers**: More layers for complex patterns
- **Neurons per layer**: Balance between capacity and overfitting
- **Activation functions**: ReLU, tanh, sigmoid
- **Regularization**: Dropout, L1/L2 regularization

## 📊 Model Evaluation

### **1. Classification Metrics**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

# ROC Curve
from sklearn.metrics import roc_curve, auc
y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
```

### **2. Regression Metrics**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For regression problems
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"Root Mean Squared Error: {rmse:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
```

### **3. Cross-Validation**

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=cv, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Learning curves
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    mlp, X_train_scaled, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

## 🔧 Feature Engineering

### **1. Feature Scaling**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (z-score normalization)
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# MinMaxScaler (scaling to [0,1] range)
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# RobustScaler (robust to outliers)
scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train)
X_test_robust = scaler_robust.transform(X_test)

print("Scaling comparison:")
print(f"StandardScaler - Mean: {X_train_std.mean():.3f}, Std: {X_train_std.std():.3f}")
print(f"MinMaxScaler - Min: {X_train_minmax.min():.3f}, Max: {X_train_minmax.max():.3f}")
print(f"RobustScaler - Median: {np.median(X_train_robust):.3f}")
```

### **2. Feature Selection**

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Univariate feature selection
selector_kbest = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector_kbest.fit_transform(X_train, y_train)
X_test_selected = selector_kbest.transform(X_test)

print(f"Selected features: {selector_kbest.get_support()}")
print(f"Feature scores: {selector_kbest.scores_}")

# Recursive feature elimination
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
selector_rfe = RFE(estimator, n_features_to_select=10)
X_train_rfe = selector_rfe.fit_transform(X_train, y_train)
X_test_rfe = selector_rfe.transform(X_test)

print(f"RFE selected features: {selector_rfe.get_support()}")
print(f"Feature ranking: {selector_rfe.ranking_}")
```

## 🎛️ Hyperparameter Tuning

### **1. Grid Search**

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Grid search with cross-validation
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
print(f"Best estimator: {grid_search.best_estimator_}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print(f"\nBest model test accuracy: {accuracy_score(y_test, y_pred_best):.3f}")
```

### **2. Random Search**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# Define parameter distributions
param_distributions = {
    'C': loguniform(1e-3, 1e3),
    'gamma': loguniform(1e-4, 1e1),
    'kernel': ['rbf', 'linear']
}

# Random search
random_search = RandomizedSearchCV(SVC(random_state=42), param_distributions, 
                                  n_iter=100, cv=5, scoring='accuracy', 
                                  random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.3f}")
```

## 💻 Practical Examples

### **1. Complete Classification Pipeline**

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Create preprocessing pipeline
numeric_features = [0, 1, 2, 3, 4]  # Example feature indices
categorical_features = [5, 6]        # Example feature indices

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create complete pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate
print("Pipeline Performance:")
print(classification_report(y_test, y_pred))
```

### **2. Model Comparison**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Neural Network': MLPClassifier(random_state=42, max_iter=500)
}

# Train and evaluate all models
results = {}
for name, model in models.items():
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

# Display results
import pandas as pd
results_df = pd.DataFrame(results).T
print("Model Comparison:")
print(results_df.round(3))
```

## 💡 Best Practices

### **1. Data Preprocessing**
- **Handle missing values** appropriately
- **Scale features** for algorithms sensitive to scale
- **Encode categorical variables** properly
- **Remove outliers** when appropriate
- **Check for data leakage**

### **2. Model Selection**
- **Start simple** with linear models
- **Use cross-validation** to estimate performance
- **Consider interpretability** vs. performance trade-offs
- **Try ensemble methods** for better performance
- **Regularize** to prevent overfitting

### **3. Evaluation**
- **Use appropriate metrics** for your problem
- **Split data** into train/validation/test sets
- **Cross-validate** to get reliable estimates
- **Check for overfitting** using learning curves
- **Compare multiple models** systematically

### **4. Deployment**
- **Save preprocessing steps** with your model
- **Monitor model performance** in production
- **Retrain models** periodically with new data
- **Version control** your models and code
- **Document** your approach and assumptions

## 📚 Resources

### **Books**
- "Introduction to Machine Learning with Python" by Andreas C. Müller and Sarah Guido
- "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Hands-On Machine Learning" by Aurélien Géron

### **Online Courses**
- [Stanford CS229: Machine Learning](http://cs229.stanford.edu/)
- [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### **Python Libraries**
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Plotting
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualization

### **Datasets**
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Built-in Datasets](https://scikit-learn.org/stable/datasets/)

---

**Happy Supervised Learning! 🚀✨**

*Supervised learning provides a solid foundation for understanding machine learning. Master these concepts and you'll be well-equipped to tackle a wide range of predictive modeling problems.*
