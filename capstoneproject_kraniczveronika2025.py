# -*- coding: utf-8 -*-
"""Business Data Analytics Project_KraniczVeronika2025

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hdCzD6WFWK5mno8-FxIoyQGaOpJYsrZh

# Business Data Analytics Project

## First steps - importing libraries and reading the file
"""

!pip install --upgrade gspread google-auth-httplib2 google-auth-oauthlib

from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

# Open GSheet by URL
worksheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1NVzwcDDxE7SlUDxQps7TKRUG35KtAg2eoKiIJxGMwyc/edit?usp=sharing').sheet1

# Get all values from the sheet
data = worksheet.get_all_values()

# Convert the list of lists to a pandas DataFrame
import pandas as pd
df = pd.DataFrame(data[1:], columns=data[0])

"""# Data Preparation and brief Exploration

##Get to know the dataset
"""

# First look at the head
display(df.head())

# Check info
print(df.info())

# Check desriptive statistics
print(df.describe())

# See columns
print(df.columns)

"""##Data cleaning

###Converting data types and handling missing values
"""

# Define columns
cols_to_convert = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

#Check non numeric values
for col in cols_to_convert:
    non_numeric = df[~df[col].astype(str).str.match(r'^-?\d+(\.\d+)?$', na=False)][col]
    print(f"{col} - count of non numeric values: {non_numeric.shape[0]}")
    if not non_numeric.empty:
        print(non_numeric.value_counts())

#Convert needed columns to numeric
for col in cols_to_convert:
    if df[col].dtype not in ['int64', 'float64']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['HeartDisease'] = pd.to_numeric(df['HeartDisease'], errors='coerce')

print("Number of original rows:", df.shape[0])
print("Number of non complete rows:", df.isnull().any(axis=1).sum())

#Convert categorical columns to 'category' dtype
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_cols:
    df[col] = df[col].astype('category')

#Double-check the updated structure
print("\nUpdated data types:")
print(df.dtypes)

print("\nCleaned dataset shape:", df.shape)

"""###One-hot encoding"""

#Check if there is high cardinality columns (>15)
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique value")
    print(df[col].value_counts().head(), '\n')

#One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# One-hot encode and drop the first category to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# Show new column names after encoding
print("New columns after One-Hot Encoding:")
print(df_encoded.columns.tolist())

# Check the updated shape
print("\nEncoded dataset shape:", df_encoded.shape)

# Preview the result
display(df_encoded.head())

"""###Looking for outliers"""

#Outlier check
import matplotlib.pyplot as plt
import seaborn as sns

for col in ['Cholesterol', 'RestingBP', 'MaxHR', 'Oldpeak']:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Function to detect outliers using the IQR method (exploration only, not for removal)
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers

# Example
outliers_chol = detect_outliers_iqr(df, 'Cholesterol')
print("Outliers in Cholesterol:", len(outliers_chol))

#Clip Oldpeak values at the 99th percentile
upper = df['Oldpeak'].quantile(0.99)
df['Oldpeak_clipped'] = df['Oldpeak'].clip(upper=upper)

# Clip Cholesterol values at the 99th percentile to reduce extreme influence
upper_chol = df['Cholesterol'].quantile(0.99)
df['Cholesterol_clipped'] = df['Cholesterol'].clip(upper=upper_chol)

#Comparison plot original Cholesterol vs. clipped Cholesterol values
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['Cholesterol'])
plt.title("Original Cholesterol")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Cholesterol_clipped'])
plt.title("Clipped Cholesterol (99th percentile)")

plt.tight_layout()
plt.show()

# Check duplicates
print(df.duplicated().any())
print(df.duplicated().sum())

# Check the shape
print(df.shape)

#Print head
print(df.head)

# Set df to final encoded and preprocessed dataset
df = df_encoded.copy()

"""#Exploratory Data Analysis


"""

# Set style for the plots
sns.set(style="whitegrid")

#Target variable distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='HeartDisease')
plt.title("Distribution of Heart Disease")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

#Distribution plots for numeric features
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

#Categorical feature counts
encoded_categorical_cols = [col for col in df.columns if col.startswith(('Sex_', 'ChestPainType_', 'RestingECG_', 'ExerciseAngina_', 'ST_Slope_'))]

for col in encoded_categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='HeartDisease')
    plt.title(f"{col} vs Heart Disease")
    plt.legend(title="HeartDisease")
    plt.show()

# Correlation heatmap centered at 0 to better highlight positive vs. negative relationships
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

#Boxplots by HeartDisease
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='HeartDisease', y=col)
    plt.title(f"{col} by Heart Disease")
    plt.show()

# Pairplot for selected numeric features + target
selected_cols = ['Age', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartDisease']
sns.pairplot(df[selected_cols], hue='HeartDisease')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

#Cross-tab for Sex vs HeartDisease
pd.crosstab(df['Sex_M'], df['HeartDisease'], normalize='index').plot(kind='bar', stacked=True)
plt.title("Heart Disease Rate by Sex")
plt.ylabel("Proportion")
plt.show()

# Filter: Only rows with HeartDisease = 1
df_hd = df[df['HeartDisease'] == 1]

# Plot Age distribution
plt.figure(figsize=(6, 4))
sns.histplot(df_hd['Age'], bins=30, kde=True, color='red')
plt.title("Age Distribution for Patients with Heart Disease")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Cross-tab and stacked bar chart
sex_hd = pd.crosstab(df['Sex_M'], df['HeartDisease'], normalize='index') #Corrected line to use 'Sex_M'

# Plot
sex_hd.plot(kind='bar', stacked=True, color=['green', 'red'])
plt.title("Heart Disease Proportion by Sex")
plt.ylabel("Proportion")
plt.xlabel("Sex")
plt.legend(["No Heart Disease", "Heart Disease"], title="HeartDisease")
plt.xticks(rotation=0)
plt.show()

# Select relevant features
selected_cols = ['Age', 'Cholesterol', 'MaxHR', 'Oldpeak', 'HeartDisease']

# Pairplot
sns.pairplot(df[selected_cols], hue='HeartDisease', palette='Set1')
plt.suptitle("Pairplot of Selected Features Colored by Heart Disease", y=1.02)
plt.show()

"""#Feature engineering"""

#Check for high cardinality cateforical features
high_cardinality = [col for col in df.columns if df[col].nunique() > 100 and df[col].dtype == 'object']
print("High-cardinality categorical columns:", high_cardinality)

"""##Scale features for Linear Model"""

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Split
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
print("Cross-validated F1 scores:", scores)

#Logistic Regression (L1 regularization)
from sklearn.preprocessing import StandardScaler

#Create X_scaled
X_scaled = StandardScaler().fit_transform(X)
logreg = LogisticRegression(penalty='l1', solver='liblinear')
logreg.fit(X_scaled, y)
importance_lr = pd.Series(logreg.coef_[0], index=X.columns)

#Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importance_rf = pd.Series(rf.feature_importances_, index=X.columns)

#Plot
importance_lr.sort_values(key=abs, ascending=False).head(10).plot(kind='barh', title="LogReg Top Features")
plt.show()
importance_rf.sort_values(ascending=False).head(10).plot(kind='barh', title="Random Forest Top Features")
plt.show()

"""#Correlation Analysis"""

#(This is for feature engineering)
# Absolute correlation with target
corr = df.corr()['HeartDisease'].abs().sort_values(ascending=False)
print("Top correlated features:\n", corr.head(10))

"""##Dimension Reduction"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.title("PCA of Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

"""##Train-test split"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

"""#SMOTE - Synthetic Minority Oversampling Technique"""

!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Original class distribution:")
print(y_train.value_counts())

print("\nSMOTE-resampled class distribution:")
print(pd.Series(y_train_smote).value_counts())

"""#Models

##Baseline and ML cLassifier
"""

# Most frequent class
baseline_pred = [y_train.mode()[0]] * len(y_test)

from sklearn.metrics import accuracy_score
baseline_acc = accuracy_score(y_test, baseline_pred)
print("Baseline Accuracy (majority class):", round(baseline_acc, 3))

"""###Evaluation Metrics"""

# Baseline model: always predict the majority class from y_train
from sklearn.metrics import accuracy_score

baseline_class = y_train.mode()[0]
baseline_pred = [baseline_class] * len(y_test)
baseline_prob = [baseline_class] * len(y_test)  # csak dummy, hogy működjön az ROC/AUC rész is

# Evaluate baseline using the same function as other models
baseline_metrics = evaluate_model(y_test, baseline_pred, baseline_prob, "Baseline (Majority Class)")

"""##Logistic Regression Models"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Fit model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report (LogReg):")
print(classification_report(y_test, y_pred_logreg))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob_logreg)
print("LogReg ROC AUC:", round(roc_auc, 3))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""###Default Logistic Regression"""

# Default Logistic Regression
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]

#Evaluation
metrics_logreg = evaluate_model(y_test, y_pred_logreg, y_prob_logreg, "Logistic Regression (Default)")

"""###Logistic Regression with class_weight"""

# Model with class_weight='balanced' to address class imbalance
logreg_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg_balanced.fit(X_train, y_train)
y_pred_logreg_bal = logreg_balanced.predict(X_test)
y_prob_logreg_bal = logreg_balanced.predict_proba(X_test)[:, 1]

# Evaluation
metrics_logreg_bal = evaluate_model(y_test, y_pred_logreg_bal, y_prob_logreg_bal, "Logistic Regression (Balanced)")

"""###Logistic Regression with SMOTE"""

#Model with SMOTE
logreg_smote = LogisticRegression(max_iter=1000)
logreg_smote.fit(X_train_smote, y_train_smote)

y_pred_logreg_smote = logreg_smote.predict(X_test)
y_prob_logreg_smote = logreg_smote.predict_proba(X_test)[:, 1]

#Evaluation
metrics_logreg_smote = evaluate_model(y_test, y_pred_logreg_smote, y_prob_logreg_smote, "Logistic Regression (SMOTE)")

"""##Random Forest Models"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
print("Random Forest ROC AUC:", roc_auc_score(y_test, y_prob_rf))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

"""###Default Random Forest Model"""

#Deafult Random Forest model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Run evaluation
auc_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest")

"""###Random Forest with class_weight"""

# Model with class_weight='balanced' to address class imbalance
rf_balanced = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_rf_bal = rf_balanced.predict(X_test)
y_prob_rf_bal = rf_balanced.predict_proba(X_test)[:, 1]

# Evaluation
metrics_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest (Default)")

"""###Random Forest with SMOTE"""

# Fit a model on SMOTE-resampled data
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_rf_smote = rf_smote.predict(X_test)
y_prob_rf_smote = rf_smote.predict_proba(X_test)[:, 1]

# Evaluate
metrics_rf_smote = evaluate_model(y_test, y_pred_rf_smote, y_prob_rf_smote, "Random Forest (SMOTE)")

"""###Random Forest with Hyperparameter Tuning"""

from sklearn.model_selection import GridSearchCV

# Hyperparameter optimization for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print("Best parameters:", grid_rf.best_params_)
print("Best CV F1-score:", grid_rf.best_score_)

# Evaluate best model on test set
best_rf = grid_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
y_prob_best_rf = best_rf.predict_proba(X_test)[:, 1]
metrics_rf_tuned = evaluate_model(y_test, y_pred_best_rf, y_prob_best_rf, "Random Forest (Tuned)")

"""##Deep Learning Model"""

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Import
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Deep Learning model for Heart Disease Prediction
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Mivel bináris osztályozás
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

"""###Prediction and Evaluation"""

# Predict
y_prob_dl = model.predict(X_test).flatten()
y_pred_dl = (y_prob_dl >= 0.5).astype(int)

# Evaluate
metrics_dl = evaluate_model(y_test, y_pred_dl, y_prob_dl, "Deep Learning (MLP)")

"""###Learning curves"""

# Learning curves
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

"""##Compare Train and Test Performance"""

def compare_train_test(model, X_train, y_train, X_test, y_test, model_name, is_keras=False):
    if is_keras:
        y_train_prob = model.predict(X_train).flatten()
        y_test_prob = model.predict(X_test).flatten()
    else:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_prob >= 0.5).astype(int)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    def get_metrics(y_true, y_pred, y_prob):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "ROC AUC": roc_auc_score(y_true, y_prob)
        }

    train_metrics = get_metrics(y_train, y_train_pred, y_train_prob)
    test_metrics = get_metrics(y_test, y_test_pred, y_test_prob)

    comparison_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])
    print(f"\nTrain vs. Test Performance for {model_name}")
    display(comparison_df.round(3))

    return train_metrics, test_metrics

"""###Run for both models"""

compare_train_test(logreg, X_train, y_train, X_test, y_test, "Logistic Regression")
compare_train_test(rf, X_train, y_train, X_test, y_test, "Random Forest")
compare_train_test(model, X_train, y_train, X_test, y_test, "Deep Learning (MLP)", is_keras=True)

"""##Final Model Comaprison"""

def evaluate_model(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n{model_name} Evaluation:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1-score: {f1:.3f} | ROC AUC: {auc:.3f}")

    # Confusion matrix
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return metrics as dict
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'ROC AUC': auc
    }

results = []

# Logistic Regression (default)
results.append(evaluate_model(y_test, y_pred_logreg, y_prob_logreg, "LogReg"))

# Logistic Regression (class_weight)
results.append(evaluate_model(y_test, y_pred_logreg_bal, y_prob_logreg_bal, "LogReg Balanced"))

# Logistic Regression (SMOTE)
results.append(evaluate_model(y_test, y_pred_logreg_smote, y_prob_logreg_smote, "LogReg SMOTE"))

# Random Forest (default)
results.append(evaluate_model(y_test, y_pred_rf, y_prob_rf, "RF"))

# Random Forest (balanced)
results.append(evaluate_model(y_test, y_pred_rf_bal, y_prob_rf_bal, "RF Balanced"))

# Random Forest (SMOTE)
results.append(evaluate_model(y_test, y_pred_rf_smote, y_prob_rf_smote, "RF SMOTE"))

# Random Forest (tuned)
results.append(evaluate_model(y_test, y_pred_best_rf, y_prob_best_rf, "RF Tuned"))

# Deep Learning
results.append(evaluate_model(y_test, y_pred_dl, y_prob_dl, "Deep Learning (MLP)"))

import pandas as pd
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-score", ascending=False)
display(results_df)

"""##Cross Validation"""

cv_scores_rf = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
cv_scores_logreg = cross_val_score(logreg, X_scaled, y, cv=5, scoring='roc_auc')

print("Random Forest CV ROC-AUC:", round(cv_scores_rf.mean(), 3))
print("Logistic Regression CV ROC-AUC:", round(cv_scores_logreg.mean(), 3))

# For logistic regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]

# For random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Run evaluations
auc_logreg = evaluate_model(y_test, y_pred_logreg, y_prob_logreg, "Logistic Regression")
auc_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest")

# Logistic Regression
evaluate_model(y_test, y_pred_logreg, y_prob_logreg, "Logistic Regression")

# Random Forest
evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest")

"""##Choosing the best model"""

#Choosing best model based on F1 score
best_model_name = results_df.iloc[0]['Model']
print(f"Thebest model based on F1 score: {best_model_name}")

"""##Saving the best model"""

import joblib
joblib.dump(best_rf, "best_model.pkl")

model.save("best_mlp_model.h5")

if best_model_name == "RF Tuned":
    joblib.dump(best_rf, "best_model.pkl")
elif best_model_name == "RF":
    joblib.dump(rf, "best_model.pkl")
elif best_model_name == "LogReg":
    joblib.dump(logreg, "best_model.pkl")
elif best_model_name == "LogReg SMOTE":
    joblib.dump(logreg_smote, "best_model.pkl")
elif best_model_name == "Deep Learning (MLP)":
    model.save("best_dl_model.h5")
else:
    print("Best model")
