# -*- coding: utf-8 -*-

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

#Convert needed columns to numeric
cols_to_convert = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease']
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#Check for missing values after conversion
print("Missing values per column:")
print(df.isnull().sum())

#Drop or fill missing values (we'll drop for simplicity)
df = df.dropna()

#Convert categorical columns to 'category' dtype
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_cols:
    df[col] = df[col].astype('category')

#Double-check the updated structure
print("\nUpdated data types:")
print(df.dtypes)

print("\nCleaned dataset shape:", df.shape)

"""###One-hot encoding"""

#One-hot encode and drop the first category to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Show new column names after encoding
print("New columns after One-Hot Encoding:")
print(df_encoded.columns)

#Check the updated shape
print("\nEncoded dataset shape:", df_encoded.shape)

#Result
df_encoded.head()

"""###Looking for outliers"""

#Outlier check
import matplotlib.pyplot as plt
import seaborn as sns

for col in ['Cholesterol', 'RestingBP', 'MaxHR', 'Oldpeak']:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

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

upper = df['Oldpeak'].quantile(0.99)
df['Oldpeak_clipped'] = df['Oldpeak'].clip(upper=upper)
df = df[~df['Cholesterol'].isin(outliers_chol['Cholesterol'])]

# Check duplicates
print(df.duplicated().any())
print(df.duplicated().sum())

# Check the shape
print(df.shape)

#Print head
print(df.head)

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
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='HeartDisease')
    plt.title(f"{col} vs Heart Disease")
    plt.legend(title="HeartDisease")
    plt.show()

#Correlation heatmap (on numeric only)
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm")
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

#Does high cholesterol necesseary mean heart disease?
#Age vs HeartDisease
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='Age', hue='HeartDisease', bins=30, kde=True, multiple="stack")
plt.title("Age Distribution by Heart Disease")
plt.show()

#Cholesterol vs HeartDisease
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='HeartDisease', y='Cholesterol')
plt.title("Cholesterol by Heart Disease")
plt.show()

#Cross-tab for Sex vs HeartDisease
pd.crosstab(df['Sex'], df['HeartDisease'], normalize='index').plot(kind='bar', stacked=True)
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
sex_hd = pd.crosstab(df['Sex'], df['HeartDisease'], normalize='index')

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

# Make sure HeartDisease is numeric/categorical
df[selected_cols] = df[selected_cols].apply(pd.to_numeric, errors='coerce')

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Split
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Logistic Regression (L1 regularization)
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

# Absolute correlation with target
corr = df_encoded.corr()['HeartDisease'].abs().sort_values(ascending=False)
print("Top correlated features:\n", corr.head(10))

"""##Normalization"""

# Use MinMaxScaler or StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

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

# No time series, so shuffle is OK
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

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

X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#Deep Learning Model

## Evaluation function
"""

def evaluate_model(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n {model_name} Evaluation:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1-score: {f1:.3f} | ROC AUC: {auc:.3f}")

    # Confusion matrix
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return auc

"""##Baseline and ML cLassifier"""

# Most frequent class
baseline_pred = [y_train.mode()[0]] * len(y_test)

from sklearn.metrics import accuracy_score
baseline_acc = accuracy_score(y_test, baseline_pred)
print("Baseline Accuracy (majority class):", round(baseline_acc, 3))

"""###Evaluation Metrics"""

baseline_pred = [y_train.mode()[0]] * len(y_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
print("Baseline Accuracy:", round(baseline_acc, 3))

"""##Logistic Regression"""

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

"""###Evaluation Metrics"""

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]

# Run evaluation
auc_logreg = evaluate_model(y_test, y_pred_logreg, y_prob_logreg, "Logistic Regression")

"""##Random Forest"""

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

"""###Evaluation Metrics"""

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
# Run evaluation
auc_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest")

"""##Compare Train and Test performance"""

def compare_train_test(model, X_train, y_train, X_test, y_test, model_name):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
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

    # Display as DataFrame
    import pandas as pd
    comparison_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])
    print(f"\n Train vs. Test Performance for {model_name}")
    display(comparison_df.round(3))

# Run for both models
compare_train_test(logreg, X_train, y_train, X_test, y_test, "Logistic Regression")
compare_train_test(rf, X_train, y_train, X_test, y_test, "Random Forest")

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

"""##Saving the best model"""

import joblib

if auc_rf > auc_logreg:
    print("Saving Random Forest as best model")
    joblib.dump(rf, 'best_model_rf.pkl')
    best_model = rf
else:
    print("Saving Logistic Regression as best model")
    joblib.dump(logreg, 'best_model_logreg.pkl')
    best_model = logreg

# Load model
model_loaded = joblib.load('best_model_rf.pkl')
preds = model_loaded.predict(X_test)
