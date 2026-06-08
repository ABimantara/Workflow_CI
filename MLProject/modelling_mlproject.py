"""
modelling.py untuk MLflow Project - Titanic Dataset
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("  MODELLING - TITANIC DATASET (MLflow Project)")
print("="*50)

train_df = pd.read_csv('titanic_preprocessing_train.csv')
test_df  = pd.read_csv('titanic_preprocessing_test.csv')

feature_cols = [c for c in train_df.columns if c != 'Survived']
X_train = train_df[feature_cols]
y_train = train_df['Survived']
X_test  = test_df[feature_cols]
y_test  = test_df['Survived']

print(f"[1/4] Data loaded. Train: {X_train.shape}, Test: {X_test.shape}")

print("[2/4] Hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5],
    'min_samples_split': [2, 5]
}

base_model  = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"  Best params: {best_params}")

print("[3/4] Evaluasi model...")
y_pred    = best_model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Feature Importance
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_imp, x='importance', y='feature')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

report = classification_report(y_test, y_pred,
    target_names=['Not Survived', 'Survived'])
with open('classification_report.txt', 'w') as f:
    f.write(report)

print(f"  Accuracy : {accuracy:.4f}")
print(f"  F1 Score : {f1:.4f}")

print("[4/4] Logging ke MLflow...")

mlflow.log_param("n_estimators",      best_params['n_estimators'])
mlflow.log_param("max_depth",         str(best_params['max_depth']))
mlflow.log_param("min_samples_split", best_params['min_samples_split'])
mlflow.log_param("dataset",           "titanic")

mlflow.log_metric("accuracy",  accuracy)
mlflow.log_metric("f1_score",  f1)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall",    recall)
mlflow.log_metric("cv_mean",   cv_scores.mean())
mlflow.log_metric("cv_std",    cv_scores.std())

mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact("feature_importance.png")
mlflow.log_artifact("classification_report.txt")

mlflow.sklearn.log_model(best_model, "model")

print("\n✅ Training selesai!")
