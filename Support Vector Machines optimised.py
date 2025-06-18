# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 08:29:21 2025

@author: pnzuza
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

# Load data
data = pd.read_csv('E:/Final/Histogram matching modelling/Extended 5 bands/VI_indices/Extended_5bands_after_histogram_matching_balanced_bands_indices.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Hyperparameter grid
param_grid = {
    'svc__C': Real(1e-4, 1e4, prior='log-uniform'),
    'svc__tol': Real(1e-4, 1e-2, prior='log-uniform'),
    'svc__class_weight': Categorical(['balanced']),
    'svc__max_iter': Categorical([100000, 500000])
}

num_iterations = 5
metrics = {metric: [] for metric in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']}
feature_importance_dfs = []  # To store feature importances per iteration

for i in range(num_iterations):
    print(f"Iteration {i+1}/{num_iterations}")
    
    # Initialize feature importance storage for this iteration
    iteration_importances = pd.Series(0, index=X.columns, name=f"Iteration_{i+1}")
    
    # Train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i, stratify=y
    )
    
    # Preprocessing with RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Variance threshold
    selector = VarianceThreshold(threshold=0.01)
    X_train_processed = selector.fit_transform(X_train_scaled)
    X_test_processed = selector.transform(X_test_scaled)
    
    selected_feature_names = np.array(X.columns)[selector.get_support()]
    print(f"Selected features after Variance Threshold ({len(selected_feature_names)}):", selected_feature_names)

    # Use LinearSVC instead of SVC
    svc = LinearSVC(dual=False, random_state=i)

    # Bayesian optimization
    pipeline = Pipeline([
        ('svc', svc)
    ])

    opt = BayesSearchCV(
        pipeline,
        param_grid,
        n_iter=30,
        cv=StratifiedKFold(5, shuffle=True, random_state=i),
        n_jobs=-1,
        scoring='accuracy'
    )

    opt.fit(X_train_processed, y_train)
    best_model = opt.best_estimator_

    # Train the model before using SelectFromModel
    best_model.named_steps['svc'].fit(X_train_processed, y_train)

    # Feature selection using coef_
    feature_selector = SelectFromModel(best_model.named_steps['svc'], prefit=True, importance_getter='coef_')
    X_train_selected = feature_selector.transform(X_train_processed)
    X_test_selected = feature_selector.transform(X_test_processed)

    # Get final selected features
    final_selected_features = selected_feature_names[feature_selector.get_support()]
    
    # Get feature importance (using absolute coefficient values)
    coefs = np.abs(best_model.named_steps['svc'].coef_)
    if coefs.ndim > 1:  # Handle multi-class classification
        feature_importances = np.mean(coefs, axis=0)
    else:
        feature_importances = coefs
    
    # Store importances for this iteration
    for feature, importance in zip(final_selected_features, feature_importances):
        iteration_importances[feature] = importance
    
    feature_importance_dfs.append(iteration_importances)

    # Train final model
    best_model.named_steps['svc'].fit(X_train_selected, y_train)
    y_pred = best_model.named_steps['svc'].predict(X_test_selected)

    # Store metrics
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['macro_precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
    metrics['macro_recall'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
    metrics['macro_f1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))

# Save feature importances to CSV
feature_importance_df = pd.DataFrame(feature_importance_dfs)
feature_importance_df.to_csv('C:/Users/pnzuza/Downloads/feature importance_values_svm_EXTENDED_5BANDS.csv', index=True)
print("\nSaved feature importances to svm_feature_importances.csv")

# Results
print(f"\nAverage Accuracy: {np.mean(metrics['accuracy']):.4f}")
print(f"Average Macro Precision: {np.mean(metrics['macro_precision']):.4f}")
print(f"Average Macro Recall: {np.mean(metrics['macro_recall']):.4f}")
print(f"Average Macro F1: {np.mean(metrics['macro_f1']):.4f}")