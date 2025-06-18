# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 08:57:14 2025

@author: Polle
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

# Load and prepare data
data = pd.read_csv('C:/Users/Polle/Downloads/All_site_indices_ (1).csv')
TARGET_COL = data.columns[-1]
X = data.drop(columns=['Site', TARGET_COL])
y = data[TARGET_COL]
sites = data['Site'].unique()

# Hyperparameter grid for SVM
param_space = {
    'C': Real(1e-4, 1e4, prior='log-uniform'),  # Regularization parameter
    'tol': Real(1e-4, 1e-2, prior='log-uniform'),  # Tolerance for stopping criteria
    'class_weight': Categorical(['balanced']),  # Handle class imbalance
    'max_iter': Categorical([100000, 500000])  # Maximum number of iterations
}

# Evaluation parameters
NUM_REPEATS = 5
CV_FOLDS = 5

# Initialize results storage
metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
           'precision_weighted', 'recall_weighted', 'f1_weighted']
site_results = {site: {metric: [] for metric in metrics} for site in sites}

for repeat in range(NUM_REPEATS):
    print(f"\n=== Repeat {repeat + 1}/{NUM_REPEATS} ===")
    
    for test_site in sites:
        # Data splitting
        train_mask = data['Site'] != test_site
        test_mask = data['Site'] == test_site
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Preprocessing: Scale the data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model tuning
        bayes_search = BayesSearchCV(
            LinearSVC(dual=False, random_state=repeat),  # Use LinearSVC
            param_space,
            n_iter=50,
            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=repeat),
            n_jobs=-1,
            scoring='accuracy',
            random_state=repeat
        )
        bayes_search.fit(X_train_scaled, y_train)

        # Feature selection using SVM coefficients
        feature_selector = SelectFromModel(
            bayes_search.best_estimator_,
            prefit=True,
            importance_getter='coef_'
        )
        X_train_selected = feature_selector.transform(X_train_scaled)
        X_test_selected = feature_selector.transform(X_test_scaled)

        # Get selected feature names
        selected_features = X.columns[feature_selector.get_support()]

        # Train final model on selected features
        final_model = bayes_search.best_estimator_
        final_model.fit(X_train_selected, y_train)
        y_pred = final_model.predict(X_test_selected)
        
        # Compute metrics
        site_results[test_site]['accuracy'].append(accuracy_score(y_test, y_pred))
        
        for average in ['macro', 'weighted']:
            site_results[test_site][f'precision_{average}'].append(
                precision_score(y_test, y_pred, average=average, zero_division=0)
            )
            site_results[test_site][f'recall_{average}'].append(
                recall_score(y_test, y_pred, average=average, zero_division=0)
            )
            site_results[test_site][f'f1_{average}'].append(
                f1_score(y_test, y_pred, average=average, zero_division=0)
            )

        print(f"Site {test_site} | Repeat {repeat+1} | Accuracy: {site_results[test_site]['accuracy'][-1]:.4f}")

# Print formatted results
print("\n\n=== Final Per-Site Metrics ===")
for site in sites:
    print(f"\nSite: {site}")
    for metric in metrics:
        values = site_results[site][metric]
        print(f"{metric.capitalize().replace('_', ' ')}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print("-" * 50)