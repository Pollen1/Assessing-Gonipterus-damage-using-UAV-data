# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 21:58:45 2025

@author: Polle
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score)
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load and prepare data
data = pd.read_csv('C:/Users/Polle/Downloads/All_site_indices_.csv')
TARGET_COL = data.columns[-1]
X = data.drop(columns=['Site', TARGET_COL])
y = data[TARGET_COL]
sites = data['Site'].unique()

# Optimization parameters
param_space = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(100, 1000),
    'subsample': Real(0.5, 1.0, prior='uniform'),
    'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
    'gamma': Real(0, 5, prior='uniform')
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

        # Model tuning
        bayes_search = BayesSearchCV(
            XGBClassifier(random_state=repeat),
            param_space,
            n_iter=50,
            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=repeat),
            n_jobs=-1,
            scoring='accuracy',
            random_state=repeat
        )
        bayes_search.fit(X_train, y_train)

        # Feature selection
        rfecv = RFECV(
            estimator=bayes_search.best_estimator_,
            step=2,
            cv=StratifiedKFold(10, shuffle=True, random_state=repeat),
            scoring='accuracy'
        )
        rfecv.fit(X_train, y_train)
        selected_features = X.columns[rfecv.support_]

        # Evaluation
        final_model = bayes_search.best_estimator_
        final_model.fit(X_train[selected_features], y_train)
        y_pred = final_model.predict(X_test[selected_features])
        
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