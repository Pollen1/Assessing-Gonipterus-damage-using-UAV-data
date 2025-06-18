# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 04:14:24 2025

@author: pnzuza
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_score, recall_score, f1_score)
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from collections import defaultdict

# Load data
data = pd.read_csv('E:/Final/Histogram matching modelling/Reduced_10 band_histogram_matching/indices_bands/Combined_histogram_normalisation_undersampling_nosit_Reduced_10bands_model_indices_bands.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Get class names dynamically
class_names = list(map(str, np.sort(y.unique())))  # Ensuring class order consistency

# Parameter space for Bayesian optimization
param_space = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(100, 1000),
    'subsample': Real(0.5, 1.0, prior='uniform'),
    'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
    'gamma': Real(0, 5, prior='uniform')
}

num_iterations = 5
metrics = {
    'accuracy': [],
    'macro_precision': [],
    'macro_recall': [],
    'macro_f1': [],
    'class_metrics': defaultdict(lambda: defaultdict(list))
}

# Storage for outputs
confusion_matrices = []
all_selected_features = []
feature_importance_dict = defaultdict(list)
gini_importance_dict = defaultdict(list)
gini_importance_list = []  # Store per-iteration Gini importance

for i in range(num_iterations):
    print(f"Iteration {i+1}/{num_iterations}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
    
    # Bayesian Optimization
    bayes_search = BayesSearchCV(
        XGBClassifier(),
        param_space,
        n_iter=50,
        cv=StratifiedKFold(5, shuffle=True, random_state=i),
        n_jobs=-1,
        scoring='accuracy',
        random_state=i
    )
    bayes_search.fit(X_train, y_train)
    
    # Feature Selection
    rfecv = RFECV(
        estimator=bayes_search.best_estimator_,
        step=2,
        cv=StratifiedKFold(10, shuffle=True, random_state=i),
        scoring='accuracy'
    )
    rfecv.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv.support_]
    all_selected_features.append(list(selected_features))  # Store selected features
    
    # Final Training
    final_model = bayes_search.best_estimator_
    final_model.fit(X_train[selected_features], y_train)
    
    # Prediction & Evaluation
    y_pred = final_model.predict(X_test[selected_features])
    
    # Store metrics
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['macro_precision'].append(precision_score(y_test, y_pred, average='macro'))
    metrics['macro_recall'].append(recall_score(y_test, y_pred, average='macro'))
    metrics['macro_f1'].append(f1_score(y_test, y_pred, average='macro'))
    
    # Per-class metrics
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    
    for idx, cls in enumerate(class_names):
        metrics['class_metrics'][cls]['precision'].append(precision[idx])
        metrics['class_metrics'][cls]['recall'].append(recall[idx])
        metrics['class_metrics'][cls]['f1'].append(f1[idx])
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=np.sort(y.unique()))
    confusion_matrices.append(conf_matrix)
    
    # Feature Importance (Permutation Importance)
    perm_importance = permutation_importance(
        final_model,
        X_test[selected_features],
        y_test,
        n_repeats=30,  # Reduced repeats for efficiency
        random_state=i
    )
    for idx, feature in enumerate(selected_features):
        feature_importance_dict[feature].append(perm_importance.importances_mean[idx])
    
    # Gini Importance
    gini_importance = final_model.feature_importances_
    for idx, feature in enumerate(selected_features):
        gini_importance_dict[feature].append(gini_importance[idx])
    
    # Store per-iteration Gini importance
    gini_importance_iteration = pd.DataFrame({
        'Iteration': [i + 1] * len(selected_features),
        'Feature': selected_features,
        'Gini_Importance': gini_importance
    })
    gini_importance_list.append(gini_importance_iteration)

# ======== Save Outputs ========

# 1. Save average confusion matrix
average_confusion = np.mean(confusion_matrices, axis=0)
confusion_df = pd.DataFrame(
    average_confusion,
    columns=class_names,
    index=class_names
)
confusion_df.to_csv('C:/Users/pnzuza/Downloads/average_confusion_matrix.csv')
print("Saved average confusion matrix to average_confusion_matrix.csv")

# 2. Save feature selection frequency table
feature_counts = pd.DataFrame({
    'Feature': X.columns,
    'Selection_Count': [sum(feat in features for features in all_selected_features) for feat in X.columns]
}).sort_values('Selection_Count', ascending=False)
feature_counts.to_csv('C:/Users/pnzuza/Downloads/feature_selection_frequency.csv', index=False)
print("Saved feature selection frequency to feature_selection_frequency.csv")

# 3. Save Gini importance (Mean)
gini_df = pd.DataFrame({
    'Feature': list(gini_importance_dict.keys()),
    'Gini_Importance_Mean': [np.mean(vals) for vals in gini_importance_dict.values()]
}).sort_values('Gini_Importance_Mean', ascending=False)
gini_df.to_csv('C:/Users/pnzuza/Downloads/gini_importance_re.csv', index=False)
print("Saved Gini importance to gini_importance.csv")

# 4. Save per-iteration selected features
features_df = pd.DataFrame({
    'Iteration': [f"Iteration {i+1}" for i in range(num_iterations)],
    'Selected_Features': ['; '.join(features) for features in all_selected_features]
})
features_df.to_csv('C:/Users/pnzuza/Downloads/selected_features_per_iteration.csv', index=False)
print("Saved per-iteration features to selected_features_per_iteration.csv")

# 5. Save Gini importance per iteration
gini_importance_all = pd.concat(gini_importance_list, ignore_index=True)
gini_importance_all.to_csv('C:/Users/pnzuza/Downloads/gini_importance_per_iteration.csv', index=False)
print("Saved per-iteration Gini importance to gini_importance_per_iteration.csv")

# ======== Reporting & Visualization ======== 
print("\nAverage Performance Metrics:")
print(f"Accuracy: {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
print(f"Macro F1: {np.mean(metrics['macro_f1']):.4f} ± {np.std(metrics['macro_f1']):.4f}")
print(f"Macro Precision: {np.mean(metrics['macro_precision']):.4f} ± {np.std(metrics['macro_precision']):.4f}")
print(f"Macro Recall: {np.mean(metrics['macro_recall']):.4f} ± {np.std(metrics['macro_recall']):.4f}")

