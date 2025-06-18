# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 22:40:08 2025

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from collections import defaultdict

# Load data
data = pd.read_csv('E:/Final/Histogram matching modelling/Extended 5 bands/VI_indices/Extended_5bands_after_histogram_matching_balanced_bands_indices.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# Get class names and ensure consistent order
classes = y.unique().tolist()
class_names = ['No Damage', 'Low', 'Medium', 'High']  # Update with your actual class names

# Parameter space for Bayesian optimization
param_space = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 30),
    'max_features': Real(0.1, 1.0, prior='uniform'),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'bootstrap': Categorical([True, False])
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
gini_impurities = []  # New list to store Gini impurities per iteration

for i in range(num_iterations):
    print(f"Iteration {i+1}/{num_iterations}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    # Bayesian Optimization
    bayes_search = BayesSearchCV(
        RandomForestClassifier(),
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
    all_selected_features.append(list(selected_features))
    
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
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    confusion_matrices.append(conf_matrix)
    
    # Feature Importance (Permutation)
    perm_importance = permutation_importance(
        final_model,
        X_test[selected_features],
        y_test,
        n_repeats=50,
        random_state=i
    )
    for idx, feature in enumerate(selected_features):
        feature_importance_dict[feature].append(perm_importance.importances_mean[idx])
    
    # Gini Impurity (Feature Importance)
    gini_importances = final_model.feature_importances_
    # Create a dictionary for current iteration's Gini importances
    iteration_gini = dict(zip(selected_features, gini_importances))
    # Initialize all features to 0 and update with current values
    gini_series = pd.Series(0.0, index=X.columns)
    gini_series.update(pd.Series(iteration_gini))
    gini_impurities.append(gini_series)

# ======== Save Outputs ========
# 1. Save average confusion matrix
average_confusion = np.mean(confusion_matrices, axis=0)
confusion_df = pd.DataFrame(
    average_confusion,
    columns=class_names,
    index=class_names
)
#confusion_df.to_csv('path_to_save_confusion_matrix.csv', index=True)

# 2. Save feature selection details
feature_counts = pd.DataFrame({
    'Feature': X.columns,
    'Selection_Count': [sum(feat in features for features in all_selected_features) for feat in X.columns]
}).sort_values('Selection_Count', ascending=False)

features_df = pd.DataFrame({
    'Iteration': [f"Iteration {i+1}" for i in range(num_iterations)],
    'Selected_Features': ['; '.join(features) for features in all_selected_features]
})
#features_df.to_csv('path_to_save_selected_features.csv', index=False)

# 3. Save Gini impurity per iteration
gini_df = pd.DataFrame(gini_impurities)
gini_df.index = [f"Iteration {i+1}" for i in range(num_iterations)]
gini_df.to_csv('C:/Users/pnzuza/Downloads/gini_impurity_values_RF_extended_5bands.csv', index=True)
print("Saved Gini impurity per iteration to gini_impurity_per_iteration.csv")

# ======== Reporting & Visualization ======== 
print("\nAverage Performance Metrics:")
print(f"Accuracy: {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
print(f"Macro F1: {np.mean(metrics['macro_f1']):.4f} ± {np.std(metrics['macro_f1']):.4f}")
print(f"Macro Precision: {np.mean(metrics['macro_precision']):.4f} ± {np.std(metrics['macro_precision']):.4f}")
print(f"Macro Recall: {np.mean(metrics['macro_recall']):.4f} ± {np.std(metrics['macro_recall']):.4f}")