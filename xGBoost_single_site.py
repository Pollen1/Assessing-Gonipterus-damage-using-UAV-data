# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 12:37:16 2025

@author: pnzuza
"""

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_score, recall_score, f1_score)
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from collections import defaultdict
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Load training data
gdf = gpd.read_file('C:/Users/pnzuza/Downloads/Sutton_stand/220209_Sutton_damage/Damage_220209_Sutton.shp')
X = gdf.drop(columns=['Damage', 'geometry'])  
y = gdf['Damage']

# Load prediction locations
prediction_gdf = gpd.read_file('C:/Users/pnzuza/Downloads/Sutton_stand/Trees_indices/Trees_indices.shp')

# Configuration
num_iterations = 5
class_names = ['No Damage', 'Low', 'Medium']  # Update with actual labels
output_dir = 'C:/Users/pnzuza/Downloads/Sutton_stand/output temp'
param_space = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(100, 1000),
    'subsample': Real(0.5, 1.0, prior='uniform'),
    'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
    'gamma': Real(0, 5, prior='uniform')
}

# Initialize results storage
metrics = defaultdict(list)
confusion_matrices = []
feature_importances = []
all_selected_features = []

# Main processing loop
for iter_num in range(num_iterations):
    print(f"\n=== Iteration {iter_num+1}/{num_iterations} ===")
    
    # 1. Stratified spatial split
    train_indices, test_indices = train_test_split(
        gdf.index, 
        test_size=0.3, 
        stratify=y,
        random_state=iter_num
    )
    
    # Save original splits
    gdf.iloc[train_indices].to_file(f"{output_dir}Train_original_iter_{iter_num+1}.shp")
    gdf.iloc[test_indices].to_file(f"{output_dir}Test_original_iter_{iter_num+1}.shp")
    
    # 2. Undersampling
    X_train = gdf.iloc[train_indices].drop(columns=['Damage', 'geometry'])
    y_train = gdf.iloc[train_indices]['Damage']
    
    rus = RandomUnderSampler(random_state=iter_num)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    
    # Save undersampled training data
    resampled_gdf = gdf.iloc[train_indices].iloc[rus.sample_indices_]
    resampled_gdf.to_file(f"{output_dir}Train_undersampled_iter_{iter_num+1}.shp")
    
    # 3. Model training
    opt = BayesSearchCV(
        XGBClassifier(),
        param_space,
        n_iter=50,
        cv=StratifiedKFold(5, shuffle=True, random_state=iter_num),
        n_jobs=-1,
        random_state=iter_num
    )
    opt.fit(X_res, y_res)
    
    # 4. Feature selection
    selector = RFECV(
        opt.best_estimator_,
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=iter_num),
        scoring='accuracy'
    )
    selector.fit(X_res, y_res)
    selected_features = X_res.columns[selector.support_]
    all_selected_features.append(list(selected_features))
    
    # 5. Final model
    final_model = opt.best_estimator_
    final_model.fit(X_res[selected_features], y_res)
    
    # 6. Store feature importance
    fi = pd.DataFrame({
        'Feature': selected_features,
        'Importance': final_model.feature_importances_,
        'Iteration': iter_num+1
    })
    feature_importances.append(fi)
    
    # 7. Test evaluation
    X_test = gdf.iloc[test_indices].drop(columns=['Damage', 'geometry'])
    y_test = gdf.iloc[test_indices]['Damage']
    y_pred = final_model.predict(X_test[selected_features])
    
    # Store metrics
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average='macro'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='macro'))
    metrics['f1'].append(f1_score(y_test, y_pred, average='macro'))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))
    
    # 8. Spatial prediction
    X_pred = prediction_gdf.drop(columns=['geometry'])[selected_features]
    prediction_gdf[f'Iter_{iter_num+1}'] = final_model.predict(X_pred)

# Post-processing
# 9. Final predictions
pred_cols = [f'Iter_{i+1}' for i in range(num_iterations)]
prediction_gdf['Final_Pred'] = prediction_gdf[pred_cols].mode(axis=1)[0]

# 10. Save results
prediction_gdf.to_file(f"{output_dir}Final_Predictions.shp")
pd.DataFrame(metrics).to_csv(f"{output_dir}performance_metrics.csv", index=False)
pd.concat(feature_importances).to_csv(f"{output_dir}feature_importances.csv", index=False)
np.save(f"{output_dir}confusion_matrices.npy", np.array(confusion_matrices))

# 11. Generate report
print("\n=== Final Report ===")
print(f"Average Accuracy: {np.mean(metrics['accuracy']):.3f} ± {np.std(metrics['accuracy']):.3f}")
print(f"Average F1 Score: {np.mean(metrics['f1']):.3f} ± {np.std(metrics['f1']):.3f}")
