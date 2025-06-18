# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 19:26:01 2025

@author: pnzuza
"""


import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score)
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from collections import defaultdict

# --------------------------
# Configuration
# --------------------------
TARGET_SITE = "Gre_Feb22"  # Specify your focus site here
#SHAPEFILE_PATH = "C:/Users/pnzuza/Downloads/Sutton_stand/Leave onesite out_LOSO/Trees_LOSO.shp"
NUM_ITERATIONS = 5

# --------------------------
# Helper Functions
# --------------------------
def truncate_features(feature_names):
    """Ensure Shapefile-compatible column names (max 10 chars)"""
    return [str(f)[:10] for f in feature_names]

def adjust_predictions(proba, valid_classes, model_classes):
    """Exclude missing classes using probability adjustment"""
    valid_mask = np.isin(model_classes, valid_classes)
    adjusted_proba = proba.copy()
    adjusted_proba[:, ~valid_mask] = -np.inf
    return model_classes[np.argmax(adjusted_proba, axis=1)]

# --------------------------
# Data Preparation
# --------------------------
# Load and prepare training data
data = pd.read_csv('C:/Users/pnzuza/Downloads/sutton/Leave one site out/All_site_indices_.csv')

# Convert all column names to strings and truncate
original_columns = data.columns.tolist()
truncated_columns = [
    (str(col)[:10] if col not in ['Site', 'Damage'] else str(col))
    for col in original_columns
]
data.columns = truncated_columns

# Split data into target and other sites
target_data = data[data['Site'] == TARGET_SITE]
other_data = data[data['Site'] != TARGET_SITE]

X_target = target_data.drop(columns=['Site', 'Damage'])
y_target = target_data['Damage']
X_other = other_data.drop(columns=['Site', 'Damage'])
y_other = other_data['Damage']

# --------------------------
# LOSO Model Training & Evaluation
# --------------------------
param_space = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(100, 1000),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 5)
}

metrics_history = defaultdict(list)
feature_importance = defaultdict(list)
final_models = []

for iteration in range(NUM_ITERATIONS):
    print(f"\n--- LOSO Iteration {iteration+1}/{NUM_ITERATIONS} ---")
    
    # Bayesian Optimization on other sites
    bayes_search = BayesSearchCV(
        XGBClassifier(),
        param_space,
        n_iter=50,
        cv=StratifiedKFold(5, shuffle=True, random_state=iteration),
        n_jobs=-1,
        scoring='accuracy',
        random_state=iteration
    )
    bayes_search.fit(X_other, y_other)
    
    # Feature Selection on other sites
    rfecv = RFECV(
        estimator=bayes_search.best_estimator_,
        step=2,
        cv=StratifiedKFold(10, shuffle=True, random_state=iteration),
        scoring='accuracy'
    )
    rfecv.fit(X_other, y_other)
    selected_features = [str(f)[:10] for f in X_other.columns[rfecv.support_].tolist()]
    
    # Final Model Training
    final_model = bayes_search.best_estimator_
    final_model.fit(X_other[selected_features], y_other)
    final_models.append((final_model, selected_features))
    
    # Predict on Target Site
    X_target_clean = X_target[selected_features].fillna(0)
    proba = final_model.predict_proba(X_target_clean)
    
    # Handle potential missing classes
    site_classes = y_target.unique()
    y_pred = adjust_predictions(proba, site_classes, final_model.classes_)
    
    # Store metrics
    metrics_history['accuracy'].append(accuracy_score(y_target, y_pred))
    metrics_history['precision'].append(precision_score(y_target, y_pred, average='macro', zero_division=0))
    metrics_history['recall'].append(recall_score(y_target, y_pred, average='macro', zero_division=0))
    metrics_history['f1'].append(f1_score(y_target, y_pred, average='macro', zero_division=0))
    
    # Track feature importance
    for feat, imp in zip(selected_features, final_model.feature_importances_):
        feature_importance[str(feat)].append(imp)

# --------------------------
# Aggregate Results
# --------------------------
print(f"\n--- Final Results for {TARGET_SITE} ---")
results = {
    'Accuracy': (np.mean(metrics_history['accuracy']), np.std(metrics_history['accuracy'])),
    'Precision': (np.mean(metrics_history['precision']), np.std(metrics_history['precision'])),
    'Recall': (np.mean(metrics_history['recall']), np.std(metrics_history['recall'])),
    'F1': (np.mean(metrics_history['f1']), np.std(metrics_history['f1']))
}

for metric, (mean, std) in results.items():
    print(f"{metric}: {mean:.3f} ± {std:.3f}")

# Save feature importance
fi_df = pd.DataFrame([
    {'Feature': feat, 'Importance': np.mean(imps)}
    for feat, imps in feature_importance.items()
]).sort_values('Importance', ascending=False)
#fi_df.to_csv("C:/Users/pnzuza/Downloads/sutton/Leave one site out/All_site_indices_feature_importance_.csv", index=False)

# --------------------------
# Spatial Prediction
# --------------------------
def loso_spatial_predict(models, shapefile_path):
    """Ensemble spatial predictions with proper type handling"""
    gdf = gpd.read_file(shapefile_path)
    
    # Convert and truncate column names
    gdf.columns = truncate_features(gdf.columns)
    
    # Initialize prediction storage
    all_preds = []
    
    for model, features in models:
        # Convert features to consistent format
        features_trunc = truncate_features(features)
        
        # Validate features using string comparisons
        missing = set(features_trunc) - set(gdf.columns)
        if missing:
            raise ValueError(f"Missing features: {list(missing)}")
        
        X_pred = gdf[features_trunc].fillna(0)
        all_preds.append(model.predict(X_pred))
    
    # Ensemble voting with type conversion
    gdf['DmgPred'] = np.round(np.mean(all_preds, axis=0)).astype(int)
    output_path = 'C:/Users/pnzuza/Downloads/sutton_stand/Leave one site out_LOSO/Ixopo_feb_damage_pred_LOSO.shp'
    gdf.to_file(output_path, driver='ESRI Shapefile')
    return output_path

# Generate predictions
try:
    pred_path = loso_spatial_predict(final_models, SHAPEFILE_PATH)
    
    # Visualize results
    gdf = gpd.read_file(pred_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(column='DmgPred', categorical=True, legend=True,
            cmap='viridis', ax=ax)
    plt.title(f"Ensemble Predictions for {TARGET_SITE}")
    plt.savefig(f"{TARGET_SITE}_predictions.png", dpi=300)
    plt.show()
    
except Exception as e:
    print(f"Spatial prediction failed: {str(e)}")