# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 06:50:56 2025

@author: pnzuza
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score)
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from collections import defaultdict

# --------------------------
# Helper Functions
# --------------------------
def truncate_features(feature_names):
    """Ensure Shapefile-compatible column names (max 10 chars)"""
    return [str(f)[:10] for f in feature_names]

def adjust_predictions(proba, valid_classes, model_classes):
    """Exclude missing classes from predictions using probability masking"""
    valid_mask = np.isin(model_classes, valid_classes)
    adjusted_proba = proba.copy()
    adjusted_proba[:, ~valid_mask] = -np.inf
    return model_classes[np.argmax(adjusted_proba, axis=1)]

# --------------------------
# Data Preparation
# --------------------------
# Load and prepare training data
data = pd.read_csv('E:/Final/Histogram matching modelling/10 band_histogram_matching/Combined_histogram_normalisation_undersampling.csv')
X = data.drop(columns=['Site', 'Damage'])
y = data['Damage']
sites = data['Site']

# Enforce Shapefile-compatible feature names
X.columns = truncate_features(X.columns)
original_features = X.columns.tolist()

# Track all classes present in training data
all_classes = np.sort(y.unique())

# --------------------------
# Model Training & Evaluation
# --------------------------
param_space = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'n_estimators': Integer(100, 1000),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 5)
}

num_iterations = 5
site_performance = defaultdict(lambda: defaultdict(list))

for iter_num in range(num_iterations):
    print(f"\n--- Iteration {iter_num+1}/{num_iterations} ---")
    
    # Train-test split
    X_train, X_test, y_train, y_test, sites_train, sites_test = train_test_split(
        X, y, sites, test_size=0.3, random_state=iter_num, stratify=y)
    
    # Bayesian Optimization
    bayes_search = BayesSearchCV(
        XGBClassifier(),
        param_space,
        n_iter=50,
        cv=StratifiedKFold(5, shuffle=True, random_state=iter_num),
        n_jobs=-1,
        scoring='accuracy',
        random_state=iter_num
    )
    bayes_search.fit(X_train, y_train)
    
    # Feature Selection
    rfecv = RFECV(
        estimator=bayes_search.best_estimator_,
        step=2,
        cv=StratifiedKFold(10, shuffle=True, random_state=iter_num),
        scoring='accuracy'
    )
    rfecv.fit(X_train, y_train)
    selected_features = X_train.columns[rfecv.support_].tolist()
    
    # Final Model
    final_model = bayes_search.best_estimator_
    final_model.fit(X_train[selected_features], y_train)
    
    # Per-Site Evaluation with Class Adjustment
    for site in sites.unique():
        site_mask = (sites_test == site)
        if not site_mask.any():
            continue
            
        X_site = X_test.loc[site_mask, selected_features]
        y_site = y_test.loc[site_mask]
        
        # Get valid classes for this site
        site_classes = y_site.unique()
        
        # Adjusted predictions
        proba = final_model.predict_proba(X_site)
        y_pred = adjust_predictions(proba, site_classes, final_model.classes_)
        
        # Store metrics
        site_performance[site]['accuracy'].append(accuracy_score(y_site, y_pred))
        site_performance[site]['precision'].append(precision_score(y_site, y_pred, average='macro', zero_division=0))
        site_performance[site]['recall'].append(recall_score(y_site, y_pred, average='macro', zero_division=0))
        site_performance[site]['f1'].append(f1_score(y_site, y_pred, average='macro', zero_division=0))

# Save performance results
performance_df = pd.DataFrame([
    {
        'Site': site,
        'Accuracy': np.mean(metrics['accuracy']),
        'Precision': np.mean(metrics['precision']),
        'Recall': np.mean(metrics['recall']),
        'F1': np.mean(metrics['f1']),
        'n_Iterations': len(metrics['accuracy'])
    } for site, metrics in site_performance.items()
])
performance_df.to_csv('C:/Users/pnzuza/Downloads/overall_model_site_performance_revised_new.csv', index=False)

# --------------------------
# Spatial Prediction Function
# --------------------------
def spatial_predict(site_name, shapefile_path, model, features, valid_classes=None):
    """
    Make spatial predictions on shapefile data
    Args:
        site_name: Output filename prefix
        shapefile_path: Path to input shapefile
        model: Trained model object
        features: List of feature names used in training
        valid_classes: Optional list of allowed classes (exclude missing classes)
    """
    # Load and validate shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf.columns = truncate_features(gdf.columns)
    features_trunc = truncate_features(features)
    
    # Validate features
    missing_features = set(features_trunc) - set(gdf.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Prepare prediction data
    X_pred = gdf[features_trunc].fillna(0)
    
    # Generate predictions with class validation
    proba = model.predict_proba(X_pred)
    valid_classes = valid_classes if valid_classes is not None else model.classes_
    predictions = adjust_predictions(proba, valid_classes, model.classes_)
    
    # Save results
    gdf['DmgPred'] = predictions
    output_path = "C:/Users/pnzuza/Downloads/Sutton_stand/overall model/"
    gdf.to_file(output_path, driver='ESRI Shapefile')
    
    return output_path

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Configuration
    TARGET_SITE = "Sutton_Feb"
    SHAPEFILE_PATH = "C:/Users/pnzuza/Downloads/Sutton_stand/overall model/Trees_overall_model.shp"
    EXCLUDE_CLASSES = [3]  # Classes to exclude for this site
    
    # Generate valid classes list
    valid_classes = [c for c in all_classes if c not in EXCLUDE_CLASSES]
    
    try:
        # Make predictions
        pred_shapefile = spatial_predict(
            site_name=TARGET_SITE,
            shapefile_path=SHAPEFILE_PATH,
            model=final_model,
            features=selected_features,
            valid_classes=valid_classes
        )
        
        # Visualize results
        gdf = gpd.read_file(pred_shapefile)
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(column='DmgPred', categorical=True, legend=True,
                cmap='viridis', ax=ax)
        plt.title(f"Damage Predictions for {TARGET_SITE}")
        plt.savefig(f"{TARGET_SITE}_predictions.png", dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Prediction failed: {str(e)}")