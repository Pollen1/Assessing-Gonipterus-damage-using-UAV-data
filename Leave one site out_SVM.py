# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 07:01:53 2025

@author: Polle
"""


import numpy as np
import rasterio
from rasterio.features import rasterize
from xgboost import XGBClassifier
import geopandas as gpd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime
import seaborn as sn
import matplotlib.pyplot as plt
import os

# ---- Configuration Parameters ----
img_RS = 'D:/Drone images/220414_Sutton/Exports/220414_sutton_multi_ortho_final.tif'
training_shp = 'C:/Users/pnzuza/Downloads/UAV results/Train.shp'
validation_shp = 'C:/Users/pnzuza/Downloads/UAV results/Test.shp'
attribute = 'Damage'
classification_image = 'C:/Users/pnzuza/Downloads/UAV results/Classification.tif'
results_txt = 'C:/Users/pnzuza/Downloads/UAV results/Classification_results.csv'

n_cores = -1      # Use all available cores
n_trials = 50     # Number of Optuna trials

# ---- Initialize results log ----
with open(results_txt, "w") as f:
    print("XGBoost Classification with Bayesian Optimization", file=f)
    print(f"Processing: {datetime.datetime.now()}", file=f)
    print("-------------------------------------------------", file=f)
    print(f"Image: {img_RS}", file=f)
    print(f"Training shape: {training_shp}", file=f)
    print(f"Validation shape: {validation_shp}", file=f)
    print(f"Attribute: {attribute}", file=f)

# ---- 1. Load and Prepare Data ----
def load_data(raster_path, shape_path, attribute):
    with rasterio.open(raster_path) as src:
        img = src.read()
        meta = src.meta
        transform = src.transform
        raster_crs = src.crs
        height, width = src.shape
    
    gdf = gpd.read_file(shape_path)
    
    # Reproject shapefile to raster CRS if needed
    if gdf.crs != raster_crs:
        print(f"Reprojecting {shape_path} to match raster CRS")
        gdf = gdf.to_crs(raster_crs)
    
    # Build shapes list, filtering out any invalid geometries
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]) if geom is not None]
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, dtype=np.uint16)
    
    return img, mask, meta, (height, width)

# ---- Load training data ----
X_img, training_mask, meta, (height, width) = load_data(img_RS, training_shp, attribute)

# Check for valid training pixels
if np.count_nonzero(training_mask) == 0:
    raise ValueError("No valid training samples found. Check shapefile and attribute field.")

# Prepare training data
X = np.moveaxis(X_img, 0, -1).reshape(-1, X_img.shape[0])[training_mask.flatten() > 0]
y = training_mask.flatten()[training_mask.flatten() > 0]

# Check unique labels and remap if necessary (e.g., [1, 2] to [0, 1])
unique_labels = np.unique(y)
print("Unique training labels before remapping:", unique_labels)
if set(unique_labels) == {1, 2}:
    print("Remapping training labels from [1, 2] to [0, 1]")
    y = y - 1

# ---- 2. Optuna Optimization Setup ----
def objective(trial):
    params = {
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        'n_estimators': trial.suggest_int("n_estimators", 50, 300, step=50),
        'gamma': trial.suggest_float("gamma", 0, 1),
        'subsample': trial.suggest_float("subsample", 0.5, 1),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1),
        'eval_metric': 'mlogloss',
        'n_jobs': n_cores,
        'use_label_encoder': False
    }
    
    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    return score

# ---- 3. Run Optuna Optimization ----
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=n_trials)

# ---- 4. Train Final Model with Best Parameters ----
best_params = study.best_params
best_params.update({
    'eval_metric': 'mlogloss',
    'n_jobs': n_cores,
    'use_label_encoder': False
})

model = XGBClassifier(**best_params)
model.fit(X, y)

# ---- 5. Predict Full Raster ----
full_array = np.moveaxis(X_img, 0, -1).reshape(-1, X_img.shape[0])
preds = model.predict(full_array).reshape(height, width).astype(np.uint8)

# Save classification result
with rasterio.open(classification_image, 'w', **meta) as dst:
    dst.write(preds, 1)

# ---- 6. Validation ----
_, validation_mask, _, _ = load_data(img_RS, validation_shp, attribute)
X_val = preds[validation_mask > 0]
y_val = validation_mask[validation_mask > 0]

# Remap validation labels if necessary
unique_val_labels = np.unique(y_val)
print("Unique validation labels before remapping:", unique_val_labels)
if set(unique_val_labels) == {1, 2}:
    print("Remapping validation labels from [1, 2] to [0, 1]")
    y_val = y_val - 1

# Generate validation metrics
cm = confusion_matrix(y_val, X_val)
report = classification_report(y_val, X_val)
oaa = accuracy_score(y_val, X_val)

# ---- Save results to log ----
with open(results_txt, "a") as f:
    print("\nOptimized Parameters:", file=f)
    for k, v in best_params.items():
        print(f"{k}: {v}", file=f)
    
    print("\nOptimization Trials:", file=f)
    print(f"Best trial value: {study.best_value:.4f}", file=f)
    
    print("\nValidation Metrics:", file=f)
    print(f"Overall Accuracy: {oaa:.2%}", file=f)
    print("\nConfusion Matrix:", file=f)
    print(cm, file=f)
    print("\nClassification Report:", file=f)
    print(report, file=f)

# ---- 7. Visualization ----
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(preds, cmap='jet')
plt.title('Classification Map')

plt.subplot(122)
sn.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')

validation_results_path = os.path.join(os.path.dirname(results_txt), "validation_results.png")
plt.savefig(validation_results_path)
plt.close()

print(f"Classification map saved to {classification_image}")
print(f"Validation results saved to {validation_results_path}")
print(f"Results log saved to {results_txt}")
