import os
from shp_to_mask import *
from RF_segmentation import *

"""
This script combines several script into one neat workflow to go from the Agisoft output orthophoto's to a
segmented image on which samples can be annotated.
"""



# Step 1: Clip rasters in QGIS so that unnecessary data is removed

# Step 2: Create a new shapefile and annotate the rows.
# Create +- 50 samples of the classes 'Vine' and 'bg', add a third 'weed' class if needed.

# Step 3: paste the correct paths here


fn_shp = "G:/Drone images/220504_Hodgsons/Metadata/220504_Hodgsons_25meter_data/220504_Hodgsons_25m_train.shp"  # path to raw shapefile
fn_ras = 'G:/Drone images/220504_Hodgsons/Analysis/Hodgsons_raw_ortho.tif' # path to clipped raster for shape
out_net = "G:/Drone images/220504_Hodgsons/Metadata/220504_Hodgsons_25meter_data/220504_Hodgsons_image_segmentation/annotation.tif"  # path to output file
split_ortho_path = "G:/Drone images/220504_Hodgsons/Metadata/220504_Hodgsons_25meter_data/220504_Hodgsons_image_segmentation"  # path to store tiled images
split_mask_path = "G:/Drone images/220504_Hodgsons/Metadata/220504_Hodgsons_25meter_data/220504_Hodgsons_image_segmentation/split_mask"  # path to store tiled masks
segment_path = "G:/Drone images/220504_Hodgsons/Metadata/220504_Hodgsons_25meter_data/220504_Hodgsons_image_segmentation/split_mask"  # path to store segmented orthophoto


# Rasterize the annotation shapefile
rasterize_shp_layer_segmentation(fn_shp, fn_ras, out_net, split_ortho_path, split_mask_path)

# Use the rasterized annotation to train RF model
segment_rows(fn_ras, out_net, save_path=segment_path, show=False)


#%%%

#from osgeo import gdal
#import geopandas as gpd
#gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')
#in_vector = 'D:/Drone images/221130_melmoth_sunny/segmentation/221130_melmoth_training_f2.shp'
#gdf = gpd.read_file(in_vector)


#%%%
 # whatever operation you now run should work