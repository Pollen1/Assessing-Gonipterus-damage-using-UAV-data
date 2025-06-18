import geopandas as gpd
from osgeo import ogr, gdal
from split_ortho_ import split_ortho_mask, split_ortho_raster

# fn_shp = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\row_annotation_3.shp"  # path to raw shapefile
# fn_ras = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\20210614_Ruffus_MS_NaDir_Clipped.tif" # path to raster for extent
# out_net = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\ruffus_annotation.tiff"
# split_ortho_path = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\split_ortho"
# split_mask_path = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\split_mask"

def rasterize_shp_layer_segmentation(fn_shp, fn_ras, out_net, split_ortho_path, split_mask_path):
    shape = gpd.read_file(fn_shp)

    # Assign numeric categories to each class
    lib = {'bg': 1, 'vine': 2, 'vine': 2, 'weed': 3, 'Weed': 3}
    shape['id'] = shape['Type'].map(lib)
    shape.to_file(fn_shp) # now the shapefile 'id' columns has a numeric value for each class

    # Rasterize shapefile
    ras_ds = gdal.Open(fn_ras)
    vec_ds = ogr.Open(fn_shp)

    lyr = vec_ds.GetLayer()

    geot = ras_ds.GetGeoTransform()

    drv_tiff = gdal.GetDriverByName("GTiff")

    chn_ras_ds = drv_tiff.Create(out_net, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
    chn_ras_ds.SetGeoTransform(geot)

    gdal.RasterizeLayer(chn_ras_ds, [1], lyr, options=['ATTRIBUTE=id'])
    chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0)
    chn_ras_ds = None

    split_ortho_raster(fn_ras, split_ortho_path, 250)
    split_ortho_mask(out_net, split_mask_path, 250)

# fn_shp = r"E:\Doctoraat\Drone\2020\Ruffus\sample locations 2.shp"
# fn_ras = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\20210614_Ruffus_MS_NaDir_Clipped.tif" # path to raster for shape
# out_net = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\sample_test.tiff"
# split_ortho_path = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\split_ortho"
# split_mask_path = r"E:\Doctoraat\Drone\2021\Ruffus\14062021\split_mask"

def rasterize_shp_layer_samples(fn_shp, fn_ras, out_net):
    # Rasterize shapefile
    ras_ds = gdal.Open(fn_ras)
    vec_ds = ogr.Open(fn_shp)

    lyr = vec_ds.GetLayer()

    geot = ras_ds.GetGeoTransform()

    drv_tiff = gdal.GetDriverByName("GTiff")

    chn_ras_ds = drv_tiff.Create(out_net, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
    chn_ras_ds.SetGeoTransform(geot)

    gdal.RasterizeLayer(chn_ras_ds, [1], lyr, options=['ATTRIBUTE=SampleN'])
    chn_ras_ds.GetRasterBand(1).SetNoDataValue(0.0)
    chn_ras_ds = None

    # split_ortho_raster(fn_ras, split_ortho_path, 250)
    # split_ortho_mask(out_net, split_mask_path, 250)

### END SCRIPT ###