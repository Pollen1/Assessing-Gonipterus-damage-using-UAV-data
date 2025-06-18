import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm

def create_tiles(img, tile_size):
    windows = {}
    for y in np.arange(0, img.shape[0], tile_size):
        for x in np.arange(0, img.shape[1], tile_size):
            windows.update({f"{y}_{x}": Window(col_off=max(0, x),
                                               row_off=max(0, y),
                                               width=min(img.shape[1]-x, tile_size),
                                               height=min(img.shape[0]-y, tile_size))})
    return windows



def split_ortho_raster(fp, save_path, tile_size):

#fp=r'E:\Doctoraat\Drone\2021\Champagne\Mumm\18072021\20210718_Mumm1_MS_Nadir_multi.tif'
    img = rasterio.open(fp)
    tiles=create_tiles(img, tile_size)

    for tile in tqdm(tiles):
        red, nir=(img.read(k, window=tiles[tile]) for k in (4, 6))
        if not red.min() == red.max():
            im = (nir - red) / (nir + red) # NDVI in range -1 to 1
            im += 1 # NDVI in range 0-2
            im[im > np.nanmax(im)]=np.nanmax(im)
            im[im < np.nanmin(im)]=0
            im[np.isnan(im)]=0
            # im[im > 1]=-1
            im *= 32767.5
            im = Image.fromarray(im)
            im.save(os.path.join(save_path,f"{tile}.tif"))

    img.close()


def split_ortho_mask(fp, save_path, tile_size):
    img=rasterio.open(fp)
    tiles=create_tiles(img, tile_size)

    for tile in tqdm(tiles):
        im = img.read(1, window=tiles[tile])
        im=Image.fromarray(im)
        im.save(os.path.join(save_path, f"{tile}.tif"))

    img.close()

