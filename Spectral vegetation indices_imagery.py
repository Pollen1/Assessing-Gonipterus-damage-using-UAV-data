# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 20:26:53 2025

@author: pnzuza
"""

import rasterio
import numpy as np
import os

# Configuration
INPUT_STACK_PATH = 'D:/Drone images/220414_Sutton/Exports/220414_sutton_multi_ortho_final.tif'  # Update with your path
OUTPUT_DIR = 'C:/Users/pnzuza/Downloads/UAV results/Indices'           # Update with your path

# Band order in the input stack (MicaSense Dual RedEdge MX)
BAND_ORDER = [
    'coastal_blue444',#Band 1
    'blue475',    # Band 2
    'green531',   # Band 3
    'green560',   # Band 4
    'red650',     #Band5
    'red668',     # Band 6
    're705',      # Band 7
    're717',      # Band 8
    're740',      # Band 9
    'nir842'      # Band 10
]

def safe_divide(numerator, denominator):
    """Safe division with zero/infinity handling"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[denominator == 0] = np.nan
        result[np.isinf(result)] = np.nan
    return result

def calculate_indices(bands):
    """Calculate all vegetation indices"""
    indices = {}
    
    # Chlorophyll Index (CI)
    indices['ci'] = safe_divide(bands['nir842'], bands['re717']) - 1
    
    # Enhanced Vegetation Index (EVI)
    evi_num = 2.5 * (bands['nir842'] - bands['red668'])
    evi_den = bands['nir842'] + 6*bands['red668'] - 7.5*bands['blue475'] + 1
    indices['evi'] = safe_divide(evi_num, evi_den)
    
    # Green NDVI (GNDVI)
    indices['gndvi'] = safe_divide(
        bands['nir842'] - bands['green560'], 
        bands['nir842'] + bands['green560']
    )
    
    # Modified Chlorophyll Absorption Ratio Index (MCARI)
    mcari_num = (bands['re705'] - bands['red668']) - 0.2*(bands['re705'] - bands['green560'])
    indices['mcari'] = mcari_num * safe_divide(bands['re705'], bands['red668'])
    
    # MTCI
    mtci_num = bands['re740'] - bands['re705']
    mtci_den = bands['re705'] - bands['red668']
    indices['mtci'] = safe_divide(mtci_num, mtci_den)
    
    # NDRE740
    indices['ndre740'] = safe_divide(
        bands['re740'] - bands['re717'], 
        bands['re740'] + bands['re717']
    )
    
    # NDRE842
    indices['ndre842'] = safe_divide(
        bands['nir842'] - bands['re717'], 
        bands['nir842'] + bands['re717']
    )
    
    # NDVI
    indices['ndvi'] = safe_divide(
        bands['nir842'] - bands['red668'], 
        bands['nir842'] + bands['red668']
    )
    
    # Photochemical Reflectance Index (PRI)
    indices['pri'] = safe_divide(
        bands['green560'] - bands['green531'], 
        bands['green560'] + bands['green531']
    )
    
    # Red Edge Position (REP)
    rep_num = (bands['red668'] + bands['re740']/2) - bands['re705']
    rep_den = bands['re740'] - bands['re705']
    indices['rep'] = bands['re705'] + 40 * safe_divide(rep_num, rep_den)
    
    # RE750/700 Ratio
    indices['re750_700'] = safe_divide(bands['re740'], bands['re705'])
    
    # TCARI/OSAVI
    tcari = 3 * (bands['re705'] - bands['red668']) - 0.2 * (bands['re705'] - bands['green560'])
    tcari *= safe_divide(bands['re705'], bands['red668'])
    osavi = 1.16 * safe_divide(
        bands['nir842'] - bands['red668'], 
        bands['nir842'] + bands['red668'] + 0.16
    )
    indices['tcari_osavi'] = safe_divide(tcari, osavi)
    
    # Anthocyanin Reflectance Index (ARI)
    indices['ARI'] = safe_divide(1, bands['green560']) - safe_divide(1, bands['re717'])
    
    # Normalized Green-Red Difference Index (NGRDI)
    indices['NGRE'] = safe_divide(
        bands['green560'] - bands['red668'], 
        bands['green560'] + bands['red668']
    )
    
    # Chlorophyll Index Green (Clgreen)
    indices['Clgreen'] = safe_divide(bands['nir842'], bands['red668']) - 1
    
    # Ratio Vegetation Index (RVI)
    indices['RVI'] = safe_divide(bands['nir842'], bands['red668'])
    
    # Optimized Soil Adjusted Vegetation Index (OSAVI)
    indices['OSAVI'] = 1.16 * safe_divide(
        bands['nir842'] - bands['red668'], 
        bands['nir842'] + bands['red668'] + 0.16
    )
    
    # Difference Vegetation Index (DVI)
    indices['DVI'] = bands['nir842'] - bands['red668']
    
    # Chlorophyll Index Red Edge (CIRE)
    indices['CIRE'] = safe_divide(bands['nir842'], bands['re717'])
    
    # Green Optimized SAVI (GOSAVI)
    indices['GOSAVI'] = 1.16 * safe_divide(
        bands['nir842'] - bands['green560'], 
        bands['nir842'] + bands['green560'] + 0.16
    )
    
    # Soil Adjusted Vegetation Index (SAVI)
    indices['SAVI'] = 1.5 * safe_divide(
        bands['nir842'] - bands['red668'], 
        bands['nir842'] + bands['red668'] + 0.5
    )
    
    # Structure Insensitive Pigment Index (SIPI)
    indices['SIPI'] = safe_divide(
        bands['nir842'] - bands['blue475'], 
        bands['nir842'] + bands['red668']
    )
    
    # Red Edge Ratio Vegetation Index (RERVI)
    indices['RERVI'] = safe_divide(bands['nir842'], bands['re717'])
    
    # Simple Ratio (SR)
    indices['SR'] = safe_divide(bands['red668'], bands['nir842'])
    
    # Leaf Chlorophyll Index (LCI)
    indices['LCI'] = safe_divide(
        bands['nir842'] - safe_divide(bands['re717'], bands['nir842'] + bands['red668']),
        1
    )
    
    # Normalized Pigment Chlorophyll Index (NPCI)
    indices['npci'] = safe_divide(
        bands['red668'] - safe_divide(bands['blue475'], bands['red668'] + bands['blue475']),
        1
    )
    
    # Simple Ratio Pigment Index (SRPI)
    indices['srpi'] = safe_divide(bands['blue475'], bands['red668'])
    
    # Nitrogen Reflectance Index (NRI)
    indices['nri'] = safe_divide(
        bands['red668'], 
        bands['red668'] + bands['green560'] + bands['blue475']
    )
    
    # Green Ratio Vegetation Index (GRVI)
    indices['grvi'] = safe_divide(
        bands['green560'] - safe_divide(bands['red668'], bands['green560'] + bands['red668']),
        1
    )
    
    # Plant Senescence Reflectance Index (PSRI)
    indices['psri'] = safe_divide(
        bands['red668'] - safe_divide(bands['green560'], bands['re717']),
        1
    )
    
    return indices

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with rasterio.open(INPUT_STACK_PATH) as src:
        # Read entire stack and validate
        if src.count != len(BAND_ORDER):
            raise ValueError(f"Expected {len(BAND_ORDER)} bands, found {src.count}")
            
        stack = src.read().astype(np.float32)
        profile = src.profile.copy()
        meta = src.meta
        
        # Extract bands to dictionary
        bands = {name: stack[i] for i, name in enumerate(BAND_ORDER)}
        
        # Calculate all indices
        indices = calculate_indices(bands)
        
        # Save individual indices
        index_profile = profile.copy()
        index_profile.update(count=1, dtype='float32', nodata=np.nan)
        
        for name, data in indices.items():
            output_path = os.path.join(OUTPUT_DIR, f'{name}.tif')
            with rasterio.open(output_path, 'w', **index_profile) as dst:
                dst.write(data, 1)
                dst.update_tags(**src.tags())
        
        # Create new layer stack
        new_stack = np.concatenate([
            stack,  # Original bands
            np.array(list(indices.values()))  # Vegetation indices
        ])
        
        # Update metadata for stack
        meta.update(count=new_stack.shape[0], dtype='float32', nodata=np.nan)
        
        # Save new stack
        stack_path = os.path.join(OUTPUT_DIR, 'full_layer_stack.tif')
        with rasterio.open(stack_path, 'w', **meta) as dst:
            dst.write(new_stack)
            dst.update_tags(**src.tags())

    print(f"Processed {len(indices)} vegetation indices")
    print(f"New stack contains {new_stack.shape[0]} bands")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()