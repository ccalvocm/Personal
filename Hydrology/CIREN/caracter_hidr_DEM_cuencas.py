#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:08:56 2021

@author: faarrosp
"""

import rasterio as rio
import fiona
import numpy as np
import rasterio.mask
import os
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon

def raster_mask(srcfile, dstfile, maskfile):
    with fiona.open(maskfile, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]
        
        with rio.open(srcfile) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                                nodata=20000)
            out_meta = src.meta
            out_meta.update({'driver': 'GTiff',
                            'height': out_image.shape[1],
                            'width': out_image.shape[2],
                            'transform': out_transform,
                            'nodata': 20000})
            with rio.open(dstfile, 'w', **out_meta) as dest:
                dest.write(out_image)
                
def raster_mask_by_geom(srcfile, dstfile, geometry):
    
    shapes = [geometry]
        
    with rio.open(srcfile) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                            nodata=20000)
        # possible drivers: JP2OpenJPEG, GTiff
        out_meta = src.meta
        out_meta.update({'driver': 'GTiff',
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        #'dtype': 'float32',
                        'transform': out_transform,
                        'nodata': 20000})
        #out_image = out_image.astype(np.float32)
        #out_image[out_image==0]=0
        
        with rio.open(dstfile, 'w', **out_meta) as dest:
            dest.write(out_image)
            
def raster_resample(src,dst,upscale_factor):
    '''
    Resamples a source raster into dest raster by upscale_factor
    src: str, path to a raster file rasterio.open()
    dst: str, path to resampled raster file rasterio.open()
    upscale_factor: float, >1 for more detail (upscale)
    '''
    with rio.open(src) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        out_meta = dataset.meta
        out_meta.update({'driver': 'GTiff',
                            'height': data.shape[1],
                            'width': data.shape[2],
                            'transform': transform})
        with rio.open(dst, 'w', **out_meta) as dest:
            dest.write(data)
            


#%% Define paths and import shapefiles
            
folder_catch = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH',
                          'Cuencas')
folder_camels = os.path.join('..', 'SIG', 'Cuencas_CAMELS')
folder_saverasters = os.path.join('.', 'outputs', 'caracter_hidr', 'rayshader',
                                  'rasters')

macrocuencasfp = os.path.join(folder_catch, 'Cuencas_DARH_2015_AOHIA_ZC.geojson')
macrocuencas = gpd.read_file(macrocuencasfp)

camelsfp = os.path.join(folder_camels,
                        'Cuencas_cabecera_MaipoRapelMataquitoMaule.shp')
camels = gpd.read_file(camelsfp)
camels = camels.to_crs('epsg:32719')

rasterfolder = os.path.join('..', 'Etapa 1 y 2', 'DEM')
rasterfp = os.path.join(rasterfolder, 'DEM Alos 5a a 8a mar.jp2')
rasterRSfp = os.path.join(folder_saverasters, 'DEM_AOHIA_ZC.tif')

# resample the raster
# raster_resample(rasterfp, rasterRSfp, 0.1)

#------------- Mask macrocuencas
srcfp = os.path.join(folder_saverasters, 'DEM_AOHIA_ZC.tif')
dstfp = os.path.join(folder_saverasters, 'macrocuencas.tif')
x1 = macrocuencas.bounds['minx'].min()
y1 = macrocuencas.bounds['miny'].min()
x2 = macrocuencas.bounds['maxx'].max()
y2 = macrocuencas.bounds['maxy'].max()
poly = Polygon([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])

# first resample the original raster
# raster_resample(rasterfp, rasterRSfp, 0.1)

# now mask the resampled raster
# raster_mask_by_geom(rasterRSfp, dstfp, poly)


#------------- mask macrocuencas por separado
# for idx in macrocuencas.index:
#     geometry = macrocuencas.loc[idx,'geometry']
#     x1 = geometry.bounds[0]
#     y1 = geometry.bounds[1]
#     x2 = geometry.bounds[2]
#     y2 = geometry.bounds[3]
#     name = macrocuencas.loc[idx, 'NOM_CUENCA']
#     srcfp = os.path.join(folder_saverasters, 'macrocuencas.tif')
#     dstfp = os.path.join(folder_saverasters, name + '.tif')
#     poly = Polygon([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
#     raster_mask_by_geom(srcfp, dstfp, poly)

#-------------- mask camels cuencas por separado
for idx in camels.index:
    geometry = camels.loc[idx,'geometry']
    x1 = geometry.bounds[0]
    y1 = geometry.bounds[1]
    x2 = geometry.bounds[2]
    y2 = geometry.bounds[3]
    name = camels.loc[idx, 'gauge_name']
    srcfp = os.path.join(folder_saverasters, 'macrocuencas.tif')
    dstfp = os.path.join(folder_saverasters, name + '.tif')
    poly = Polygon([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
    raster_mask_by_geom(srcfp, dstfp, geometry)
    
#%% 
