#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:44:41 2021

@author: faarrosp
"""

import geopandas as gpd
import contextily as ctx
import os
import pandas as pd
import rasterio as rio
from matplotlib import pyplot as plt
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
import rasterio.mask
from rasterio.plot import show
from rasterio.enums import Resampling
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

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

def raster_reproject(srcfile, dstfile, crs_dst):
    with rio.open(srcfile) as src:
        transform, width, height = calculate_default_transform(
            src.crs, crs_dst, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs_dst,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(dstfile, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs_dst,
                    resampling=Resampling.nearest)

def raster_mask(srcfile, dstfile, maskfile):
    with fiona.open(maskfile, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]
        
        with rio.open(srcfile) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                                nodata=9999)

            out_meta = src.meta
            out_image.astype(np.float32)
            out_image[out_image==9999]=np.nan
            out_meta.update({'driver': 'GTiff',
                            'height': out_image.shape[1],
                            'width': out_image.shape[2],
                            'dtype': 'float32',
                            'transform': out_transform,
                            'nodata': np.nan})
            with rio.open(dstfile, 'w', **out_meta) as dest:
                dest.write(out_image)

def raster_mask_by_geom(srcfile, dstfile, geometry):
    
    shapes = [geometry]
        
    with rio.open(srcfile) as src:
        meta=src.meta
        
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True,
                            nodata=0)
        # possible drivers: JP2OpenJPEG, GTiff
        out_meta = src.meta
        out_meta.update({'driver': 'JP2OpenJPEG',
                        'height': out_image.shape[1],
                        'width': out_image.shape[2],
                        #'dtype': 'float32',
                        'transform': out_transform,
                        'nodata': 0})
        #out_image = out_image.astype(np.float32)
        out_image[out_image==0]=0
        
        with rio.open(dstfile, 'w', **out_meta) as dest:
            dest.write(out_image)



def filter_gdf_by_gdf(gdf1,gdf2,crs):
    gdf1 = gdf1.to_crs(crs)
    gdf2 = gdf2.to_crs(crs)
    gdf1 = gpd.sjoin(gdf1,gdf2, how = 'inner')
    return gdf1

# paths and folders -- stations
folder_stations = os.path.join('..', 'Etapa 1 y 2', 'datos', 'datosDGA', 'Q')
sta_maipo = os.path.join(folder_stations, 'Maipo', 'Maipo_cr2corregido_Q.xlsx')
gdf_maipo = pd.read_excel(sta_maipo, sheet_name='info estacion')

# paths and folders -- subcatchment
folder_catch = os.path.join('..','Etapa 1 y 2','GIS','Cuencas_DARH')
gdf_subcatch = os.path.join(folder_catch,'Subcuencas',
                         'SubCuencas_DARH_2015_AOHIA_ZC.geojson')
gdf_subcatch = gpd.read_file(gdf_subcatch)

# paths and folders -- catchment
gdf_catch = os.path.join(folder_catch, 'Cuencas',
                         'Cuencas_DARH_2015_AOHIA_ZC.geojson')
gdf_catch = gpd.read_file(gdf_catch)

# ------- Determine the catchments that do not have flow-gauging stations
gdf_sta = gpd.GeoDataFrame(gdf_maipo, crs = 'epsg:32719',
                           geometry =gpd.points_from_xy(gdf_maipo['UTM Este'],
                                                        gdf_maipo['UTM Norte']))

dst_folder = os.path.join('./outputs/no_controladas')

def extract_no_controlled(stations_gdf, subcatchment_gdf, epsg):
    # -------- Filter by
    # subset_scatch_lvl1 = gdf_subcatch[gdf_subcatch['COD_CUENCA'].isin(['1300'])]
    # subset_scatch_lvl2 = filter_gdf_by_gdf(subset_scatch_lvl1,
    #                                   gdf_sta, 'epsg:32719')
    # no_controlled = subset_scatch_lvl1.drop(subset_scatch_lvl2.index.unique())
    
    # subset_catch = filter_gdf_by_gdf(gdf_catch, gdf_sta, 'epsg:32719')
    
    
    # # fig, ax = plt.subplots()
    # # subset_scatch_lvl2.plot(ax=ax,ec='black',fc='green',lw=1,ls='--')
    # # no_controlled.plot(ax=ax, ec = 'black', fc='red', lw=1)
    # # subset_catch.plot(ax=ax,fc='none')
    # # gdf_sta.plot(ax=ax, color='yellow')
    
    # # # -------- Save the no_controlled geometries individually
    # dst_folder = os.path.join('./outputs/no_controladas')
    # no_controlled.reset_index(inplace=True, drop=True)
    
    # for i in no_controlled.index:
    #     filename = os.path.join(dst_folder, no_controlled.loc[i,'COD_DGA']) + '.geojson'
    #     no_controlled[i:i+1].to_file(filename, driver='GeoJSON')
    pass

#------------------------------
# generate the masks
# src_dem = os.path.join('..', 'Etapa 1 y 2', 'DEM', 'DEM Alos 5a a 8a mar.jp2')
# dst_folder = os.path.join('./outputs/no_controladas/DEM')

# for idx in gdf_catch.index:
#     name = gdf_catch.loc[idx,'NOM_CUENCA']
#     geom = gdf_catch.loc[idx, 'geometry']
#     dst_file = os.path.join(dst_folder, name + '.jp2')
#     raster_mask_by_geom(src_dem, dst_file, geom)
#------------------------------




from pysheds.grid import Grid
demfolder = os.path.join('.' ,'outputs', 'no_controladas', 'DEM')
demfile = os.path.join(demfolder, 'Río Maipo.jp2')
demres = os.path.join(demfolder, 'Río Maipo_resampled.jp2')

# raster_resample(demfile, demres, 0.1)
# files = os.listdir(demfolder)
# file = files[2]
#%%
# for file in files[2:3]:
# filename = os.path.join(demfolder,'Río Maipo_resampled.jp2')
# # # Instantiate grid from raster
# grid = Grid.from_raster(filename, data_name='dem')
# # plt.imshow(grid.view('dem'))
# # # Resolve flats and compute flow directions
# grid.resolve_flats(data='dem', out_name='inflated_dem')
# grid.flowdir('inflated_dem', out_name='dir')
# plt.imshow(grid.view('inflated_dem'))

# depressions = grid.detect_depressions('dem')
# grid.fill_depressions(data='dem', out_name='flooded_dem')
# depressions = grid.detect_depressions('flooded_dem')
# plt.imshow(grid.view('flooded_dem'))

# flats = grid.detect_flats('flooded_dem')
# grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
# flats = grid.detect_flats('inflated_dem')
# print(flats.any())
# plt.imshow(grid.view('inflated_dem'))

#%%

