#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:20:43 2021

@author: faarrosp
"""

import geopandas as gpd
import contextily as ctx
import os
import rasterio as rio
from matplotlib import pyplot as plt
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona
import rasterio.mask
from rasterio.plot import show
from rasterio.enums import Resampling


#------------- Paths and files
folder_BH = os.path.join('..', 'SIG', 'REH5796_Proyecto_SIG_BH1', '4_Anexos',
                      '3_Archivos_raster', '3_Forzantes_1985-2015')
file = os.path.join(folder_BH, 'Pma_85-15_Chile.tif')
filereproj = os.path.join(folder_BH, 'Pma_85-15_Chile_epsg32719.tif')
folder_catch = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH',
                          'Cuencas')
catchments = os.path.join(folder_catch, 'Cuencas_DARH_2015_AOHIA_ZC.geojson')

raster_masked = os.path.join(folder_BH,
                             'Pma_85-15_Chile_epsg32719_AOHIA_ZC.tif')
raster_rsed = os.path.join(folder_BH,
                           'Pma_85-15_Chile_epsg32719_AOHIA_ZC_rs.tif')

folder_saved_imgs = os.path.join('./','outputs', 'caracter_hidr', 'pp')
folder_camels = os.path.join('..', 'SIG', 'Cuencas_CAMELS')

catchments_camels = os.path.join(folder_camels,
                                 'Cuencas_cabecera_MaipoRapelMataquitoMaule.shp')
#------------- Import raster with rasterio
# dataset = rio.open(file)
# plt.imshow(dataset.read(1), cmap = 'Blues')
# plt.show()
# dataset.close()


#%%
'''
#------------- Reproject the source (1 time)

dst_crs = 'EPSG:32719'
with rio.open(file) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(filereproj, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
            '''
'''            
#------------- Clip the raster with the catchment shapefile
folder_catch = os.path.join('..', 'Etapa 1 y 2', 'GIS', 'Cuencas_DARH',
                          'Cuencas')
catchments = os.path.join(folder_catch, 'Cuencas_DARH_2015_AOHIA_ZC.geojson')

raster_masked = os.path.join(folder_BH,
                             'Pma_85-15_Chile_epsg32719_AOHIA_ZC.tif')

with fiona.open(catchments, 'r') as shapefile:
    shapes = [feature['geometry'] for feature in shapefile]
    
    with rio.open(filereproj) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
        out_meta.update({'driver': 'GTiff',
                         'height': out_image.shape[1],
                         'width': out_image.shape[2],
                         'transform': out_transform})
        with rio.open(raster_masked, 'w', **out_meta) as dest:
            dest.write(out_image)'''

#%%
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

ext_img = 'pdf'
fs = (8.5,11)
fc = 'none'
ls = '--'
ec = 'black'
crs = 'EPSG:32719'
colors = 'blue'
provider = ctx.providers.Esri.WorldTerrain
zoom = 9
levels1 = 5
levels2 = 15

# ------------ Plot the raster and catchments
with rio.open(raster_rsed) as src:
    
    #--------- Cuencas en estudio (las 4 juntas)
    catch = gpd.read_file(catchments)
    
    fig = plt.figure(figsize = fs)
    ax = fig.add_subplot(111)
    catch.plot(ax = ax, fc=fc, ec = ec, ls=ls)
    ctx.add_basemap(ax = ax, crs = 'EPSG:32719',
                    source = provider, zoom=9)
    image = show(src, ax = ax, vmin=0, contour=False, cmap='Blues')
    norm = Normalize(vmin=0, vmax=src.read().max())
    sm = ScalarMappable(norm = norm, cmap = 'Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = fig.colorbar(sm, cax=cax, shrink=1.0)
    cbar.ax.set_ylabel('Precipitación (mm)')
                  # colors=colors,
                  #  levels = levels1,
                  #  contour_label_kws={'fmt': '%1.0f',
                  #                    'fontsize': 'medium'})
    ax.set_title('\n'.join(['Precipitación media anual',
                         'ABHN 2017',
                         'Cuencas en estudio']))
    
    

    ax.set_xlabel('Coordenada Este UTM (m)')
    ax.set_ylabel('Coordenada Norte UTM (m)')
    
    filename = os.path.join(folder_saved_imgs,
                            'pp_media_85-15_macrocuencas' + '.' + ext_img)
    plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                pad_inches = 0)
    
    plt.close()
#%%    
   
    #---------- Cuencas en estudio (cada una por separado)
    for idx in catch.index:
        fig = plt.figure(figsize = fs)
        ax = fig.add_subplot(111)
        
        #----- Mask the precipitation according to the catchment
        geometry = catch.loc[[idx], 'geometry']
        out_image, out_transform = rasterio.mask.mask(src, geometry, crop=True,
                                                      )
        out_meta = src.meta
        out_meta.update({'driver': 'GTiff',
                          'height': out_image.shape[1],
                          'width': out_image.shape[2],
                          'transform': out_transform})
        tempfile = os.path.join(folder_saved_imgs, 'temp.tif')
        with rio.open(tempfile, 'w', **out_meta) as dest:
            dest.write(out_image)
            
        with rio.open(tempfile) as temp:
            name = catch.loc[idx,'NOM_CUENCA']
            catch.loc[[idx],'geometry'].plot(ax=ax, fc=fc, ec=ec, ls=ls)
            ctx.add_basemap(ax = ax, crs = crs, source = provider, zoom = zoom)
            show(temp, contour=False, ax = ax, cmap='Blues', vmin=0)
            norm = Normalize(vmin=0, vmax=temp.read().max())
            sm = ScalarMappable(norm = norm, cmap = 'Blues')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.5)
            cbar = fig.colorbar(sm, cax=cax, shrink=1.0)
            cbar.ax.set_ylabel('Precipitación (mm)')
            ax.set_xlabel('Coordenada Este UTM (m)')
            ax.set_ylabel('Coordenada Norte UTM (m)')
            ax.set_title('\n'.join(['Precipitación media anual',
                              'ABHN 2017',
                              'Cuenca ' + name]))
            filename = os.path.join(folder_saved_imgs, 'pp_media_85-15_' + \
                                    name + '.' + ext_img)
            plt.show()
            plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                        pad_inches = 0)
            plt.close()
    
    #---------- Cuencas CAMELS cabecera
    camels = gpd.read_file(catchments_camels)
    camels = camels.to_crs('EPSG:32719')
    
    for idx in camels.index:
        fig = plt.figure(figsize = fs)
        ax = fig.add_subplot(111)
        
        #----- Mask the precipitation according to the catchment
        geometry = camels.loc[[idx], 'geometry']
        out_image, out_transform = rasterio.mask.mask(src, geometry, crop=True,
                                                      )
        out_meta = src.meta
        out_meta.update({'driver': 'GTiff',
                          'height': out_image.shape[1],
                          'width': out_image.shape[2],
                          'transform': out_transform})
        tempfile = os.path.join(folder_saved_imgs, 'temp.tif')
        with rio.open(tempfile, 'w', **out_meta) as dest:
            dest.write(out_image)
            
        with rio.open(tempfile) as temp:
            name = camels.loc[idx,'gauge_name']
            camels.loc[[idx],'geometry'].plot(ax=ax, fc=fc, ec=ec, ls=ls)
            ctx.add_basemap(ax = ax, crs = crs, source = provider, zoom = zoom)
            show(temp, contour=False, ax = ax, cmap='Blues')
            norm = Normalize(vmin=0, vmax=temp.read().max())
            sm = ScalarMappable(norm = norm, cmap = 'Blues')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.5)
            cbar = fig.colorbar(sm, cax=cax, shrink=1.0)
            cbar.ax.set_ylabel('Precipitación (mm)')
            ax.set_xlabel('Coordenada Este UTM (m)')
            ax.set_ylabel('Coordenada Norte UTM (m)')
            ax.set_title('\n'.join(['Precipitación media anual',
                              'ABHN 2017',
                              'Cuenca ' + name]))
            filename = os.path.join(folder_saved_imgs, 'pp_media_85-15_' + \
                                    name + '.' + ext_img)
            plt.show()
            plt.savefig(filename, format=ext_img, bbox_inches = 'tight',
                        pad_inches = 0)
            plt.close()
            
            
        

#%%
'''
# ------------ Resample

with rio.open(raster_masked) as dataset:
    upscale_factor = 10
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
    with rio.open(raster_rsed, 'w', **out_meta) as dest:
        dest.write(data)
'''    